#include "ternary_dataset.h"
#include "ternary_json.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void set_err(char *err, size_t cap, const char *fmt, ...) {
    if (!err || cap == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, cap, fmt, ap);
    va_end(ap);
}

static char *read_whole_file(const char *path, size_t *out_len, char *err, size_t err_cap) {
    FILE *f = fopen(path, "rb");
    if (!f) { set_err(err, err_cap, "no se pudo abrir '%s'", path); return NULL; }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); set_err(err, err_cap, "no se pudo leer '%s'", path); return NULL; }
    long size = ftell(f);
    if (size < 0) { fclose(f); set_err(err, err_cap, "no se pudo leer el tamano de '%s'", path); return NULL; }
    rewind(f);
    char *buf = malloc((size_t)size + 1);
    if (!buf) { fclose(f); set_err(err, err_cap, "sin memoria"); return NULL; }
    size_t got = (size > 0) ? fread(buf, 1, (size_t)size, f) : 0;
    fclose(f);
    if (got != (size_t)size) { free(buf); set_err(err, err_cap, "error leyendo '%s'", path); return NULL; }
    buf[size] = '\0';
    *out_len = (size_t)size;
    return buf;
}

/* ===================================================================
 * Buffers de crecimiento por duplicacion: uno para el texto renderizado
 * de UNA conversacion (TextBuf), otro para la coleccion de ejemplos ya
 * terminados, separados por NUL (ExampleCollector).
 * ===================================================================*/
typedef struct { char *buf; size_t len, cap; } TextBuf;

static int tb_push(TextBuf *tb, const char *data, size_t n) {
    if (n == 0) return 1;
    if (tb->len + n > tb->cap) {
        size_t newcap = tb->cap ? tb->cap * 2 : 128;
        while (newcap < tb->len + n) newcap *= 2;
        char *nb = realloc(tb->buf, newcap);
        if (!nb) return 0;
        tb->buf = nb; tb->cap = newcap;
    }
    memcpy(tb->buf + tb->len, data, n);
    tb->len += n;
    return 1;
}
static int tb_push_str(TextBuf *tb, const char *s) { return tb_push(tb, s, strlen(s)); }

typedef struct { uint8_t *buf; size_t len, cap; size_t count; } ExampleCollector;

static int ec_push_example(ExampleCollector *ec, const char *text, size_t text_len) {
    size_t needed = ec->len + text_len + 1;  /* +1 por el separador NUL */
    if (needed > ec->cap) {
        size_t newcap = ec->cap ? ec->cap * 2 : 256;
        while (newcap < needed) newcap *= 2;
        uint8_t *nb = realloc(ec->buf, newcap);
        if (!nb) return 0;
        ec->buf = nb; ec->cap = newcap;
    }
    if (text_len) memcpy(ec->buf + ec->len, text, text_len);
    ec->len += text_len;
    ec->buf[ec->len++] = 0;
    ec->count++;
    return 1;
}

/* ===================================================================
 * Registros JSON: el formato se detecta POR REGISTRO segun la clave que traiga
 * (orden de prioridad):
 *   "chat"                       formato propio: [{"user":"..."},{"model":"..."}, ...]
 *   "messages"                   ChatML / OpenAI: [{"role":..., "content":...}, ...]
 *   "conversations"              ShareGPT: [{"from":..., "value":...}, ...] (tambien acepta role/content)
 *   "prompt" + "completion"      prompt-completion (cadenas, o listas de mensajes role/content)
 *   "instruction" [+ "input"] + "output"   Alpaca
 *   "text"                       texto plano: el ejemplo es el texto tal cual, sin marcadores
 * Todos los formatos conversacionales se renderizan igual (ver ternary_dataset.h).
 * ===================================================================*/
static const TernaryJNode *jget(const TernaryJNode *obj, const char *key) {
    const TernaryJNode *found = NULL;
    for (size_t i = 0; i < obj->n; i++)
        if (strcmp(obj->keys[i], key) == 0) found = &obj->kids[i];   /* si se repite, gana la ultima */
    return found;
}

static int ieq(const char *a, size_t an, const char *b) {
    size_t bn = strlen(b);
    if (an != bn) return 0;
    for (size_t i = 0; i < an; i++) {
        char c = a[i];
        if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
        if (c != b[i]) return 0;
    }
    return 1;
}

/* Normaliza un rol (sin distinguir mayusculas). Devuelve el nombre canonico o NULL si no es uno conocido.
 * user/human -> user; assistant/gpt/model -> assistant; system, console, terminal, archive, log, result, function_call, tool tal cual. */
static const char *canon_role(const char *r, size_t n) {
    static const char *const same[] = { "system", "console", "terminal", "archive", "log", "result", "function_call", "tool" };
    if (ieq(r, n, "user") || ieq(r, n, "human")) return "user";
    if (ieq(r, n, "assistant") || ieq(r, n, "gpt") || ieq(r, n, "model")) return "assistant";
    for (size_t i = 0; i < sizeof same / sizeof same[0]; i++) if (ieq(r, n, same[i])) return same[i];
    return NULL;
}

static int push_turn(TextBuf *tb, const char *role, size_t role_len, const char *content, size_t content_len) {
    return tb_push_str(tb, "<|") && tb_push(tb, role, role_len) && tb_push_str(tb, "|>\n")
        && tb_push(tb, content, content_len) && tb_push_str(tb, "\n");
}

/* ---- razonamiento ------------------------------------------------------------------------------------------------
 * En el texto de un turno del asistente las etiquetas <think>...</think> (tambien <thinking>) y <|think|>...<|/think|> se
 * normalizan a <|think|> ... <|/think|> (TERNARY_DATASET_REASONING_KEEP) o se quitan junto con lo que encierran
 * (TERNARY_DATASET_REASONING_DROP, para entrenar un modelo que no razona con un dataset que si). Deben estar balanceadas: un
 * bloque abierto dos veces, un cierre sin apertura o un bloque sin cerrar invalida el registro. Ademas, un mensaje del asistente
 * puede traer el razonamiento aparte, en 'reasoning_content', 'reasoning' o 'thinking' (cadena): con KEEP se escribe antes de la
 * respuesta como "<|think|>\n<razonamiento>\n<|/think|>\n\n<respuesta>" (el formato de DeepSeek-R1); con DROP se ignora. */
typedef struct { int drop_reasoning; } RenderCtx;

static const char *const THINK_FIELDS[] = { "reasoning_content", "reasoning", "thinking" };

static const TernaryJNode *reasoning_field(const TernaryJNode *obj) {
    for (size_t i = 0; i < sizeof THINK_FIELDS / sizeof THINK_FIELDS[0]; i++) {
        const TernaryJNode *n = jget(obj, THINK_FIELDS[i]);
        if (n && n->type == TJ_STR && n->str_len > 0) return n;
    }
    return NULL;
}

typedef struct { const char *tag; size_t len; int open; } ThinkTag;
static const ThinkTag THINK_TAGS[] = {
    { "<think>", 7, 1 }, { "</think>", 8, 0 }, { "<thinking>", 10, 1 }, { "</thinking>", 11, 0 },
    { "<|think|>", 9, 1 }, { "<|/think|>", 10, 0 },
};

/* Copia s[0..n) a out convirtiendo (o quitando) las etiquetas de razonamiento. 0 con mensaje en err si estan mal balanceadas. */
static int convert_think(TextBuf *out, const char *s, size_t n, int drop, char *err, size_t cap) {
    int inside = 0;
    size_t i = 0, copied = 0;                       /* s[copied..i) pendiente de copiar */
    while (i < n) {
        const char *lt = memchr(s + i, '<', n - i);
        if (!lt) break;
        i = (size_t)(lt - s);
        const ThinkTag *hit = NULL;
        for (size_t t = 0; t < sizeof THINK_TAGS / sizeof THINK_TAGS[0]; t++)
            if (THINK_TAGS[t].len <= n - i && memcmp(s + i, THINK_TAGS[t].tag, THINK_TAGS[t].len) == 0) { hit = &THINK_TAGS[t]; break; }
        if (!hit) { i++; continue; }
        if (hit->open == inside) {
            set_err(err, cap, inside ? "razonamiento: etiqueta de apertura dentro de un bloque ya abierto"
                                     : "razonamiento: etiqueta de cierre sin apertura");
            return 0;
        }
        if (!inside || !drop) { if (!tb_push(out, s + copied, i - copied)) { set_err(err, cap, "sin memoria"); return 0; } }
        if (!drop) {
            if (!tb_push_str(out, hit->open ? "<|think|>" : "<|/think|>")) { set_err(err, cap, "sin memoria"); return 0; }
        }
        i += hit->len;
        if (drop && !hit->open) while (i < n && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) i++;   /* espacio que seguia al bloque */
        copied = i;
        inside = hit->open;
    }
    if (inside) { set_err(err, cap, "razonamiento: bloque sin cerrar"); return 0; }
    if (!tb_push(out, s + copied, n - copied)) { set_err(err, cap, "sin memoria"); return 0; }
    return 1;
}

/* Un turno: rol conocido -> nombre canonico (desconocido tal cual). Si es del asistente se aplica lo de razonamiento. 'reasoning'
 * (puede ser NULL) es el campo aparte del mensaje. */
static int push_turn_rc(const RenderCtx *rc, TextBuf *tb, const char *role, size_t role_len, const TernaryJNode *reasoning,
                        const char *content, size_t content_len, char *err, size_t cap) {
    const char *c = canon_role(role, role_len);
    if (c) { role = c; role_len = strlen(c); }
    int ok;
    if (c && strcmp(c, "assistant") == 0) {
        TextBuf ct = {0};
        ok = 1;
        if (reasoning && !rc->drop_reasoning)
            ok = tb_push_str(&ct, "<|think|>\n") && tb_push(&ct, reasoning->str, reasoning->str_len) && tb_push_str(&ct, "\n<|/think|>\n\n");
        if (!ok) { free(ct.buf); set_err(err, cap, "sin memoria"); return 0; }
        if (!convert_think(&ct, content, content_len, rc->drop_reasoning, err, cap)) { free(ct.buf); return 0; }
        ok = push_turn(tb, role, role_len, ct.buf ? ct.buf : "", ct.len);
        free(ct.buf);
    } else {
        ok = push_turn(tb, role, role_len, content, content_len);
    }
    if (!ok) { set_err(err, cap, "sin memoria"); return 0; }
    return 1;
}

/* Lista de mensajes {rkey: rol, ckey: contenido}; agrega los turnos a tb. */
static int render_msg_list(const RenderCtx *rc, const TernaryJNode *arr, const char *rkey, const char *ckey, const char *what,
                           TextBuf *tb, size_t *count, char *err, size_t cap) {
    if (arr->type != TJ_ARR) { set_err(err, cap, "'%s' debe ser una lista", what); return 0; }
    for (size_t i = 0; i < arr->n; i++) {
        const TernaryJNode *m = &arr->kids[i];
        if (m->type != TJ_OBJ) { set_err(err, cap, "elemento de '%s' no es un objeto", what); return 0; }
        const TernaryJNode *r = jget(m, rkey), *c = jget(m, ckey);
        if (!r || !c || r->type != TJ_STR || c->type != TJ_STR) {
            if (strcmp(rkey, "role") == 0) set_err(err, cap, "mensaje sin 'role' y/o sin 'content'");
            else set_err(err, cap, "elemento de '%s' sin '%s' y/o sin '%s' (cadenas)", what, rkey, ckey);
            return 0;
        }
        if (!push_turn_rc(rc, tb, r->str, r->str_len, reasoning_field(m), c->str, c->str_len, err, cap)) return 0;
        (*count)++;
    }
    return 1;
}

/* Roles validos en "chat": user, model, system, console, terminal, archive, log, result, function_call, tool (sin distinguir mayusculas).
 * Devuelve el nombre con que se renderiza (model -> assistant) o NULL. ('human', 'gpt' y 'assistant' son de otros formatos.) */
static const char *chat_role(const char *k) {
    size_t n = strlen(k);
    if (ieq(k, n, "user")) return "user";
    if (ieq(k, n, "model")) return "assistant";
    static const char *const same[] = { "system", "console", "terminal", "archive", "log", "result", "function_call", "tool" };
    for (size_t i = 0; i < sizeof same / sizeof same[0]; i++) if (ieq(k, n, same[i])) return same[i];
    return NULL;
}

/* "chat": cada elemento es un objeto con UNA sola clave = el rol, y su valor = el texto. */
static int render_chat(const RenderCtx *rc, const TernaryJNode *arr, TextBuf *tb, size_t *count, char *err, size_t cap) {
    if (arr->type != TJ_ARR) { set_err(err, cap, "'chat' debe ser una lista"); return 0; }
    for (size_t i = 0; i < arr->n; i++) {
        const TernaryJNode *t = &arr->kids[i];
        if (t->type != TJ_OBJ || t->n != 1) {
            set_err(err, cap, "turno %zu de 'chat': debe ser un objeto con una sola clave (el rol)", i + 1); return 0;
        }
        const char *role = chat_role(t->keys[0]);
        if (!role) {
            set_err(err, cap, "turno %zu de 'chat': rol '%s' no valido (user, model, system, console, terminal, archive, log, result, function_call, tool)",
                    i + 1, t->keys[0]);
            return 0;
        }
        if (t->kids[0].type != TJ_STR) { set_err(err, cap, "turno %zu de 'chat': el texto debe ser una cadena", i + 1); return 0; }
        char e2[200];
        if (!push_turn_rc(rc, tb, role, strlen(role), NULL, t->kids[0].str, t->kids[0].str_len, e2, sizeof e2)) {
            set_err(err, cap, "turno %zu de 'chat': %s", i + 1, e2); return 0;
        }
        (*count)++;
    }
    return 1;
}

/* ShareGPT: [{"from": "human"|"gpt"|"system"|..., "value": "..."}, ...]; tambien acepta elementos {"role","content"}. */
static int render_sharegpt(const RenderCtx *rc, const TernaryJNode *arr, TextBuf *tb, size_t *count, char *err, size_t cap) {
    if (arr->type != TJ_ARR) { set_err(err, cap, "'conversations' debe ser una lista"); return 0; }
    for (size_t i = 0; i < arr->n; i++) {
        const TernaryJNode *m = &arr->kids[i];
        if (m->type != TJ_OBJ) { set_err(err, cap, "elemento de 'conversations' no es un objeto"); return 0; }
        const TernaryJNode *r = jget(m, "from"), *c = jget(m, "value");
        if (!r) { r = jget(m, "role"); c = jget(m, "content"); }
        if (!r || !c || r->type != TJ_STR || c->type != TJ_STR) {
            set_err(err, cap, "elemento %zu de 'conversations' sin ('from','value') ni ('role','content') como cadenas", i + 1);
            return 0;
        }
        if (!push_turn_rc(rc, tb, r->str, r->str_len, reasoning_field(m), c->str, c->str_len, err, cap)) return 0;
        (*count)++;
    }
    return 1;
}

/* Un lado de prompt/completion: una cadena (un turno con el rol dado; 'reasoning' = campo aparte del registro) o una lista de
 * mensajes {"role","content"}. */
static int render_side(const RenderCtx *rc, const TernaryJNode *n, const char *role, const TernaryJNode *reasoning, const char *what,
                       TextBuf *tb, size_t *count, char *err, size_t cap) {
    if (n->type == TJ_STR) {
        if (!push_turn_rc(rc, tb, role, strlen(role), reasoning, n->str, n->str_len, err, cap)) return 0;
        (*count)++;
        return 1;
    }
    if (n->type == TJ_ARR) return render_msg_list(rc, n, "role", "content", what, tb, count, err, cap);
    set_err(err, cap, "'%s' debe ser una cadena o una lista de mensajes", what);
    return 0;
}

/* Alpaca: instruction [+ input] -> turno user (input en un parrafo aparte si no esta vacio); output -> turno assistant. */
static int render_alpaca(const RenderCtx *rc, const TernaryJNode *ins, const TernaryJNode *inp, const TernaryJNode *out,
                         const TernaryJNode *reasoning, TextBuf *tb, size_t *count, char *err, size_t cap) {
    if (ins->type != TJ_STR || !out || out->type != TJ_STR) { set_err(err, cap, "Alpaca: 'instruction' y 'output' deben ser cadenas"); return 0; }
    if (inp && inp->type != TJ_STR && inp->type != TJ_NULL) { set_err(err, cap, "Alpaca: 'input' debe ser una cadena"); return 0; }
    int has_input = inp && inp->type == TJ_STR && inp->str_len > 0;
    int ok = tb_push_str(tb, "<|user|>\n") && tb_push(tb, ins->str, ins->str_len)
          && (!has_input || (tb_push_str(tb, "\n\n") && tb_push(tb, inp->str, inp->str_len)))
          && tb_push_str(tb, "\n");
    if (!ok) { set_err(err, cap, "sin memoria"); return 0; }
    if (!push_turn_rc(rc, tb, "assistant", 9, reasoning, out->str, out->str_len, err, cap)) return 0;
    *count += 2;
    return 1;
}

/* Un registro JSON ya parseado -> un ejemplo en 'ec'. */
static int render_record(const RenderCtx *rc, const TernaryJNode *rec, ExampleCollector *ec, char *err, size_t cap) {
    if (rec->type != TJ_OBJ) { set_err(err, cap, "el registro no es un objeto JSON"); return 0; }
    TextBuf tb = {0};
    size_t turns = 0;
    int ok = 0, plain = 0;
    const TernaryJNode *n, *m;
    if ((n = jget(rec, "chat"))) {
        if (!render_chat(rc, n, &tb, &turns, err, cap)) goto done;
        if (turns == 0) { set_err(err, cap, "'chat' vacio"); goto done; }
    } else if ((n = jget(rec, "messages"))) {
        if (!render_msg_list(rc, n, "role", "content", "messages", &tb, &turns, err, cap)) goto done;
        if (turns == 0) { set_err(err, cap, "objeto sin 'messages' o con 'messages' vacio"); goto done; }
    } else if ((n = jget(rec, "conversations"))) {
        if (!render_sharegpt(rc, n, &tb, &turns, err, cap)) goto done;
        if (turns == 0) { set_err(err, cap, "'conversations' vacio"); goto done; }
    } else if ((n = jget(rec, "prompt"))) {
        if (!(m = jget(rec, "completion"))) { set_err(err, cap, "'prompt' sin 'completion'"); goto done; }
        if (!render_side(rc, n, "user", NULL, "prompt", &tb, &turns, err, cap)) goto done;
        if (!render_side(rc, m, "assistant", reasoning_field(rec), "completion", &tb, &turns, err, cap)) goto done;
        if (turns == 0) { set_err(err, cap, "'prompt' y 'completion' vacios"); goto done; }
    } else if ((n = jget(rec, "instruction"))) {
        if (!render_alpaca(rc, n, jget(rec, "input"), jget(rec, "output"), reasoning_field(rec), &tb, &turns, err, cap)) goto done;
    } else if ((n = jget(rec, "text"))) {
        if (n->type != TJ_STR) { set_err(err, cap, "'text' debe ser una cadena"); goto done; }
        if (n->str_len == 0) { set_err(err, cap, "'text' vacio"); goto done; }
        if (!tb_push(&tb, n->str, n->str_len)) { set_err(err, cap, "sin memoria"); goto done; }
        plain = 1;      /* texto tal cual: sin marcadores de turno ni <|end|> */
    } else {
        set_err(err, cap, "registro sin ninguno de los campos conocidos (chat, messages, conversations, prompt/completion, instruction/output, text)");
        goto done;
    }
    if (!plain && !tb_push_str(&tb, "<|end|>\n")) { set_err(err, cap, "sin memoria"); goto done; }
    ok = ec_push_example(ec, tb.buf, tb.len);
    if (!ok) set_err(err, cap, "sin memoria");
done:
    free(tb.buf);
    return ok;
}

/* Parsea el texto de UN registro y lo agrega. */
static int parse_record(const RenderCtx *rc, const char *s, size_t len, ExampleCollector *ec, char *err, size_t cap) {
    TernaryJNode *root = ternary_jdom_parse(s, len, err, cap);
    if (!root) return 0;
    int ok = render_record(rc, root, ec, err, cap);
    ternary_jdom_free(root);
    return ok;
}

/* ===================================================================
 * Un formato por funcion
 * ===================================================================*/
static int load_text(const char *path, int whole_block, ExampleCollector *ec, char *err, size_t err_cap) {
    size_t flen;
    char *data = read_whole_file(path, &flen, err, err_cap);
    if (!data) return 0;
    int ok = 1;
    if (whole_block) {
        if (!ec_push_example(ec, data, flen)) { ok = 0; set_err(err, err_cap, "sin memoria"); }
    } else {
        size_t i = 0;
        while (i < flen) {
            size_t start = i;
            while (i < flen && data[i] != '\n') i++;
            size_t end = i;
            if (end > start && data[end - 1] == '\r') end--;  /* CRLF */
            if (end > start) {  /* saltea lineas vacias */
                if (!ec_push_example(ec, data + start, end - start)) {
                    ok = 0; set_err(err, err_cap, "sin memoria"); break;
                }
            }
            if (i < flen) i++;  /* saltar '\n' */
        }
    }
    free(data);
    return ok;
}

static int load_json_array(const char *path, int strict, const RenderCtx *rc, ExampleCollector *ec, size_t *out_skipped,
                            char *err, size_t err_cap) {
    size_t flen;
    char *data = read_whole_file(path, &flen, err, err_cap);
    if (!data) return 0;
    TernaryJsonParser p;
    ternary_json_init(&p, data, flen);
    ternary_json_skip_ws(&p);
    int ok = 1;
    if (!ternary_json_expect(&p, '[', err, err_cap)) { free(data); return 0; }
    ternary_json_skip_ws(&p);
    if (!ternary_json_match(&p, ']')) {
        for (;;) {
            /* Se delimita el elemento saltando el valor completo (un error de sintaxis JSON real hace fallar la
             * carga, strict o no: ahi no hay forma segura de seguir); despues se interpreta ese trozo como registro.
             * Si no cumple el esquema, strict=1 falla y strict=0 lo cuenta como saltado y sigue. */
            size_t start = p.pos;
            char skip_err[256];
            if (!ternary_json_skip_value(&p, skip_err, sizeof(skip_err))) {
                set_err(err, err_cap, "elemento invalido y no se pudo saltear: %s", skip_err);
                ok = 0; break;
            }
            char item_err[256];
            if (!parse_record(rc, data + start, p.pos - start, ec, item_err, sizeof(item_err))) {
                if (strict) { set_err(err, err_cap, "%s", item_err); ok = 0; break; }
                if (out_skipped) (*out_skipped)++;
            }
            ternary_json_skip_ws(&p);
            if (ternary_json_match(&p, ',')) { ternary_json_skip_ws(&p); continue; }
            break;
        }
        if (ok && !ternary_json_expect(&p, ']', err, err_cap)) ok = 0;
    }
    free(data);
    return ok;
}

static int load_jsonl(const char *path, int strict, const RenderCtx *rc, ExampleCollector *ec, size_t *out_skipped,
                       char *err, size_t err_cap) {
    size_t flen;
    char *data = read_whole_file(path, &flen, err, err_cap);
    if (!data) return 0;
    size_t i = 0, line_no = 0;
    int ok = 1;
    while (i < flen) {
        size_t start = i;
        while (i < flen && data[i] != '\n') i++;
        size_t end = i;
        if (end > start && data[end - 1] == '\r') end--;
        line_no++;
        if (i < flen) i++;

        size_t j = start;
        while (j < end && (data[j] == ' ' || data[j] == '\t')) j++;
        if (j >= end) continue;  /* linea en blanco */

        char item_err[256];
        if (!parse_record(rc, data + start, end - start, ec, item_err, sizeof(item_err))) {
            if (strict) { set_err(err, err_cap, "linea %zu: %s", line_no, item_err); ok = 0; break; }
            if (out_skipped) (*out_skipped)++;
        }
    }
    free(data);
    return ok;
}

int ternary_dataset_load(const char *path, int format, int strict,
                          uint8_t *out_buf, size_t out_cap, size_t *out_len, size_t *out_count,
                          size_t *out_skipped, char *err, size_t err_cap) {
    return ternary_dataset_load_ex(path, format, strict, 0, out_buf, out_cap, out_len, out_count, out_skipped, err, err_cap);
}

int ternary_dataset_load_ex(const char *path, int format, int strict, int flags,
                             uint8_t *out_buf, size_t out_cap, size_t *out_len, size_t *out_count,
                             size_t *out_skipped, char *err, size_t err_cap) {
    RenderCtx rc = { (flags & TERNARY_DATASET_REASONING_DROP) != 0 };
    if (!path || !out_len || !out_count) { set_err(err, err_cap, "argumentos nulos"); return 0; }
    ExampleCollector ec; memset(&ec, 0, sizeof(ec));
    size_t skipped = 0;
    int ok;
    switch (format) {
        case TERNARY_DATASET_TEXT_LINES: ok = load_text(path, 0, &ec, err, err_cap); break;
        case TERNARY_DATASET_TEXT_WHOLE: ok = load_text(path, 1, &ec, err, err_cap); break;
        case TERNARY_DATASET_JSON:       ok = load_json_array(path, strict, &rc, &ec, &skipped, err, err_cap); break;
        case TERNARY_DATASET_JSONL:      ok = load_jsonl(path, strict, &rc, &ec, &skipped, err, err_cap); break;
        default: set_err(err, err_cap, "formato desconocido %d", format); free(ec.buf); return 0;
    }
    if (!ok) { free(ec.buf); return 0; }
    *out_len = ec.len;
    *out_count = ec.count;
    if (out_skipped) *out_skipped = skipped;
    if (out_buf && out_cap >= ec.len && ec.len > 0) memcpy(out_buf, ec.buf, ec.len);
    free(ec.buf);
    return 1;
}
