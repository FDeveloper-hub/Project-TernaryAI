#include "ternary_json.h"
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

static void set_err(char *err, size_t cap, const char *fmt, ...) {
    if (!err || cap == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, cap, fmt, ap);
    va_end(ap);
}

void ternary_json_init(TernaryJsonParser *p, const char *s, size_t len) {
    p->s = s; p->len = len; p->pos = 0;
}

void ternary_json_skip_ws(TernaryJsonParser *p) {
    while (p->pos < p->len) {
        char c = p->s[p->pos];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') p->pos++;
        else break;
    }
}

int ternary_json_peek(const TernaryJsonParser *p, char *out) {
    if (p->pos >= p->len) return 0;
    *out = p->s[p->pos];
    return 1;
}

int ternary_json_match(TernaryJsonParser *p, char c) {
    char got;
    if (!ternary_json_peek(p, &got) || got != c) return 0;
    p->pos++;
    return 1;
}

int ternary_json_expect(TernaryJsonParser *p, char c, char *err, size_t err_cap) {
    if (ternary_json_match(p, c)) return 1;
    char got;
    if (ternary_json_peek(p, &got))
        set_err(err, err_cap, "se esperaba '%c' en la posicion %zu, se encontro '%c'", c, p->pos, got);
    else
        set_err(err, err_cap, "se esperaba '%c' pero el archivo termino antes (posicion %zu)", c, p->pos);
    return 0;
}

/* Buffer de bytes con crecimiento por duplicacion (mismo patron que
 * src/ternary_tokenizer.c). */
static int buf_push_bytes(char **buf, size_t *len, size_t *cap, const char *data, size_t n) {
    if (*len + n > *cap) {
        size_t newcap = *cap ? *cap * 2 : 64;
        while (newcap < *len + n) newcap *= 2;
        char *nb = realloc(*buf, newcap);
        if (!nb) return 0;
        *buf = nb; *cap = newcap;
    }
    memcpy(*buf + *len, data, n);
    *len += n;
    return 1;
}

static int utf8_encode(uint32_t cp, char *out, int *out_n) {
    if (cp <= 0x7F) { out[0] = (char)cp; *out_n = 1; }
    else if (cp <= 0x7FF) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        *out_n = 2;
    } else if (cp <= 0xFFFF) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        *out_n = 3;
    } else if (cp <= 0x10FFFF) {
        out[0] = (char)(0xF0 | (cp >> 18));
        out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[3] = (char)(0x80 | (cp & 0x3F));
        *out_n = 4;
    } else return 0;
    return 1;
}

static int parse_hex4(TernaryJsonParser *p, uint32_t *out, char *err, size_t err_cap) {
    if (p->pos + 4 > p->len) { set_err(err, err_cap, "\\u incompleto cerca de la posicion %zu", p->pos); return 0; }
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) {
        char c = p->s[p->pos + (size_t)i];
        v <<= 4;
        if (c >= '0' && c <= '9') v |= (uint32_t)(c - '0');
        else if (c >= 'a' && c <= 'f') v |= (uint32_t)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= (uint32_t)(c - 'A' + 10);
        else { set_err(err, err_cap, "escape \\u con digito hexadecimal invalido en la posicion %zu", p->pos); return 0; }
    }
    p->pos += 4;
    *out = v;
    return 1;
}

int ternary_json_parse_string(TernaryJsonParser *p, char **out, size_t *out_len,
                               char *err, size_t err_cap) {
    if (!ternary_json_expect(p, '"', err, err_cap)) return 0;
    char *buf = NULL; size_t len = 0, cap = 0;
    for (;;) {
        char c;
        if (!ternary_json_peek(p, &c)) {
            set_err(err, err_cap, "string sin cerrar (el archivo termino antes de la comilla final)");
            free(buf); return 0;
        }
        if (c == '"') { p->pos++; break; }
        if (c == '\\') {
            p->pos++;
            char esc;
            if (!ternary_json_peek(p, &esc)) {
                set_err(err, err_cap, "escape incompleto al final del archivo"); free(buf); return 0;
            }
            p->pos++;
            char lit;
            switch (esc) {
                case '"': lit = '"'; goto push_lit;
                case '\\': lit = '\\'; goto push_lit;
                case '/': lit = '/'; goto push_lit;
                case 'b': lit = '\b'; goto push_lit;
                case 'f': lit = '\f'; goto push_lit;
                case 'n': lit = '\n'; goto push_lit;
                case 'r': lit = '\r'; goto push_lit;
                case 't': lit = '\t'; goto push_lit;
                push_lit:
                    if (!buf_push_bytes(&buf, &len, &cap, &lit, 1)) {
                        set_err(err, err_cap, "sin memoria"); free(buf); return 0;
                    }
                    break;
                case 'u': {
                    uint32_t cp;
                    if (!parse_hex4(p, &cp, err, err_cap)) { free(buf); return 0; }
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        /* par subrogado: debe seguir \uDCxx..\uDFxx */
                        char c1, c2;
                        if (p->pos + 6 > p->len || p->s[p->pos] != '\\' || p->s[p->pos + 1] != 'u') {
                            set_err(err, err_cap, "par subrogado UTF-16 incompleto en la posicion %zu", p->pos);
                            free(buf); return 0;
                        }
                        p->pos += 2;
                        uint32_t low;
                        if (!parse_hex4(p, &low, err, err_cap)) { free(buf); return 0; }
                        if (low < 0xDC00 || low > 0xDFFF) {
                            set_err(err, err_cap, "par subrogado UTF-16 invalido en la posicion %zu", p->pos);
                            free(buf); return 0;
                        }
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                        (void)c1; (void)c2;
                    }
                    char enc[4]; int n;
                    if (!utf8_encode(cp, enc, &n)) {
                        set_err(err, err_cap, "codepoint invalido en escape \\u"); free(buf); return 0;
                    }
                    if (!buf_push_bytes(&buf, &len, &cap, enc, (size_t)n)) {
                        set_err(err, err_cap, "sin memoria"); free(buf); return 0;
                    }
                    break;
                }
                default:
                    set_err(err, err_cap, "escape desconocido '\\%c' en la posicion %zu", esc, p->pos);
                    free(buf); return 0;
            }
        } else {
            p->pos++;
            if (!buf_push_bytes(&buf, &len, &cap, &c, 1)) {
                set_err(err, err_cap, "sin memoria"); free(buf); return 0;
            }
        }
    }
    *out = buf;      /* puede ser NULL si len==0 -- valido, el llamador solo debe mirar *out_len */
    *out_len = len;
    return 1;
}

static int skip_string(TernaryJsonParser *p, char *err, size_t err_cap) {
    char *tmp; size_t tmp_len;
    if (!ternary_json_parse_string(p, &tmp, &tmp_len, err, err_cap)) return 0;
    free(tmp);
    return 1;
}

static int skip_number(TernaryJsonParser *p) {
    char c;
    while (ternary_json_peek(p, &c) &&
           ((c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E')) {
        p->pos++;
    }
    return 1;
}

static int expect_literal(TernaryJsonParser *p, const char *lit, char *err, size_t err_cap) {
    size_t n = strlen(lit);
    if (p->pos + n > p->len || memcmp(p->s + p->pos, lit, n) != 0) {
        set_err(err, err_cap, "se esperaba '%s' en la posicion %zu", lit, p->pos);
        return 0;
    }
    p->pos += n;
    return 1;
}

int ternary_json_skip_value(TernaryJsonParser *p, char *err, size_t err_cap) {
    ternary_json_skip_ws(p);
    char c;
    if (!ternary_json_peek(p, &c)) { set_err(err, err_cap, "valor esperado, el archivo termino"); return 0; }
    if (c == '"') return skip_string(p, err, err_cap);
    if (c == '{') {
        p->pos++;
        ternary_json_skip_ws(p);
        if (ternary_json_match(p, '}')) return 1;
        for (;;) {
            ternary_json_skip_ws(p);
            if (!skip_string(p, err, err_cap)) return 0;   /* clave */
            ternary_json_skip_ws(p);
            if (!ternary_json_expect(p, ':', err, err_cap)) return 0;
            if (!ternary_json_skip_value(p, err, err_cap)) return 0;
            ternary_json_skip_ws(p);
            if (ternary_json_match(p, ',')) continue;
            return ternary_json_expect(p, '}', err, err_cap);
        }
    }
    if (c == '[') {
        p->pos++;
        ternary_json_skip_ws(p);
        if (ternary_json_match(p, ']')) return 1;
        for (;;) {
            if (!ternary_json_skip_value(p, err, err_cap)) return 0;
            ternary_json_skip_ws(p);
            if (ternary_json_match(p, ',')) continue;
            return ternary_json_expect(p, ']', err, err_cap);
        }
    }
    if (c == 't') return expect_literal(p, "true", err, err_cap);
    if (c == 'f') return expect_literal(p, "false", err, err_cap);
    if (c == 'n') return expect_literal(p, "null", err, err_cap);
    if (c == '-' || (c >= '0' && c <= '9')) return skip_number(p);
    set_err(err, err_cap, "caracter '%c' inesperado en la posicion %zu", c, p->pos);
    return 0;
}


/* ======================================================================
 * DOM
 * ====================================================================== */
#define JDOM_MAX_DEPTH 64

static int jdom_value(TernaryJsonParser *p, TernaryJNode *out, int depth, char *err, size_t err_cap);

static void jdom_free_contents(TernaryJNode *n) {
    if (!n) return;
    free(n->str);
    for (size_t i = 0; i < n->n; i++) {
        if (n->keys) free(n->keys[i]);
        if (n->kids) jdom_free_contents(&n->kids[i]);
    }
    free(n->keys);
    free(n->kids);
    memset(n, 0, sizeof(*n));
}

void ternary_jdom_free(TernaryJNode *root) {
    if (!root) return;
    jdom_free_contents(root);
    free(root);
}

static int jdom_grow(TernaryJNode *n, size_t *cap, int with_keys) {
    if (n->n < *cap) return 1;
    size_t nc = *cap ? *cap * 2 : 4;
    TernaryJNode *k = (TernaryJNode *)realloc(n->kids, nc * sizeof(TernaryJNode));
    if (!k) return 0;
    n->kids = k;
    if (with_keys) {
        char **ks = (char **)realloc(n->keys, nc * sizeof(char *));
        if (!ks) return 0;
        n->keys = ks;
    }
    *cap = nc;
    return 1;
}

static int jdom_value(TernaryJsonParser *p, TernaryJNode *out, int depth, char *err, size_t err_cap) {
    memset(out, 0, sizeof(*out));
    if (depth > JDOM_MAX_DEPTH) { set_err(err, err_cap, "JSON demasiado anidado (mas de %d niveles)", JDOM_MAX_DEPTH); return 0; }
    ternary_json_skip_ws(p);
    char c;
    if (!ternary_json_peek(p, &c)) { set_err(err, err_cap, "valor esperado, el texto termino"); return 0; }
    if (c == '"') {
        out->type = TJ_STR;
        if (!ternary_json_parse_string(p, &out->str, &out->str_len, err, err_cap)) { out->type = TJ_NULL; return 0; }
        char *z = (char *)realloc(out->str, out->str_len + 1);
        if (!z) { free(out->str); out->str = NULL; out->type = TJ_NULL; set_err(err, err_cap, "sin memoria"); return 0; }
        z[out->str_len] = '\0';
        out->str = z;
        return 1;
    }
    if (c == '{' || c == '[') {
        const int is_obj = (c == '{');
        const char close = is_obj ? '}' : ']';
        out->type = is_obj ? TJ_OBJ : TJ_ARR;
        p->pos++;
        size_t cap = 0;
        ternary_json_skip_ws(p);
        if (ternary_json_match(p, close)) return 1;
        for (;;) {
            if (!jdom_grow(out, &cap, is_obj)) { set_err(err, err_cap, "sin memoria"); jdom_free_contents(out); return 0; }
            if (is_obj) {
                ternary_json_skip_ws(p);
                char *key; size_t klen;
                if (!ternary_json_parse_string(p, &key, &klen, err, err_cap)) { jdom_free_contents(out); return 0; }
                char *kz = (char *)realloc(key, klen + 1);
                if (!kz) { free(key); set_err(err, err_cap, "sin memoria"); jdom_free_contents(out); return 0; }
                kz[klen] = '\0';
                out->keys[out->n] = kz;
                ternary_json_skip_ws(p);
                if (!ternary_json_expect(p, ':', err, err_cap)) { free(kz); jdom_free_contents(out); return 0; }
            }
            if (!jdom_value(p, &out->kids[out->n], depth + 1, err, err_cap)) {
                if (is_obj) free(out->keys[out->n]);
                jdom_free_contents(out);
                return 0;
            }
            out->n++;
            ternary_json_skip_ws(p);
            if (ternary_json_match(p, ',')) continue;
            if (!ternary_json_expect(p, close, err, err_cap)) { jdom_free_contents(out); return 0; }
            return 1;
        }
    }
    if (c == 't' || c == 'f' || c == 'n') {
        if (c == 't') { if (!expect_literal(p, "true", err, err_cap)) return 0; out->type = TJ_BOOL; out->b = 1; }
        else if (c == 'f') { if (!expect_literal(p, "false", err, err_cap)) return 0; out->type = TJ_BOOL; out->b = 0; }
        else { if (!expect_literal(p, "null", err, err_cap)) return 0; out->type = TJ_NULL; }
        return 1;
    }
    if (c == '-' || (c >= '0' && c <= '9')) {
        size_t start = p->pos;
        int integral = 1;
        skip_number(p);
        size_t n = p->pos - start;
        char tmp[64];
        if (n == 0 || n >= sizeof(tmp)) { set_err(err, err_cap, "numero invalido en la posicion %zu", start); return 0; }
        memcpy(tmp, p->s + start, n); tmp[n] = '\0';
        for (size_t k = 0; k < n; k++) if (tmp[k] == '.' || tmp[k] == 'e' || tmp[k] == 'E') integral = 0;
        char *end = NULL;
        out->type = TJ_NUM;
        out->num = strtod(tmp, &end);
        if (end != tmp + n) { set_err(err, err_cap, "numero invalido '%s' en la posicion %zu", tmp, start); out->type = TJ_NULL; return 0; }
        if (integral) {
            errno = 0;
            long long v = strtoll(tmp, &end, 10);
            if (errno == 0 && end == tmp + n) { out->is_int = 1; out->i = (int64_t)v; }
        }
        return 1;
    }
    set_err(err, err_cap, "caracter '%c' inesperado en la posicion %zu", c, p->pos);
    return 0;
}

TernaryJNode *ternary_jdom_parse(const char *s, size_t len, char *err, size_t err_cap) {
    TernaryJsonParser p;
    ternary_json_init(&p, s, len);
    TernaryJNode *root = (TernaryJNode *)calloc(1, sizeof(TernaryJNode));
    if (!root) { set_err(err, err_cap, "sin memoria"); return NULL; }
    if (!jdom_value(&p, root, 0, err, err_cap)) { free(root); return NULL; }
    ternary_json_skip_ws(&p);
    if (p.pos != p.len) {
        set_err(err, err_cap, "texto sobrante despues del valor JSON (posicion %zu)", p.pos);
        ternary_jdom_free(root);
        return NULL;
    }
    return root;
}

const TernaryJNode *ternary_jdom_get(const TernaryJNode *obj, const char *key) {
    if (!obj || obj->type != TJ_OBJ) return NULL;
    for (size_t i = obj->n; i-- > 0;) if (strcmp(obj->keys[i], key) == 0) return &obj->kids[i];
    return NULL;
}

int ternary_jdom_get_int(const TernaryJNode *obj, const char *key, int64_t *out) {
    const TernaryJNode *n = ternary_jdom_get(obj, key);
    if (!n || n->type != TJ_NUM || !n->is_int) return 0;
    *out = n->i;
    return 1;
}

int ternary_jdom_get_float(const TernaryJNode *obj, const char *key, float *out) {
    const TernaryJNode *n = ternary_jdom_get(obj, key);
    if (!n || n->type != TJ_NUM) return 0;
    *out = (float)n->num;
    return 1;
}

int ternary_jdom_get_bool(const TernaryJNode *obj, const char *key, int *out) {
    const TernaryJNode *n = ternary_jdom_get(obj, key);
    if (!n || n->type != TJ_BOOL) return 0;
    *out = n->b;
    return 1;
}

const char *ternary_jdom_get_str(const TernaryJNode *obj, const char *key) {
    const TernaryJNode *n = ternary_jdom_get(obj, key);
    return (n && n->type == TJ_STR) ? n->str : NULL;
}

const TernaryJNode *ternary_jdom_at(const TernaryJNode *arr, size_t i) {
    if (!arr || arr->type != TJ_ARR || i >= arr->n) return NULL;
    return &arr->kids[i];
}

/* ======================================================================
 * Constructor de texto
 * ====================================================================== */
void ternary_jbuf_init(TernaryJBuf *b) { b->buf = NULL; b->len = 0; b->cap = 0; b->failed = 0; }
void ternary_jbuf_free(TernaryJBuf *b) { free(b->buf); ternary_jbuf_init(b); }

static void jbuf_put(TernaryJBuf *b, const char *s, size_t n) {
    if (b->failed) return;
    if (!buf_push_bytes(&b->buf, &b->len, &b->cap, s, n)) b->failed = 1;
}

void ternary_jbuf_raw(TernaryJBuf *b, const char *s) { jbuf_put(b, s, strlen(s)); }

const char *ternary_jbuf_cstr(TernaryJBuf *b) {
    if (b->failed) return NULL;
    if (!buf_push_bytes(&b->buf, &b->len, &b->cap, "", 1)) { b->failed = 1; return NULL; }
    b->len--;                                   /* el NUL queda fuera del largo */
    return b->buf;
}

static void jbuf_sep(TernaryJBuf *b) {
    if (b->len == 0) return;
    char c = b->buf[b->len - 1];
    if (c != '{' && c != '[' && c != ',' && c != ':') jbuf_put(b, ",", 1);
}

void ternary_jbuf_open(TernaryJBuf *b, char c) { jbuf_sep(b); jbuf_put(b, &c, 1); }
void ternary_jbuf_close(TernaryJBuf *b, char c) { jbuf_put(b, &c, 1); }

static void jbuf_string_body(TernaryJBuf *b, const char *s) {
    jbuf_put(b, "\"", 1);
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        char tmp[8];
        switch (*p) {
            case '"': jbuf_put(b, "\\\"", 2); break;
            case '\\': jbuf_put(b, "\\\\", 2); break;
            case '\n': jbuf_put(b, "\\n", 2); break;
            case '\r': jbuf_put(b, "\\r", 2); break;
            case '\t': jbuf_put(b, "\\t", 2); break;
            default:
                if (*p < 0x20) { snprintf(tmp, sizeof tmp, "\\u%04x", *p); jbuf_put(b, tmp, 6); }
                else jbuf_put(b, (const char *)p, 1);
        }
    }
    jbuf_put(b, "\"", 1);
}

void ternary_jbuf_str(TernaryJBuf *b, const char *s) { jbuf_sep(b); jbuf_string_body(b, s); }
void ternary_jbuf_key(TernaryJBuf *b, const char *key) { jbuf_sep(b); jbuf_string_body(b, key); jbuf_put(b, ":", 1); }

void ternary_jbuf_int(TernaryJBuf *b, int64_t v) {
    char tmp[32];
    jbuf_sep(b);
    int n = snprintf(tmp, sizeof tmp, "%lld", (long long)v);
    jbuf_put(b, tmp, (size_t)n);
}

void ternary_jbuf_float(TernaryJBuf *b, float v) {
    char tmp[48];
    jbuf_sep(b);
    if (v != v || v > 3.4028235e38f || v < -3.4028235e38f) { b->failed = 1; return; }   /* JSON no tiene NaN ni inf */
    int n = 0;
    for (int prec = 1; prec <= 9; prec++) {      /* la representacion mas corta que vuelve exacta al mismo float */
        n = snprintf(tmp, sizeof tmp, "%.*g", prec, (double)v);
        if (strtof(tmp, NULL) == v) break;
    }
    jbuf_put(b, tmp, (size_t)n);
}

void ternary_jbuf_bool(TernaryJBuf *b, int v) { jbuf_sep(b); jbuf_put(b, v ? "true" : "false", v ? 4 : 5); }
void ternary_jbuf_kint(TernaryJBuf *b, const char *key, int64_t v) { ternary_jbuf_key(b, key); ternary_jbuf_int(b, v); }
void ternary_jbuf_kfloat(TernaryJBuf *b, const char *key, float v) { ternary_jbuf_key(b, key); ternary_jbuf_float(b, v); }
void ternary_jbuf_kbool(TernaryJBuf *b, const char *key, int v) { ternary_jbuf_key(b, key); ternary_jbuf_bool(b, v); }
void ternary_jbuf_kstr(TernaryJBuf *b, const char *key, const char *v) { ternary_jbuf_key(b, key); ternary_jbuf_str(b, v); }

/* ¿El arreglo que abre en s[i] ('[') contiene solo escalares? (mira hasta su cierre, saltando strings) */
static int pretty_scalar_array(const char *s, size_t len, size_t i) {
    int in_str = 0;
    for (size_t k = i + 1; k < len; k++) {
        char c = s[k];
        if (in_str) { if (c == '\\') k++; else if (c == '"') in_str = 0; continue; }
        if (c == '"') in_str = 1;
        else if (c == '{' || c == '[') return 0;
        else if (c == ']') return 1;
    }
    return 1;
}

char *ternary_json_pretty(const char *s, size_t len, size_t *out_len) {
    TernaryJBuf b; ternary_jbuf_init(&b);
    int depth = 0, in_str = 0, inline_arr = 0;
    for (size_t i = 0; i < len; i++) {
        char c = s[i];
        if (in_str) {
            jbuf_put(&b, &c, 1);
            if (c == '\\' && i + 1 < len) { jbuf_put(&b, &s[++i], 1); }
            else if (c == '"') in_str = 0;
            continue;
        }
        switch (c) {
            case '"': in_str = 1; jbuf_put(&b, &c, 1); break;
            case '{': case '[': {
                jbuf_put(&b, &c, 1);
                if (c == '[' && pretty_scalar_array(s, len, i)) { inline_arr = 1; break; }
                int empty = (i + 1 < len && (s[i + 1] == '}' || s[i + 1] == ']'));
                depth++;
                if (!empty) { jbuf_put(&b, "\n", 1); for (int k = 0; k < depth; k++) jbuf_put(&b, "  ", 2); }
                break;
            }
            case '}': case ']': {
                if (c == ']' && inline_arr) { inline_arr = 0; jbuf_put(&b, &c, 1); break; }
                depth--;
                int empty = (i > 0 && (s[i - 1] == '{' || s[i - 1] == '['));
                if (!empty) { jbuf_put(&b, "\n", 1); for (int k = 0; k < depth; k++) jbuf_put(&b, "  ", 2); }
                jbuf_put(&b, &c, 1);
                break;
            }
            case ',':
                jbuf_put(&b, ",", 1);
                if (inline_arr) jbuf_put(&b, " ", 1);
                else { jbuf_put(&b, "\n", 1); for (int k = 0; k < depth; k++) jbuf_put(&b, "  ", 2); }
                break;
            case ':': jbuf_put(&b, ": ", 2); break;
            case ' ': case '\t': case '\n': case '\r': break;
            default: jbuf_put(&b, &c, 1);
        }
    }
    jbuf_put(&b, "\n", 1);
    if (b.failed) { ternary_jbuf_free(&b); return NULL; }
    char *z = (char *)realloc(b.buf, b.len + 1);
    if (!z) { free(b.buf); return NULL; }
    z[b.len] = '\0';
    if (out_len) *out_len = b.len;
    return z;
}
