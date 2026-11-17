#include "ternary_tokenizer.h"
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

/* ===================================================================
 * Tabla hash generica uint64 -> int64 (open addressing, sondeo lineal).
 * Se usa dos veces: contar frecuencias de pares durante el entrenamiento
 * (valor = conteo) y, ya entrenado, buscar el id de una fusion a partir
 * de su par (valor = id). EMPTY_KEY = UINT64_MAX: nunca choca con una
 * clave real porque left/right son ids >= 0 (caben en 32 bits cada uno).
 * ===================================================================*/
#define EMPTY_KEY UINT64_MAX

typedef struct {
    uint64_t *keys;
    int64_t *vals;
    size_t cap;   /* siempre potencia de 2 */
    size_t len;
} PairMap;

static uint64_t pack_pair(int32_t left, int32_t right) {
    return ((uint64_t)(uint32_t)left << 32) | (uint64_t)(uint32_t)right;
}

static uint64_t hash64(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

static int pairmap_init(PairMap *m, size_t cap_hint) {
    size_t cap = 16;
    while (cap < cap_hint * 2) cap <<= 1;
    m->keys = malloc(cap * sizeof(uint64_t));
    m->vals = malloc(cap * sizeof(int64_t));
    if (!m->keys || !m->vals) { free(m->keys); free(m->vals); return 0; }
    for (size_t i = 0; i < cap; i++) m->keys[i] = EMPTY_KEY;
    m->cap = cap;
    m->len = 0;
    return 1;
}

static void pairmap_free(PairMap *m) {
    free(m->keys); free(m->vals);
    m->keys = NULL; m->vals = NULL; m->cap = m->len = 0;
}

static int pairmap_grow(PairMap *m) {
    size_t newcap = m->cap * 2;
    uint64_t *nk = malloc(newcap * sizeof(uint64_t));
    int64_t *nv = malloc(newcap * sizeof(int64_t));
    if (!nk || !nv) { free(nk); free(nv); return 0; }
    for (size_t i = 0; i < newcap; i++) nk[i] = EMPTY_KEY;
    size_t mask = newcap - 1;
    for (size_t i = 0; i < m->cap; i++) {
        if (m->keys[i] == EMPTY_KEY) continue;
        size_t j = hash64(m->keys[i]) & mask;
        while (nk[j] != EMPTY_KEY) j = (j + 1) & mask;
        nk[j] = m->keys[i]; nv[j] = m->vals[i];
    }
    free(m->keys); free(m->vals);
    m->keys = nk; m->vals = nv; m->cap = newcap;
    return 1;
}

/* Suma 1 al conteo de 'key' (creandola en 0 si no existia). */
static int pairmap_incr(PairMap *m, uint64_t key) {
    if ((m->len + 1) * 10 >= m->cap * 7) { if (!pairmap_grow(m)) return 0; }
    size_t mask = m->cap - 1;
    size_t i = hash64(key) & mask;
    while (m->keys[i] != EMPTY_KEY && m->keys[i] != key) i = (i + 1) & mask;
    if (m->keys[i] == EMPTY_KEY) { m->keys[i] = key; m->vals[i] = 0; m->len++; }
    m->vals[i]++;
    return 1;
}

/* Asocia 'key' -> value (se usa para construir la tabla de lookup final,
 * cada par se inserta una sola vez, no incrementa). */
static int pairmap_set(PairMap *m, uint64_t key, int64_t value) {
    if ((m->len + 1) * 10 >= m->cap * 7) { if (!pairmap_grow(m)) return 0; }
    size_t mask = m->cap - 1;
    size_t i = hash64(key) & mask;
    while (m->keys[i] != EMPTY_KEY && m->keys[i] != key) i = (i + 1) & mask;
    if (m->keys[i] == EMPTY_KEY) { m->keys[i] = key; m->len++; }
    m->vals[i] = value;
    return 1;
}

static int pairmap_get(const PairMap *m, uint64_t key, int64_t *out_value) {
    if (m->cap == 0) return 0;
    size_t mask = m->cap - 1;
    size_t i = hash64(key) & mask;
    while (m->keys[i] != EMPTY_KEY) {
        if (m->keys[i] == key) { *out_value = m->vals[i]; return 1; }
        i = (i + 1) & mask;
    }
    return 0;
}

/* ===================================================================
 * Tokenizador
 * ===================================================================*/
typedef struct { int32_t left, right; } TokenPair;

/* Tokens especiales: cadenas de bytes reservadas con ids FIJOS justo
 * despues de los 256 bytes (ids 256 .. 256+S-1, en el orden en que se
 * agregaron); las fusiones aprendidas empiezan en 256+S. El BPE nunca
 * parte ni fusiona un token especial: en el entrenamiento cada aparicion
 * es una frontera que ninguna fusion cruza, y en encode se emite su id
 * tal cual. Sin tokens especiales (S=0) todo es identico a antes. */
struct TernaryTokenizer {
    size_t vocab_size_wanted;   /* total: 256 + S + fusiones */
    size_t num_merges;          /* fusiones REALMENTE logradas */
    TokenPair *merges;          /* [num_merges]; token id 256+S+i */
    PairMap lookup;             /* (left,right) -> id, solo para encode; vacia hasta entrenar */
    int trained;
    size_t num_special;
    uint8_t **special;          /* [num_special] cadenas (no terminadas en NUL) */
    size_t *special_len;
};

TernaryTokenizer *ternary_tokenizer_create(size_t vocab_size, char *err, size_t err_cap) {
    if (vocab_size < 256) {
        set_err(err, err_cap, "vocab_size debe ser >= 256 (los 256 bytes crudos siempre estan), se pidio %zu",
                vocab_size);
        return NULL;
    }
    TernaryTokenizer *t = calloc(1, sizeof(TernaryTokenizer));
    if (!t) { set_err(err, err_cap, "sin memoria"); return NULL; }
    t->vocab_size_wanted = vocab_size;
    t->num_merges = 0;
    t->merges = NULL;
    memset(&t->lookup, 0, sizeof(t->lookup));
    t->trained = 0;
    return t;
}

void ternary_tokenizer_free(TernaryTokenizer *t) {
    if (!t) return;
    free(t->merges);
    pairmap_free(&t->lookup);
    for (size_t i = 0; i < t->num_special; i++) free(t->special[i]);
    free(t->special);
    free(t->special_len);
    free(t);
}

static int build_lookup(TernaryTokenizer *t, char *err, size_t err_cap) {
    if (!pairmap_init(&t->lookup, t->num_merges)) {
        set_err(err, err_cap, "sin memoria construyendo la tabla de codificacion");
        return 0;
    }
    for (size_t i = 0; i < t->num_merges; i++) {
        uint64_t key = pack_pair(t->merges[i].left, t->merges[i].right);
        if (!pairmap_set(&t->lookup, key, (int64_t)(256 + t->num_special + i))) {
            set_err(err, err_cap, "sin memoria construyendo la tabla de codificacion");
            return 0;
        }
    }
    return 1;
}

/* Longitud del token especial que empieza en text[i] (el MAS LARGO si
 * varios coinciden), o 0 si ninguno; *which recibe su indice. */
static size_t match_special(const TernaryTokenizer *t, const uint8_t *text, size_t i, size_t n, size_t *which) {
    size_t best = 0;
    for (size_t k = 0; k < t->num_special; k++) {
        size_t L = t->special_len[k];
        if (L > best && L <= n - i && memcmp(text + i, t->special[k], L) == 0) { best = L; *which = k; }
    }
    return best;
}

int ternary_tokenizer_add_special(TernaryTokenizer *t, const uint8_t *text, size_t len, char *err, size_t err_cap) {
    if (!t || !text) { set_err(err, err_cap, "argumentos nulos"); return 0; }
    if (t->trained) { set_err(err, err_cap, "los tokens especiales se agregan antes de entrenar (despues cambiarian los ids de las fusiones)"); return 0; }
    if (len == 0) { set_err(err, err_cap, "un token especial no puede ser vacio"); return 0; }
    if (256 + t->num_special + 1 > t->vocab_size_wanted) {
        set_err(err, err_cap, "no cabe otro token especial: vocab_size=%zu ya esta ocupado por los 256 bytes y %zu especiales",
                t->vocab_size_wanted, t->num_special);
        return 0;
    }
    for (size_t k = 0; k < t->num_special; k++) {
        if (t->special_len[k] == len && memcmp(t->special[k], text, len) == 0) {
            set_err(err, err_cap, "token especial repetido");
            return 0;
        }
    }
    uint8_t **ns = realloc(t->special, (t->num_special + 1) * sizeof(uint8_t *));
    if (!ns) { set_err(err, err_cap, "sin memoria"); return 0; }
    t->special = ns;
    size_t *nl = realloc(t->special_len, (t->num_special + 1) * sizeof(size_t));
    if (!nl) { set_err(err, err_cap, "sin memoria"); return 0; }
    t->special_len = nl;
    uint8_t *copy = malloc(len);
    if (!copy) { set_err(err, err_cap, "sin memoria"); return 0; }
    memcpy(copy, text, len);
    t->special[t->num_special] = copy;
    t->special_len[t->num_special] = len;
    t->num_special++;
    return 1;
}

size_t ternary_tokenizer_num_special(const TernaryTokenizer *t) { return t ? t->num_special : 0; }

int ternary_tokenizer_special_text(const TernaryTokenizer *t, size_t index, uint8_t *out, size_t out_cap,
                                    size_t *out_len, char *err, size_t err_cap) {
    if (!t || !out_len) { set_err(err, err_cap, "argumentos nulos"); return 0; }
    if (index >= t->num_special) { set_err(err, err_cap, "indice de token especial fuera de rango (%zu de %zu)", index, t->num_special); return 0; }
    *out_len = t->special_len[index];
    if (out && out_cap >= *out_len) memcpy(out, t->special[index], *out_len);
    return 1;
}

int32_t ternary_tokenizer_special_id(const TernaryTokenizer *t, const uint8_t *text, size_t len) {
    if (!t || !text) return -1;
    for (size_t k = 0; k < t->num_special; k++)
        if (t->special_len[k] == len && memcmp(t->special[k], text, len) == 0) return (int32_t)(256 + k);
    return -1;
}

int ternary_tokenizer_train(TernaryTokenizer *t, const uint8_t *corpus, size_t corpus_len,
                             char *err, size_t err_cap) {
    if (!t) { set_err(err, err_cap, "tokenizador nulo"); return 0; }
    if (t->trained) { set_err(err, err_cap, "este tokenizador ya fue entrenado"); return 0; }
    if (corpus_len == 0 && t->vocab_size_wanted > 256) {
        set_err(err, err_cap, "corpus vacio: no hay nada de que aprender fusiones");
        return 0;
    }

    if (t->vocab_size_wanted < 256 + t->num_special) {
        set_err(err, err_cap, "vocab_size=%zu no alcanza para 256 bytes + %zu tokens especiales", t->vocab_size_wanted, t->num_special);
        return 0;
    }
    size_t merges_wanted = t->vocab_size_wanted - 256 - t->num_special;
    t->merges = merges_wanted ? malloc(merges_wanted * sizeof(TokenPair)) : NULL;
    if (merges_wanted && !t->merges) { set_err(err, err_cap, "sin memoria"); return 0; }

    int32_t *symbols = malloc(corpus_len * sizeof(int32_t));
    if (corpus_len && !symbols) { set_err(err, err_cap, "sin memoria"); free(t->merges); t->merges = NULL; return 0; }
    /* Cada aparicion de un token especial en el corpus pasa a ser un
     * centinela -1 (ninguna fusion lo cruza ni lo cuenta). */
    size_t n = 0;
    for (size_t i = 0; i < corpus_len; ) {
        size_t which = 0, L = t->num_special ? match_special(t, corpus, i, corpus_len, &which) : 0;
        if (L) { symbols[n++] = -1; i += L; }
        else { symbols[n++] = (int32_t)corpus[i]; i++; }
    }

    size_t made = 0;
    while (made < merges_wanted && n > 1) {
        PairMap counts;
        if (!pairmap_init(&counts, n)) {
            set_err(err, err_cap, "sin memoria contando pares"); free(symbols); free(t->merges); t->merges = NULL; return 0;
        }
        for (size_t i = 0; i + 1 < n; i++) {
            if (symbols[i] < 0 || symbols[i + 1] < 0) continue;
            if (!pairmap_incr(&counts, pack_pair(symbols[i], symbols[i + 1]))) {
                set_err(err, err_cap, "sin memoria contando pares");
                pairmap_free(&counts); free(symbols); free(t->merges); t->merges = NULL; return 0;
            }
        }
        int64_t best_count = 0; uint64_t best_key = 0; int found = 0;
        for (size_t i = 0; i < counts.cap; i++) {
            if (counts.keys[i] == EMPTY_KEY) continue;
            int64_t c = counts.vals[i];
            if (!found || c > best_count || (c == best_count && counts.keys[i] < best_key)) {
                best_count = c; best_key = counts.keys[i]; found = 1;
            }
        }
        pairmap_free(&counts);
        if (!found || best_count < 2) break;  /* no queda compresion real que ganar */

        int32_t left = (int32_t)(best_key >> 32);
        int32_t right = (int32_t)(uint32_t)best_key;
        int32_t new_id = (int32_t)(256 + t->num_special + made);
        t->merges[made].left = left;
        t->merges[made].right = right;
        made++;

        int32_t *out = malloc(n * sizeof(int32_t));
        if (!out) { set_err(err, err_cap, "sin memoria fusionando"); free(symbols); free(t->merges); t->merges = NULL; return 0; }
        size_t on = 0;
        for (size_t i = 0; i < n; ) {
            if (i + 1 < n && symbols[i] == left && symbols[i + 1] == right) { out[on++] = new_id; i += 2; }
            else { out[on++] = symbols[i]; i += 1; }
        }
        free(symbols);
        symbols = out;
        n = on;
    }
    free(symbols);

    t->num_merges = made;
    t->trained = 1;
    return build_lookup(t, err, err_cap);
}

size_t ternary_tokenizer_vocab_size(const TernaryTokenizer *t) { return t ? 256 + t->num_special + t->num_merges : 0; }
size_t ternary_tokenizer_num_merges(const TernaryTokenizer *t) { return t ? t->num_merges : 0; }

/* BPE de un segmento sin tokens especiales. Devuelve un buffer malloc
 * con *out_n ids (NULL si no hay memoria o el segmento es vacio con
 * *out_n=0 y retorno distinto de NULL no aplica: ver llamador). */
static int32_t *bpe_segment(const TernaryTokenizer *t, const uint8_t *text, size_t text_len, size_t *out_n) {
    int32_t *ids = malloc(text_len * sizeof(int32_t));
    if (!ids) return NULL;
    for (size_t i = 0; i < text_len; i++) ids[i] = (int32_t)text[i];
    size_t n = text_len;

    for (;;) {
        int64_t best_id = -1;
        int32_t best_left = -1, best_right = -1;
        for (size_t i = 0; i + 1 < n; i++) {
            int64_t id;
            if (pairmap_get(&t->lookup, pack_pair(ids[i], ids[i + 1]), &id)) {
                if (best_id < 0 || id < best_id) { best_id = id; best_left = ids[i]; best_right = ids[i + 1]; }
            }
        }
        if (best_id < 0) break;
        int32_t *out = malloc(n * sizeof(int32_t));
        if (!out) { free(ids); return NULL; }
        size_t on = 0;
        for (size_t i = 0; i < n; ) {
            if (i + 1 < n && ids[i] == best_left && ids[i + 1] == best_right) { out[on++] = (int32_t)best_id; i += 2; }
            else { out[on++] = ids[i]; i += 1; }
        }
        free(ids);
        ids = out;
        n = on;
    }
    *out_n = n;
    return ids;
}

int ternary_tokenizer_encode_ex(const TernaryTokenizer *t, const uint8_t *text, size_t text_len, int parse_special,
                                 int32_t *out_ids, size_t out_cap, size_t *out_len,
                                 char *err, size_t err_cap) {
    if (!t || !out_len) { set_err(err, err_cap, "argumentos nulos"); return 0; }
    if (text_len == 0) { *out_len = 0; return 1; }
    int use_special = parse_special && t->num_special > 0;

    int32_t *res = NULL; size_t res_n = 0, res_cap = 0;
    size_t seg_start = 0, i = 0;
    while (seg_start < text_len) {
        size_t which = 0, L = 0;
        if (use_special) {
            while (i < text_len && !(L = match_special(t, text, i, text_len, &which))) i++;
        } else {
            i = text_len;
        }
        /* segmento [seg_start, i) sin especiales, luego el especial (si hay) */
        size_t seg_n = 0;
        int32_t *seg = NULL;
        if (i > seg_start) {
            seg = bpe_segment(t, text + seg_start, i - seg_start, &seg_n);
            if (!seg) { set_err(err, err_cap, "sin memoria"); free(res); return 0; }
        }
        size_t add = seg_n + (L ? 1 : 0);
        if (res_n + add > res_cap) {
            size_t nc = res_cap ? res_cap : 64;
            while (nc < res_n + add) nc *= 2;
            int32_t *nr = realloc(res, nc * sizeof(int32_t));
            if (!nr) { set_err(err, err_cap, "sin memoria"); free(seg); free(res); return 0; }
            res = nr; res_cap = nc;
        }
        if (seg_n) memcpy(res + res_n, seg, seg_n * sizeof(int32_t));
        res_n += seg_n;
        free(seg);
        if (L) { res[res_n++] = (int32_t)(256 + which); i += L; }
        seg_start = i;
    }

    *out_len = res_n;
    if (out_ids && out_cap >= res_n && res_n) memcpy(out_ids, res, res_n * sizeof(int32_t));
    free(res);
    return 1;
}

int ternary_tokenizer_encode(const TernaryTokenizer *t, const uint8_t *text, size_t text_len,
                              int32_t *out_ids, size_t out_cap, size_t *out_len,
                              char *err, size_t err_cap) {
    return ternary_tokenizer_encode_ex(t, text, text_len, 1, out_ids, out_cap, out_len, err, err_cap);
}

/* Buffer de bytes con crecimiento por duplicacion, para decode. */
static int push_byte(uint8_t **buf, size_t *len, size_t *cap, uint8_t b) {
    if (*len >= *cap) {
        size_t newcap = *cap ? *cap * 2 : 64;
        uint8_t *nb = realloc(*buf, newcap);
        if (!nb) return 0;
        *buf = nb; *cap = newcap;
    }
    (*buf)[(*len)++] = b;
    return 1;
}

/* Expande un id a bytes con una pila explicita (no recursion): una
 * cadena de fusiones podria tener miles de eslabones y el dispositivo
 * objetivo (dispositivo ARMv7 de 32 bits) tiene poco stack -- no vale la pena arriesgarse. */
static int expand_token(const TernaryTokenizer *t, int32_t id, int skip_special, uint8_t **buf, size_t *len, size_t *cap,
                         int32_t *stack, size_t stack_cap) {
    size_t sp = 0;
    stack[sp++] = id;
    while (sp > 0) {
        int32_t cur = stack[--sp];
        if (cur >= 0 && cur < 256) {
            if (!push_byte(buf, len, cap, (uint8_t)cur)) return 0;
            continue;
        }
        if (cur < 256) continue;  /* id invalido: se ignora */
        size_t rel = (size_t)(cur - 256);
        if (rel < t->num_special) {  /* token especial: sus bytes, o nada si se piden omitir */
            if (!skip_special) {
                for (size_t b = 0; b < t->special_len[rel]; b++)
                    if (!push_byte(buf, len, cap, t->special[rel][b])) return 0;
            }
            continue;
        }
        size_t idx = rel - t->num_special;
        if (idx >= t->num_merges) continue;  /* id invalido: se ignora */
        if (sp + 2 > stack_cap) return 0;  /* no deberia pasar: stack_cap = num_merges+1 alcanza siempre */
        stack[sp++] = t->merges[idx].right;  /* right primero: left sale primero de la pila (LIFO) */
        stack[sp++] = t->merges[idx].left;
    }
    return 1;
}

int ternary_tokenizer_decode_ex(const TernaryTokenizer *t, const int32_t *ids, size_t num_ids, int skip_special,
                                 uint8_t *out_bytes, size_t out_cap, size_t *out_len,
                                 char *err, size_t err_cap) {
    if (!t || !out_len) { set_err(err, err_cap, "argumentos nulos"); return 0; }
    uint8_t *buf = NULL; size_t len = 0, cap = 0;
    size_t stack_cap = t->num_merges + 2;
    int32_t *stack = malloc(stack_cap * sizeof(int32_t));
    if (!stack) { set_err(err, err_cap, "sin memoria"); return 0; }
    for (size_t i = 0; i < num_ids; i++) {
        if (!expand_token(t, ids[i], skip_special, &buf, &len, &cap, stack, stack_cap)) {
            set_err(err, err_cap, "sin memoria decodificando");
            free(buf); free(stack); return 0;
        }
    }
    free(stack);
    *out_len = len;
    if (out_bytes && out_cap >= len && len > 0) memcpy(out_bytes, buf, len);
    free(buf);
    return 1;
}

int ternary_tokenizer_decode(const TernaryTokenizer *t, const int32_t *ids, size_t num_ids,
                              uint8_t *out_bytes, size_t out_cap, size_t *out_len,
                              char *err, size_t err_cap) {
    return ternary_tokenizer_decode_ex(t, ids, num_ids, 0, out_bytes, out_cap, out_len, err, err_cap);
}

/* ===================================================================
 * Guardado / carga: tokenizer.json (texto, igual que en HuggingFace)
 *
 * {
 *   "format": "ternary-bpe", "format_version": 1,
 *   "vocab_size": 512,                       <- el pedido (el logrado es 256 + especiales + fusiones)
 *   "special_tokens": [ {"id":256,"content":"<|user|>"}, ... ],   <- "content_hex" si no es UTF-8 valido
 *   "merges": [ [left, right], ... ]         <- la fusion i crea el id 256 + num_special + i
 * }
 * =================================================================== */
static int utf8_valid(const uint8_t *p, size_t n) {
    for (size_t i = 0; i < n;) {
        uint8_t c = p[i];
        size_t k = c < 0x80 ? 1 : (c >= 0xC2 && c <= 0xDF) ? 2 : (c >= 0xE0 && c <= 0xEF) ? 3 : (c >= 0xF0 && c <= 0xF4) ? 4 : 0;
        if (!k || i + k > n) return 0;
        for (size_t j = 1; j < k; j++) if ((p[i + j] & 0xC0) != 0x80) return 0;
        if (c == 0xE0 && p[i + 1] < 0xA0) return 0;
        if (c == 0xED && p[i + 1] > 0x9F) return 0;
        if (c == 0xF0 && p[i + 1] < 0x90) return 0;
        if (c == 0xF4 && p[i + 1] > 0x8F) return 0;
        i += k;
    }
    return 1;
}

int ternary_tokenizer_save(const TernaryTokenizer *t, const char *path, char *err, size_t err_cap) {
    if (!t || !path) { set_err(err, err_cap, "argumentos nulos"); return 0; }
    TernaryJBuf j; ternary_jbuf_init(&j);
    ternary_jbuf_open(&j, '{');
    ternary_jbuf_kstr(&j, "format", "ternary-bpe");
    ternary_jbuf_kint(&j, "format_version", 1);
    ternary_jbuf_kint(&j, "vocab_size", (int64_t)t->vocab_size_wanted);
    ternary_jbuf_key(&j, "special_tokens");
    ternary_jbuf_open(&j, '[');
    for (size_t k = 0; k < t->num_special; k++) {
        ternary_jbuf_open(&j, '{');
        ternary_jbuf_kint(&j, "id", (int64_t)(256 + k));
        if (utf8_valid(t->special[k], t->special_len[k]) && memchr(t->special[k], 0, t->special_len[k]) == NULL) {
            char *z = malloc(t->special_len[k] + 1);
            if (!z) { ternary_jbuf_free(&j); set_err(err, err_cap, "sin memoria"); return 0; }
            memcpy(z, t->special[k], t->special_len[k]); z[t->special_len[k]] = '\0';
            ternary_jbuf_kstr(&j, "content", z);
            free(z);
        } else {
            char *hex = malloc(t->special_len[k] * 2 + 1);
            if (!hex) { ternary_jbuf_free(&j); set_err(err, err_cap, "sin memoria"); return 0; }
            for (size_t q = 0; q < t->special_len[k]; q++) snprintf(hex + 2 * q, 3, "%02x", t->special[k][q]);
            ternary_jbuf_kstr(&j, "content_hex", hex);
            free(hex);
        }
        ternary_jbuf_close(&j, '}');
    }
    ternary_jbuf_close(&j, ']');
    ternary_jbuf_key(&j, "merges");
    ternary_jbuf_open(&j, '[');
    for (size_t k = 0; k < t->num_merges; k++) {
        ternary_jbuf_open(&j, '[');
        ternary_jbuf_int(&j, t->merges[k].left);
        ternary_jbuf_int(&j, t->merges[k].right);
        ternary_jbuf_close(&j, ']');
    }
    ternary_jbuf_close(&j, ']');
    ternary_jbuf_close(&j, '}');
    size_t plen = 0;
    char *pretty = j.failed ? NULL : ternary_json_pretty(j.buf, j.len, &plen);
    ternary_jbuf_free(&j);
    if (!pretty) { set_err(err, err_cap, "sin memoria"); return 0; }
    size_t pl = strlen(path);
    char *tmp = malloc(pl + 5);
    if (!tmp) { free(pretty); set_err(err, err_cap, "sin memoria"); return 0; }
    memcpy(tmp, path, pl); memcpy(tmp + pl, ".tmp", 5);
    FILE *f = fopen(tmp, "wb");
    int ok = f != NULL;
    if (!f) set_err(err, err_cap, "no se pudo abrir '%s' para escritura", path);
    if (f) {
        ok = fwrite(pretty, 1, plen, f) == plen;
        if (fclose(f) != 0) ok = 0;
        if (ok && rename(tmp, path) != 0) ok = 0;
        if (!ok) { remove(tmp); set_err(err, err_cap, "error escribiendo '%s'", path); }
    }
    free(tmp); free(pretty);
    return ok;
}

static TernaryTokenizer *tokenizer_from_text(char *text, size_t len, const char *path, char *err, size_t err_cap);

TernaryTokenizer *ternary_tokenizer_load(const char *path, char *err, size_t err_cap) {
    if (!path) { set_err(err, err_cap, "path nulo"); return NULL; }
    FILE *f = fopen(path, "rb");
    if (!f) { set_err(err, err_cap, "no se pudo abrir '%s'", path); return NULL; }
    char *text = NULL; size_t len = 0;
    if (fseek(f, 0, SEEK_END) == 0) {
        long sz = ftell(f);
        if (sz >= 0 && sz <= (1L << 28) && fseek(f, 0, SEEK_SET) == 0) {
            text = malloc((size_t)sz + 1);
            if (text && fread(text, 1, (size_t)sz, f) == (size_t)sz) len = (size_t)sz; else { free(text); text = NULL; }
        }
    }
    fclose(f);
    if (!text) { set_err(err, err_cap, "no se pudo leer '%s'", path); return NULL; }
    return tokenizer_from_text(text, len, path, err, err_cap);
}

/* Igual que ternary_tokenizer_load pero desde un texto en memoria (p. ej. la seccion de un modelo autocontenido); 'label' solo sirve en los mensajes. */
TernaryTokenizer *ternary_tokenizer_load_mem(const char *text_in, size_t len, const char *label, char *err, size_t err_cap) {
    char *text = (char *)malloc(len + 1);
    if (!text) { set_err(err, err_cap, "sin memoria"); return NULL; }
    memcpy(text, text_in, len); text[len] = 0;
    return tokenizer_from_text(text, len, label ? label : "tokenizer", err, err_cap);
}

/* Interpreta el JSON de un tokenizer.json; libera 'text' (malloc). */
static TernaryTokenizer *tokenizer_from_text(char *text, size_t len, const char *path, char *err, size_t err_cap) {
    char jerr[160] = "";
    TernaryJNode *root = ternary_jdom_parse(text, len, jerr, sizeof jerr);
    free(text);
    if (!root) { set_err(err, err_cap, "'%s' no es un JSON valido: %s", path, jerr); return NULL; }
    TernaryTokenizer *t = NULL;
    const char *fmt = ternary_jdom_get_str(root, "format");
    int64_t ver = 0, vocab_wanted = 0;
    const TernaryJNode *sp = ternary_jdom_get(root, "special_tokens"), *mg = ternary_jdom_get(root, "merges");
    if (!fmt || strcmp(fmt, "ternary-bpe") != 0) { set_err(err, err_cap, "'%s' no es un tokenizer.json de ternaryai (format != 'ternary-bpe')", path); goto done; }
    if (!ternary_jdom_get_int(root, "format_version", &ver) || ver != 1) { set_err(err, err_cap, "'%s': format_version %lld no soportada", path, (long long)ver); goto done; }
    if (!ternary_jdom_get_int(root, "vocab_size", &vocab_wanted) || vocab_wanted < 256 || vocab_wanted > ((int64_t)1 << 31) ||
        !sp || sp->type != TJ_ARR || !mg || mg->type != TJ_ARR) { set_err(err, err_cap, "'%s': faltan campos o son invalidos (vocab_size, special_tokens, merges)", path); goto done; }
    if ((int64_t)sp->n > vocab_wanted || (int64_t)mg->n > vocab_wanted || 256 + (int64_t)sp->n + (int64_t)mg->n > vocab_wanted) {
        set_err(err, err_cap, "'%s' es inconsistente (especiales/fusiones exceden el vocabulario)", path); goto done;
    }
    t = calloc(1, sizeof(TernaryTokenizer));
    if (!t) { set_err(err, err_cap, "sin memoria"); goto done; }
    t->vocab_size_wanted = (size_t)vocab_wanted;
    for (size_t k = 0; k < sp->n; k++) {
        const TernaryJNode *e = &sp->kids[k];
        int64_t id = -1;
        const char *content = ternary_jdom_get_str(e, "content"), *hex = ternary_jdom_get_str(e, "content_hex");
        uint8_t *buf = NULL; size_t L = 0;
        if (e->type != TJ_OBJ || !ternary_jdom_get_int(e, "id", &id) || id != (int64_t)(256 + k) || (!content && !hex)) {
            set_err(err, err_cap, "'%s': token especial %zu invalido (id esperado %zu)", path, k, 256 + k);
            ternary_tokenizer_free(t); t = NULL; goto done;
        }
        if (content) {
            const TernaryJNode *c = ternary_jdom_get(e, "content");
            L = c->str_len; buf = malloc(L ? L : 1); if (buf) memcpy(buf, content, L);
        } else {
            size_t hl = strlen(hex);
            L = hl / 2; buf = malloc(L ? L : 1);
            for (size_t q = 0; buf && q < L; q++) {
                unsigned v; if (sscanf(hex + 2 * q, "%2x", &v) != 1) { free(buf); buf = NULL; }
                else buf[q] = (uint8_t)v;
            }
            if (hl % 2) { free(buf); buf = NULL; }
        }
        if (!buf || L == 0 || L > (1u << 16)) {
            free(buf); set_err(err, err_cap, "'%s': token especial %zu invalido o vacio", path, k);
            ternary_tokenizer_free(t); t = NULL; goto done;
        }
        t->trained = 0;  /* add_special exige no entrenado; se restaura abajo */
        int ok = ternary_tokenizer_add_special(t, buf, L, err, err_cap);
        free(buf);
        if (!ok) { ternary_tokenizer_free(t); t = NULL; goto done; }
    }
    t->trained = 1;
    t->num_merges = mg->n;
    t->merges = t->num_merges ? malloc(t->num_merges * sizeof(TokenPair)) : NULL;
    if (t->num_merges && !t->merges) { set_err(err, err_cap, "sin memoria"); ternary_tokenizer_free(t); t = NULL; goto done; }
    for (size_t i = 0; i < t->num_merges; i++) {
        const TernaryJNode *pr = &mg->kids[i];
        if (pr->type != TJ_ARR || pr->n != 2 || !pr->kids[0].is_int || !pr->kids[1].is_int) {
            set_err(err, err_cap, "'%s': la fusion %zu no es un par de enteros", path, i);
            ternary_tokenizer_free(t); t = NULL; goto done;
        }
        int64_t l = pr->kids[0].i, r = pr->kids[1].i;
        int64_t lim = 256 + (int64_t)t->num_special + (int64_t)i, base = 256 + (int64_t)t->num_special;
        /* cada fusion solo puede usar bytes o fusiones anteriores (nunca un especial ni un id futuro) */
        int lok = (l >= 0 && l < 256) || (l >= base && l < lim), rok = (r >= 0 && r < 256) || (r >= base && r < lim);
        if (!lok || !rok) {
            set_err(err, err_cap, "'%s' es corrupto: la fusion %zu usa ids invalidos", path, i);
            ternary_tokenizer_free(t); t = NULL; goto done;
        }
        t->merges[i].left = (int32_t)l; t->merges[i].right = (int32_t)r;
    }
    if (!build_lookup(t, err, err_cap)) { ternary_tokenizer_free(t); t = NULL; }
done:
    ternary_jdom_free(root);
    return t;
}
