#define _POSIX_C_SOURCE 200809L
#include "ternary_tfile.h"
#include "ternary_json.h"
#include "ternary_model_io.h"
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

static void set_err(char *err, size_t cap, const char *fmt, ...) {
    if (!err || !cap) return;
    va_list ap; va_start(ap, fmt); vsnprintf(err, cap, fmt, ap); va_end(ap);
}

/* ---- dtypes ------------------------------------------------------------------------------------------- */
static const struct { const char *name; size_t size; } DT[] = {
    {"F32", 4}, {"I32", 4}, {"I8", 1}, {"U8", 1}, {"TRIT5", 0},
};

const char *ternary_tdtype_name(TernaryTDType t) { return DT[t].name; }
size_t ternary_tdtype_bytes(TernaryTDType t, size_t n) {
    return t == TTD_TRIT5 ? ternary_packed_trits_bytes(n) : n * DT[t].size;
}
static int dtype_by_name(const char *s, TernaryTDType *out) {
    for (int i = 0; i < 5; i++) if (strcmp(DT[i].name, s) == 0) { *out = (TernaryTDType)i; return 1; }
    return 0;
}

/* producto de la forma con control de desbordamiento */
static int shape_nelem(const size_t *shape, size_t nd, size_t *out) {
    size_t n = 1;
    for (size_t i = 0; i < nd; i++) {
        if (shape[i] != 0 && n > SIZE_MAX / shape[i]) return 0;
        n *= shape[i];
    }
    *out = n;
    return 1;
}

/* ====================================================================================================== */
/* Escritor                                                                                               */
/* ====================================================================================================== */
typedef struct {
    char *name;
    TernaryTDType dtype;
    size_t nd, shape[TERNARY_TFILE_MAX_DIMS];
    size_t nbytes;
    const void *data;
    uint8_t *owned;
} WEntry;

struct TernaryTWriter {
    WEntry *e; size_t n, cap;
    char **mk, **mv; size_t nm, mcap;
};

TernaryTWriter *ternary_tw_create(void) { return (TernaryTWriter *)calloc(1, sizeof(TernaryTWriter)); }

void ternary_tw_free(TernaryTWriter *w) {
    if (!w) return;
    for (size_t i = 0; i < w->n; i++) { free(w->e[i].name); free(w->e[i].owned); }
    for (size_t i = 0; i < w->nm; i++) { free(w->mk[i]); free(w->mv[i]); }
    free(w->e); free(w->mk); free(w->mv); free(w);
}

int ternary_tw_meta(TernaryTWriter *w, const char *key, const char *value) {
    if (!w || !key || !value || !strcmp(key, "format") || !strcmp(key, "version")) return 0;
    if (w->nm == w->mcap) {
        size_t nc = w->mcap ? w->mcap * 2 : 4;
        char **a = (char **)realloc(w->mk, nc * sizeof(char *)); if (!a) return 0; w->mk = a;
        char **b = (char **)realloc(w->mv, nc * sizeof(char *)); if (!b) return 0; w->mv = b;
        w->mcap = nc;
    }
    w->mk[w->nm] = strdup(key); w->mv[w->nm] = strdup(value);
    if (!w->mk[w->nm] || !w->mv[w->nm]) { free(w->mk[w->nm]); free(w->mv[w->nm]); return 0; }
    w->nm++;
    return 1;
}

static int tw_push(TernaryTWriter *w, const char *name, TernaryTDType dt, const size_t *shape, size_t nd,
                   const void *data, uint8_t *owned) {
    if (!w || !name || !name[0] || nd > TERNARY_TFILE_MAX_DIMS) { free(owned); return 0; }
    size_t nelem;
    if (!shape_nelem(shape, nd, &nelem)) { free(owned); return 0; }
    for (size_t i = 0; i < w->n; i++) if (!strcmp(w->e[i].name, name)) { free(owned); return 0; }   /* nombre repetido */
    if (w->n == w->cap) {
        size_t nc = w->cap ? w->cap * 2 : 16;
        WEntry *a = (WEntry *)realloc(w->e, nc * sizeof(WEntry));
        if (!a) { free(owned); return 0; }
        w->e = a; w->cap = nc;
    }
    WEntry *e = &w->e[w->n];
    memset(e, 0, sizeof(*e));
    e->name = strdup(name);
    if (!e->name) { free(owned); return 0; }
    e->dtype = dt; e->nd = nd;
    for (size_t i = 0; i < nd; i++) e->shape[i] = shape[i];
    e->nbytes = ternary_tdtype_bytes(dt, nelem);
    e->data = data; e->owned = owned;
    w->n++;
    return 1;
}

int ternary_tw_add(TernaryTWriter *w, const char *name, TernaryTDType dt, const size_t *shape, size_t nd, const void *data) {
    if (dt == TTD_TRIT5) return 0;       /* los trits entran por ternary_tw_add_trits */
    return tw_push(w, name, dt, shape, nd, data, NULL);
}

int ternary_tw_add_trits(TernaryTWriter *w, const char *name, const size_t *shape, size_t nd, const TernaryVec *v) {
    size_t nelem;
    if (!v || nd > TERNARY_TFILE_MAX_DIMS || !shape_nelem(shape, nd, &nelem) || nelem != v->n) return 0;
    size_t nb = ternary_packed_trits_bytes(nelem);
    uint8_t *buf = (uint8_t *)malloc(nb ? nb : 1);
    if (!buf) return 0;
    ternary_pack_trits(v, nelem, buf);
    return tw_push(w, name, TTD_TRIT5, shape, nd, buf, buf);
}

int ternary_tw_add_f32(TernaryTWriter *w, const char *name, const float *data, size_t n) {
    return ternary_tw_add(w, name, TTD_F32, &n, 1, data);
}
int ternary_tw_add_i8(TernaryTWriter *w, const char *name, const int8_t *data, size_t n) {
    return ternary_tw_add(w, name, TTD_I8, &n, 1, data);
}

static size_t align_up(size_t x) { return (x + (TERNARY_TFILE_ALIGN - 1)) & ~(size_t)(TERNARY_TFILE_ALIGN - 1); }

/* Escribe todo el archivo en 'f'. */
static int tw_emit(const TernaryTWriter *w, FILE *f) {
    TernaryJBuf j; ternary_jbuf_init(&j);
    ternary_jbuf_open(&j, '{');
    ternary_jbuf_key(&j, "__metadata__");
    ternary_jbuf_open(&j, '{');
    ternary_jbuf_kstr(&j, "format", TERNARY_TFILE_FORMAT);
    char ver[16]; snprintf(ver, sizeof ver, "%d", TERNARY_TFILE_VERSION);
    ternary_jbuf_kstr(&j, "version", ver);
    for (size_t i = 0; i < w->nm; i++) ternary_jbuf_kstr(&j, w->mk[i], w->mv[i]);
    ternary_jbuf_close(&j, '}');
    size_t off = 0;
    for (size_t i = 0; i < w->n; i++) {
        const WEntry *e = &w->e[i];
        size_t start = align_up(off), end = start + e->nbytes;
        ternary_jbuf_key(&j, e->name);
        ternary_jbuf_open(&j, '{');
        ternary_jbuf_kstr(&j, "dtype", ternary_tdtype_name(e->dtype));
        ternary_jbuf_key(&j, "shape");
        ternary_jbuf_open(&j, '[');
        for (size_t d = 0; d < e->nd; d++) ternary_jbuf_int(&j, (int64_t)e->shape[d]);
        ternary_jbuf_close(&j, ']');
        ternary_jbuf_key(&j, "data_offsets");
        ternary_jbuf_open(&j, '[');
        ternary_jbuf_int(&j, (int64_t)start); ternary_jbuf_int(&j, (int64_t)end);
        ternary_jbuf_close(&j, ']');
        ternary_jbuf_close(&j, '}');
        off = end;
    }
    ternary_jbuf_close(&j, '}');
    if (j.failed) { ternary_jbuf_free(&j); return 0; }
    size_t total = 8 + j.len, pad = align_up(total) - total;
    uint64_t n = (uint64_t)(j.len + pad);
    uint8_t lenb[8];
    for (int i = 0; i < 8; i++) lenb[i] = (uint8_t)(n >> (8 * i));    /* little-endian explicito */
    static const char SPACES[65] = "                                                                ";
    static const uint8_t ZEROS[64] = {0};
    int ok = fwrite(lenb, 1, 8, f) == 8 && fwrite(j.buf, 1, j.len, f) == j.len && fwrite(SPACES, 1, pad, f) == pad;
    ternary_jbuf_free(&j);
    off = 0;
    for (size_t i = 0; ok && i < w->n; i++) {
        const WEntry *e = &w->e[i];
        size_t start = align_up(off);
        ok = fwrite(ZEROS, 1, start - off, f) == start - off;
        if (ok && e->nbytes) ok = fwrite(e->data, 1, e->nbytes, f) == e->nbytes;
        off = start + e->nbytes;
    }
    return ok;
}

int ternary_tw_write_file(const TernaryTWriter *w, const char *path) {
    if (!w || !path) return 0;
    size_t pl = strlen(path);
    char *tmp = (char *)malloc(pl + 5);
    if (!tmp) return 0;
    memcpy(tmp, path, pl); memcpy(tmp + pl, ".tmp", 5);
    FILE *f = fopen(tmp, "wb");
    if (!f) { free(tmp); return 0; }
    int ok = tw_emit(w, f);
    if (fclose(f) != 0) ok = 0;
    if (ok && rename(tmp, path) != 0) ok = 0;
    if (!ok) remove(tmp);
    free(tmp);
    return ok;
}

int ternary_tw_serialize(const TernaryTWriter *w, uint8_t **out, size_t *out_len) {
#if defined(_WIN32)
    (void)w; (void)out; (void)out_len; return 0;
#else
    char *buf = NULL; size_t size = 0;
    FILE *f = open_memstream(&buf, &size);
    if (!f) return 0;
    int ok = tw_emit(w, f);
    if (fclose(f) != 0) ok = 0;
    if (!ok) { free(buf); return 0; }
    *out = (uint8_t *)buf; *out_len = size;
    return 1;
#endif
}

/* ====================================================================================================== */
/* Lector                                                                                                 */
/* ====================================================================================================== */
struct TernaryTReader {
    const uint8_t *base; size_t len;     /* archivo / buffer completo */
    int mapped; uint8_t *owned;          /* mapped: munmap(base,len); owned: free(owned) */
    TernaryTInfo *t; char **names; size_t n;
    char **mk, **mv; size_t nm;
};

void ternary_tr_close(TernaryTReader *r) {
    if (!r) return;
#if !defined(_WIN32)
    if (r->mapped) munmap((void *)r->base, r->len);
#endif
    free(r->owned);
    for (size_t i = 0; i < r->n; i++) free(r->names[i]);
    for (size_t i = 0; i < r->nm; i++) { free(r->mk[i]); free(r->mv[i]); }
    free(r->t); free(r->names); free(r->mk); free(r->mv); free(r);
}

static int cmp_info(const void *a, const void *b) {
    return strcmp(((const TernaryTInfo *)a)->name, ((const TernaryTInfo *)b)->name);
}

typedef struct { size_t start, end; } Span;
static int cmp_span(const void *a, const void *b) {
    const Span *x = (const Span *)a, *y = (const Span *)b;
    return x->start < y->start ? -1 : x->start > y->start;
}

static TernaryTReader *parse(const uint8_t *data, size_t len, char *err, size_t cap) {
    if (len < 8) { set_err(err, cap, "archivo demasiado corto (%zu bytes) para ser un model.ternary", len); return NULL; }
    uint64_t hn = 0;
    for (int i = 0; i < 8; i++) hn |= (uint64_t)data[i] << (8 * i);
    if (hn > len - 8) { set_err(err, cap, "cabecera de %llu bytes no cabe en el archivo (%zu bytes): truncado o corrupto", (unsigned long long)hn, len); return NULL; }
    if (hn > ((uint64_t)1 << 28)) { set_err(err, cap, "cabecera de %llu bytes: absurdamente grande", (unsigned long long)hn); return NULL; }
    if ((8 + hn) % TERNARY_TFILE_ALIGN != 0) { set_err(err, cap, "el inicio de los datos (%llu) no esta alineado a %d bytes", (unsigned long long)(8 + hn), TERNARY_TFILE_ALIGN); return NULL; }
    const uint8_t *blob = data + 8 + hn;
    const size_t blob_len = len - 8 - (size_t)hn;
    char jerr[160] = "";
    TernaryJNode *root = ternary_jdom_parse((const char *)data + 8, (size_t)hn, jerr, sizeof jerr);
    if (!root) { set_err(err, cap, "cabecera JSON invalida: %s", jerr); return NULL; }
    if (root->type != TJ_OBJ) { set_err(err, cap, "la cabecera JSON no es un objeto"); ternary_jdom_free(root); return NULL; }

    TernaryTReader *r = (TernaryTReader *)calloc(1, sizeof(TernaryTReader));
    Span *spans = (Span *)calloc(root->n ? root->n : 1, sizeof(Span));
    if (!r || !spans) { set_err(err, cap, "sin memoria"); free(r); free(spans); ternary_jdom_free(root); return NULL; }
    r->base = data; r->len = len;
    r->t = (TernaryTInfo *)calloc(root->n ? root->n : 1, sizeof(TernaryTInfo));
    r->names = (char **)calloc(root->n ? root->n : 1, sizeof(char *));
    size_t meta_cap = 1;
    { const TernaryJNode *mdn = ternary_jdom_get(root, "__metadata__"); if (mdn && mdn->type == TJ_OBJ) meta_cap += mdn->n; }
    r->mk = (char **)calloc(meta_cap, sizeof(char *));
    r->mv = (char **)calloc(meta_cap, sizeof(char *));
    if (!r->t || !r->names || !r->mk || !r->mv) { set_err(err, cap, "sin memoria"); goto fail; }

    for (size_t k = 0; k < root->n; k++) {
        const char *key = root->keys[k];
        const TernaryJNode *v = &root->kids[k];
        if (!strcmp(key, "__metadata__")) {
            if (v->type != TJ_OBJ) { set_err(err, cap, "__metadata__ no es un objeto"); goto fail; }
            for (size_t m = 0; m < v->n; m++) {
                if (v->kids[m].type != TJ_STR) { set_err(err, cap, "__metadata__.%s no es un string", v->keys[m]); goto fail; }
                r->mk[r->nm] = strdup(v->keys[m]); r->mv[r->nm] = strdup(v->kids[m].str);
                if (!r->mk[r->nm] || !r->mv[r->nm]) { set_err(err, cap, "sin memoria"); goto fail; }
                r->nm++;
            }
            continue;
        }
        if (v->type != TJ_OBJ) { set_err(err, cap, "tensor '%s': la entrada no es un objeto", key); goto fail; }
        for (size_t q = 0; q < r->n; q++)
            if (!strcmp(r->names[q], key)) { set_err(err, cap, "tensor '%s' repetido en la cabecera", key); goto fail; }
        const char *dts = ternary_jdom_get_str(v, "dtype");
        TernaryTDType dt;
        if (!dts) { set_err(err, cap, "tensor '%s': falta dtype", key); goto fail; }
        if (!dtype_by_name(dts, &dt)) { set_err(err, cap, "tensor '%s': dtype desconocido '%s' (archivo de una version futura?)", key, dts); goto fail; }
        const TernaryJNode *sh = ternary_jdom_get(v, "shape"), *off = ternary_jdom_get(v, "data_offsets");
        if (!sh || sh->type != TJ_ARR || sh->n > TERNARY_TFILE_MAX_DIMS) { set_err(err, cap, "tensor '%s': shape invalida", key); goto fail; }
        if (!off || off->type != TJ_ARR || off->n != 2 || !off->kids[0].is_int || !off->kids[1].is_int ||
            off->kids[0].i < 0 || off->kids[1].i < off->kids[0].i) { set_err(err, cap, "tensor '%s': data_offsets invalidos", key); goto fail; }
        TernaryTInfo *ti = &r->t[r->n];
        r->names[r->n] = strdup(key);
        if (!r->names[r->n]) { set_err(err, cap, "sin memoria"); goto fail; }
        ti->name = r->names[r->n];
        ti->dtype = dt; ti->ndims = sh->n;
        for (size_t d = 0; d < sh->n; d++) {
            if (!sh->kids[d].is_int || sh->kids[d].i < 0) { set_err(err, cap, "tensor '%s': shape invalida", key); r->n++; goto fail; }
            ti->shape[d] = (size_t)sh->kids[d].i;
        }
        r->n++;
        if (!shape_nelem(ti->shape, ti->ndims, &ti->nelem)) { set_err(err, cap, "tensor '%s': shape desborda", key); goto fail; }
        ti->nbytes = ternary_tdtype_bytes(dt, ti->nelem);
        size_t s0 = (size_t)off->kids[0].i, s1 = (size_t)off->kids[1].i;
        if (s1 - s0 != ti->nbytes) { set_err(err, cap, "tensor '%s': data_offsets miden %zu bytes pero %s x %zu elementos ocupan %zu", key, s1 - s0, dts, ti->nelem, ti->nbytes); goto fail; }
        if (s0 % TERNARY_TFILE_ALIGN != 0) { set_err(err, cap, "tensor '%s': inicio %zu sin alinear a %d", key, s0, TERNARY_TFILE_ALIGN); goto fail; }
        if (s1 > blob_len) { set_err(err, cap, "tensor '%s' llega al byte %zu pero el bloque de datos tiene %zu: archivo truncado", key, s1, blob_len); goto fail; }
        ti->data = blob + s0;
        spans[r->n - 1].start = s0; spans[r->n - 1].end = s1;
    }
    {
        qsort(spans, r->n, sizeof(Span), cmp_span);
        for (size_t i = 1; i < r->n; i++)
            if (spans[i].start < spans[i - 1].end) { set_err(err, cap, "hay tensores que se solapan en el archivo"); goto fail; }
    }
    {
        const char *fmt = ternary_tr_meta(r, "format"), *ver = ternary_tr_meta(r, "version");
        if (!fmt || strcmp(fmt, TERNARY_TFILE_FORMAT)) { set_err(err, cap, "__metadata__.format no es '%s'", TERNARY_TFILE_FORMAT); goto fail; }
        char want[16]; snprintf(want, sizeof want, "%d", TERNARY_TFILE_VERSION);
        if (!ver || strcmp(ver, want)) { set_err(err, cap, "version de formato '%s' no soportada (esta libreria lee la %s)", ver ? ver : "?", want); goto fail; }
    }
    qsort(r->t, r->n, sizeof(TernaryTInfo), cmp_info);
    free(spans);
    ternary_jdom_free(root);
    return r;
fail:
    free(spans);
    ternary_jdom_free(root);
    ternary_tr_close(r);
    return NULL;
}

TernaryTReader *ternary_tr_from_memory(const void *data, size_t len, char *err, size_t err_cap) {
    return parse((const uint8_t *)data, len, err, err_cap);
}

TernaryTReader *ternary_tr_open(const char *path, char *err, size_t err_cap) {
#if !defined(_WIN32)
    int fd = open(path, O_RDONLY);
    if (fd < 0) { set_err(err, err_cap, "no se puede abrir '%s': %s", path, strerror(errno)); return NULL; }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 0) { set_err(err, err_cap, "no se puede leer '%s'", path); close(fd); return NULL; }
    size_t len = (size_t)st.st_size;
    if (len == 0) { set_err(err, err_cap, "'%s' esta vacio", path); close(fd); return NULL; }
    void *m = mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, 0);
    if (m != MAP_FAILED) {
        close(fd);
        TernaryTReader *r = parse((const uint8_t *)m, len, err, err_cap);
        if (!r) { munmap(m, len); return NULL; }
        r->mapped = 1;
        return r;
    }
    uint8_t *buf = (uint8_t *)malloc(len);
    size_t got = 0;
    while (buf && got < len) {
        ssize_t k = read(fd, buf + got, len - got);
        if (k <= 0) break;
        got += (size_t)k;
    }
    close(fd);
    if (!buf || got != len) { free(buf); set_err(err, err_cap, "no se puede leer '%s'", path); return NULL; }
#else
    FILE *f = fopen(path, "rb");
    if (!f) { set_err(err, err_cap, "no se puede abrir '%s'", path); return NULL; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); set_err(err, err_cap, "'%s' esta vacio", path); return NULL; }
    size_t len = (size_t)sz;
    uint8_t *buf = (uint8_t *)malloc(len);
    if (!buf || fread(buf, 1, len, f) != len) { free(buf); fclose(f); set_err(err, err_cap, "no se puede leer '%s'", path); return NULL; }
    fclose(f);
#endif
    TernaryTReader *r = parse(buf, len, err, err_cap);
    if (!r) { free(buf); return NULL; }
    r->owned = buf;
    return r;
}

size_t ternary_tr_count(const TernaryTReader *r) { return r ? r->n : 0; }
const TernaryTInfo *ternary_tr_at(const TernaryTReader *r, size_t i) { return (r && i < r->n) ? &r->t[i] : NULL; }

const TernaryTInfo *ternary_tr_find(const TernaryTReader *r, const char *name) {
    if (!r || !name) return NULL;
    TernaryTInfo key; memset(&key, 0, sizeof key); key.name = name;
    return (const TernaryTInfo *)bsearch(&key, r->t, r->n, sizeof(TernaryTInfo), cmp_info);
}

const char *ternary_tr_meta(const TernaryTReader *r, const char *key) {
    if (!r) return NULL;
    for (size_t i = 0; i < r->nm; i++) if (!strcmp(r->mk[i], key)) return r->mv[i];
    return NULL;
}

static void shape_str(char *out, size_t cap, const size_t *s, size_t nd) {
    size_t o = 0;
    o += (size_t)snprintf(out + o, o < cap ? cap - o : 0, "[");
    for (size_t i = 0; i < nd && o < cap; i++) o += (size_t)snprintf(out + o, cap - o, i ? ",%zu" : "%zu", s[i]);
    if (o < cap) snprintf(out + o, cap - o, "]");
}

const TernaryTInfo *ternary_tr_expect(const TernaryTReader *r, const char *name, TernaryTDType dt,
                                      const size_t *shape, size_t nd, char *err, size_t err_cap) {
    const TernaryTInfo *t = ternary_tr_find(r, name);
    if (!t) { set_err(err, err_cap, "falta el tensor '%s' en model.ternary", name); return NULL; }
    int same = t->dtype == dt && t->ndims == nd;
    for (size_t i = 0; same && i < nd; i++) same = t->shape[i] == shape[i];
    if (!same) {
        char a[96], b[96];
        shape_str(a, sizeof a, shape, nd); shape_str(b, sizeof b, t->shape, t->ndims);
        set_err(err, err_cap, "tensor '%s': se esperaba %s %s y el archivo trae %s %s", name, ternary_tdtype_name(dt), a, ternary_tdtype_name(t->dtype), b);
        return NULL;
    }
    return t;
}

int ternary_tinfo_read_trits(const TernaryTInfo *t, TernaryVec *out) {
    if (!t || t->dtype != TTD_TRIT5 || !out || out->n != t->nelem) return 0;
    return ternary_unpack_trits(t->data, t->nelem, out);
}

float ternary_tinfo_f32(const TernaryTInfo *t, size_t i) {
    float v;
    memcpy(&v, t->data + 4 * i, 4);
    return v;
}
