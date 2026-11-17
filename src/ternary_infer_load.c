/* Carga de modelos: model.ternary (TRIT5, F32) + config.json -> TernaryInferModel. */
#define _POSIX_C_SOURCE 200809L
#include "ternary_mem.h"
#include "ternary_infer_internal.h"
#include "ternary_json.h"
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <sys/stat.h>

#ifndef TERNARYAI_INFER_QK_LIM_MULT
#define TERNARYAI_INFER_QK_LIM_MULT 128      /* igual que TERNARY_GQA_QK_LIM_MULT de la libreria */
#endif

#ifndef TERNARYAI_FUSED
static void set_err(char *err, size_t cap, const char *fmt, ...) {
    if (!err || !cap) return;
    va_list ap; va_start(ap, fmt); vsnprintf(err, cap, fmt, ap); va_end(ap);
}
#endif

#ifndef TERNARYAI_FUSED   /* la libreria fusionada no lee archivos ni config: el modelo viene generado en C */
/* ---------------------------------------------------------------- archivos */
static char *read_file(const char *path, size_t *len_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz < 0 || fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    char *buf = (char *)ternary_malloc64((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    if (got != (size_t)sz) { free(buf); return NULL; }
    buf[sz] = 0;
    if (len_out) *len_out = (size_t)sz;
    return buf;
}

static uint64_t fnv1a(uint64_t h, const void *data, size_t n) {
    const uint8_t *p = (const uint8_t *)data;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

/* ---------------------------------------------------------------- lector de model.ternary */
typedef enum { TD_F32 = 0, TD_I32, TD_I8, TD_U8, TD_TRIT5 } TDType;
typedef struct { const char *name; TDType dt; size_t nd, shape[8]; size_t nelem, off0, off1; } TEnt;
typedef struct {
    char *buf; size_t len;
    const uint8_t *blob; size_t blob_len;
    TEnt *e; size_t n;
    TernaryJNode *root;
} TFile;

static void tf_close(TFile *t) {
    if (t->root) ternary_jdom_free(t->root);
    free(t->e); free(t->buf);
    memset(t, 0, sizeof(*t));
}

static int tf_open(TFile *t, const char *path, char *err, size_t cap) {
    memset(t, 0, sizeof(*t));
    t->buf = read_file(path, &t->len);
    if (!t->buf) { set_err(err, cap, "no se puede leer '%s'", path); return 0; }
    if (t->len < 8) { set_err(err, cap, "'%s' demasiado corto", path); tf_close(t); return 0; }
    uint64_t hn = 0;
    for (int i = 0; i < 8; i++) hn |= (uint64_t)(uint8_t)t->buf[i] << (8 * i);
    if (hn > t->len - 8 || (8 + hn) % 64 != 0) { set_err(err, cap, "cabecera de model.ternary invalida o truncada"); tf_close(t); return 0; }
    char jerr[160] = "";
    t->root = ternary_jdom_parse(t->buf + 8, (size_t)hn, jerr, sizeof jerr);
    if (!t->root || t->root->type != TJ_OBJ) { set_err(err, cap, "cabecera JSON invalida: %s", jerr); tf_close(t); return 0; }
    t->blob = (const uint8_t *)t->buf + 8 + hn;
    t->blob_len = t->len - 8 - (size_t)hn;
    t->e = (TEnt *)ternary_calloc64(t->root->n + 1, sizeof(TEnt));
    if (!t->e) { set_err(err, cap, "sin memoria"); tf_close(t); return 0; }
    for (size_t k = 0; k < t->root->n; k++) {
        const char *key = t->root->keys[k];
        const TernaryJNode *v = &t->root->kids[k];
        if (!strcmp(key, "__metadata__")) continue;
        if (v->type != TJ_OBJ) { set_err(err, cap, "tensor '%s': entrada invalida", key); tf_close(t); return 0; }
        const char *ds = ternary_jdom_get_str(v, "dtype");
        TEnt *e = &t->e[t->n];
        e->name = key;
        if (!ds) { set_err(err, cap, "tensor '%s': falta dtype", key); tf_close(t); return 0; }
        if (!strcmp(ds, "F32")) e->dt = TD_F32; else if (!strcmp(ds, "I32")) e->dt = TD_I32; else if (!strcmp(ds, "I8")) e->dt = TD_I8;
        else if (!strcmp(ds, "U8")) e->dt = TD_U8; else if (!strcmp(ds, "TRIT5")) e->dt = TD_TRIT5;
        else { set_err(err, cap, "tensor '%s': dtype desconocido '%s'", key, ds); tf_close(t); return 0; }
        const TernaryJNode *sh = ternary_jdom_get(v, "shape"), *off = ternary_jdom_get(v, "data_offsets");
        if (!sh || sh->type != TJ_ARR || sh->n > 8 || !off || off->type != TJ_ARR || off->n != 2 || !off->kids[0].is_int || !off->kids[1].is_int ||
            off->kids[0].i < 0 || off->kids[1].i < off->kids[0].i) { set_err(err, cap, "tensor '%s': forma u offsets invalidos", key); tf_close(t); return 0; }
        e->nd = sh->n; e->nelem = 1;
        for (size_t d = 0; d < sh->n; d++) {
            if (!sh->kids[d].is_int || sh->kids[d].i < 0) { set_err(err, cap, "tensor '%s': forma invalida", key); tf_close(t); return 0; }
            e->shape[d] = (size_t)sh->kids[d].i; e->nelem *= e->shape[d];
        }
        e->off0 = (size_t)off->kids[0].i; e->off1 = (size_t)off->kids[1].i;
        size_t need = e->dt == TD_TRIT5 ? (e->nelem + 4) / 5 : e->nelem * (e->dt == TD_F32 || e->dt == TD_I32 ? 4 : 1);
        if (e->off1 - e->off0 != need || e->off1 > t->blob_len) { set_err(err, cap, "tensor '%s': tamano incoherente o archivo truncado", key); tf_close(t); return 0; }
        t->n++;
    }
    return 1;
}

static const TEnt *tf_find(const TFile *t, const char *name) {
    for (size_t i = 0; i < t->n; i++) if (!strcmp(t->e[i].name, name)) return &t->e[i];
    return NULL;
}

static int tf_f32(const TFile *t, const char *name, float *out, char *err, size_t cap) {
    const TEnt *e = tf_find(t, name);
    if (!e || e->dt != TD_F32 || e->nelem != 1) { set_err(err, cap, "falta el tensor F32 '%s' (escala)", name); return 0; }
    memcpy(out, t->blob + e->off0, 4);
    return 1;
}

static int8_t TRIT_LUT[243][5];
static int trit_lut_ready = 0;
static void trit_lut_init(void) {
    if (trit_lut_ready) return;
    for (int b = 0; b < 243; b++) { int v = b; for (int k = 0; k < 5; k++) { TRIT_LUT[b][k] = (int8_t)(v % 3 - 1); v /= 3; } }
    trit_lut_ready = 1;
}

/* Lee un tensor TRIT5 de forma [rows, cols] a int8 {-1,0,1} (malloc). */
static int8_t *tf_trits(const TFile *t, const char *name, size_t rows, size_t cols, char *err, size_t cap) {
    const TEnt *e = tf_find(t, name);
    if (!e) { set_err(err, cap, "falta el tensor '%s' en model.ternary", name); return NULL; }
    if (e->dt != TD_TRIT5 || e->nelem != rows * cols) {
        set_err(err, cap, "tensor '%s': se esperaban %zu x %zu trits y el archivo trae %zu elementos", name, rows, cols, e->nelem); return NULL;
    }
    trit_lut_init();
    int8_t *w = (int8_t *)ternary_malloc64((size_t)rows * (size_t)cols + 1);
    if (!w) { set_err(err, cap, "sin memoria"); return NULL; }
    const uint8_t *src = t->blob + e->off0;
    size_t n = rows * cols;
    for (size_t i = 0; i < n; i++) {
        const uint8_t b = src[i / 5];
        if (b > 242) { set_err(err, cap, "tensor '%s': byte %u fuera de 0..242 (archivo corrupto)", name, b); free(w); return NULL; }
        w[i] = TRIT_LUT[b][i % 5];
    }
    return w;
}

/* ---------------------------------------------------------------- origen de los tensores: carpeta (model.ternary) o .ternam */
typedef struct { const TFile *tf; const TnamFile *tn; TernaryInferRecList *rec; } Src;

/* ---------------------------------------------------------------- matrices (2 bits por peso) */
#endif

static inline size_t pk_index(int layout, int cp, int o, int i, int *shift) {
    if (layout == TERNARYAI_INFER_LAYOUT_AVX) {
        const int g = i >> 4, j = (i & 15) >> 2, c = i & 3, rb = o >> 4, r = o & 15;
        *shift = 2 * j;
        return (((size_t)rb * (size_t)(cp >> 4) + (size_t)g) * 16 + (size_t)r) * 4 + (size_t)c;
    } else if (layout == TERNARYAI_INFER_LAYOUT_NEON) {
        const int g = i >> 6, n = (cp - 64 * g < 64) ? cp - 64 * g : 64, B = n >> 2, r = i & 63;    /* grupo de n columnas = B bytes; campo j, byte c */
        const int j = r / B, c = r % B;
        *shift = 2 * j;
        return (size_t)o * (size_t)(cp >> 2) + (size_t)(16 * g + c);
    } else {
        const int T = cp >> 3, p = i / T, t = i % T;          /* palabra de 16 bits t de la fila, campo p (0..7) */
        *shift = 2 * (p & 3);
        return (size_t)o * (size_t)(cp >> 2) + (size_t)(2 * t + (p >> 2));
    }
}

static void pack_codes(TernaryInferMat *m, const int8_t *w) {
    const size_t bytes = (size_t)m->rb * 16 * (size_t)m->cp / 4;
    memset(m->w, 0x55, bytes);                       /* todo en codigo 1 (peso 0): filas y columnas de relleno */
    for (int o = 0; o < m->rows; o++)
        for (int i = 0; i < m->cols; i++) {
            int sh; const size_t k = pk_index(m->layout, m->cp, o, i, &sh);
            m->w[k] = (uint8_t)((m->w[k] & ~(3u << sh)) | ((unsigned)(w[(size_t)o * m->cols + i] + 1) << sh));
        }
}

int ternaryai_infer_mat_build(TernaryInferMat *m, const int8_t *w, int rows, int cols, float scale, int layout) {
    memset(m, 0, sizeof(*m));
    m->rows = rows; m->cols = cols; m->scale = scale; m->layout = layout; m->owned = 1;
    m->cp = (cols + 15) & ~15;
    m->rb = (rows + 15) / 16;
    const size_t bytes = (size_t)m->rb * 16 * (size_t)m->cp / 4;
    m->w = (uint8_t *)ternary_aligned_alloc(64, (bytes + 63) / 64 * 64);
    m->S = (int32_t *)ternary_calloc64((size_t)m->rb * 16 + 16, sizeof(int32_t));
    if (!m->w || !m->S) { ternaryai_infer_mat_free(m); return 0; }
    pack_codes(m, w);
    for (int o = 0; o < rows; o++) { int s = 0; for (int i = 0; i < cols; i++) s += w[(size_t)o * cols + i]; m->S[o] = s; }
    return 1;
}

/* Cambia el orden de los pesos en el sitio (mismo tamano): se usa si el nucleo activo cambia despues de cargar. */
int ternaryai_infer_mat_relayout(TernaryInferMat *m, int layout) {
    if (!m->w || m->layout == layout) return 1;
    int8_t *tmp = (int8_t *)ternary_malloc64((size_t)m->rows * (size_t)m->cols + 1);
    if (!tmp) return 0;
    for (int o = 0; o < m->rows; o++)
        for (int i = 0; i < m->cols; i++) {
            int sh; const size_t k = pk_index(m->layout, m->cp, o, i, &sh);
            tmp[(size_t)o * m->cols + i] = (int8_t)(((m->w[k] >> sh) & 3) - 1);
        }
    m->layout = layout;
    pack_codes(m, tmp);
    free(tmp);
    return 1;
}

void ternaryai_infer_mat_free(TernaryInferMat *m) {
    if (m->owned) free(m->w);
    if (m->owned || m->s_owned) free(m->S);
    memset(m, 0, sizeof(*m));
}

#ifndef TERNARYAI_FUSED
static size_t mat_bytes(const TernaryInferMat *m) {
    return (size_t)m->rb * 16 * (size_t)m->cp / 4 + ((size_t)m->rb * 16) * 4;
}

static int load_mat(const Src *t, const char *base, int rows, int cols, TernaryInferMat *m, int avx, char *err, size_t cap) {
    if (t->tn) {                                     /* .ternam: los pesos ya vienen empaquetados; se usan en el sitio */
        const TnamEnt *e = ternaryai_tnam_find(t->tn, base);
        if (!e || e->kind != 1) { set_err(err, cap, "falta el tensor '%s' en el modelo .ternam", base); return 0; }
        memset(m, 0, sizeof(*m));
        m->rows = rows; m->cols = cols; m->cp = (cols + 15) & ~15; m->rb = (rows + 15) / 16; m->layout = t->tn->layout; m->owned = 0;
        const uint64_t wl = (uint64_t)m->rb * 16 * (uint64_t)m->cp / 4, sl = (uint64_t)m->rb * 16 * 4, sl1 = (uint64_t)m->rb * 16 * 2;
        const int v1 = t->tn->ver == 1;                /* .ternam v1: S en int16 */
        if (e->rows != rows || e->cols != cols || e->w_len != wl || e->s_len != (v1 ? sl1 : sl)) {
            set_err(err, cap, "tensor '%s': se esperaban %d x %d y el archivo trae %d x %d", base, rows, cols, e->rows, e->cols); return 0;
        }
        const uint32_t sb = e->scale_bits; memcpy(&m->scale, &sb, 4);
        m->w = t->tn->base + e->w_off; if (v1) {
            const int16_t *s16 = (const int16_t *)(const void *)(t->tn->base + e->s_off);
            m->S = (int32_t *)ternary_calloc64((size_t)m->rb * 16 + 16, sizeof(int32_t));
            if (!m->S) { set_err(err, cap, "sin memoria"); return 0; }
            for (size_t i = 0; i < (size_t)m->rb * 16; i++) m->S[i] = s16[i];
            m->s_owned = 1;
        } else m->S = (int32_t *)(void *)(t->tn->base + e->s_off);
        if (t->rec && !ternaryai_infer_rec_push(t->rec, base, 1, rows, cols, e->scale_bits, m->w, (size_t)wl, m->S, (size_t)sl, NULL)) { set_err(err, cap, "sin memoria"); return 0; }
        return 1;
    }
    char nm[256]; float sc;
    snprintf(nm, sizeof nm, "%s.weight", base);
    int8_t *w = tf_trits(t->tf, nm, (size_t)rows, (size_t)cols, err, cap);
    if (!w) return 0;
    snprintf(nm, sizeof nm, "%s.scale", base);
    if (!tf_f32(t->tf, nm, &sc, err, cap)) { free(w); return 0; }
    int ok = ternaryai_infer_mat_build(m, w, rows, cols, sc, avx);   /* avx = orden de pesos del nucleo activo */
    free(w);
    if (!ok) { set_err(err, cap, "sin memoria"); return 0; }
    if (t->rec) {
        uint32_t sb; memcpy(&sb, &sc, 4);
        if (!ternaryai_infer_rec_push(t->rec, base, 1, rows, cols, sb, m->w, (size_t)m->rb * 16 * (size_t)m->cp / 4, m->S, (size_t)m->rb * 16 * 4, NULL)) { set_err(err, cap, "sin memoria"); return 0; }
    }
    return 1;
}

static int load_norm(const Src *t, const char *name, int dim, float eps, TernaryInferNorm *n, char *err, size_t cap) {
    char nm[256];
    int8_t *g = NULL; int own = 0;
    if (t->tn) {
        snprintf(nm, sizeof nm, "%s.gain", name);
        const TnamEnt *e = ternaryai_tnam_find(t->tn, nm);
        if (!e || e->kind != 3 || e->cols != dim || e->w_len != (uint64_t)dim) { set_err(err, cap, "falta la ganancia '%s' (o su tamano no coincide) en el modelo .ternam", nm); return 0; }
        const uint32_t sb = e->scale_bits; memcpy(&n->scale, &sb, 4);
        g = (int8_t *)(t->tn->base + e->w_off);
        for (int j = 0; j < dim; j++) if (g[j] < -1 || g[j] > 1) { set_err(err, cap, "ganancia '%s': valores fuera de {-1,0,1}", nm); return 0; }
        if (t->rec && !ternaryai_infer_rec_push(t->rec, nm, 3, 1, dim, e->scale_bits, g, (size_t)dim, NULL, 0, NULL)) { set_err(err, cap, "sin memoria"); return 0; }
    } else {
        snprintf(nm, sizeof nm, "%s.gain.weight", name);
        g = tf_trits(t->tf, nm, 1, (size_t)dim, err, cap);
        if (!g) return 0;
        own = 1;
        snprintf(nm, sizeof nm, "%s.gain.scale", name);
        if (!tf_f32(t->tf, nm, &n->scale, err, cap)) { free(g); return 0; }
    }
    n->dim = dim; n->eps = eps;
    n->gmul = (float *)ternary_malloc64((size_t)dim * sizeof(float));
    if (!n->gmul) { if (own) free(g); set_err(err, cap, "sin memoria"); return 0; }
    for (int j = 0; j < dim; j++) n->gmul[j] = (float)g[j];
    if (own) {
        if (t->rec) {                                /* el registro se queda con g */
            uint32_t sb; memcpy(&sb, &n->scale, 4);
            snprintf(nm, sizeof nm, "%s.gain", name);
            if (!ternaryai_infer_rec_push(t->rec, nm, 3, 1, dim, sb, g, (size_t)dim, NULL, 0, g)) { free(g); set_err(err, cap, "sin memoria"); return 0; }
        } else free(g);
    }
    return 1;
}

/* ---------------------------------------------------------------- configuracion */
static int cfg_int(const TernaryJNode *o, const char *k, int *out, int required, char *err, size_t cap, const char *where) {
    int64_t v;
    if (ternary_jdom_get_int(o, k, &v)) { *out = (int)v; return 1; }
    if (required) set_err(err, cap, "config.json: falta '%s' en %s", k, where);
    return !required;
}

static int check_u8_act(const TernaryJNode *layer, const char *name, char *err, size_t cap) {
    const char *a = ternary_jdom_get_str(layer, "activation");
    if (a && strcmp(a, "uint8") != 0) { set_err(err, cap, "capa '%s': activacion '%s' no soportada (solo uint8)", name, a); return 0; }
    return 1;
}

static void gqa_exp_table(TernaryInferGqa *g) {
    const int lim = TERNARYAI_INFER_QK_LIM_MULT * g->hd;
    const size_t len = 2 * (size_t)lim + 1;
    const float inv_temp = 1.0f / sqrtf((float)g->hd);
    size_t k0 = len - 1;
    float *tab = (float *)ternary_malloc64(len * sizeof(float));
    for (size_t k = 0; k < len; k++) {
        float e = (float)exp(-(double)k * (double)inv_temp);
        tab[k] = (e < 1e-20f) ? 0.0f : e;
    }
    for (size_t k = 0; k < len; k++) if (tab[k] == 0.0f) { k0 = k; break; }
    g->exp_tab = (float *)ternary_malloc64((k0 + 1) * sizeof(float));
    memcpy(g->exp_tab, tab, k0 * sizeof(float));
    g->exp_tab[k0] = 0.0f;
    g->exp_k0 = (uint32_t)k0;
    g->qk_lim = lim;
    free(tab);
}

static int load_sub(TernaryInferModel *m, const Src *t, const TernaryJNode *lay, TernaryInferSub *s, int avx, char *err, size_t cap) {
    const char *type = ternary_jdom_get_str(lay, "type"), *name = ternary_jdom_get_str(lay, "name");
    const TernaryJNode *c = ternary_jdom_get(lay, "config");
    if (!type || !name || !c) { set_err(err, cap, "config.json: capa sin type/name/config"); return 0; }
    if (!check_u8_act(lay, name, err, cap)) return 0;
    char base[256];
    if (!strcmp(type, "rmsnorm")) {
        int dim; float eps = 1e-5f;
        if (!cfg_int(c, "dim", &dim, 1, err, cap, name)) return 0;
        ternary_jdom_get_float(c, "eps", &eps);
        if (dim != m->dim) { set_err(err, cap, "capa '%s': dim %d != dim del embedding %d", name, dim, m->dim); return 0; }
        s->kind = TERNARYAI_INFER_SUB_NORM;
        return load_norm(t, name, dim, eps, &s->norm, err, cap);
    }
    if (!strcmp(type, "gqa")) {
        int E, nh, nkv, rope_mode = 1, qs = 2, window = 0, sinks = 0, smear = 0; float base_f = 10.0f;
        if (!cfg_int(c, "embed_dim", &E, 1, err, cap, name) || !cfg_int(c, "n_heads", &nh, 1, err, cap, name) ||
            !cfg_int(c, "n_kv_heads", &nkv, 1, err, cap, name)) return 0;
        cfg_int(c, "rope_mode", &rope_mode, 0, err, cap, name); cfg_int(c, "qk_shift", &qs, 0, err, cap, name);
        cfg_int(c, "window", &window, 0, err, cap, name); cfg_int(c, "sinks", &sinks, 0, err, cap, name);
        ternary_jdom_get_float(c, "rope_base", &base_f); ternary_jdom_get_bool(c, "smear", &smear);
        if (rope_mode != 1) { set_err(err, cap, "capa '%s': solo se soporta rope_mode 1 (RoPE entera con puntaje entero)", name); return 0; }
        if (E != m->dim || nh <= 0 || nkv <= 0 || nh % nkv || E % nh || (E / nh) % 2) { set_err(err, cap, "capa '%s': geometria GQA invalida", name); return 0; }
        TernaryInferGqa *g = &s->gqa;
        s->kind = TERNARYAI_INFER_SUB_GQA;
        if (nkv * (E / nh) > 4096) { set_err(err, cap, "capa '%s': n_kv_heads*head_dim mayor que 4096 no soportado", name); return 0; }
        g->n_heads = nh; g->n_kv = nkv; g->hd = E / nh; g->group = nh / nkv; g->qw = nh * g->hd; g->kvw = nkv * g->hd;
        g->shift = qs; g->window = window; g->sinks = sinks; g->smear = smear; g->rope_base = base_f;
        if (m->rope_hd == 0) { m->rope_hd = g->hd; m->rope_base = base_f; }
        else if (m->rope_hd != g->hd || m->rope_base != base_f) { set_err(err, cap, "capas GQA con distinto head_dim o rope_base: no soportado"); return 0; }
        snprintf(base, sizeof base, "%s.q_proj", name); if (!load_mat(t, base, g->qw, E, &g->wq, avx, err, cap)) return 0;
        snprintf(base, sizeof base, "%s.k_proj", name); if (!load_mat(t, base, g->kvw, E, &g->wk, avx, err, cap)) return 0;
        snprintf(base, sizeof base, "%s.v_proj", name); if (!load_mat(t, base, g->kvw, E, &g->wv, avx, err, cap)) return 0;
        snprintf(base, sizeof base, "%s.o_proj", name); if (!load_mat(t, base, E, g->qw, &g->wo, avx, err, cap)) return 0;
        gqa_exp_table(g);
        return 1;
    }
    if (!strcmp(type, "swiglu")) {
        int dim, hid, gm = 2;
        if (!cfg_int(c, "dim", &dim, 1, err, cap, name) || !cfg_int(c, "hidden", &hid, 1, err, cap, name)) return 0;
        cfg_int(c, "gate_mode", &gm, 0, err, cap, name);
        if (dim != m->dim || gm < 0 || gm > 2) { set_err(err, cap, "capa '%s': dim o gate_mode invalidos", name); return 0; }
        TernaryInferFfn *f = &s->ffn;
        s->kind = TERNARYAI_INFER_SUB_FFN;
        f->dim = dim; f->hidden = hid; f->gate_mode = gm;
        snprintf(base, sizeof base, "%s.up_proj", name); if (!load_mat(t, base, hid, dim, &f->up, avx, err, cap)) return 0;
        snprintf(base, sizeof base, "%s.down_proj", name); if (!load_mat(t, base, dim, hid, &f->down, avx, err, cap)) return 0;
        if (gm != 2) { snprintf(base, sizeof base, "%s.gate_proj", name); if (!load_mat(t, base, hid, dim, &f->gate, avx, err, cap)) return 0; }
        return 1;
    }
    set_err(err, cap, "tipo de capa '%s' no soportado por el motor (capa '%s')", type, name);
    return 0;
}

#endif

#ifndef TERNARYAI_FUSED
static void sub_free(TernaryInferSub *s) {
    if (s->kind == TERNARYAI_INFER_SUB_NORM) free(s->norm.gmul);
    else if (s->kind == TERNARYAI_INFER_SUB_GQA) { ternaryai_infer_mat_free(&s->gqa.wq); ternaryai_infer_mat_free(&s->gqa.wk); ternaryai_infer_mat_free(&s->gqa.wv); ternaryai_infer_mat_free(&s->gqa.wo); free(s->gqa.exp_tab); }
    else { ternaryai_infer_mat_free(&s->ffn.up); ternaryai_infer_mat_free(&s->ffn.down); ternaryai_infer_mat_free(&s->ffn.gate); }
}

#endif

void ternaryai_infer_model_free(TernaryInferModel *m) {
    if (!m) return;
#ifdef TERNARYAI_FUSED
    return;                                  /* el modelo incrustado es estatico y vive tanto como la libreria */
#else
    for (int i = 0; i < m->n_top; i++) {
        if (m->top[i].kind == TERNARYAI_INFER_TOP_NORM) free(m->top[i].norm.gmul);
        else { for (int j = 0; j < m->top[i].n_sub; j++) sub_free(&m->top[i].sub[j]); free(m->top[i].sub); }
    }
    free(m->top); if (m->emb_owned) free(m->emb); ternaryai_infer_mat_free(&m->head); free(m->rope_c); free(m->rope_s);
    if (m->tok) ternary_tokenizer_free(m->tok);
    ternaryai_tnam_release(m->blob, m->blob_len, m->blob_mmap);
    free(m->dir); free(m);
#endif
}

/* ---------------------------------------------------------------- orden de pesos segun el nucleo activo */
void ternaryai_infer_model_sync(TernaryInferModel *m) {
    const int want = ternaryai_infer_kern->layout;
    if (m->layout == want) return;
    ternaryai_infer_mat_relayout(&m->head, want);
    for (int i = 0; i < m->n_top; i++) {
        if (m->top[i].kind != TERNARYAI_INFER_TOP_RESIDUAL) continue;
        for (int j = 0; j < m->top[i].n_sub; j++) {
            TernaryInferSub *s = &m->top[i].sub[j];
            if (s->kind == TERNARYAI_INFER_SUB_GQA) { ternaryai_infer_mat_relayout(&s->gqa.wq, want); ternaryai_infer_mat_relayout(&s->gqa.wk, want); ternaryai_infer_mat_relayout(&s->gqa.wv, want); ternaryai_infer_mat_relayout(&s->gqa.wo, want); }
            else if (s->kind == TERNARYAI_INFER_SUB_FFN) { ternaryai_infer_mat_relayout(&s->ffn.up, want); ternaryai_infer_mat_relayout(&s->ffn.down, want); if (s->ffn.gate_mode != 2) ternaryai_infer_mat_relayout(&s->ffn.gate, want); }
        }
    }
    m->layout = want;
}

/* ---------------------------------------------------------------- RoPE perezosa */
void ternaryai_infer_rope_ensure(TernaryInferModel *m, int pos) {
    if (pos < m->rope_len || m->rope_hd == 0) return;
    int want = m->rope_len ? m->rope_len : m->seq_len;
    while (want <= pos) want *= 2;
    const int P = m->rope_hd / 2;
    int16_t *c = (int16_t *)realloc(m->rope_c, (size_t)want * P * sizeof(int16_t));
    int16_t *s = (int16_t *)realloc(m->rope_s, (size_t)want * P * sizeof(int16_t));
    if (!c || !s) { free(c == m->rope_c ? NULL : c); return; }
    m->rope_c = c; m->rope_s = s;
    for (int p = m->rope_len; p < want; p++)
        for (int i = 0; i < P; i++) {
            const double theta = pow((double)m->rope_base, -2.0 * (double)i / (double)m->rope_hd);
            const double ang = (double)p * theta;
            c[(size_t)p * P + i] = (int16_t)lround(cos(ang) * 256.0);
            s[(size_t)p * P + i] = (int16_t)lround(sin(ang) * 256.0);
        }
    m->rope_len = want;
}

#ifndef TERNARYAI_FUSED
/* ---------------------------------------------------------------- carga principal */
typedef struct { char *cfg; size_t cfg_len; char *tok; size_t tok_len; } ExportInfo;

static int path_is_file(const char *p) {
    struct stat st;
    if (stat(p, &st) == 0) return S_ISREG(st.st_mode) ? 1 : 0;
    return 0;
}

static TernaryInferModel *model_load_impl(const char *dir, TernaryInferRecList *rec, ExportInfo *xi, char *err, size_t errcap) {
    if (!dir) { set_err(err, errcap, "ruta nula"); return NULL; }
    char path[1024];
    const int is_tnam = path_is_file(dir);
    TFile tf; memset(&tf, 0, sizeof tf);
    TnamFile tn; memset(&tn, 0, sizeof tn);
    char *ctext = NULL; size_t clen = 0;
    if (is_tnam) {
        if (!ternaryai_tnam_open(&tn, dir, err, errcap)) return NULL;
        ctext = (char *)ternary_malloc64(tn.cfg_len + 1);
        if (!ctext) { set_err(err, errcap, "sin memoria"); ternaryai_tnam_close(&tn); ternaryai_tnam_release(tn.base, tn.len, tn.is_mmap); return NULL; }
        memcpy(ctext, tn.cfg, tn.cfg_len); ctext[tn.cfg_len] = 0; clen = tn.cfg_len;
    } else {
        snprintf(path, sizeof path, "%s/config.json", dir);
        ctext = read_file(path, &clen);
        if (!ctext) { set_err(err, errcap, "no se puede leer '%s'", path); return NULL; }
    }
    char jerr[160] = "";
    TernaryJNode *cfg = ternary_jdom_parse(ctext, clen, jerr, sizeof jerr);
    if (!cfg) { set_err(err, errcap, "config.json invalido: %s", jerr); free(ctext); if (is_tnam) { ternaryai_tnam_close(&tn); ternaryai_tnam_release(tn.base, tn.len, tn.is_mmap); } return NULL; }
    TernaryInferModel *m = NULL; int ok = 0;
    if (!is_tnam) {
        snprintf(path, sizeof path, "%s/model.ternary", dir);
        if (!tf_open(&tf, path, err, errcap)) { ternary_jdom_free(cfg); free(ctext); return NULL; }
    }
    const Src src = { is_tnam ? NULL : &tf, is_tnam ? &tn : NULL, rec };
    const char *fmt = ternary_jdom_get_str(cfg, "format");
    int64_t ver = 0; ternary_jdom_get_int(cfg, "format_version", &ver);
    const TernaryJNode *layers = ternary_jdom_get(cfg, "layers");
    if (!fmt || strcmp(fmt, "ternary") || ver != 2 || !layers || layers->type != TJ_ARR || layers->n < 2) {
        set_err(err, errcap, "config.json: se espera format 'ternary', format_version 2 y una lista de capas"); goto done;
    }
    m = (TernaryInferModel *)ternary_calloc64(1, sizeof(TernaryInferModel));
    if (!m) { set_err(err, errcap, "sin memoria"); goto done; }
    m->dir = strdup(dir);
    ternaryai_infer_backend_name();                 /* fija el backend por defecto si aun no hay uno */
    const int avx = is_tnam ? tn.layout : ternaryai_infer_kern->layout;     /* un .ternam trae sus pesos en su orden; el nucleo activo los reordena si hace falta */
    m->layout = avx;
    trit_lut_init();
    /* capa 0: embedding */
    {
        const TernaryJNode *l0 = &layers->kids[0], *c = ternary_jdom_get(l0, "config");
        const char *type = ternary_jdom_get_str(l0, "type"), *name = ternary_jdom_get_str(l0, "name");
        if (!type || strcmp(type, "embedding") || !c || !name) { set_err(err, errcap, "la primera capa debe ser un embedding"); goto done; }
        int V, E, S;
        if (!cfg_int(c, "vocab_size", &V, 1, err, errcap, name) || !cfg_int(c, "embed_dim", &E, 1, err, errcap, name) ||
            !cfg_int(c, "seq_len", &S, 1, err, errcap, name)) goto done;
        m->vocab = V; m->dim = E; m->seq_len = S;
        char nm[256]; snprintf(nm, sizeof nm, "%s.weight", name);
        m->emb_rb = (E + 3) / 4;
        if (is_tnam) {
            const TnamEnt *e = ternaryai_tnam_find(&tn, name);
            if (!e || e->kind != 2 || e->rows != V || e->cols != E || e->w_len != (uint64_t)V * (uint64_t)m->emb_rb) { set_err(err, errcap, "falta el embedding '%s' (o su forma no coincide) en el modelo .ternam", name); goto done; }
            const uint32_t sb = e->scale_bits; memcpy(&m->emb_scale, &sb, 4);
            m->emb = tn.base + e->w_off; m->emb_owned = 0;
            if (rec && !ternaryai_infer_rec_push(rec, name, 2, V, E, e->scale_bits, m->emb, (size_t)V * (size_t)m->emb_rb, NULL, 0, NULL)) { set_err(err, errcap, "sin memoria"); goto done; }
        } else {
            int8_t *e8 = tf_trits(&tf, nm, (size_t)V, (size_t)E, err, errcap); if (!e8) goto done;
            snprintf(nm, sizeof nm, "%s.scale", name);
            if (!tf_f32(&tf, nm, &m->emb_scale, err, errcap)) { free(e8); goto done; }
            m->emb = (uint8_t *)ternary_malloc64((size_t)V * (size_t)m->emb_rb); m->emb_owned = 1;
            if (!m->emb) { free(e8); set_err(err, errcap, "sin memoria"); goto done; }
            memset(m->emb, 0x55, (size_t)V * (size_t)m->emb_rb);
            for (int v = 0; v < V; v++)
                for (int d = 0; d < E; d++) {
                    uint8_t *by = &m->emb[(size_t)v * (size_t)m->emb_rb + (size_t)(d >> 2)]; const int sh = 2 * (d & 3);
                    *by = (uint8_t)((*by & ~(3u << sh)) | ((unsigned)(e8[(size_t)v * E + d] + 1) << sh));
                }
            free(e8);
            if (rec) { uint32_t sb; memcpy(&sb, &m->emb_scale, 4); if (!ternaryai_infer_rec_push(rec, name, 2, V, E, sb, m->emb, (size_t)V * (size_t)m->emb_rb, NULL, 0, NULL)) { set_err(err, errcap, "sin memoria"); goto done; } }
        }
    }
    /* capas intermedias y cabeza */
    {
        const size_t nl = layers->n;
        const TernaryJNode *last = &layers->kids[nl - 1];
        const char *lt = ternary_jdom_get_str(last, "type");
        if (!lt || strcmp(lt, "lm_head")) { set_err(err, errcap, "la ultima capa debe ser lm_head"); goto done; }
        m->top = (TernaryInferTop *)ternary_calloc64(nl, sizeof(TernaryInferTop));
        if (!m->top) { set_err(err, errcap, "sin memoria"); goto done; }
        for (size_t li = 1; li + 1 < nl; li++) {
            const TernaryJNode *lay = &layers->kids[li], *c = ternary_jdom_get(lay, "config");
            const char *type = ternary_jdom_get_str(lay, "type"), *name = ternary_jdom_get_str(lay, "name");
            if (!type || !name || !c) { set_err(err, errcap, "config.json: capa %zu sin type/name/config", li); goto done; }
            TernaryInferTop *tp = &m->top[m->n_top];
            if (!strcmp(type, "residual")) {
                const TernaryJNode *sl = ternary_jdom_get(c, "layers");
                if (!sl || sl->type != TJ_ARR || sl->n == 0 || sl->n > TERNARYAI_INFER_MAX_SUB) { set_err(err, errcap, "residual '%s': lista de sub-capas invalida", name); goto done; }
                tp->kind = TERNARYAI_INFER_TOP_RESIDUAL;
                tp->sub = (TernaryInferSub *)ternary_calloc64(sl->n, sizeof(TernaryInferSub));
                if (!tp->sub) { set_err(err, errcap, "sin memoria"); goto done; }
                for (size_t j = 0; j < sl->n; j++) {
                    tp->n_sub = (int)j + 1;
                    TernaryInferSub *sb = &tp->sub[j];
                    if (!load_sub(m, &src, &sl->kids[j], sb, avx, err, errcap)) { m->n_top++; goto done; }
                    if (j == 0 && sb->kind != TERNARYAI_INFER_SUB_NORM) { set_err(err, errcap, "residual '%s': la primera sub-capa debe ser rmsnorm", name); m->n_top++; goto done; }
                    if (sb->kind == TERNARYAI_INFER_SUB_GQA) sb->gqa_idx = m->n_attn++;
                    if (sb->kind == TERNARYAI_INFER_SUB_FFN) m->n_ffn++;
                }
                const TernaryInferSub *lastsub = &tp->sub[tp->n_sub - 1];
                if (lastsub->kind == TERNARYAI_INFER_SUB_NORM) { set_err(err, errcap, "residual '%s': la ultima sub-capa no puede ser rmsnorm", name); m->n_top++; goto done; }
                m->n_residual++;
            } else if (!strcmp(type, "rmsnorm")) {
                if (!check_u8_act(lay, name, err, errcap)) goto done;
                int dim; float eps = 1e-5f;
                if (!cfg_int(c, "dim", &dim, 1, err, errcap, name)) goto done;
                ternary_jdom_get_float(c, "eps", &eps);
                if (dim != m->dim) { set_err(err, errcap, "rmsnorm '%s': dim distinta", name); goto done; }
                tp->kind = TERNARYAI_INFER_TOP_NORM;
                if (!load_norm(&src, name, dim, eps, &tp->norm, err, errcap)) { m->n_top++; goto done; }
            } else { set_err(err, errcap, "capa '%s' de tipo '%s' no soportada", name, type); goto done; }
            m->n_top++;
        }
        /* el lm_head debe ir precedido de un rmsnorm (su salida cuantizada es la entrada de la cabeza) */
        if (m->n_top == 0 || m->top[m->n_top - 1].kind != TERNARYAI_INFER_TOP_NORM) { set_err(err, errcap, "antes de lm_head debe haber un rmsnorm"); goto done; }
        const TernaryJNode *c = ternary_jdom_get(last, "config");
        const char *name = ternary_jdom_get_str(last, "name");
        int V, D;
        if (!c || !name || !cfg_int(c, "vocab", &V, 1, err, errcap, name) || !cfg_int(c, "dim", &D, 1, err, errcap, name)) goto done;
        if (V != m->vocab || D != m->dim) { set_err(err, errcap, "lm_head '%s': vocab/dim no coinciden con el embedding", name); goto done; }
        if (!load_mat(&src, name, V, D, &m->head, avx, err, errcap)) goto done;
    }
    /* estadisticas */
    m->max_rows = m->head.rows; m->max_cols4 = m->head.cp; m->max_rows_body = m->dim;
    for (int i = 0; i < m->n_top; i++) {
        if (m->top[i].kind != TERNARYAI_INFER_TOP_RESIDUAL) continue;
        for (int j = 0; j < m->top[i].n_sub; j++) {
            TernaryInferSub *s = &m->top[i].sub[j];
            const TernaryInferMat *ms[4]; int nm = 0;
            if (s->kind == TERNARYAI_INFER_SUB_GQA) { ms[nm++] = &s->gqa.wq; ms[nm++] = &s->gqa.wk; ms[nm++] = &s->gqa.wv; ms[nm++] = &s->gqa.wo;
                if (s->gqa.qw > m->max_qw) m->max_qw = s->gqa.qw;
                if (s->gqa.kvw > m->max_kvw) m->max_kvw = s->gqa.kvw;
                if (s->gqa.hd > m->max_hd) m->max_hd = s->gqa.hd; }
            else if (s->kind == TERNARYAI_INFER_SUB_FFN) { ms[nm++] = &s->ffn.up; ms[nm++] = &s->ffn.down; if (s->ffn.gate_mode != 2) ms[nm++] = &s->ffn.gate; if (s->ffn.hidden > m->max_hidden) m->max_hidden = s->ffn.hidden; }
            for (int k = 0; k < nm; k++) { if (ms[k]->rows > m->max_rows) m->max_rows = ms[k]->rows; if (ms[k]->rows > m->max_rows_body) m->max_rows_body = ms[k]->rows; if (ms[k]->cp > m->max_cols4) m->max_cols4 = ms[k]->cp; }
        }
    }
    /* peso en memoria y huella */
    m->weight_bytes = (size_t)m->vocab * (size_t)m->emb_rb + mat_bytes(&m->head);
    for (int i = 0; i < m->n_top; i++) if (m->top[i].kind == TERNARYAI_INFER_TOP_RESIDUAL) for (int j = 0; j < m->top[i].n_sub; j++) {
        TernaryInferSub *s = &m->top[i].sub[j];
        if (s->kind == TERNARYAI_INFER_SUB_GQA) m->weight_bytes += mat_bytes(&s->gqa.wq) + mat_bytes(&s->gqa.wk) + mat_bytes(&s->gqa.wv) + mat_bytes(&s->gqa.wo);
        else if (s->kind == TERNARYAI_INFER_SUB_FFN) m->weight_bytes += mat_bytes(&s->ffn.up) + mat_bytes(&s->ffn.down) + (s->ffn.gate_mode != 2 ? mat_bytes(&s->ffn.gate) : 0);
    }
    for (int i = 0; i < m->n_top; i++) {
        if (m->top[i].kind == TERNARYAI_INFER_TOP_NORM) m->weight_bytes += (size_t)m->top[i].norm.dim * 4;
        else for (int j = 0; j < m->top[i].n_sub; j++) if (m->top[i].sub[j].kind == TERNARYAI_INFER_SUB_NORM) m->weight_bytes += (size_t)m->top[i].sub[j].norm.dim * 4;
    }
    if (is_tnam) m->fingerprint = tn.fingerprint;
    else { uint64_t h = 1469598103934665603ULL; h = fnv1a(h, ctext, clen); h = fnv1a(h, tf.buf, tf.len); m->fingerprint = h; }
    ternaryai_infer_rope_ensure(m, m->seq_len - 1);
    /* tokenizador (opcional) */
    {
        char terr[160] = "";
        char *tok_text = NULL; size_t tok_len = 0;
        if (is_tnam) {
            if (tn.tok_len) { tok_text = (char *)ternary_malloc64(tn.tok_len + 1); if (tok_text) { memcpy(tok_text, tn.tok, tn.tok_len); tok_text[tn.tok_len] = 0; tok_len = tn.tok_len; } }
        } else {
            snprintf(path, sizeof path, "%s/tokenizer.json", dir);
            tok_text = read_file(path, &tok_len);               /* NULL si no hay tokenizador */
        }
        if (tok_text) {
            m->tok = ternary_tokenizer_load_mem(tok_text, tok_len, "tokenizer.json", terr, sizeof terr);
            if (m->tok) {
                m->chat_user_id = ternary_tokenizer_special_id(m->tok, (const uint8_t *)"<|user|>", 8);
                m->chat_asst_id = ternary_tokenizer_special_id(m->tok, (const uint8_t *)"<|assistant|>", 13);
                m->chat_end_id = ternary_tokenizer_special_id(m->tok, (const uint8_t *)"<|end|>", 7);
            }
        }
        if (xi) {                                                /* para exportar: el texto tal cual */
            xi->cfg = ctext; xi->cfg_len = clen; ctext = NULL;
            xi->tok = tok_text; xi->tok_len = tok_len; tok_text = NULL;
        }
        free(tok_text);
    }
    if (is_tnam) { m->blob = tn.base; m->blob_len = tn.len; m->blob_mmap = tn.is_mmap; tn.base = NULL; }     /* el modelo se queda con el archivo en memoria */
    ok = 1;
done:
    tf_close(&tf);
    if (is_tnam) { ternaryai_tnam_close(&tn); if (tn.base) ternaryai_tnam_release(tn.base, tn.len, tn.is_mmap); }
    ternary_jdom_free(cfg); free(ctext);
    if (!ok) { ternaryai_infer_model_free(m); return NULL; }
    return m;
}

#endif

#ifdef TERNARYAI_FUSED
/* Libreria fusionada: el unico modelo es el incrustado y su estructura (dimensiones, capas, punteros a los pesos) la genera
 * tn.export_fused como C estatico (ternaryai_fused_model_data): no hay config.json ni indice que interpretar, ni carga de modelos
 * externos ni exportacion. */
extern TernaryInferModel ternaryai_fused_model_data;
extern const unsigned char ternaryai_fused_blob[];
extern const size_t ternaryai_fused_tok_off, ternaryai_fused_tok_len;      /* tokenizador dentro del blob (len 0 = ninguno) */
TernaryInferModel *ternaryai_infer_fused_model(void) {
    static TernaryInferModel *m = NULL;
    static int tried = 0;
    if (!tried) {
        tried = 1;
#if defined(TERNARYAI_FIXED_AVX512)
        if (!ternaryai_infer_cpu_has_avx512()) return NULL;
#endif
        ternaryai_infer_backend_name();                 /* fija el backend por defecto si aun no hay uno */
        TernaryInferModel *d = &ternaryai_fused_model_data;
        if (d->layout != ternaryai_infer_kern->layout) return NULL;     /* el modelo no esta en el orden de pesos del nucleo de esta libreria */
#ifndef TERNARYAI_FUSED_NO_TOKENIZER
        if (ternaryai_fused_tok_len) {
            char terr[160] = "";
            char *tok_text = (char *)ternary_malloc64(ternaryai_fused_tok_len + 1);
            if (tok_text) {
                memcpy(tok_text, ternaryai_fused_blob + ternaryai_fused_tok_off, ternaryai_fused_tok_len); tok_text[ternaryai_fused_tok_len] = 0;
                d->tok = ternary_tokenizer_load_mem(tok_text, ternaryai_fused_tok_len, "tokenizer.json", terr, sizeof terr);
                if (d->tok) {
                    d->chat_user_id = ternary_tokenizer_special_id(d->tok, (const uint8_t *)"<|user|>", 8);
                    d->chat_asst_id = ternary_tokenizer_special_id(d->tok, (const uint8_t *)"<|assistant|>", 13);
                    d->chat_end_id = ternary_tokenizer_special_id(d->tok, (const uint8_t *)"<|end|>", 7);
                }
                free(tok_text);
            }
        }
#endif
        ternaryai_infer_rope_ensure(d, d->seq_len - 1);
        m = d;
    }
    return m;
}
#else
TernaryInferModel *ternaryai_infer_model_load(const char *path, char *err, size_t errcap) {
#if defined(TERNARYAI_FIXED_AVX512)
    if (!ternaryai_infer_cpu_has_avx512()) { set_err(err, errcap, "esta libreria se compilo solo para AVX-512 y esta CPU no lo tiene"); return NULL; }
#endif
    return model_load_impl(path, NULL, NULL, err, errcap);
}
#endif

/* Reordena los pesos de todas las matrices al orden 'layout' (en el sitio). */
int ternaryai_infer_model_relayout(TernaryInferModel *m, int layout) {
    int ok = ternaryai_infer_mat_relayout(&m->head, layout);
    for (int i = 0; i < m->n_top; i++) {
        if (m->top[i].kind != TERNARYAI_INFER_TOP_RESIDUAL) continue;
        for (int j = 0; j < m->top[i].n_sub; j++) {
            TernaryInferSub *s = &m->top[i].sub[j];
            if (s->kind == TERNARYAI_INFER_SUB_GQA) ok = ternaryai_infer_mat_relayout(&s->gqa.wq, layout) && ternaryai_infer_mat_relayout(&s->gqa.wk, layout) && ternaryai_infer_mat_relayout(&s->gqa.wv, layout) && ternaryai_infer_mat_relayout(&s->gqa.wo, layout) && ok;
            else if (s->kind == TERNARYAI_INFER_SUB_FFN) ok = ternaryai_infer_mat_relayout(&s->ffn.up, layout) && ternaryai_infer_mat_relayout(&s->ffn.down, layout) && (s->ffn.gate_mode == 2 || ternaryai_infer_mat_relayout(&s->ffn.gate, layout)) && ok;
        }
    }
    if (ok) m->layout = layout;
    return ok;
}

#ifndef TERNARYAI_FUSED
/* Escribe un modelo autocontenido (.ternam) con los pesos en el orden 'layout'. */
int ternaryai_infer_export_ternam(const char *src, const char *out, int layout, char *err, size_t errcap) {
    if (!src || !out) { set_err(err, errcap, "ruta nula"); return TERNARYAI_INFER_E_ARG; }
    int no_tok = 0;
    if (layout >= 0) { no_tok = (layout & TERNARYAI_INFER_EXPORT_NO_TOKENIZER) != 0; layout &= 0xFF; }
    if (layout == TERNARYAI_INFER_LAYOUT_AUTO) { ternaryai_infer_backend_name(); layout = ternaryai_infer_kern->layout; }
    if (layout != TERNARYAI_INFER_LAYOUT_ROW && layout != TERNARYAI_INFER_LAYOUT_AVX && layout != TERNARYAI_INFER_LAYOUT_NEON) { set_err(err, errcap, "orden de pesos invalido"); return TERNARYAI_INFER_E_ARG; }
    if (err && errcap) err[0] = 0;
    TernaryInferRecList rl; memset(&rl, 0, sizeof rl);
    ExportInfo xi; memset(&xi, 0, sizeof xi);
    TernaryInferModel *m = model_load_impl(src, &rl, &xi, err, errcap);
    if (!m) { free(xi.cfg); free(xi.tok); ternaryai_infer_rec_free(&rl); return TERNARYAI_INFER_E_IO; }
    int rc = TERNARYAI_INFER_OK;
    if (!ternaryai_infer_model_relayout(m, layout)) { set_err(err, errcap, "sin memoria"); rc = TERNARYAI_INFER_E_NOMEM; }
    else if (!ternaryai_tnam_write(out, xi.cfg, xi.cfg_len, no_tok ? NULL : xi.tok, no_tok ? 0 : xi.tok_len, layout, m->fingerprint, &rl, err, errcap)) rc = TERNARYAI_INFER_E_IO;
    ternaryai_infer_model_free(m); free(xi.cfg); free(xi.tok); ternaryai_infer_rec_free(&rl);
    return rc;
}


/* ---------------------------------------------------------------- generador del modelo fusionado (C estatico)
 * Carga un .ternam y escribe un .c con la estructura completa del modelo como datos estaticos: dimensiones, capas, ganancias, tablas de
 * exponenciales y punteros a los pesos DENTRO del blob (ternaryai_fused_blob). La libreria fusionada compila este archivo en vez de
 * interpretar config.json y el indice del .ternam. */
typedef struct { FILE *f; const uint8_t *blob; size_t blob_len; int err; } GenCtx;

static void gen_mat(GenCtx *g, const char *ind, const char *field, const TernaryInferMat *m) {
    const long wo = (long)(m->w - g->blob), so = (long)((const uint8_t *)m->S - g->blob);
    if (m->w < g->blob || (size_t)wo > g->blob_len || (const uint8_t *)m->S < g->blob || (size_t)so > g->blob_len) { g->err = 1; return; }
    fprintf(g->f, "%s.%s = { .rows = %d, .cols = %d, .cp = %d, .rb = %d, .layout = %d, .owned = 0, .s_owned = 0, .scale = %a,\n"
                  "%s    .w = (uint8_t *)(ternaryai_fused_blob + %ld), .S = (int32_t *)(ternaryai_fused_blob + %ld) },\n",
            ind, field, m->rows, m->cols, m->cp, m->rb, m->layout, (double)m->scale, ind, wo, so);
}

static void gen_norm_data(GenCtx *g, const char *id, const TernaryInferNorm *n) {
    fprintf(g->f, "static float %s[%d] = {", id, n->dim > 0 ? n->dim : 1);
    for (int j = 0; j < n->dim; j++) fprintf(g->f, "%s%d.0f", j ? "," : "", (int)n->gmul[j]);
    fprintf(g->f, "};\n");
}

static void gen_norm(GenCtx *g, const char *ind, const char *field, const char *id, const TernaryInferNorm *n) {
    fprintf(g->f, "%s.%s = { .dim = %d, .eps = %a, .scale = %a, .gmul = %s },\n", ind, field, n->dim, (double)n->eps, (double)n->scale, id);
}

int ternaryai_infer_export_fused_c(const char *ternam, const char *out_c, int with_tok, char *err, size_t errcap) {
    if (!ternam || !out_c) { set_err(err, errcap, "ruta nula"); return TERNARYAI_INFER_E_ARG; }
    if (err && errcap) err[0] = 0;
    TernaryInferModel *m = model_load_impl(ternam, NULL, NULL, err, errcap);
    if (!m) return TERNARYAI_INFER_E_IO;
    int rc = TERNARYAI_INFER_E_IO;
    TnamFile tn; memset(&tn, 0, sizeof tn);
    FILE *f = NULL;
    if (!m->blob || m->blob_len == 0) { set_err(err, errcap, "'%s' no es un .ternam", ternam); goto done; }
    if (!ternaryai_tnam_open(&tn, ternam, err, errcap)) goto done;
    f = fopen(out_c, "wb");
    if (!f) { set_err(err, errcap, "no se puede escribir '%s'", out_c); goto done; }
    GenCtx g = { f, m->blob, m->blob_len, 0 };
    fprintf(f, "/* Generado por ternaryai_infer_export_fused_c: estructura estatica del modelo incrustado (no editar). */\n"
               "#include \"ternary_infer_internal.h\"\n"
               "extern const unsigned char ternaryai_fused_blob[];\n"
               "const size_t ternaryai_fused_tok_off = %zu, ternaryai_fused_tok_len = %zu;\n",
            (with_tok && tn.tok_len) ? (size_t)((const uint8_t *)tn.tok - tn.base) : (size_t)0, (with_tok && tn.tok_len) ? tn.tok_len : (size_t)0);
    /* datos por capa */
    for (int i = 0; i < m->n_top; i++) {
        char id[64];
        if (m->top[i].kind == TERNARYAI_INFER_TOP_NORM) { snprintf(id, sizeof id, "gm_%d", i); gen_norm_data(&g, id, &m->top[i].norm); continue; }
        for (int j = 0; j < m->top[i].n_sub; j++) {
            const TernaryInferSub *s = &m->top[i].sub[j];
            snprintf(id, sizeof id, "gm_%d_%d", i, j);
            if (s->kind == TERNARYAI_INFER_SUB_NORM) gen_norm_data(&g, id, &s->norm);
            else if (s->kind == TERNARYAI_INFER_SUB_GQA) {
                fprintf(f, "static float et_%d_%d[%u] = {", i, j, (unsigned)s->gqa.exp_k0 + 1u);
                for (uint32_t k = 0; k <= s->gqa.exp_k0; k++) fprintf(f, "%s%a", k ? "," : "", (double)s->gqa.exp_tab[k]);
                fprintf(f, "};\n");
            }
        }
        fprintf(f, "static TernaryInferSub sub_%d[%d] = {\n", i, m->top[i].n_sub);
        for (int j = 0; j < m->top[i].n_sub; j++) {
            const TernaryInferSub *s = &m->top[i].sub[j];
            fprintf(f, "  { .kind = %d, .gqa_idx = %d,\n", (int)s->kind, s->gqa_idx);
            snprintf(id, sizeof id, "gm_%d_%d", i, j);
            if (s->kind == TERNARYAI_INFER_SUB_NORM) gen_norm(&g, "    ", "norm", id, &s->norm);
            else if (s->kind == TERNARYAI_INFER_SUB_GQA) {
                const TernaryInferGqa *q = &s->gqa;
                fprintf(f, "    .gqa = { .n_heads = %d, .n_kv = %d, .hd = %d, .group = %d, .qw = %d, .kvw = %d, .shift = %d, .window = %d, .sinks = %d, .smear = %d, .rope_base = %a,\n",
                        q->n_heads, q->n_kv, q->hd, q->group, q->qw, q->kvw, q->shift, q->window, q->sinks, q->smear, (double)q->rope_base);
                gen_mat(&g, "      ", "wq", &q->wq); gen_mat(&g, "      ", "wk", &q->wk); gen_mat(&g, "      ", "wv", &q->wv); gen_mat(&g, "      ", "wo", &q->wo);
                fprintf(f, "      .exp_tab = et_%d_%d, .exp_k0 = %u, .qk_lim = %d },\n", i, j, (unsigned)q->exp_k0, (int)q->qk_lim);
            } else {
                const TernaryInferFfn *q = &s->ffn;
                fprintf(f, "    .ffn = { .dim = %d, .hidden = %d, .gate_mode = %d,\n", q->dim, q->hidden, q->gate_mode);
                gen_mat(&g, "      ", "up", &q->up); gen_mat(&g, "      ", "down", &q->down);
                if (q->gate_mode != 2) gen_mat(&g, "      ", "gate", &q->gate);
                fprintf(f, "    },\n");
            }
            fprintf(f, "  },\n");
        }
        fprintf(f, "};\n");
    }
    fprintf(f, "static TernaryInferTop tops[%d] = {\n", m->n_top > 0 ? m->n_top : 1);
    for (int i = 0; i < m->n_top; i++) {
        if (m->top[i].kind == TERNARYAI_INFER_TOP_NORM) { char id[64]; snprintf(id, sizeof id, "gm_%d", i); fprintf(f, "  { .kind = %d,\n", (int)m->top[i].kind); gen_norm(&g, "    ", "norm", id, &m->top[i].norm); fprintf(f, "  },\n"); }
        else fprintf(f, "  { .kind = %d, .n_sub = %d, .sub = sub_%d },\n", (int)m->top[i].kind, m->top[i].n_sub, i);
    }
    fprintf(f, "};\n");
    fprintf(f, "TernaryInferModel ternaryai_fused_model_data = {\n"
               "  .vocab = %d, .dim = %d, .seq_len = %d, .emb_scale = %a, .emb = (uint8_t *)(ternaryai_fused_blob + %ld), .emb_rb = %d, .emb_owned = 0,\n"
               "  .blob = (uint8_t *)ternaryai_fused_blob, .blob_len = %zu, .blob_mmap = 2, .layout = %d,\n",
            m->vocab, m->dim, m->seq_len, (double)m->emb_scale, (long)(m->emb - m->blob), m->emb_rb, m->blob_len, m->layout);
    if (m->emb < m->blob || (size_t)(m->emb - m->blob) > m->blob_len) g.err = 1;
    gen_mat(&g, "  ", "head", &m->head);
    fprintf(f, "  .n_top = %d, .top = tops, .n_attn = %d, .n_ffn = %d, .n_residual = %d,\n"
               "  .max_rows = %d, .max_rows_body = %d, .max_cols4 = %d, .max_hidden = %d, .max_qw = %d, .max_kvw = %d, .max_hd = %d,\n"
               "  .rope_hd = %d, .rope_base = %a, .fingerprint = %lluULL, .weight_bytes = %zu,\n"
               "  .chat_user_id = %d, .chat_asst_id = %d, .chat_end_id = %d,\n};\n",
            m->n_top, m->n_attn, m->n_ffn, m->n_residual, m->max_rows, m->max_rows_body, m->max_cols4, m->max_hidden, m->max_qw, m->max_kvw, m->max_hd,
            m->rope_hd, (double)m->rope_base, (unsigned long long)m->fingerprint, m->weight_bytes,
            with_tok ? m->chat_user_id : -1, with_tok ? m->chat_asst_id : -1, with_tok ? m->chat_end_id : -1);
    if (g.err || ferror(f)) { set_err(err, errcap, "no se pudo generar el modelo fusionado (puntero fuera del blob o error de escritura)"); goto done; }
    rc = TERNARYAI_INFER_OK;
done:
    if (f) fclose(f);
    if (tn.base) { ternaryai_tnam_close(&tn); ternaryai_tnam_release(tn.base, tn.len, tn.is_mmap); }
    ternaryai_infer_model_free(m);
    return rc;
}

#endif

void ternaryai_infer_model_info(const TernaryInferModel *m, TernaryInferModelInfo *o) {
    memset(o, 0, sizeof(*o));
    if (!m) return;
    o->vocab = m->vocab; o->dim = m->dim; o->seq_len = m->seq_len;
    o->n_residual = m->n_residual; o->n_attn = m->n_attn; o->n_ffn = m->n_ffn;
    o->has_tokenizer = m->tok != NULL; o->weight_bytes = m->weight_bytes; o->fingerprint = m->fingerprint;
    int ga = 0, gf = 0;
    for (int i = 0; i < m->n_top; i++) {
        if (m->top[i].kind != TERNARYAI_INFER_TOP_RESIDUAL) continue;
        for (int j = 0; j < m->top[i].n_sub; j++) {
            const TernaryInferSub *s = &m->top[i].sub[j];
            if (s->kind == TERNARYAI_INFER_SUB_GQA && !ga) { ga = 1; o->n_heads = s->gqa.n_heads; o->n_kv_heads = s->gqa.n_kv; o->head_dim = s->gqa.hd; o->smear = s->gqa.smear; o->window = s->gqa.window; o->sinks = s->gqa.sinks; }
            if (s->kind == TERNARYAI_INFER_SUB_FFN && !gf) { gf = 1; o->hidden = s->ffn.hidden; o->gate_mode = s->ffn.gate_mode; }
        }
    }
}
