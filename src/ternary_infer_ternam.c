/* Modelo autocontenido (.ternam): un solo archivo con la configuracion, el tokenizador (opcional) y los pesos ya empaquetados a 2 bits
 * en el orden de un nucleo (TERNARYAI_INFER_LAYOUT_*), listo para mmap.
 *
 *   0   64   cabecera: "TNAM", u32 version (1), u32 layout, u32 n_secciones, u64 huella del modelo, u64 tamano total, u64 checksum
 *            (FNV-1a de todo el archivo con el campo checksum a cero), relleno
 *   64       tabla de secciones: n x { u32 tipo, u32 reservado, u64 desplazamiento, u64 largo }
 *   ...      secciones (cada una empieza en multiplo de 64): 1 CONFIG (config.json tal cual), 2 TOKENIZER (tokenizer.json tal cual, opcional),
 *            4 DATOS (tensores; cada uno empieza en multiplo de 64), 3 INDICE (JSON: clave de tensor -> tipo, forma, escala y desplazamientos)
 * Todo en little-endian. */
#define _POSIX_C_SOURCE 200809L
#include "ternary_mem.h"
#include "ternary_infer_internal.h"
#include "ternary_json.h"
#include <stdarg.h>
#include <stdio.h>
#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#define TNAM_MMAP 1
#endif

#define TNAM_VERSION 2u      /* v2: la suma por fila S es int32 (v1: int16; se sigue leyendo) */
#define SEC_CONFIG 1u
#define SEC_TOKENIZER 2u
#define SEC_INDEX 3u
#define SEC_DATA 4u
#define HDR 64
#define CK_OFF 32

static void set_err(char *err, size_t cap, const char *fmt, ...) {
    if (!err || !cap) return;
    va_list ap; va_start(ap, fmt); vsnprintf(err, cap, fmt, ap); va_end(ap);
}
static uint64_t fnv(uint64_t h, const void *d, size_t n) {
    const uint8_t *p = (const uint8_t *)d;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}
static void put32(uint8_t *p, uint32_t v) { for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i)); }
static void put64(uint8_t *p, uint64_t v) { for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i)); }
static uint32_t get32(const uint8_t *p) { uint32_t v = 0; for (int i = 0; i < 4; i++) v |= (uint32_t)p[i] << (8 * i); return v; }
static uint64_t get64(const uint8_t *p) { uint64_t v = 0; for (int i = 0; i < 8; i++) v |= (uint64_t)p[i] << (8 * i); return v; }
static uint64_t al64(uint64_t x) { return (x + 63) & ~(uint64_t)63; }

/* ---------------------------------------------------------------- lista de registros (para exportar) */
int ternaryai_infer_rec_push(TernaryInferRecList *l, const char *key, int kind, int rows, int cols, uint32_t scale_bits,
                             const void *w, size_t wlen, const int32_t *S, size_t slen, int8_t *own) {
    if (l->n == l->cap) {
        const int nc = l->cap ? l->cap * 2 : 32;
        TernaryInferRec *v = (TernaryInferRec *)realloc(l->v, (size_t)nc * sizeof(*v));
        if (!v) return 0;
        l->v = v; l->cap = nc;
    }
    TernaryInferRec *r = &l->v[l->n++];
    memset(r, 0, sizeof(*r));
    snprintf(r->key, sizeof r->key, "%s", key);
    r->kind = kind; r->rows = rows; r->cols = cols; r->scale_bits = scale_bits; r->w = w; r->wlen = wlen; r->S = S; r->slen = slen; r->own = own;
    return 1;
}
void ternaryai_infer_rec_free(TernaryInferRecList *l) {
    for (int i = 0; i < l->n; i++) free(l->v[i].own);
    free(l->v); memset(l, 0, sizeof(*l));
}

/* ---------------------------------------------------------------- escritura */
static int json_key(char *dst, size_t cap, const char *k) {
    size_t o = 0;
    for (; *k && o + 3 < cap; k++) { if (*k == '"' || *k == '\\') dst[o++] = '\\'; dst[o++] = (*k < 32) ? '_' : *k; }
    dst[o] = 0;
    return (int)o;
}

int ternaryai_tnam_write(const char *path, const char *cfg, size_t cfg_len, const char *tok, size_t tok_len, int layout, uint64_t fingerprint,
                         const TernaryInferRecList *rl, char *err, size_t cap) {
    const int nsec = tok_len ? 4 : 3;
    uint64_t *w_off = (uint64_t *)ternary_calloc64((size_t)(rl->n ? rl->n : 1), 8), *s_off = (uint64_t *)ternary_calloc64((size_t)(rl->n ? rl->n : 1), 8);
    char *idx = NULL; size_t idx_len = 0, idx_cap = 0;
    FILE *f = NULL; int ok = 0;
    if (!w_off || !s_off) { set_err(err, cap, "sin memoria"); goto done; }
    uint64_t pos = al64(HDR + (uint64_t)nsec * 24);
    const uint64_t cfg_off = pos; pos = al64(pos + cfg_len);
    const uint64_t tok_off = pos; if (tok_len) pos = al64(pos + tok_len);
    const uint64_t data_off = pos;
    for (int i = 0; i < rl->n; i++) {
        w_off[i] = pos; pos = al64(pos + rl->v[i].wlen);
        if (rl->v[i].kind == 1) { s_off[i] = pos; pos = al64(pos + rl->v[i].slen); }
    }
    const uint64_t data_len = pos - data_off;
    /* indice JSON */
    idx_cap = 256 + (size_t)rl->n * 512; idx = (char *)ternary_malloc64(idx_cap);
    if (!idx) { set_err(err, cap, "sin memoria"); goto done; }
    idx[idx_len++] = '{';
    for (int i = 0; i < rl->n; i++) {
        const TernaryInferRec *r = &rl->v[i]; char k[2 * sizeof r->key];
        json_key(k, sizeof k, r->key);
        idx_len += (size_t)snprintf(idx + idx_len, idx_cap - idx_len,
            "%s\"%s\":{\"kind\":%d,\"rows\":%d,\"cols\":%d,\"scale_bits\":%u,\"w_off\":%llu,\"w_len\":%llu,\"s_off\":%llu,\"s_len\":%llu}",
            i ? "," : "", k, r->kind, r->rows, r->cols, (unsigned)r->scale_bits, (unsigned long long)w_off[i], (unsigned long long)r->wlen,
            (unsigned long long)s_off[i], (unsigned long long)(r->kind == 1 ? r->slen : 0));
    }
    idx[idx_len++] = '}';
    const uint64_t idx_off = pos; pos = al64(pos + idx_len);
    const uint64_t total = pos;
    /* cabecera y tabla */
    uint8_t *head = (uint8_t *)ternary_calloc64(1, (size_t)(al64(HDR + (uint64_t)nsec * 24)));
    if (!head) { set_err(err, cap, "sin memoria"); goto done; }
    memcpy(head, "TNAM", 4); put32(head + 4, TNAM_VERSION); put32(head + 8, (uint32_t)layout); put32(head + 12, (uint32_t)nsec);
    put64(head + 16, fingerprint); put64(head + 24, total); put64(head + CK_OFF, 0);
    { uint8_t *t = head + HDR; int k = 0;
      put32(t + 24 * k, SEC_CONFIG); put64(t + 24 * k + 8, cfg_off); put64(t + 24 * k + 16, cfg_len); k++;
      if (tok_len) { put32(t + 24 * k, SEC_TOKENIZER); put64(t + 24 * k + 8, tok_off); put64(t + 24 * k + 16, tok_len); k++; }
      put32(t + 24 * k, SEC_DATA); put64(t + 24 * k + 8, data_off); put64(t + 24 * k + 16, data_len); k++;
      put32(t + 24 * k, SEC_INDEX); put64(t + 24 * k + 8, idx_off); put64(t + 24 * k + 16, idx_len); }
    const size_t head_len = (size_t)al64(HDR + (uint64_t)nsec * 24);
    f = fopen(path, "wb");
    if (!f) { set_err(err, cap, "no se puede crear '%s'", path); free(head); goto done; }
    uint64_t h = 1469598103934665603ULL, at = 0;
    static const uint8_t zero[64] = { 0 };
#define EMIT(P, N) do { const size_t n_ = (size_t)(N); if (n_) { if (fwrite((P), 1, n_, f) != n_) { set_err(err, cap, "error al escribir '%s'", path); free(head); goto done; } h = fnv(h, (P), n_); at += n_; } } while (0)
#define PAD_TO(OFF) do { while (at < (OFF)) { const size_t n2 = (size_t)(((OFF) - at) < 64 ? ((OFF) - at) : 64); EMIT(zero, n2); } } while (0)
    EMIT(head, head_len);
    PAD_TO(cfg_off); EMIT(cfg, cfg_len); PAD_TO(tok_off); if (tok_len) EMIT(tok, tok_len);
    for (int i = 0; i < rl->n; i++) {
        const TernaryInferRec *r = &rl->v[i];
        PAD_TO(w_off[i]); EMIT(r->w, r->wlen);
        if (r->kind == 1) { PAD_TO(s_off[i]); EMIT(r->S, r->slen); }
    }
    PAD_TO(idx_off); EMIT(idx, idx_len); PAD_TO(total);
    /* el checksum se calculo con el campo en cero: se escribe ahora */
    uint8_t ck[8]; put64(ck, h);
    if (fseek(f, CK_OFF, SEEK_SET) != 0 || fwrite(ck, 1, 8, f) != 8) { set_err(err, cap, "error al escribir '%s'", path); free(head); goto done; }
    free(head);
    ok = 1;
done:
    if (f && fclose(f) != 0) ok = 0;
    free(w_off); free(s_off); free(idx);
    if (!ok && err && cap && !err[0]) set_err(err, cap, "error al escribir '%s'", path);
    return ok;
#undef EMIT
#undef PAD_TO
}

/* ---------------------------------------------------------------- lectura */
void ternaryai_tnam_release(uint8_t *base, size_t len, int is_mmap) {
    if (!base || is_mmap == 2) return;          /* 2: bloque incrustado en la libreria (solo lectura, no se libera) */
#ifdef TNAM_MMAP
    if (is_mmap) { munmap(base, len); return; }
#endif
    (void)len; (void)is_mmap;
    free(base);
}

void ternaryai_tnam_close(TnamFile *t) {          /* libera el indice (el archivo en memoria lo conserva quien lo reclame) */
    for (int i = 0; i < t->n; i++) free(t->e[i].key);
    free(t->e);
    t->e = NULL; t->n = 0;
}

#ifdef TERNARYAI_FUSED
/* Libreria fusionada: el modelo esta incrustado (ternaryai_fused_blob); no se abre ningun archivo. */
extern const unsigned char ternaryai_fused_blob[];
extern const size_t ternaryai_fused_blob_len;
static int map_file(const char *path, TnamFile *t, char *err, size_t cap) {
    (void)path; (void)err; (void)cap;
    t->base = (uint8_t *)(uintptr_t)ternaryai_fused_blob; t->len = ternaryai_fused_blob_len; t->is_mmap = 2;
    return 1;
}
#else
static int map_file(const char *path, TnamFile *t, char *err, size_t cap) {
#ifdef TNAM_MMAP
    const int fd = open(path, O_RDONLY);
    if (fd < 0) { set_err(err, cap, "no se puede abrir '%s'", path); return 0; }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < HDR) { close(fd); set_err(err, cap, "'%s' no es un modelo .ternam (demasiado corto)", path); return 0; }
    /* privado y con escritura: si el nucleo activo pide otro orden de pesos, se reordenan en el sitio (copia al escribir), sin tocar el archivo */
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p != MAP_FAILED) { t->base = (uint8_t *)p; t->len = (size_t)st.st_size; t->is_mmap = 1; return 1; }
#endif
    FILE *f = fopen(path, "rb");
    if (!f) { set_err(err, cap, "no se puede abrir '%s'", path); return 0; }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); set_err(err, cap, "no se puede leer '%s'", path); return 0; }
    const long sz = ftell(f);
    if (sz < HDR || fseek(f, 0, SEEK_SET) != 0) { fclose(f); set_err(err, cap, "'%s' no es un modelo .ternam (demasiado corto)", path); return 0; }
    uint8_t *b = (uint8_t *)ternary_aligned_alloc(64, ((size_t)sz + 63) / 64 * 64);
    if (!b || fread(b, 1, (size_t)sz, f) != (size_t)sz) { free(b); fclose(f); set_err(err, cap, "no se puede leer '%s'", path); return 0; }
    fclose(f);
    t->base = b; t->len = (size_t)sz; t->is_mmap = 0;
    return 1;
}
#endif

int ternaryai_tnam_open(TnamFile *t, const char *path, char *err, size_t cap) {
    memset(t, 0, sizeof(*t));
    if (!map_file(path, t, err, cap)) return 0;
    const uint8_t *b = t->base; const uint64_t len = t->len;
    int ok = 0;
    if (memcmp(b, "TNAM", 4) != 0) { set_err(err, cap, "'%s' no es un modelo .ternam (cabecera invalida)", path); goto fail; }
    t->ver = (int)get32(b + 4);
    if (t->ver != 1 && t->ver != (int)TNAM_VERSION) { set_err(err, cap, "'%s': version de formato %u no soportada", path, (unsigned)get32(b + 4)); goto fail; }
    t->layout = (int)get32(b + 8);
    const uint32_t nsec = get32(b + 12);
    t->fingerprint = get64(b + 16);
    if (get64(b + 24) != len) { set_err(err, cap, "'%s' esta truncado o tiene datos de mas (tamano %llu, esperado %llu)", path, (unsigned long long)len, (unsigned long long)get64(b + 24)); goto fail; }
    if (t->layout != TERNARYAI_INFER_LAYOUT_ROW && t->layout != TERNARYAI_INFER_LAYOUT_AVX && t->layout != TERNARYAI_INFER_LAYOUT_NEON) { set_err(err, cap, "'%s': orden de pesos %d desconocido", path, t->layout); goto fail; }
    if (nsec < 3 || nsec > 8 || HDR + (uint64_t)nsec * 24 > len) { set_err(err, cap, "'%s': tabla de secciones invalida", path); goto fail; }
    { /* checksum */
        uint64_t h = 1469598103934665603ULL;
        h = fnv(h, b, CK_OFF); { static const uint8_t z8[8] = { 0 }; h = fnv(h, z8, 8); }
        h = fnv(h, b + CK_OFF + 8, (size_t)(len - CK_OFF - 8));
        if (h != get64(b + CK_OFF)) { set_err(err, cap, "'%s' esta corrupto (el checksum no coincide)", path); goto fail; }
    }
    const uint8_t *idx = NULL; uint64_t idx_len = 0;
    for (uint32_t i = 0; i < nsec; i++) {
        const uint8_t *s = b + HDR + 24 * (size_t)i;
        const uint32_t kind = get32(s); const uint64_t off = get64(s + 8), n = get64(s + 16);
        if (off % 64 != 0 || off > len || n > len - off) { set_err(err, cap, "'%s': seccion %u fuera del archivo", path, (unsigned)kind); goto fail; }
        if (kind == SEC_CONFIG) { t->cfg = (const char *)b + off; t->cfg_len = (size_t)n; }
        else if (kind == SEC_TOKENIZER) { t->tok = (const char *)b + off; t->tok_len = (size_t)n; }
        else if (kind == SEC_INDEX) { idx = b + off; idx_len = n; }
    }
    if (!t->cfg || !t->cfg_len || !idx) { set_err(err, cap, "'%s': faltan la configuracion o el indice", path); goto fail; }
    char jerr[160] = "";
    TernaryJNode *root = ternary_jdom_parse((const char *)idx, (size_t)idx_len, jerr, sizeof jerr);
    if (!root || root->type != TJ_OBJ) { if (root) ternary_jdom_free(root); set_err(err, cap, "'%s': indice invalido: %s", path, jerr); goto fail; }
    t->e = (TnamEnt *)ternary_calloc64(root->n + 1, sizeof(TnamEnt));
    if (!t->e) { ternary_jdom_free(root); set_err(err, cap, "sin memoria"); goto fail; }
    for (size_t k = 0; k < root->n; k++) {
        const TernaryJNode *v = &root->kids[k];
        int64_t kind, rows, cols, sb, wo, wl, so, sl;
        if (v->type != TJ_OBJ || !ternary_jdom_get_int(v, "kind", &kind) || !ternary_jdom_get_int(v, "rows", &rows) || !ternary_jdom_get_int(v, "cols", &cols) ||
            !ternary_jdom_get_int(v, "scale_bits", &sb) || !ternary_jdom_get_int(v, "w_off", &wo) || !ternary_jdom_get_int(v, "w_len", &wl) ||
            !ternary_jdom_get_int(v, "s_off", &so) || !ternary_jdom_get_int(v, "s_len", &sl) || kind < 1 || kind > 3 || rows < 0 || cols < 0 ||
            wo < 0 || wl < 0 || so < 0 || sl < 0 || sb < 0 || sb > 0xFFFFFFFFLL) {
            ternary_jdom_free(root); set_err(err, cap, "'%s': entrada de indice invalida", path); goto fail;
        }
        if ((uint64_t)wo % 64 != 0 || (uint64_t)wo > len || (uint64_t)wl > len - (uint64_t)wo ||
            (kind == 1 && ((uint64_t)so % 64 != 0 || (uint64_t)so > len || (uint64_t)sl > len - (uint64_t)so))) {
            ternary_jdom_free(root); set_err(err, cap, "'%s': un tensor cae fuera del archivo", path); goto fail;
        }
        TnamEnt *e = &t->e[t->n];
        e->key = strdup(root->keys[k]);
        if (!e->key) { ternary_jdom_free(root); set_err(err, cap, "sin memoria"); goto fail; }
        e->kind = (int)kind; e->rows = (int)rows; e->cols = (int)cols; e->scale_bits = (uint32_t)sb;
        e->w_off = (uint64_t)wo; e->w_len = (uint64_t)wl; e->s_off = (uint64_t)so; e->s_len = (uint64_t)sl;
        t->n++;
    }
    ternary_jdom_free(root);
    ok = 1;
fail:
    if (!ok) { ternaryai_tnam_close(t); ternaryai_tnam_release(t->base, t->len, t->is_mmap); memset(t, 0, sizeof(*t)); }
    return ok;
}

const TnamEnt *ternaryai_tnam_find(const TnamFile *t, const char *key) {
    for (int i = 0; i < t->n; i++) if (!strcmp(t->e[i].key, key)) return &t->e[i];
    return NULL;
}
