#ifndef TERNARYAI_INFER_INTERNAL_H
#define TERNARYAI_INFER_INTERNAL_H
#include "ternary_infer.h"
#include "ternary_tokenizer.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Las sumas en float se hacen siempre en el mismo orden que la libreria (16 carriles) y se compila con
 * -ffp-contract=off: el escalar y el AVX-512 dan los mismos bits. */

#define TERNARYAI_INFER_MAX_SUB 8
#ifndef TERNARYAI_INFER_K16_MAX
#define TERNARYAI_INFER_K16_MAX 32767     /* mayor |clave| que cabe en la copia int16 (se reduce solo para probar la promocion a int32) */
#endif

/* ---------------------------------------------------------------- pesos
 * z[o] = sum_i w[o][i] * (q[i] - zp) = sum_i w[o][i]*q[i] - zp*S[o]   (entero exacto; S = npos - nneg de la fila). */
/* Los pesos viven SOLO a 2 bits (codigo = w+1 en {0,1,2}; el relleno es 1 = peso 0), en uno de dos ordenes segun el nucleo activo:
 *   LAYOUT_ROW: fila a fila, cp/4 bytes por fila = T = cp/8 palabras de 16 bits (little-endian); la palabra t guarda 8 codigos: el campo
 *               p (bits 2p..2p+1) es la columna t + p*T. Asi cada campo se extrae con (palabra >> 2p) & 3 en carriles de 16 bits.
 *   LAYOUT_AVX: [bloque de 16 filas][grupo g de 16 columnas][fila r][byte c]; el byte guarda las columnas 16g+4j+c (j = campo 0..3)
 * Ambos ocupan rb*16*cp/4 bytes. El producto se hace con los codigos tal cual (sin desempaquetar a int8 en memoria):
 *   sum(w q) = sum(code q) - sum(q)  y  z = sum(w q) - zp*S. */
typedef struct {
    int rows, cols;       /* salida, entrada */
    int cp;               /* cols redondeado a multiplo de 16 (los nucleos leen q hasta cp) */
    int rb;               /* bloques de 16 filas */
    int layout;
    int owned;            /* 1 = w y S son de la matriz (malloc); 0 = apuntan al archivo .ternam en memoria */
    int s_owned;          /* 1 = S es de la matriz aunque w apunte al archivo (.ternam v1: S se ensancha a int32 al cargar) */
    float scale;
    uint8_t *w;           /* rb*16*cp/4 bytes, alineado a 64 */
    int32_t *S;           /* rb*16: npos - nneg de cada fila (relleno con 0) */
} TernaryInferMat;

typedef struct { int dim; float eps, scale; float *gmul; } TernaryInferNorm;           /* gmul[j] en {-1, 0, 1} */

typedef struct {
    int n_heads, n_kv, hd, group, qw, kvw;
    int shift;            /* qk_shift: puntaje = (q.k) >> (16 + shift) */
    int window, sinks;    /* de la configuracion del modelo (0 = atencion completa) */
    int smear;
    float rope_base;
    TernaryInferMat wq, wk, wv, wo;
    float *exp_tab; uint32_t exp_k0; int32_t qk_lim;
} TernaryInferGqa;

typedef struct { int dim, hidden, gate_mode; TernaryInferMat up, down, gate; } TernaryInferFfn;

typedef enum { TERNARYAI_INFER_SUB_NORM = 0, TERNARYAI_INFER_SUB_GQA, TERNARYAI_INFER_SUB_FFN } TernaryInferSubKind;
typedef struct { TernaryInferSubKind kind; TernaryInferNorm norm; TernaryInferGqa gqa; TernaryInferFfn ffn; int gqa_idx; } TernaryInferSub;

typedef enum { TERNARYAI_INFER_TOP_RESIDUAL = 0, TERNARYAI_INFER_TOP_NORM } TernaryInferTopKind;
typedef struct { TernaryInferTopKind kind; int n_sub; TernaryInferSub *sub; TernaryInferNorm norm; } TernaryInferTop;   /* sub: n_sub elementos (exactos) */

struct TernaryInferModel {
    int vocab, dim, seq_len;
    float emb_scale; uint8_t *emb;    /* vocab filas de emb_rb bytes: 4 pesos por byte (codigo w+1, columna 4j+k en los bits 2k) */
    int emb_rb;                       /* (dim+3)/4 */
    int emb_owned;                    /* 0 = emb apunta al archivo .ternam */
    uint8_t *blob; size_t blob_len; int blob_mmap;   /* archivo .ternam en memoria (si se cargo uno) */
    int layout;                       /* orden de las matrices en memoria (el del nucleo activo) */
    TernaryInferMat head;                       /* vocab x dim */
    int n_top; TernaryInferTop *top;
    int n_attn, n_ffn, n_residual;
    int max_rows;                     /* mayor numero de filas de cualquier matriz */
    int max_rows_body;                /* idem sin la cabeza (la cabeza escribe en los z de la sesion): tamano de zi/zi2 de cada carril */
    int max_cols4;                    /* mayor cp */
    int max_hidden, max_qw, max_kvw, max_hd;
    /* RoPE: tablas de punto fijo (escala 256), crecen perezosamente; una por cabeza de cabeza distinta (aqui: una por modelo) */
    int rope_hd; float rope_base; int rope_len; int16_t *rope_c, *rope_s;
    uint64_t fingerprint;
    size_t weight_bytes;
    TernaryTokenizer *tok;
    int chat_user_id, chat_asst_id, chat_end_id;
    char *dir;
};

/* ---------------------------------------------------------------- memoria de trabajo de un carril (un token en vuelo) */
typedef struct {
    float *x, *y, *f;
    uint8_t *q8, *q8b;
    int32_t *zi, *zi2, *qr, *kr, *sh;
    float *attn, *vrow, *hbuf;
} TernaryInferScratch;
#define TERNARYAI_INFER_MAX_LANES 16

/* ---------------------------------------------------------------- sesion */
typedef struct {
    int tp;               /* paso de la cache traspuesta (multiplo de 16, >= max_context + 16) */
    int32_t *kT;          /* n_kv*hd filas de tp: K rotada, traspuesta (int32). NULL mientras todas las claves caben en k16: se crea solo si hace falta */
    int8_t *v8;           /* n_kv * max_context * hd: V = (float)v8 * vs[posicion] (V ya es round(v/sc)*sc: se guarda entera) */
    float *vs;            /* max_context: escala de V de cada posicion */
    int32_t *prev;        /* kvw: clave anterior (smear) */
    int16_t *k16;         /* copia de kT en int16, de a pares de filas: [fila/2][pos][fila%2] (camino rapido de los puntajes) */
    int32_t kmax;         /* mayor |clave| escrita desde el ultimo reinicio (cota conservadora) */
    int k16kind;          /* 1 = pares intercalados (nucleo AVX-512), 2 = plano [fila][pos] (portable) */
    int k16ok;            /* 0 si alguna clave no cabe en int16 o hd/kvw son impares: solo el camino int64 */
} TernaryInferLayerCache;

struct TernaryInferSession {
    const TernaryInferModel *m;
    TernaryInferSessionOpts o;
    int n;                /* claves en la cache */
    int next_pos;         /* posicion absoluta del proximo token */
    TernaryInferLayerCache *lc;     /* n_attn */
    int32_t *hist; int hist_n, hist_cap;
    uint64_t rng;
    /* estado del ultimo token procesado */
    int have_head;
    int32_t *z;           /* vocab: z enteros de la cabeza */
    float head_k;         /* head.scale * sx: logit = head_k * z */
    /* scratch */
    float *x, *y, *f, *tmp;
    uint8_t *q8; float qs; int32_t qzp;
    int32_t *zi, *zi2, *qr, *kr, *sh;
    int64_t *scr;
    float *attn, *vrow;
    int32_t *sc; float *ee;
    float *hbuf;          /* hidden */
    uint8_t *q8b;         /* segundo buffer u8 (salida de atencion / hidden) */
    float *lg;            /* vocab: logits float (bajo demanda) */
    TernaryInferScratch *xs; int xs_n;   /* carriles extra (prefill por bloques), creados al necesitarlos */
    int oom;              /* una asignacion tardia (promocion de K a int32) fallo: la sesion ya no es fiable */
    float *probs_rec;     /* record_attn: n_attn * n_heads * (max_context) */
    int *probs_n;         /* n_attn: claves visibles del ultimo token */
};

/* ---------------------------------------------------------------- nucleos enteros (escalar / AVX-512) */
typedef struct {
    const char *name;
    /* z[o] = sum w q - zp*S (o < rows). q con relleno: lee hasta cp bytes. */
    void (*matvec)(const TernaryInferMat *m, const uint8_t *q, int32_t zp, int32_t *z);
    /* indice del mayor valor de z[0..n) (el primero en empate) */
    int (*argmax)(const int32_t *z, int n);
    /* puntajes de G cabezas contra claves [start, start+cnt) de kT (hd filas de paso tp): sc[b][j] = clip((q.k) >> (16+shift)).
     * scr: int64 de al menos cnt (solo el nucleo escalar lo usa). */
    void (*scores)(const int32_t *kT, int tp, int hd, int start, int cnt, const int32_t *const *q, int G,
                   int shift, int32_t lim, int32_t **sc, int64_t *scr);
    /* opcionales (NULL = version escalar); mismos bits que la version escalar */
    void (*quant)(const float *x, int n, uint8_t *q, float *scale, int32_t *zp);
    void (*relu2)(const int32_t *z, int n, float k, float *h);   /* h[j] = a > 0 ? a*a : 0, a = k*(float)z[j] */
    /* ee[j] = tab[min(mmax - sc[j], K0)], devuelve la suma en el orden canonico de 16 carriles; mmax = max(sc) */
    float (*softmax)(const int32_t *sc, int cnt, const float *tab, uint32_t K0, float *ee);
    void *unused_pv;      /* (reservado) */
    /* igual que scores pero con claves int16 intercaladas por pares (vpmaddwd), acumulador int32: solo valido si
     * hd*max|q|*max|k| < 2^31 (lo comprueba el motor); mismos resultados que scores en ese caso */
    void (*scores16)(const int16_t *k16, int tp, int hd, int start, int cnt, const int32_t *const *q, int G,
                     int shift, int32_t lim, int32_t **sc);
    /* acc[c][d] += ee[c][j] * ((float)v[c][j*hd + d] * vs[j]) para nc cadenas independientes (cada una en orden de j, sin FMA) */
    void (*pv_multi)(int nc, const float *const *ee, int cnt, const int8_t *const *v, const float *vs, int hd, float *const *acc);
    /* z[b][o] = sum w q[b] - zp[b]*S[o] para B carriles (B <= TERNARYAI_INFER_MAX_LANES): los pesos se leen una vez */
    void (*matmul)(const TernaryInferMat *m, const uint8_t *const *q, const int32_t *zp, int32_t *const *z, int B);
    int layout;           /* orden de pesos que espera (TERNARYAI_INFER_LAYOUT_*) */
    /* opcional: igual que scores16 pero con claves int16 PLANAS [fila d][posicion] (paso tp); mismos valores que scores16_planar del motor */
    void (*scores16p)(const int16_t *k16, int tp, int hd, int start, int cnt, const int32_t *const *q, int G, int shift, int32_t lim, int32_t **sc);
} TernaryInferKernels;

extern const TernaryInferKernels ternaryai_infer_k_scalar;
extern const TernaryInferKernels ternaryai_infer_k_avx512_vnni;
extern const TernaryInferKernels ternaryai_infer_k_avx512_bw;
extern const TernaryInferKernels ternaryai_infer_k_neon;          /* completo (pruebas y bench) */
extern const TernaryInferKernels ternaryai_infer_k_neon_hybrid;   /* el que usa el backend NEON */
extern const TernaryInferKernels *ternaryai_infer_kern;    /* activo */

/* ---------------------------------------------------------------- utilidades compartidas */
int ternaryai_infer_mat_build(TernaryInferMat *m, const int8_t *w, int rows, int cols, float scale, int layout);
int ternaryai_infer_mat_relayout(TernaryInferMat *m, int layout);
int ternaryai_infer_kt_promote(TernaryInferLayerCache *c, int kvw);   /* crea kT (desde k16 si vale); 0 si no hay memoria */
void ternaryai_infer_model_sync(TernaryInferModel *m);    /* deja las matrices en el orden del nucleo activo */
void ternaryai_infer_mat_free(TernaryInferMat *m);
void ternaryai_infer_rope_ensure(TernaryInferModel *m, int pos);
void ternaryai_infer_quantize_row(const float *x, int n, uint8_t *q, float *scale, int32_t *zp);
void ternaryai_infer_rmsnorm_row(const TernaryInferNorm *nm, const float *x, float *out);
int ternaryai_infer_cpu_vnni(void);
int ternaryai_infer_cpu_has_neon(void);
extern long long ternaryai_infer_n_fast, ternaryai_infer_n_slow;   /* estadistica: puntajes por el camino int16 / int64 */
void ternaryai_infer_step_many(TernaryInferSession **ss, const int32_t *ids, int n, int *rcs);
void ternaryai_infer_matmul(const TernaryInferMat *m, const uint8_t *const *q, const int32_t *zp, int32_t *const *z, int B);

/* perfilador */
extern int ternaryai_infer_prof_on;
int ternaryai_infer_prof_id(const char *name);
void ternaryai_infer_prof_begin(int id);
void ternaryai_infer_prof_end(int id);
#define TERNARYAI_INFER_PROF_BEGIN(id) do { if (ternaryai_infer_prof_on) ternaryai_infer_prof_begin(id); } while (0)
#define TERNARYAI_INFER_PROF_END(id)   do { if (ternaryai_infer_prof_on) ternaryai_infer_prof_end(id); } while (0)

/* ---------------------------------------------------------------- modelo autocontenido (.ternam) */
typedef struct { char key[200]; int kind; int rows, cols; uint32_t scale_bits; const void *w; size_t wlen; const int32_t *S; size_t slen; int8_t *own; } TernaryInferRec;
typedef struct { TernaryInferRec *v; int n, cap; } TernaryInferRecList;       /* tensores en el orden en que se cargaron (para exportar) */
int ternaryai_infer_rec_push(TernaryInferRecList *l, const char *key, int kind, int rows, int cols, uint32_t scale_bits, const void *w, size_t wlen,
                             const int32_t *S, size_t slen, int8_t *own);   /* kind: 1 matriz, 2 embedding, 3 ganancia de norma (int8) */
void ternaryai_infer_rec_free(TernaryInferRecList *l);

typedef struct { char *key; int kind; int rows, cols; uint32_t scale_bits; uint64_t w_off, w_len, s_off, s_len; } TnamEnt;
typedef struct { uint8_t *base; size_t len; int is_mmap; int ver; int layout; uint64_t fingerprint; const char *cfg, *tok; size_t cfg_len, tok_len; TnamEnt *e; int n; } TnamFile;
int ternaryai_tnam_open(TnamFile *t, const char *path, char *err, size_t cap);
void ternaryai_tnam_close(TnamFile *t);                                        /* solo el indice */
void ternaryai_tnam_release(uint8_t *base, size_t len, int is_mmap);           /* el archivo en memoria */
const TnamEnt *ternaryai_tnam_find(const TnamFile *t, const char *key);
int ternaryai_tnam_write(const char *path, const char *cfg, size_t cfg_len, const char *tok, size_t tok_len, int layout, uint64_t fingerprint,
                         const TernaryInferRecList *rl, char *err, size_t cap);
int ternaryai_infer_model_relayout(TernaryInferModel *m, int layout);

#endif
