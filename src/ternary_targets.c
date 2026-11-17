#include "ternary_targets.h"
#include "ternary_threadpool.h"
#include "ternary_neon_policy.h"   /* incluye ternary_neon.h y ternary_avx512.h */
#include <string.h>

/* Referencia escalar: ids[i] en [lo, hi)  <=>  (uint32)(ids[i] - lo) < (uint32)(hi - lo)  (lo < hi). */
size_t ternary_targets_find_range_scalar(const int32_t *ids, size_t from, size_t n, int32_t lo, int32_t hi) {
    if (lo >= hi) return n;
    const uint32_t span = (uint32_t)hi - (uint32_t)lo;
    for (size_t i = from; i < n; i++)
        if ((uint32_t)ids[i] - (uint32_t)lo < span) return i;
    return n;
}

/* Los primeros ids se miran en escalar (si los marcadores estan juntos, que es lo habitual en conversaciones, no se paga la puesta en
 * marcha del nucleo vectorial); el resto va al nucleo vectorial: AVX-512 -> NEON -> escalar. La sonda es de 8 ids con AVX-512 y de 16 con NEON
 * (en el dispositivo, con 8 ids y marcadores muy juntos el NEON salia mas lento que el escalar; el escalar del A53 ya es rapido y el NEON de 16 ids por
 * vuelta necesita mas trabajo para compensar su arranque). El resultado es el mismo con cualquier longitud de sonda. */
#define TARGETS_PROBE_AVX 8
#define TARGETS_PROBE_NEON 16
size_t ternary_targets_find_range(const int32_t *ids, size_t from, size_t n, int32_t lo, int32_t hi) {
    if (lo >= hi || from >= n) return n;
    const uint32_t span = (uint32_t)hi - (uint32_t)lo;
    const int avx = ternary_avx512_enabled();
    const int neon = !avx && TN_ON(SCAN);
    const size_t probe = neon ? TARGETS_PROBE_NEON : TARGETS_PROBE_AVX;
    const size_t e = (n - from > probe) ? from + probe : n;
    for (size_t i = from; i < e; i++)
        if ((uint32_t)ids[i] - (uint32_t)lo < span) return i;
    if (e == n) return n;
    if (avx) return ternary_avx512_find_range(ids, e, n, lo, hi);
    if (neon) return ternary_neon_find_range(ids, e, n, lo, hi);
    return ternary_targets_find_range_scalar(ids, e, n, lo, hi);
}

static void fill_ignored(int32_t *y, size_t n) { memset(y, 0xFF, n * sizeof(int32_t)); }   /* 0xFFFFFFFF = -1 */

int ternary_targets_build(const int32_t *ids, size_t n, const TernaryTargetsCfg *cfg, int32_t *y) {
    if (n == 0) return 1;
    const int32_t lo = cfg->special_lo, hi = cfg->special_hi;
    const int has_special = lo < hi && cfg->kind;
    const int init = cfg->unmarked_train ? 1 : 0;
    int cur = init;
    /* el token 0 no tiene objetivo, pero si es un marcador abre su tramo */
    if (has_special && ids[0] >= lo && ids[0] < hi) {
        const int kd = cfg->kind[ids[0] - lo];
        cur = kd == TERNARY_TARGETS_TRAIN ? 1 : kd == TERNARY_TARGETS_MASK ? 0 : kd == TERNARY_TARGETS_END ? init : cur;
    }
    y[n - 1] = -1;
    size_t j = 1;
    while (j < n) {
        const size_t k = has_special ? ternary_targets_find_range(ids, j, n, lo, hi) : n;
        /* tokens normales [j, k): se entrenan si el tramo actual lo es */
        if (k > j) {
            if (cur) memcpy(y + (j - 1), ids + j, (k - j) * sizeof(int32_t)); else fill_ignored(y + (j - 1), k - j);
        }
        if (k >= n) break;
        /* ids[k] es especial: cierra el tramo actual (se entrena si el tramo se entrena) y, si es marcador, abre otro */
        y[k - 1] = cur ? ids[k] : -1;
        const int kd = cfg->kind[ids[k] - lo];
        if (kd == TERNARY_TARGETS_TRAIN) cur = 1;
        else if (kd == TERNARY_TARGETS_MASK) cur = 0;
        else if (kd == TERNARY_TARGETS_END) cur = init;
        j = k + 1;
    }
    return 1;
}

typedef struct { const int32_t *ids; const size_t *offs; const TernaryTargetsCfg *cfg; int32_t *y; } BatchCtx;

static void batch_range(size_t b, size_t e, void *ctx) {
    const BatchCtx *c = (const BatchCtx *)ctx;
    for (size_t i = b; i < e; i++)
        ternary_targets_build(c->ids + c->offs[i], c->offs[i + 1] - c->offs[i], c->cfg, c->y + c->offs[i]);
}

int ternary_targets_build_batch(const int32_t *ids, const size_t *offs, size_t n_ex, const TernaryTargetsCfg *cfg, int32_t *y) {
    BatchCtx c = { ids, offs, cfg, y };
    /* reparto por rangos de ejemplos; por debajo de ~64 mil tokens se hace en el hilo que llama (no paga la sincronizacion) */
    TernaryThreadPool *pool = (offs[n_ex] - offs[0] >= 65536) ? ternary_global_pool() : NULL;
    ternary_parallel_ranges(pool, 0, n_ex, 0, batch_range, &c);
    return 1;
}
