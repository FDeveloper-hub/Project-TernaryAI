/* Operaciones en float compartidas por los dos nucleos (el orden de las sumas es el de la libreria),
 * seleccion de backend y perfilador. */
#define _POSIX_C_SOURCE 200809L
#include "ternary_mem.h"
#include "ternary_infer_internal.h"
#include "ternary_threadpool.h"
#include <math.h>
#include <stdio.h>
#include <time.h>

uint64_t ternaryai_infer_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* ---------------------------------------------------------------- CPU y backend */
int ternaryai_infer_cpu_has_avx512(void) {
#if defined(__x86_64__) && defined(__GNUC__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vl") &&
           __builtin_cpu_supports("avx512dq");
#else
    return 0;
#endif
}
int ternaryai_infer_cpu_vnni(void) {
#if defined(__x86_64__) && defined(__GNUC__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512vnni");
#else
    return 0;
#endif
}

/* NEON (ARMv7/ARMv8): el procesador debe declararlo (HWCAP_NEON) y el binario compilarse con NEON */
#if (defined(__arm__) || defined(__aarch64__)) && defined(__linux__)
#include <sys/auxv.h>
#endif
int ternaryai_infer_cpu_has_neon(void) {
#if defined(__ARM_NEON) && defined(__aarch64__)
    return 1;
#elif defined(__ARM_NEON) && defined(__arm__) && defined(__linux__)
    return (getauxval(AT_HWCAP) & (1UL << 12)) != 0;    /* HWCAP_NEON */
#else
    return 0;
#endif
}

/* Libreria de solo motor con vectorizacion FIJA (se decide al exportar, desde Python): se compila con TERNARYAI_FIXED_SCALAR,
 * TERNARYAI_FIXED_AVX512 o TERNARYAI_FIXED_NEON; los otros nucleos ni se compilan ni se enlazan, y no hay seleccion en ejecucion. Sin
 * ninguna de las macros (libternaryai normal) todos los nucleos existen y se elige al cargar. */
#if (defined(TERNARYAI_FIXED_SCALAR) + defined(TERNARYAI_FIXED_AVX512) + defined(TERNARYAI_FIXED_NEON)) > 1
#error "TERNARYAI_FIXED_SCALAR, TERNARYAI_FIXED_AVX512 y TERNARYAI_FIXED_NEON son excluyentes"
#endif

#if defined(TERNARYAI_FIXED_SCALAR)
const TernaryInferKernels *ternaryai_infer_kern = &ternaryai_infer_k_scalar;
int ternaryai_infer_set_backend(TernaryInferBackend b) {
    return (b == TERNARYAI_INFER_BACKEND_AVX512 || b == TERNARYAI_INFER_BACKEND_NEON) ? TERNARYAI_INFER_E_UNSUPPORTED : TERNARYAI_INFER_BACKEND_SCALAR;
}
int ternaryai_infer_get_backend(void) { return TERNARYAI_INFER_BACKEND_SCALAR; }
const char *ternaryai_infer_backend_name(void) { return ternaryai_infer_k_scalar.name; }
#elif defined(TERNARYAI_FIXED_NEON)
const TernaryInferKernels *ternaryai_infer_kern = &ternaryai_infer_k_neon_hybrid;
int ternaryai_infer_set_backend(TernaryInferBackend b) {
    if (b == TERNARYAI_INFER_BACKEND_SCALAR || b == TERNARYAI_INFER_BACKEND_AVX512) return TERNARYAI_INFER_E_UNSUPPORTED;
    return ternaryai_infer_cpu_has_neon() ? TERNARYAI_INFER_BACKEND_NEON : TERNARYAI_INFER_E_UNSUPPORTED;
}
int ternaryai_infer_get_backend(void) { return TERNARYAI_INFER_BACKEND_NEON; }
const char *ternaryai_infer_backend_name(void) { return ternaryai_infer_kern->name; }
#elif defined(TERNARYAI_FIXED_AVX512)
#if defined(__AVX512VNNI__)
const TernaryInferKernels *ternaryai_infer_kern = &ternaryai_infer_k_avx512_vnni;
#else
const TernaryInferKernels *ternaryai_infer_kern = &ternaryai_infer_k_avx512_bw;
#endif
int ternaryai_infer_set_backend(TernaryInferBackend b) {
    if (b == TERNARYAI_INFER_BACKEND_SCALAR || b == TERNARYAI_INFER_BACKEND_NEON) return TERNARYAI_INFER_E_UNSUPPORTED;
    return ternaryai_infer_cpu_has_avx512() ? TERNARYAI_INFER_BACKEND_AVX512 : TERNARYAI_INFER_E_UNSUPPORTED;
}
int ternaryai_infer_get_backend(void) { return TERNARYAI_INFER_BACKEND_AVX512; }
const char *ternaryai_infer_backend_name(void) { return ternaryai_infer_kern->name; }
#else
const TernaryInferKernels *ternaryai_infer_kern = &ternaryai_infer_k_scalar;
static int backend_init = 0;

/* NEON solo entra en "auto" si el nucleo gana al escalar (medido en el dispositivo con tests/bench_infer; -DTERNARYAI_INFER_NEON_AUTO=0 lo deja
 * solo como opcion explicita). */
#ifndef TERNARYAI_INFER_NEON_AUTO
#define TERNARYAI_INFER_NEON_AUTO 1
#endif

int ternaryai_infer_set_backend(TernaryInferBackend b) {
    backend_init = 1;
    if (b == TERNARYAI_INFER_BACKEND_AUTO) {
        const char *e = getenv("TERNARYAI_NO_AVX512"), *n = getenv("TERNARYAI_NO_NEON");
        if (ternaryai_infer_cpu_has_avx512() && !(e && e[0] && e[0] != '0')) b = TERNARYAI_INFER_BACKEND_AVX512;
        else if (TERNARYAI_INFER_NEON_AUTO && ternaryai_infer_cpu_has_neon() && !(n && n[0] && n[0] != '0')) b = TERNARYAI_INFER_BACKEND_NEON;
        else b = TERNARYAI_INFER_BACKEND_SCALAR;
    }
    if (b == TERNARYAI_INFER_BACKEND_AVX512) {
        if (!ternaryai_infer_cpu_has_avx512()) return TERNARYAI_INFER_E_UNSUPPORTED;
        ternaryai_infer_kern = ternaryai_infer_cpu_vnni() ? &ternaryai_infer_k_avx512_vnni : &ternaryai_infer_k_avx512_bw;
        return TERNARYAI_INFER_BACKEND_AVX512;
    }
    if (b == TERNARYAI_INFER_BACKEND_NEON) {
        if (!ternaryai_infer_cpu_has_neon()) return TERNARYAI_INFER_E_UNSUPPORTED;
        ternaryai_infer_kern = &ternaryai_infer_k_neon_hybrid;
        return TERNARYAI_INFER_BACKEND_NEON;
    }
    ternaryai_infer_kern = &ternaryai_infer_k_scalar;
    return TERNARYAI_INFER_BACKEND_SCALAR;
}
int ternaryai_infer_get_backend(void) {
    if (!backend_init) ternaryai_infer_set_backend(TERNARYAI_INFER_BACKEND_AUTO);
    return ternaryai_infer_kern == &ternaryai_infer_k_scalar ? TERNARYAI_INFER_BACKEND_SCALAR
         : ternaryai_infer_kern == &ternaryai_infer_k_neon_hybrid ? TERNARYAI_INFER_BACKEND_NEON : TERNARYAI_INFER_BACKEND_AVX512;
}
const char *ternaryai_infer_backend_name(void) {
    if (!backend_init) ternaryai_infer_set_backend(TERNARYAI_INFER_BACKEND_AUTO);
    return ternaryai_infer_kern->name;
}
#endif

/* ---------------------------------------------------------------- cuantizacion uint8 por fila (como ternary_u8_quantize) */
static void quant_scalar(const float *x, int n, uint8_t *q, float *scale, int32_t *zp);
void ternaryai_infer_quantize_row(const float *x, int n, uint8_t *q, float *scale, int32_t *zp) {
    if (ternaryai_infer_kern && ternaryai_infer_kern->quant) ternaryai_infer_kern->quant(x, n, q, scale, zp); else quant_scalar(x, n, q, scale, zp);
}
static void quant_scalar(const float *x, int n, uint8_t *q, float *scale, int32_t *zp) {
    /* minimo y maximo por 16 carriles independientes (el compilador los vectoriza; min/max es exacto en cualquier orden) */
    float lo16[16] = {0}, hi16[16] = {0};
    int i = 0;
    for (; i + 16 <= n; i += 16)
        for (int l = 0; l < 16; l++) { const float v = x[i + l]; lo16[l] = v < lo16[l] ? v : lo16[l]; hi16[l] = v > hi16[l] ? v : hi16[l]; }
    float lo = 0.0f, hi = 0.0f;
    for (int l = 0; l < 16; l++) { lo = lo16[l] < lo ? lo16[l] : lo; hi = hi16[l] > hi ? hi16[l] : hi; }
    for (; i < n; i++) { const float v = x[i]; lo = v < lo ? v : lo; hi = v > hi ? v : hi; }
    const float sc = (hi - lo) / 255.0f;
    if (!(sc > 0.0f) || !isfinite(sc)) { *scale = 1.0f; *zp = 0; memset(q, 0, (size_t)n); return; }
    int z = (int)rintf(-lo / sc);
    if (z < 0) z = 0;
    if (z > 255) z = 255;
    *scale = sc; *zp = z;
    const float inv = 1.0f / sc, zf = (float)z;
    /* x*inv esta en [-255, 255] (lo <= 0 <= hi): sumar y restar 1.5*2^23 redondea al par, igual que cvtss2si, y vectoriza */
    for (int k = 0; k < n; k++) {
        float t = x[k] * inv;
        t = (t + 12582912.0f) - 12582912.0f;
        float qf = t + zf;
        qf = qf < 0.0f ? 0.0f : qf;
        qf = qf > 255.0f ? 255.0f : qf;
        q[k] = (uint8_t)qf;
    }
}

/* ---------------------------------------------------------------- RMSNorm de una fila */
static inline double reduce_d16(const double *d) {
    double v[8], t4[4], t2[2];
    for (int l = 0; l < 8; l++) v[l] = d[l] + d[l + 8];
    for (int l = 0; l < 4; l++) t4[l] = v[l] + v[l + 4];
    for (int l = 0; l < 2; l++) t2[l] = t4[l] + t4[l + 2];
    return t2[0] + t2[1];
}

void ternaryai_infer_rmsnorm_row(const TernaryInferNorm *nm, const float *x, float *out) {
    const int D = nm->dim;
    double d[16] = {0};
    int j = 0;
    for (; j + 16 <= D; j += 16)
        for (int l = 0; l < 16; l++) d[l] += (double)x[j + l] * (double)x[j + l];
    for (int l = 0; j + l < D; l++) d[l] += (double)x[j + l] * (double)x[j + l];
    const double ss = reduce_d16(d);
    const float inv = (float)(1.0 / sqrt(ss / (double)D + (double)nm->eps));
    const float f = nm->scale * inv;
    for (int k = 0; k < D; k++) out[k] = nm->gmul[k] * (f * x[k]);
}

/* ---------------------------------------------------------------- perfilador */
#define PROF_MAX 64
int ternaryai_infer_prof_on = 0;
static struct { const char *name; uint64_t calls, total, t0; } prof[PROF_MAX];
static int prof_n = 0;

int ternaryai_infer_prof_id(const char *name) {
    for (int i = 0; i < prof_n; i++) if (prof[i].name == name || !strcmp(prof[i].name, name)) return i;
    if (prof_n >= PROF_MAX) return PROF_MAX - 1;
    prof[prof_n].name = name;
    return prof_n++;
}
void ternaryai_infer_prof_begin(int id) { prof[id].t0 = ternaryai_infer_now_ns(); }
void ternaryai_infer_prof_end(int id) { prof[id].total += ternaryai_infer_now_ns() - prof[id].t0; prof[id].calls++; }
void ternaryai_infer_prof_enable(int on) { ternaryai_infer_prof_on = on; }
void ternaryai_infer_prof_reset(void) { for (int i = 0; i < PROF_MAX; i++) { prof[i].calls = 0; prof[i].total = 0; } }
int ternaryai_infer_prof_report(char *buf, size_t cap) {
    size_t o = 0;
    if (!buf || !cap) return 0;
    o += (size_t)snprintf(buf + o, cap - o, "%-18s %12s %14s %10s\n", "ambito", "llamadas", "total (us)", "ns/llam");
    for (int i = 0; i < prof_n && o < cap; i++)
        o += (size_t)snprintf(buf + o, cap - o, "%-18s %12llu %14.1f %10.0f\n", prof[i].name, (unsigned long long)prof[i].calls,
                              (double)prof[i].total / 1e3, prof[i].calls ? (double)prof[i].total / (double)prof[i].calls : 0.0);
    if (o < cap) o += (size_t)snprintf(buf + o, cap - o, "puntajes de atencion: camino int16 %lld, camino int64 %lld\n", ternaryai_infer_n_fast, ternaryai_infer_n_slow);
    return (int)(o < cap ? o : cap - 1);
}

/* ---------------------------------------------------------------- producto por varios carriles
 * Repartido en hilos por BLOQUES DE FILAS de la matriz: cada hilo calcula z[b][r0..r1) con una vista de la matriz que empieza en la fila r0
 * (en los dos ordenes de pesos, la fila r0 -multiplo de 16- empieza en el byte r0*cp/4). Los z son enteros exactos: el resultado no depende del reparto. */
#define MM_TASK_BLOCKS 4         /* bloques de 16 filas por tarea (el nucleo AVX-512 trabaja de a 4) */
#define MM_MIN_WORK 1048576      /* filas*columnas*carriles minimos para repartir una matriz entre hilos (con menos, la sincronizacion cuesta mas que el calculo) */
typedef struct { const TernaryInferMat *m; const uint8_t *const *q; const int32_t *zp; int32_t *const *z; int B; } MmCtx;

static void mm_range(size_t t0, size_t t1, void *c_) {
    const MmCtx *c = (const MmCtx *)c_;
    const TernaryInferMat *m = c->m;
    const int r0 = (int)(t0 * (size_t)MM_TASK_BLOCKS * 16);
    int r1 = (int)(t1 * (size_t)MM_TASK_BLOCKS * 16);
    if (r1 > m->rb * 16) r1 = m->rb * 16;
    TernaryInferMat v = *m;
    v.w = m->w + (size_t)r0 * (size_t)m->cp / 4;
    v.S = m->S + r0;
    v.rows = (r1 < m->rows ? r1 : m->rows) - r0;
    v.rb = (r1 - r0) / 16;
    int32_t *zz[TERNARYAI_INFER_MAX_LANES];
    for (int b = 0; b < c->B; b++) zz[b] = c->z[b] + r0;
    if (c->B > 1 && ternaryai_infer_kern->matmul) ternaryai_infer_kern->matmul(&v, c->q, c->zp, zz, c->B);
    else for (int b = 0; b < c->B; b++) ternaryai_infer_kern->matvec(&v, c->q[b], c->zp[b], zz[b]);
}

void ternaryai_infer_matmul(const TernaryInferMat *m, const uint8_t *const *q, const int32_t *zp, int32_t *const *z, int B) {
    TernaryThreadPool *pool = ternary_get_compute_pool();
    if (pool && ternary_threadpool_num_threads(pool) > 1 && (size_t)m->rows * (size_t)m->cp * (size_t)(B > 0 ? B : 1) >= MM_MIN_WORK && m->rb > MM_TASK_BLOCKS) {
        MmCtx c = { m, q, zp, z, B };
        ternary_parallel_ranges(pool, 0, ((size_t)m->rb + MM_TASK_BLOCKS - 1) / MM_TASK_BLOCKS, 1, mm_range, &c);
        return;
    }
    if (B > 1 && ternaryai_infer_kern->matmul) { ternaryai_infer_kern->matmul(m, q, zp, z, B); return; }
    for (int b = 0; b < B; b++) ternaryai_infer_kern->matvec(m, q[b], zp[b], z[b]);
}

/* Hilos del motor: el mismo reparto que el resto de la libreria (0 o 1 = un solo hilo; mismo resultado con cualquier numero). */
void ternaryai_infer_set_threads(int n) { ternary_set_num_threads(n); }
int ternaryai_infer_get_threads(void) { return ternary_get_num_threads(); }
