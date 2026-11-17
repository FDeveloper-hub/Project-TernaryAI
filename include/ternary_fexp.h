#ifndef TERNARY_FEXP_H
#define TERNARY_FEXP_H
/*
 * Exponencial negativa de un ENTERO sin libm ni tabla (idea de FlashAttention-4: la unidad de exponenciales es el
 * cuello de botella, asi que se emula con aritmetica de punto flotante normal; aqui se adapta al caso ternario, donde
 * el argumento es siempre un entero k >= 0 = (maximo del puntaje) - (puntaje)):
 *
 *     ternary_fexp_neg(k, c) = exp(-k*t)      con c = t * log2(e)  (c se calcula UNA vez por capa, en float)
 *
 * Camino (todas las operaciones en float, sin FMA, en este orden; el nucleo AVX-512 hace exactamente las mismas):
 *     y  = -((float)k * c)                    (y <= 0)
 *     y < TERNARY_FEXP_CUTOFF  ->  0          (equivale al recorte < 1e-20 de la tabla antigua; evita denormales)
 *     fl = rint(y)                            (al par mas cercano; exacto)
 *     f  = y - fl                             (en [-0.5, 0.5]; exacto)
 *     r  = 1 + f * P(f),  P por Horner con 6 coeficientes (minimax de (2^f - 1)/f, Cephes exp2f)
 *     resultado = r * 2^fl                    (2^fl por los bits del exponente; exacto)
 * Error relativo maximo medido 9.8e-8 (~ el redondeo de float). Deterministico: mismos bits en cualquier CPU con
 * IEEE-754 (compilar sin contraccion FMA: -std=c11 ya la apaga).
 */
#include <math.h>
#include <stdint.h>
#include <string.h>

#define TERNARY_LOG2E_F        1.4426950408889634f
#define TERNARY_FEXP_CUTOFF    (-66.0f)
#define TERNARY_FEXP_P0        1.535336188319500e-4f
#define TERNARY_FEXP_P1        1.339887440266574e-3f
#define TERNARY_FEXP_P2        9.618437357674640e-3f
#define TERNARY_FEXP_P3        5.550332471162809e-2f
#define TERNARY_FEXP_P4        2.402264791363012e-1f
#define TERNARY_FEXP_P5        6.931472028550421e-1f

/* 2^y para y en [CUTOFF, 0] (el llamador ya descarto y < CUTOFF). */
static inline float ternary_fexp2_core(float y) {
    const float fl = rintf(y);
    const float f = y - fl;
    float p = TERNARY_FEXP_P0;
    p = p * f; p = p + TERNARY_FEXP_P1;
    p = p * f; p = p + TERNARY_FEXP_P2;
    p = p * f; p = p + TERNARY_FEXP_P3;
    p = p * f; p = p + TERNARY_FEXP_P4;
    p = p * f; p = p + TERNARY_FEXP_P5;
    float r = f * p; r = r + 1.0f;
    const int32_t e = ((int32_t)fl + 127) << 23;      /* 2^fl como float (fl en [-66, 0] -> exponente normal) */
    float two; memcpy(&two, &e, sizeof(two));
    return r * two;
}

static inline float ternary_fexp_neg(int32_t k, float c) {
    const float y = -((float)k * c);
    if (y < TERNARY_FEXP_CUTOFF) return 0.0f;
    return ternary_fexp2_core(y);
}

/* Suma de 16 parciales en un orden fijo (el mismo arbol que usa el nucleo AVX-512): (l, l+8), luego (l, l+4), ... */
static inline float ternary_sum16(const float *part) {
    float t8[8], t4[4];
    for (int l = 0; l < 8; l++) t8[l] = part[l] + part[l + 8];
    for (int l = 0; l < 4; l++) t4[l] = t8[l] + t8[l + 4];
    const float a = t4[0] + t4[2], b = t4[1] + t4[3];
    return a + b;
}

/* ---- Reducciones canonicas (el MISMO orden en escalar y en AVX-512, para que den los mismos bits) -------------------
 * Suma de |x|: 2 acumuladores de 16 carriles (a0 recibe los bloques pares y el resto, a1 los impares de los pares de
 * bloques completos), luego a0+a1 por carril y ternary_sum16. */
static inline float ternary_canon_sum_abs(const float *x, size_t n) {
    float a0[16] = {0}, a1[16] = {0}, p[16];
    const size_t nt = (n + 15) / 16;
    size_t t = 0;
    for (; t + 2 <= nt && 16 * (t + 2) <= n; t += 2)
        for (int l = 0; l < 16; l++) { a0[l] += fabsf(x[16 * t + l]); a1[l] += fabsf(x[16 * t + 16 + l]); }
    for (; t < nt; t++)
        for (int l = 0; l < 16; l++) if (16 * t + (size_t)l < n) a0[l] += fabsf(x[16 * t + l]);
    for (int l = 0; l < 16; l++) p[l] = a0[l] + a1[l];
    return ternary_sum16(p);
}
/* Reduccion de 16 parciales double (carriles 0..7 = a0, 8..15 = a1): (a0+a1), luego (l, l+4), (l, l+2), (0+1). */
static inline double ternary_canon_reduce_d16(const double *d) {
    double v[8], t4[4], t2[2];
    for (int l = 0; l < 8; l++) v[l] = d[l] + d[l + 8];
    for (int l = 0; l < 4; l++) t4[l] = v[l] + v[l + 4];
    for (int l = 0; l < 2; l++) t2[l] = t4[l] + t4[l + 2];
    return t2[0] + t2[1];
}
/* Suma de cuadrados en double con 16 carriles (x*x es exacto en double). */
static inline double ternary_canon_sumsq(const float *x, size_t n) {
    double d[16] = {0};
    for (size_t j = 0; j < n; j++) d[j & 15] += (double)x[j] * (double)x[j];
    return ternary_canon_reduce_d16(d);
}

/* exp(x) para x <= 0 con la exponencial compartida (la usan la entropia cruzada y su nucleo AVX-512): y = x*log2(e),
 * y < CUTOFF -> 0. Error relativo ~1e-7. */
static inline float ternary_expf_nonpos(float x) {
    const float y = x * TERNARY_LOG2E_F;
    return (y < TERNARY_FEXP_CUTOFF) ? 0.0f : ternary_fexp2_core(y);
}
/* sum_v (double)g[v]*(double)z[v] en 16 carriles (un mul y un add separados por carril): orden canonico. */
static inline double ternary_canon_dot_gz(const float *g, const int32_t *z, size_t n) {
    double d[16] = {0};
    for (size_t j = 0; j < n; j++) d[j & 15] += (double)g[j] * (double)z[j];
    return ternary_canon_reduce_d16(d);
}
/* Igual con el factor sx: ((double)g * sx) * (double)z (la ruta uint8 de la cabeza). */
static inline double ternary_canon_dot_gsz(const float *g, const int32_t *z, size_t n, double sx) {
    double d[16] = {0};
    for (size_t j = 0; j < n; j++) d[j & 15] += (double)g[j] * sx * (double)z[j];
    return ternary_canon_reduce_d16(d);
}

#endif /* TERNARY_FEXP_H */
