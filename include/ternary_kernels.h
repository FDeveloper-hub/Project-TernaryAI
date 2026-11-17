#ifndef TERNARY_KERNELS_H
#define TERNARY_KERNELS_H
/*
 * Nucleos numericos de Dense (backward por bloques), escritos para que el resultado sea IDENTICO BIT A BIT al
 * bucle escalar de siempre:
 *   - grad_w[o][i] += sum_s g[s][o] * x[s][i]     (suma en orden ascendente de muestra s, un redondeo por mul y otro por add)
 *   - grad_x[s][i] += sum_o gs[s][o] * w[o][i]    (suma en orden ascendente de fila o)
 * Implementacion portable (C puro con acumuladores en arreglos de tamano fijo, que los compiladores vectorizan).
 *
 * Bloqueo de cache: las columnas se recorren por tiles; el tile de x expandido (S x CT floats) queda en L1 mientras
 * se recorren TODAS las filas de salida, en vez de releer x entero por cada fila.
 * Prefetch: se pide con antelacion la siguiente fila de W (PLD en ARM) mientras se calcula la actual.
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define TK_PREFETCH(p) __builtin_prefetch((const void *)(p), 0, 3)

/* Ancho de columnas por tile (multiplo de 16). S*CT*4 bytes de x deben caber en L1 (32 KB en el A53). */
#ifndef TERNARY_KERNEL_CT
#define TERNARY_KERNEL_CT 128
#endif

/* row[i] += sum_s g[s] * x[s*ld + i]  para i en [0, n). Orden ascendente de s por elemento. */
static inline void tk_gradw_row(float *restrict row, size_t n, size_t S,
                                const float *restrict g, const float *restrict x, size_t ld) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        float a[16];
        memcpy(a, row + i, sizeof(a));
        for (size_t s = 0; s < S; s++) {
            const float gs = g[s];
            const float *xr = x + s * ld + i;
            for (int k = 0; k < 16; k++) a[k] += gs * xr[k];
        }
        memcpy(row + i, a, sizeof(a));
    }
    for (; i < n; i++) {
        float a = row[i];
        for (size_t s = 0; s < S; s++) a += g[s] * x[s * ld + i];
        row[i] = a;
    }
}

/* Filas de salida por bloque en grad_x: el bloque de W expandido a float (OB x 8 columnas) ocupa OB*32 B = 16 KB y se
 * reutiliza para TODAS las muestras desde L1 (antes: se releia W por filas con paso de fila y se re-expandia una vez
 * por cada grupo de 4 muestras). Los acumuladores pasan de un bloque al siguiente por memoria (gx), en orden de o
 * ascendente, asi que la suma de cada elemento es la misma de siempre. */
#ifndef TERNARY_GRADX_OB
#define TERNARY_GRADX_OB 512
#endif

/* Un grupo de hasta 4 muestras sobre un bloque de nb filas ya expandido (wsc[r*8 .. r*8+8) = w[o0+r][8 columnas]).
 * gp[r*4 + t]: gradiente (por escala) de la muestra s0+t en la fila o0+r, contiguo por fila. */
static inline void tk_gradx_group(float *const *gx, size_t col, size_t s0, size_t nS, size_t nb,
                                  const float *restrict gp, const float *restrict wsc) {
    float acc[4][8];
    for (size_t t = 0; t < nS; t++) memcpy(acc[t], gx[s0 + t] + col, sizeof(acc[t]));
    for (size_t r = 0; r < nb; r++) {
        const float *w = wsc + r * 8;
        for (size_t t = 0; t < nS; t++) {
            const float gv = gp[r * 4 + t];
            for (int k = 0; k < 8; k++) acc[t][k] += gv * w[k];
        }
    }
    for (size_t t = 0; t < nS; t++) memcpy(gx[s0 + t] + col, acc[t], sizeof(acc[t]));
}

/* gx[s][col + j] += sum_o g[s][o] * w[o][col + j] para j en [0, n8*8) (n8 bytes de columnas), o ascendente.
 * mb/sb: byte de la primera columna de la fila 0 de cada plano; wstride: bytes entre filas de W.
 * gsG: gradientes por escala en orden de GRUPO: gsG[(s/4)*outd*4 + o*4 + s%4] (los huecos de un ultimo grupo
 * incompleto no se leen). wsc: espacio de trabajo de TERNARY_GRADX_OB*8 floats. */
static inline void tk_gradx_cols(float *const *gx, size_t col, size_t n8, size_t S, size_t outd,
                                 const float *restrict gsG, const unsigned char *mb, const unsigned char *sb,
                                 size_t wstride, const float (*lm)[8], const float (*ls)[8], float *restrict wsc) {
    for (size_t j = 0; j < n8; j++) {
        for (size_t o0 = 0; o0 < outd; o0 += TERNARY_GRADX_OB) {
            const size_t nb = (outd - o0 < TERNARY_GRADX_OB) ? (outd - o0) : TERNARY_GRADX_OB;
            /* W de este bloque: una sola vez, w = m*s + 0.0f (normaliza -0 a +0, igual que la expansion de siempre) */
            for (size_t r = 0; r < nb; r++) {
                TK_PREFETCH(mb + j + (o0 + r + 8) * wstride); TK_PREFETCH(sb + j + (o0 + r + 8) * wstride);
                const unsigned m = mb[j + (o0 + r) * wstride], sg = sb[j + (o0 + r) * wstride];
                for (int k = 0; k < 8; k++) wsc[r * 8 + k] = lm[m][k] * ls[sg][k] + 0.0f;
            }
            for (size_t s0 = 0; s0 < S; s0 += 4)
                tk_gradx_group(gx, col + j * 8, s0, (S - s0 < 4) ? (S - s0) : 4, nb,
                               gsG + (s0 >> 2) * outd * 4 + o0 * 4, wsc);
        }
    }
}

/* VARIANTE ANTERIOR (recorre W por filas con paso de fila y la re-expande por cada grupo de 4 muestras); se conserva para
 * que el benchmark del dispositivo decida entre las dos (TERNARYAI_GRADX_KERNEL=rows|blocked). Lee gsG en orden de grupo:
 * gsT = gsG completo (orden de grupo); el paso entre filas dentro de un grupo es 4.
 * Para 8 columnas (un byte de mask y uno de sign por fila de W): gx[s][0..8) += gsT[o*S + s] * w[o][0..8),
 * o ascendente, para s en [s0, s0+nS) con nS <= 4. lm/ls: tablas por byte (lut_m[b][k] = bit k, lut_s[b][k] = 1-2*bit).
 * w = m*s + 0.0f (normaliza -0 a +0, igual que la expansion de siempre). */
static inline void tk_gradx_byte_rows(float *const *gx, size_t col, size_t s0, size_t nS,
                                 size_t outd, const float *restrict gsT, size_t gstride,
                                 const unsigned char *mb, const unsigned char *sb, size_t wstride,
                                 const float (*lm)[8], const float (*ls)[8]) {
    float acc[4][8];
    for (size_t t = 0; t < nS; t++) memcpy(acc[t], gx[s0 + t] + col, sizeof(acc[t]));
    for (size_t o = 0; o < outd; o++) {
        TK_PREFETCH(mb + (o + 8) * wstride); TK_PREFETCH(sb + (o + 8) * wstride);
        const unsigned m = mb[o * wstride], sg = sb[o * wstride];
        float w[8];
        for (int k = 0; k < 8; k++) w[k] = lm[m][k] * ls[sg][k] + 0.0f;
        const float *g = gsT + o * gstride;
        for (size_t t = 0; t < nS; t++) {
            const float gv = g[t];
            for (int k = 0; k < 8; k++) acc[t][k] += gv * w[k];
        }
    }
    for (size_t t = 0; t < nS; t++) memcpy(gx[s0 + t] + col, acc[t], sizeof(acc[t]));
}

/* gx[s][col + j] += sum_o gsT[o*gstride+s] * w[o][col + j] para j en [0, n8*8): n8 bytes de columnas. mb/sb apuntan al byte
 * de la primera columna de la fila 0; wstride: bytes entre filas de W. */
static inline void tk_gradx_cols_rows(float *const *gx, size_t col, size_t n8, size_t S, size_t outd,
                                 const float *restrict gsT, const unsigned char *mb, const unsigned char *sb,
                                 size_t wstride, const float (*lm)[8], const float (*ls)[8]) {
    for (size_t j = 0; j < n8; j++) {
        size_t s0 = 0;
        for (; s0 + 4 <= S; s0 += 4) tk_gradx_byte_rows(gx, col + j * 8, s0, 4, outd, gsT + (s0 >> 2) * outd * 4, 4, mb + j, sb + j, wstride, lm, ls);
        if (s0 < S) tk_gradx_byte_rows(gx, col + j * 8, s0, S - s0, outd, gsT + (s0 >> 2) * outd * 4, 4, mb + j, sb + j, wstride, lm, ls);
    }
}

/* Expande n trits (planos mask/sign, bit 0 = primer elemento, little-endian) a floats {-1,0,+1}. */
static inline void tk_expand_trits(const uint64_t *mask, const uint64_t *sign, size_t n, float *out,
                                   const float (*lm)[8], const float (*ls)[8]) {
    const size_t nbytes = n / 8;
    const unsigned char *mb = (const unsigned char *)mask, *sb = (const unsigned char *)sign;
    size_t j = 0;
    for (; j < nbytes; j++) {
        const float *restrict m = lm[mb[j]], *restrict s = ls[sb[j]];
        float *restrict o = out + j * 8;
        for (int k = 0; k < 8; k++) o[k] = m[k] * s[k] + 0.0f;
    }
    for (size_t i = nbytes * 8; i < n; i++) {
        int mbit = (int)((mb[i >> 3] >> (i & 7)) & 1u);
        int sbit = (int)((sb[i >> 3] >> (i & 7)) & 1u);
        out[i] = (float)(mbit * (1 - 2 * sbit));
    }
}

#endif /* TERNARY_KERNELS_H */
