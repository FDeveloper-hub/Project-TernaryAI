#ifndef TERNARY_SOFTMAX_H
#define TERNARY_SOFTMAX_H
#include <stddef.h>
#include <stdint.h>
/*
 * Softmax comun, con entradas ENTERAS (los puntajes q.k de dos vectores
 * ternarios son enteros en [-d, d]) -- "softmax ternaria" en el sentido
 * de que la funcion es la de siempre, lo ternario esta en sus
 * entradas, y eso la simplifica:
 *
 *   y_i = exp(s_i*t) / sum_j exp(s_j*t)        t = inv_temp (en atencion, 1/sqrt(d))
 *       = exp(-(m - s_i)*t) / sum_j exp(-(m - s_j)*t),   m = max_j s_j
 *
 * Como s es entero, la diferencia (m - s_i) es un entero en [0, 2d] y
 * exp(-k*t) sale de una TABLA de 2d+1 valores calculada una sola vez: no
 * hay expf en el camino caliente, y el resultado es determinista.
 *
 * La salida son probabilidades float (suman 1). NO se ternarizan: eso
 * seria otra operacion (seleccion dura), no una simplificacion de esta.
 * Sin pesos propios: los pesos ternarios de la atencion viven en las
 * proyecciones Wq/Wk/Wv/Wo de GQA, no aqui.
 */
typedef struct {
    int32_t max_abs_score;   /* d: se esperan puntajes en [-d, d] */
    float inv_temp;
    size_t table_len;        /* 2d + 1 */
    float *table;            /* table[k] = exp(-k*inv_temp), k = 0..2d */
} TernarySoftmax;

TernarySoftmax *ternary_softmax_alloc(int32_t max_abs_score, float inv_temp);
void ternary_softmax_free(TernarySoftmax *s);

/* scores: n enteros. Solo las primeras n_valid posiciones participan
 * (mascara causal de la fila i: n_valid = i+1); el resto sale con
 * probabilidad exactamente 0. n_valid==0 -> todo ceros. Un puntaje fuera
 * de [-d, d] no rompe nada: la diferencia se recorta al final de la
 * tabla (se trata como un puntaje aun mas bajo). */
void ternary_softmax_forward(const TernarySoftmax *s, const int32_t *scores, size_t n,
                              size_t n_valid, float *out);

/* y: salida del forward. g = dL/dy. Escribe dL/ds_i (derivada respecto
 * al puntaje ENTERO, ya incluye el factor inv_temp):
 *   dL/ds_i = inv_temp * y_i * (g_i - sum_j g_j*y_j)
 * Las posiciones enmascaradas tienen y=0, asi que salen con gradiente 0. */
void ternary_softmax_backward(const TernarySoftmax *s, const float *y, const float *g, size_t n,
                               float *out_ds);

#endif
