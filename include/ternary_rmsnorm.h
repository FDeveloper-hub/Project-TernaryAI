#ifndef TERNARY_RMSNORM_H
#define TERNARY_RMSNORM_H

#include "ternary_dense.h"
#include <stddef.h>

/*
 * RMSNorm por fila:  out[i][j] = escala * g[j] * x[i][j] / sqrt(mean_j(x[i][j]^2) + eps)
 *
 *   - La entrada es FLOAT (el crudo escalado de la capa anterior, o el flujo residual): normalizar
 *     solo tiene sentido donde hay magnitud, porque el signo de un vector no cambia al normalizarlo.
 *   - g[j] es una ganancia POR DIMENSION en {-1,0,1} (una matriz ternaria de 1 x dim, con votos como
 *     cualquier otra matriz del proyecto) y 'escala' es el escalar float por capa que ya tienen
 *     Dense/GQA. Inicialmente g[j] = +1 y escala = 1 (RMSNorm puro). La interfaz de capa del stack solo
 *     soporta matrices ternarias + una escala escalar; una ganancia float por dimension (como en un
 *     RMSNorm "normal") exigiria extender el contenedor, el formato de archivo y la fachada Python.
 *   - Se calcula por FILA: la entrada es (rows, dim) y cada una de las 'rows' filas se normaliza por
 *     separado (en una secuencia, una fila por posicion); la ganancia es compartida por todas.
 *   - Es 100% float en forward y backward (un producto punto de largo 'dim' por fila, una raiz
 *     cuadrada y un recíproco): ver el comentario del forward para el costo.
 */
/* Learning rate FIJO de la escala de RMSNorm. La capa ignora el 'scale_lr' que le pasa
 * ternary_stack_apply_gradients (el de Dense, 0.01): con ese valor la escala se desplomaba al piso (0.02)
 * antes de que la ganancia ternaria llegara a votar (medido en tests/test_rmsnorm_layer.c). */
#ifndef TERNARY_RMSNORM_SCALE_LR
#define TERNARY_RMSNORM_SCALE_LR 0.001f
#endif

typedef struct {
    size_t rows;           /* filas independientes (posiciones) */
    size_t dim;            /* largo de cada fila */
    float eps;             /* > 0, evita dividir por cero */
    TernaryDense *gain;    /* 1 x dim; 'gain->scale' es la escala float de la capa */
} TernaryRMSNorm;

/* gain = +1 en todas las dimensiones, escala 1. NULL si no hay memoria o si rows, dim == 0 o eps <= 0. */
TernaryRMSNorm *ternary_rmsnorm_alloc(size_t rows, size_t dim, float eps);
void ternary_rmsnorm_free(TernaryRMSNorm *n);

/* x y out: rows*dim floats (pueden ser el mismo buffer). Acumula la suma de cuadrados en double. */
void ternary_rmsnorm_forward(const TernaryRMSNorm *n, const float *x, float *out);

/* Backward. x: la MISMA entrada del forward (se recalcula 1/rms, no se guarda estado).
 *   grad_gain_out : dim floats, se SOBREESCRIBE con dL/dg[j] (derivada tratando g como real, sumada
 *                   sobre las filas); es lo que votan los pesos ternarios.
 *   grad_x_out    : rows*dim floats, se ACUMULA (puede ser NULL).
 *   grad_scale_out: se ACUMULA dL/descala (puede ser NULL). */
void ternary_rmsnorm_backward(const TernaryRMSNorm *n, const float *x, const float *grad_out,
                              float *grad_gain_out, float *grad_x_out, float *grad_scale_out);

#endif /* TERNARY_RMSNORM_H */
