#ifndef TERNARY_HEAD_H
#define TERNARY_HEAD_H
#include "ternary_core.h"
#include "ternary_dense.h"
#include "ternary_u8.h"
#include <stddef.h>
#include <stdint.h>
/*
 * Cabeza de salida de un modelo de lenguaje ("lm_head"): proyecta CADA posicion de dim a vocab logits con
 * UNA matriz ternaria compartida por todas las posiciones (vocab x dim, con votos, sin bias) y la escala
 * float por matriz de siempre:
 *
 *     logits[p][v] = escala * (W[v] . x[p])          x[p] ternario, dot entero por popcount
 *
 * Es el mismo calculo que Dense, pero POR FILA (posicion) en vez de sobre toda la entrada junta -- por eso no
 * se puede usar un Dense normal: mezclaria posiciones. La escala SI influye en la salida (es la
 * "temperatura" de los logits) y aprende por gradiente exacto.
 *
 * Entrada: UN TernaryVec de rows*dim (todas las posiciones concatenadas, igual que GQA/SwiGLU/RMSNorm).
 * Salida: rows*vocab floats (logits, sin softmax: la perdida -- ver ternary_loss.h -- la calcula el
 * llamador con el softmax y el gradiente; igual que el resto del proyecto, que entrega dL/d(salida
 * escalada de la ultima capa) a ternary_stack_backward).
 *
 * Gradiente exacto (no hay cuantizacion entre la entrada y los logits): dL/dW[v][i] = sum_p g[p][v]*x[p][i],
 * dL/dx[p][i] = escala * sum_v g[p][v]*W[v][i], dL/descala = sum_{p,v} g[p][v]*z[p][v].
 */
#define TERNARY_HEAD_MAX_DIM 65536
#define TERNARY_HEAD_MAX_VOCAB 1048576

typedef struct {
    size_t rows, dim, vocab;
    TernaryDense *w;        /* dim -> vocab, sin bias; w->scale es la escala de la cabeza */
    TernaryVec *x_row;      /* dim */
    int32_t *z;             /* rows*vocab: Z entero crudo propio; SE RESERVA al primer forward sin out_raw (con out_raw el
                             * Z vive en el buffer del llamador y no se duplica) */
    const int32_t *zcur;    /* Z del ultimo forward (z o el out_raw del llamador): lo que lee el backward */
    /* Ruta fusionada (ternary_head_train_xent): trozos de HEAD_BLOCK posiciones, reservados al primer uso. */
    float *c_logits, *c_grad;
    int32_t *c_z;
    /* Reservado UNA vez (no en el hot path): una entrada ternaria por posicion, para tratar las posiciones como
     * "muestras" de un bloque de Dense (forward_block / backward_block) en vez de recorrerlas de a una. */
    TernaryVec **xpos;      /* rows vectores de dim */
    const float **gview;    /* rows punteros (vistas de grad_out) -- escritos en cada backward */
    float **gxview;         /* rows punteros (vistas de grad_x)  -- idem */
    const int32_t **zview;  /* rows punteros (vistas de z) */
    float **oview;          /* rows punteros (vistas de la salida) */
    int32_t **rview;        /* rows punteros (vistas de z, escribible) */
    const TernaryVec **xcv; /* rows punteros const a xpos */
    /* Estado de la normalizacion por elemento del gradiente (ver TernaryElemNorm en ternary_dense.h): 4 bits por peso + una escala
     * por fila. Se reserva al primer apply_gradients con head_beta2 > 0 y va en el checkpoint (training_state.ternary). */
    TernaryElemNorm *vn;
} TernaryHead;

/* NULL si alguna dimension es 0 o excede el maximo, o sin memoria. Los pesos arrancan en 0. */
TernaryHead *ternary_head_alloc(size_t rows, size_t dim, size_t vocab);
void ternary_head_free(TernaryHead *h);

/* x: rows*dim ternarios. out: rows*vocab floats (logits). out_raw_opt (opcional): Z entero crudo. */
void ternary_head_forward(TernaryHead *h, const TernaryVec *x, float *out, int32_t *out_raw_opt);

/* Entrada uint8 (ver ternary_u8.h): x tiene rows filas de dim bytes, cada una con su escala y zero-point. Mismas
 * convenciones que arriba: logits[p][v] = escala * sx_p * (sum_i W[v][i]*(q_pi - zp_p)); out_raw = Z entero. */
void ternary_head_forward_u8(TernaryHead *h, const TernaryU8 *x, float *out, int32_t *out_raw_opt);
void ternary_head_backward_u8(TernaryHead *h, const TernaryU8 *x, const float *grad_out,
                              float *grad_w_out, float *grad_x_out, float *grad_scale_out);

/* grad_out: rows*vocab = dL/d(logits). grad_w_out: vocab*dim, SOBREESCRITO con el total de la muestra.
 * grad_x_out: rows*dim, se ACUMULA. *grad_scale_out se ACUMULA. Requiere el forward inmediato. */
void ternary_head_backward(TernaryHead *h, const TernaryVec *x, const float *grad_out,
                            float *grad_w_out, float *grad_x_out, float *grad_scale_out);

/* Entrenamiento fusionado con entropia cruzada, SIN materializar rows*vocab: por trozos de 48 posiciones calcula
 * logits, perdida, gradiente y el backward de la cabeza. Mismos bits que forward + ternary_xent + (grad*=inv_n) +
 * ternary_head_backward (mismos trozos, mismo orden de sumas). inv_n: factor que se multiplica al gradiente.
 * grad_w_out se SOBREESCRIBE; grad_x_out y *grad_scale_out se ACUMULAN. Devuelve la perdida media de la fila
 * (como ternary_xent) o NAN si algun target >= vocab (sin tocar nada). n_valid/n_correct opcionales. */
/* Version con entrada uint8 (la activacion entre capas): misma perdida/gradientes, fila a fila. */
double ternary_head_train_xent_u8(TernaryHead *h, const TernaryU8 *x, const int32_t *targets, float inv_n,
                                  float *grad_w_out, float *grad_x_out, float *grad_scale_out,
                                  size_t *n_valid_out, size_t *n_correct_out);
double ternary_head_train_xent(TernaryHead *h, const TernaryVec *x, const int32_t *targets, float inv_n,
                               float *grad_w_out, float *grad_x_out, float *grad_scale_out,
                               size_t *n_valid_out, size_t *n_correct_out);

#endif /* TERNARY_HEAD_H */
