#ifndef TERNARY_CONV2D_H
#define TERNARY_CONV2D_H

#include "ternary_core.h"
#include <stdint.h>

/*
 * Convolucion 2D ternaria, layout HWC (igual que ternary_image.h).
 *
 * Puesta al dia (igual que TernaryDense, con los mismos 3 hallazgos de
 * la verificacion contra Python):
 *   - matriz UNICA de kernels (out_channels*kernel_size), no un TernaryVec
 *     por canal -- mismo motivo que Dense: evita el relleno de palabra
 *     minima por kernel cuando kernel_size es chico.
 *   - escala estilo BitNet por capa (antes no existia -- precision pobre
 *     sin ella, igual que se midio en Dense).
 *   - bias entrenable via votos (antes el bias existia pero era estatico
 *     para siempre -- el mismo bug que encontramos en Dense).
 *   - backward completo (antes Conv2D era forward-only).
 */
typedef struct {
    size_t in_height, in_width, in_channels;
    size_t kernel_h, kernel_w, out_channels;
    size_t stride;
    size_t out_height, out_width;
    size_t kernel_size; /* = kernel_h*kernel_w*in_channels, cacheado */

    TernaryVec *kernels;   /* matriz unica, longitud out_channels*kernel_size */
    TernaryVec *bias;      /* longitud out_channels, NULL si no hay bias */
    int8_t *counter_kernels; /* plano, longitud out_channels*kernel_size */
    int8_t *counter_bias;    /* longitud out_channels, NULL si no hay bias */
    float scale;

    TernaryVec *patch_scratch; /* buffer de ventana reservado una vez */
    /* Buffers de trabajo reservados UNA vez (no en el hot path); su contenido se rehace en cada llamada. */
    float *img_f;              /* in_height*in_width*in_channels: la imagen expandida a {-1,0,+1} */
    float *patch_f;            /* kernel_size: la ventana expandida */
    uint64_t *kcache;          /* out_channels * 2 * kwords: kernels extraidos (mask, sign) para el forward */
} TernaryConv2D;

TernaryConv2D *ternary_conv2d_alloc(size_t in_height, size_t in_width, size_t in_channels,
                                     size_t kernel_h, size_t kernel_w, size_t out_channels,
                                     size_t stride, int use_bias);
void ternary_conv2d_free(TernaryConv2D *layer);

int  ternary_conv2d_get_weight(const TernaryConv2D *layer, size_t o, size_t k);
void ternary_conv2d_set_weight(TernaryConv2D *layer, size_t o, size_t k, int value);

/* out: out_height*out_width*out_channels floats (salida escalada).
 * out_raw_opt (puede ser NULL): mismo tamano, Z crudo antes de escalar --
 * se reusa en backward para el gradiente de la escala, evitando repetir
 * el ternary_dot (misma optimizacion que en Dense). */
void ternary_conv2d_forward(const TernaryConv2D *layer, const TernaryVec *image,
                             float *out, int32_t *out_raw_opt);

/*
 * Backward. grad_out: dL/d(salida escalada), longitud
 * out_height*out_width*out_channels.
 * Calcula:
 *   - grad_kernels_out[o*kernel_size+k] = dL/dkernel[o][k] (para apply_gradients)
 *   - grad_bias_out[o]  += dL/dbias[o]                      (se ACUMULA)
 *   - *grad_scale_out   += dL/dscale                        (se ACUMULA)
 * NO propaga gradiente a la imagen de entrada (no hace falta: en este
 * proyecto Conv2D es siempre la primera capa, sobre la imagen cruda).
 */
void ternary_conv2d_backward(const TernaryConv2D *layer, const TernaryVec *image,
                              const float *grad_out, float *grad_kernels_out,
                              float *grad_bias_out, float *grad_scale_out,
                              const int32_t *z_raw_opt);

/* Aplica gradientes acumulados (ya promediados/sumados por el llamador
 * sobre el batch) a los kernels, al bias, y a la escala. */
void ternary_conv2d_apply_kernel_gradients(TernaryConv2D *layer, size_t o, const float *grad_kernel);
void ternary_conv2d_apply_bias_gradients(TernaryConv2D *layer, const float *grad_bias);
void ternary_conv2d_apply_scale_gradient(TernaryConv2D *layer, float grad_scale, float scale_lr);

#endif /* TERNARY_CONV2D_H */
