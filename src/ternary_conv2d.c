#include "ternary_mem.h"
#include "ternary_conv2d.h"

#include "ternary_image.h"
#include "ternary_avx512.h"
#include "ternary_fexp.h"
#include "ternary_dense.h"
#include "ternary_threadpool.h" /* ternary_r2_update_row */
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Misma tecnica que ternary_dense.c: extraer una fila (aqui, un kernel)
 * de la matriz unica desplazando palabras, sin desempaquetar nunca a
 * una representacion mas ancha. Duplicada a proposito (ver nota al pie
 * del archivo) en vez de compartir codigo con ternary_dense.c, para no
 * crear un acoplamiento cruzado entre dos tipos de capa independientes
 * por una funcion de 15 lineas. */
static void extract_bits(const uint64_t *src, size_t src_words, size_t offset,
                          size_t n, uint64_t *dst) {
    size_t dst_words = (n + 63) / 64;
    size_t word_off = offset >> 6;
    size_t bit_off = offset & 63;
    for (size_t i = 0; i < dst_words; i++) {
        uint64_t lo = (word_off + i < src_words) ? src[word_off + i] : 0;
        uint64_t hi = (word_off + i + 1 < src_words) ? src[word_off + i + 1] : 0;
        dst[i] = (bit_off == 0) ? lo : (lo >> bit_off) | (hi << (64 - bit_off));
    }
    size_t rem = n % 64;
    if (rem != 0) {
        uint64_t keep_mask = ((uint64_t)1 << rem) - 1;
        dst[dst_words - 1] &= keep_mask;
    }
}

static void extract_kernel(const TernaryConv2D *layer, size_t o, TernaryVec *dst) {
    size_t offset = o * layer->kernel_size;
    extract_bits(layer->kernels->mask, layer->kernels->words, offset, layer->kernel_size, dst->mask);
    extract_bits(layer->kernels->sign, layer->kernels->words, offset, layer->kernel_size, dst->sign);
}

TernaryConv2D *ternary_conv2d_alloc(size_t in_height, size_t in_width, size_t in_channels,
                                     size_t kernel_h, size_t kernel_w, size_t out_channels,
                                     size_t stride, int use_bias) {
    TernaryConv2D *layer = (TernaryConv2D *)ternary_malloc64(sizeof(TernaryConv2D));
    if (!layer) return NULL;

    layer->in_height = in_height; layer->in_width = in_width; layer->in_channels = in_channels;
    layer->kernel_h = kernel_h; layer->kernel_w = kernel_w; layer->out_channels = out_channels;
    layer->stride = stride;
    layer->out_height = (in_height - kernel_h) / stride + 1;
    layer->out_width = (in_width - kernel_w) / stride + 1;
    layer->kernel_size = kernel_h * kernel_w * in_channels;
    layer->scale = 1.0f;

    layer->kernels = ternary_vec_alloc(out_channels * layer->kernel_size);
    layer->counter_kernels = (int8_t *)ternary_calloc64(out_channels * layer->kernel_size, sizeof(int8_t));
    layer->patch_scratch = ternary_vec_alloc(layer->kernel_size);
    layer->img_f = (float *)ternary_malloc64(in_height * in_width * in_channels * sizeof(float) + 1);
    layer->patch_f = (float *)ternary_malloc64(layer->kernel_size * sizeof(float) + 1);
    layer->kcache = (uint64_t *)ternary_malloc64(out_channels * 2 * ((layer->kernel_size + 63) / 64) * sizeof(uint64_t) + 1);

    if (use_bias) {
        layer->bias = ternary_vec_alloc(out_channels);
        layer->counter_bias = (int8_t *)ternary_calloc64(out_channels, sizeof(int8_t));
    } else {
        layer->bias = NULL;
        layer->counter_bias = NULL;
    }
    return layer;
}

void ternary_conv2d_free(TernaryConv2D *layer) {
    if (!layer) return;
    ternary_vec_free(layer->kernels);
    free(layer->counter_kernels);
    ternary_vec_free(layer->patch_scratch);
    free(layer->img_f); free(layer->patch_f); free(layer->kcache);
    if (layer->bias) ternary_vec_free(layer->bias);
    free(layer->counter_bias);
    free(layer);
}

int ternary_conv2d_get_weight(const TernaryConv2D *layer, size_t o, size_t k) {
    return ternary_vec_get(layer->kernels, o * layer->kernel_size + k);
}
void ternary_conv2d_set_weight(TernaryConv2D *layer, size_t o, size_t k, int value) {
    ternary_vec_set(layer->kernels, o * layer->kernel_size + k, value);
}

/* Expande la imagen a floats {-1,0,+1} (una vez por llamada). */
static void expand_image(const TernaryConv2D *layer, const TernaryVec *image) {
    const size_t n = layer->in_height * layer->in_width * layer->in_channels;
    float *f = layer->img_f;
    if (ternary_avx512_enabled()) { ternary_avx512_expand_trits(image->mask, image->sign, n, f); return; }
    for (size_t i = 0; i < n; i++) {
        int m = (int)((image->mask[i >> 6] >> (i & 63)) & 1u);
        int sg = (int)((image->sign[i >> 6] >> (i & 63)) & 1u);
        f[i] = (float)(m * (1 - 2 * sg));
    }
}

/* Ventana (oh, ow) en floats, mismo orden que extract_patch (kh, kw, c). En HWC, (kw, c) de una fila del
 * kernel son contiguos en la imagen: una copia por fila. */
static void gather_patch_f(const TernaryConv2D *layer, size_t oh, size_t ow, float *dst) {
    const size_t rowlen = layer->kernel_w * layer->in_channels;
    const size_t base_h = oh * layer->stride, base_w = ow * layer->stride;
    for (size_t kh = 0; kh < layer->kernel_h; kh++) {
        const size_t src = ternary_image_index(base_h + kh, base_w, 0, layer->in_width, layer->in_channels);
        memcpy(dst + kh * rowlen, layer->img_f + src, rowlen * sizeof(float));
    }
}

/* Repartido en hilos: forward por filas de salida (cada tarea con su propia ventana); backward por CANALES de salida (cada tarea recorre
 * las posiciones en orden: mismo orden de sumas en gradientes y sesgo) y los terminos de la escala se suman despues en el orden de siempre. */
typedef struct { const TernaryConv2D *l; float *out; int32_t *raw; size_t kwords; } ConvF;
static void conv_fwd_rows(size_t h0, size_t h1, void *c_) {
    ConvF *c = (ConvF *)c_;
    const TernaryConv2D *layer = c->l; const size_t kwords = c->kwords;
    TernaryVec *patch = ternary_vec_alloc(layer->kernel_size);
    float *patch_f = (float *)ternary_malloc64(layer->kernel_size * sizeof(float));
    if (!patch || !patch_f) abort();
    TernaryVec kern; kern.n = layer->kernel_size; kern.words = kwords;
    for (size_t oh = h0; oh < h1; oh++) {
        for (size_t ow = 0; ow < layer->out_width; ow++) {
            gather_patch_f(layer, oh, ow, patch_f);
            memset(patch->mask, 0, kwords * sizeof(uint64_t)); memset(patch->sign, 0, kwords * sizeof(uint64_t));
            if (ternary_avx512_enabled()) ternary_avx512_pack_thr(patch_f, layer->kernel_size, 0.0f, patch->mask, patch->sign);
            else
            for (size_t k = 0; k < layer->kernel_size; k++) {
                const float v = patch_f[k];
                if (v != 0.0f) {
                    const uint64_t b = (uint64_t)1 << (k & 63);
                    patch->mask[k >> 6] |= b;
                    if (v < 0.0f) patch->sign[k >> 6] |= b;
                }
            }
            for (size_t o = 0; o < layer->out_channels; o++) {
                kern.mask = layer->kcache + (2 * o) * kwords; kern.sign = layer->kcache + (2 * o + 1) * kwords;
                int32_t z = ternary_dot(&kern, patch);
                if (layer->bias) z += ternary_vec_get(layer->bias, o);
                size_t out_idx = ternary_image_index(oh, ow, o, layer->out_width, layer->out_channels);
                if (c->raw) c->raw[out_idx] = z;
                c->out[out_idx] = layer->scale * (float)z;
            }
        }
    }
    free(patch_f); ternary_vec_free(patch);
}

void ternary_conv2d_forward(const TernaryConv2D *layer, const TernaryVec *image,
                             float *out, int32_t *out_raw_opt) {
    const size_t kwords = (layer->kernel_size + 63) / 64;
    TernaryVec kern;
    kern.n = layer->kernel_size; kern.words = kwords;
    /* Los kernels se extraen UNA vez por llamada (antes: una vez por posicion y por canal). */
    for (size_t o = 0; o < layer->out_channels; o++) {
        kern.mask = layer->kcache + (2 * o) * kwords; kern.sign = layer->kcache + (2 * o + 1) * kwords;
        extract_kernel(layer, o, &kern);
    }
    expand_image(layer, image);
    ConvF c = { layer, out, out_raw_opt, kwords };
    ternary_parallel_ranges(ternary_get_compute_pool(), 0, layer->out_height, 0, conv_fwd_rows, &c);
}

typedef struct { const TernaryConv2D *l; const float *go; const int32_t *z; float *gk, *gb, *terms; } ConvB;
static void conv_bwd_channels(size_t o0, size_t o1, void *c_) {
    ConvB *c = (ConvB *)c_;
    const TernaryConv2D *layer = c->l; const size_t KS = layer->kernel_size;
    float *patch_f = (float *)ternary_malloc64(KS * sizeof(float));
    if (!patch_f) abort();
    for (size_t oh = 0; oh < layer->out_height; oh++) {
        for (size_t ow = 0; ow < layer->out_width; ow++) {
            gather_patch_f(layer, oh, ow, patch_f);
            for (size_t o = o0; o < o1; o++) {
                size_t out_idx = ternary_image_index(oh, ow, o, layer->out_width, layer->out_channels);
                float g = c->go[out_idx]; /* mismo criterio que Dense: sin escala aqui */
                c->terms[out_idx] = g * (float)c->z[out_idx];
                if (c->gb) c->gb[o] += g;
                float *restrict grad_row = c->gk + o * KS;
                if (ternary_avx512_enabled()) ternary_avx512_outer_acc(grad_row, KS, g, patch_f);
                else for (size_t k = 0; k < KS; k++) grad_row[k] += g * patch_f[k];
            }
        }
    }
    free(patch_f);
}

void ternary_conv2d_backward(const TernaryConv2D *layer, const TernaryVec *image,
                              const float *grad_out, float *grad_kernels_out,
                              float *grad_bias_out, float *grad_scale_out,
                              const int32_t *z_raw_opt) {
    const size_t n = layer->out_height * layer->out_width * layer->out_channels;
    int32_t *z_own = NULL;
    if (!z_raw_opt) {
        /* Sin Z crudo: se recalcula con el forward (repartido en hilos); el resto del backward es el mismo camino. */
        float *tmp = (float *)ternary_malloc64(n * sizeof(float));
        z_own = (int32_t *)ternary_malloc64(n * sizeof(int32_t));
        if (!tmp || !z_own) abort();
        ternary_conv2d_forward(layer, image, tmp, z_own);
        free(tmp);
        z_raw_opt = z_own;
    }
    expand_image(layer, image);
    float *terms = (float *)ternary_malloc64(n * sizeof(float));
    if (!terms) abort();
    ConvB c = { layer, grad_out, z_raw_opt, grad_kernels_out, grad_bias_out, terms };
    ternary_parallel_ranges(ternary_get_compute_pool(), 0, layer->out_channels, 1, conv_bwd_channels, &c);
    float gs = *grad_scale_out;
    for (size_t i = 0; i < n; i++) gs += terms[i];             /* orden de siempre: posicion, canal */
    *grad_scale_out = gs;
    free(terms); free(z_own);
}


void ternary_conv2d_apply_kernel_gradients(TernaryConv2D *layer, size_t o, const float *grad_kernel) {
    const TernaryUpdateRule rule = ternary_update_rule_get();
    ternary_r2_update_row(layer->kernels, layer->counter_kernels, o * layer->kernel_size, grad_kernel,
                          layer->kernel_size, &rule);   /* regla R2, un filtro = una fila */
}

void ternary_conv2d_apply_bias_gradients(TernaryConv2D *layer, const float *grad_bias) {
    if (!layer->bias) return;
    const TernaryUpdateRule rule = ternary_update_rule_get();
    ternary_r2_update_row(layer->bias, layer->counter_bias, 0, grad_bias, layer->out_channels, &rule);
}

void ternary_conv2d_apply_scale_gradient(TernaryConv2D *layer, float grad_scale, float scale_lr) {
    float new_scale = layer->scale - scale_lr * grad_scale;
    if (new_scale < 0.02f) new_scale = 0.02f;
    layer->scale = new_scale;
}


