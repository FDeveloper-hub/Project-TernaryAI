#include "ternary_mem.h"
#include "ternary_layer.h"
#include "ternary_residual.h"
#include "ternary_model_io.h"
#include <stdlib.h>
#include <string.h>

/* export/import al formato model.ternary (definidas al final del archivo) */
static int dense_export(const void *self, const TernaryExportCtx *cx, const char *prefix, TernaryJBuf *cfg);
static void *dense_import(const TernaryImportCtx *cx, const char *prefix, const TernaryJNode *cfg);
static int conv2d_export(const void *self, const TernaryExportCtx *cx, const char *prefix, TernaryJBuf *cfg);
static void *conv2d_import(const TernaryImportCtx *cx, const char *prefix, const TernaryJNode *cfg);
static int emb_export(const void *self, const TernaryExportCtx *cx, const char *prefix, TernaryJBuf *cfg);
static void *emb_import(const TernaryImportCtx *cx, const char *prefix, const TernaryJNode *cfg);
static int gqa_export(const void *self, const TernaryExportCtx *cx, const char *prefix, TernaryJBuf *cfg);
static void *gqa_import(const TernaryImportCtx *cx, const char *prefix, const TernaryJNode *cfg);
static int rms_export(const void *self, const TernaryExportCtx *cx, const char *prefix, TernaryJBuf *cfg);
static void *rms_import(const TernaryImportCtx *cx, const char *prefix, const TernaryJNode *cfg);
static int swiglu_export(const void *self, const TernaryExportCtx *cx, const char *prefix, TernaryJBuf *cfg);
static void *swiglu_import(const TernaryImportCtx *cx, const char *prefix, const TernaryJNode *cfg);
static int head_export(const void *self, const TernaryExportCtx *cx, const char *prefix, TernaryJBuf *cfg);
static void *head_import(const TernaryImportCtx *cx, const char *prefix, const TernaryJNode *cfg);

/* ===================================================================
 * Dense
 * =================================================================== */

static size_t dense_input_len(const void *self) { return ((const TernaryDense *)self)->input_dim; }
static size_t dense_output_len(const void *self) { return ((const TernaryDense *)self)->output_dim; }
static size_t dense_num_matrices(const void *self) { (void)self; return 1; }
static size_t dense_grad_rows(const void *self, size_t m) { (void)m; return ((const TernaryDense *)self)->output_dim; }
static size_t dense_grad_row_width(const void *self, size_t m) { (void)m; return ((const TernaryDense *)self)->input_dim; }
static int dense_has_bias(const void *self, size_t m) { (void)m; return ((const TernaryDense *)self)->bias != NULL; }
static int dense_propagates_grad_x(const void *self) { (void)self; return 1; }
static TernaryDType dense_input_dtype(const void *self) { (void)self; return TERNARY_DTYPE_TERNARY; }
static size_t dense_accepts_u8(const void *self) { return ((const TernaryDense *)self)->input_dim; }

static void dense_forward(const void *self, const TernaryTensor *x, float *out, int32_t *out_raw_opt,
                           const TernaryCapabilities *caps) {
    if (x->dtype == TERNARY_DTYPE_UINT8) { (void)caps; ternary_dense_forward_u8((const TernaryDense *)self, x->data_u8, out, out_raw_opt); return; }
    ternary_dense_forward_auto((const TernaryDense *)self, x->data_t, out, out_raw_opt, caps);
}

static void dense_backward(const void *self, const TernaryTensor *x, const float *grad_out,
                            float *const *grad_w_out, float *grad_x_out,
                            float *grad_scale_out, float *const *grad_bias_out,
                            const int32_t *z_raw_opt) {
    if (x->dtype == TERNARY_DTYPE_UINT8) {
        /* ternary_dense_backward_u8 ACUMULA; este adaptador sobreescribe grad_w y grad_bias */
        const TernaryDense *d = (const TernaryDense *)self;
        memset(grad_w_out[0], 0, d->output_dim * d->input_dim * sizeof(float));
        if (grad_bias_out[0]) memset(grad_bias_out[0], 0, d->output_dim * sizeof(float));
        if (grad_x_out) memset(grad_x_out, 0, d->input_dim * sizeof(float));
        if (grad_scale_out) *grad_scale_out = 0.0f;
        ternary_dense_backward_u8(d, x->data_u8, grad_out, grad_w_out[0], grad_x_out, grad_scale_out, grad_bias_out[0], z_raw_opt);
        return;
    }
    ternary_dense_backward((const TernaryDense *)self, x->data_t, grad_out, grad_w_out[0], grad_x_out,
                            grad_scale_out, grad_bias_out[0], z_raw_opt);
}

static void dense_forward_block(const void *self, size_t S, const TernaryTensor *const *xs,
                                float *const *outs, int32_t *const *raws, const TernaryCapabilities *caps) {
    const TernaryVec *vx[S ? S : 1];
    for (size_t s = 0; s < S; s++) vx[s] = xs[s]->data_t;
    ternary_dense_forward_block((const TernaryDense *)self, S, vx, outs, raws, caps);
}

static void dense_backward_acc(const void *self, const TernaryTensor *x, const float *grad_out,
                                float *const *grad_w_out, float *grad_x_out,
                                float *grad_scale_out, float *const *grad_bias_out,
                                const int32_t *z_raw_opt) {
    if (x->dtype == TERNARY_DTYPE_UINT8) {
        ternary_dense_backward_u8((const TernaryDense *)self, x->data_u8, grad_out, grad_w_out[0], grad_x_out,
                                  grad_scale_out, grad_bias_out[0], z_raw_opt);
        return;
    }
    ternary_dense_backward_accumulate((const TernaryDense *)self, x->data_t, grad_out, grad_w_out[0], grad_x_out,
                                       grad_scale_out, grad_bias_out[0], z_raw_opt);
}

static void dense_backward_block(const void *self, size_t S, const TernaryTensor *const *xs,
                                 const float *const *grad_out, const int32_t *const *z_raw,
                                 float *const *grad_w_out, float *const *grad_x,
                                 float *grad_scale_out, float *const *grad_bias_out) {
    const TernaryVec *vx[S ? S : 1];
    for (size_t s = 0; s < S; s++) vx[s] = xs[s]->data_t;
    ternary_dense_backward_block((const TernaryDense *)self, S, vx, grad_out, z_raw,
                                  grad_w_out[0], grad_x, grad_scale_out, grad_bias_out[0]);
}

static void dense_apply_row_gradients(void *self, size_t m, size_t row, const float *grad_row) {
    (void)m;
    ternary_dense_apply_row_gradients((TernaryDense *)self, row, grad_row);
}

static void dense_apply_matrix_gradients(void *self, size_t m, const float *grad_w) {
    (void)m;
    ternary_dense_apply_matrix_gradients((TernaryDense *)self, grad_w);
}

static void dense_apply_bias_gradients(void *self, size_t m, const float *grad_bias) {
    (void)m;
    ternary_dense_apply_bias_gradients((TernaryDense *)self, grad_bias);
}

static void dense_apply_scale_gradient(void *self, size_t m, float grad_scale, float scale_lr) {
    (void)m;
    ternary_dense_apply_scale_gradient((TernaryDense *)self, grad_scale, scale_lr);
}

static void dense_destroy(void *self) { ternary_dense_free((TernaryDense *)self); }

static const TernaryLayerVTable TERNARY_DENSE_VTABLE = {
    "dense",
    dense_input_len, dense_output_len, dense_num_matrices, dense_grad_rows, dense_grad_row_width,
    dense_has_bias, dense_propagates_grad_x, dense_input_dtype,
    dense_forward, dense_backward,
    dense_apply_row_gradients, dense_apply_bias_gradients, dense_apply_scale_gradient,
    dense_destroy,
    dense_backward_acc,
    dense_backward_block,
    dense_apply_matrix_gradients,
    dense_forward_block,
    NULL, /* state_bytes: sin estado compartible */
    NULL, /* bind_state */
    dense_accepts_u8,
    NULL, /* output_row_len */
    dense_export, dense_import,
};

TernaryLayer ternary_layer_from_dense(TernaryDense *layer) {
    TernaryLayer l; l.impl = layer; l.vt = &TERNARY_DENSE_VTABLE; return l;
}

/* ===================================================================
 * Conv2D
 * =================================================================== */

static size_t conv2d_input_len(const void *self) {
    const TernaryConv2D *c = (const TernaryConv2D *)self;
    return c->in_height * c->in_width * c->in_channels;
}
static size_t conv2d_output_len(const void *self) {
    const TernaryConv2D *c = (const TernaryConv2D *)self;
    return c->out_height * c->out_width * c->out_channels;
}
static size_t conv2d_num_matrices(const void *self) { (void)self; return 1; }
static size_t conv2d_grad_rows(const void *self, size_t m) { (void)m; return ((const TernaryConv2D *)self)->out_channels; }
static size_t conv2d_grad_row_width(const void *self, size_t m) { (void)m; return ((const TernaryConv2D *)self)->kernel_size; }
static int conv2d_has_bias(const void *self, size_t m) { (void)m; return ((const TernaryConv2D *)self)->bias != NULL; }
/* Conv2D nunca calcula gradiente hacia su imagen de entrada (ver
 * ternary_conv2d_backward) -- ternary_stack_add() usa esto para
 * rechazar ponerla en cualquier posicion que no sea la 0. */
static int conv2d_propagates_grad_x(const void *self) { (void)self; return 0; }
static TernaryDType conv2d_input_dtype(const void *self) { (void)self; return TERNARY_DTYPE_TERNARY; }

static void conv2d_forward(const void *self, const TernaryTensor *x, float *out, int32_t *out_raw_opt,
                            const TernaryCapabilities *caps) {
    (void)caps; /* Conv2D no tiene variante GPU todavia -- ver README */
    ternary_conv2d_forward((const TernaryConv2D *)self, x->data_t, out, out_raw_opt);
}

static void conv2d_backward(const void *self, const TernaryTensor *x, const float *grad_out,
                             float *const *grad_w_out, float *grad_x_out,
                             float *grad_scale_out, float *const *grad_bias_out,
                             const int32_t *z_raw_opt) {
    (void)grad_x_out; /* Conv2D no lo produce -- ver propagates_grad_x=0 */
    ternary_conv2d_backward((const TernaryConv2D *)self, x->data_t, grad_out, grad_w_out[0],
                             grad_bias_out[0], grad_scale_out, z_raw_opt);
}

static void conv2d_apply_row_gradients(void *self, size_t m, size_t row, const float *grad_row) {
    (void)m;
    ternary_conv2d_apply_kernel_gradients((TernaryConv2D *)self, row, grad_row);
}
static void conv2d_apply_bias_gradients(void *self, size_t m, const float *grad_bias) {
    (void)m;
    ternary_conv2d_apply_bias_gradients((TernaryConv2D *)self, grad_bias);
}
static void conv2d_apply_scale_gradient(void *self, size_t m, float grad_scale, float scale_lr) {
    (void)m;
    ternary_conv2d_apply_scale_gradient((TernaryConv2D *)self, grad_scale, scale_lr);
}
static void conv2d_destroy(void *self) { ternary_conv2d_free((TernaryConv2D *)self); }

static const TernaryLayerVTable TERNARY_CONV2D_VTABLE = {
    "conv2d",
    conv2d_input_len, conv2d_output_len, conv2d_num_matrices, conv2d_grad_rows, conv2d_grad_row_width,
    conv2d_has_bias, conv2d_propagates_grad_x, conv2d_input_dtype,
    conv2d_forward, conv2d_backward,
    conv2d_apply_row_gradients, conv2d_apply_bias_gradients, conv2d_apply_scale_gradient,
    conv2d_destroy,
    NULL, /* backward_accumulate: no soportado */
    NULL, /* backward_block: no soportado */
    NULL, /* apply_matrix_gradients: no soportado */
    NULL, /* forward_block: no soportado */
    NULL, /* state_bytes: sin estado compartible */
    NULL, /* bind_state */
    NULL, /* accepts_u8 */
    NULL, /* output_row_len */
    conv2d_export, conv2d_import,
};

TernaryLayer ternary_layer_from_conv2d(TernaryConv2D *layer) {
    TernaryLayer l; l.impl = layer; l.vt = &TERNARY_CONV2D_VTABLE; return l;
}

/* ===================================================================
 * Registro de tipos de capa (ver ternary_layer.h)
 * =================================================================== */

#include <math.h>
#include <stdarg.h>
#include <stdio.h>

static void set_err(char *err, size_t cap, const char *fmt, ...) {
    if (!err || cap == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, cap, fmt, ap);
    va_end(ap);
}

/* ===================================================================
 * Embedding: entrada = ids (dtype ID), salida = seq_len*embed_dim floats.
 * Es la primera capa que NO recibe un TernaryVec -- ver ternary_tensor.h.
 * =================================================================== */
static size_t emb_input_len(const void *self) { return ((const TernaryEmbedding *)self)->seq_len; }
static size_t emb_output_len(const void *self) {
    const TernaryEmbedding *e = (const TernaryEmbedding *)self;
    return e->seq_len * e->embed_dim;
}
static size_t emb_num_matrices(const void *self) { (void)self; return 1; }
static size_t emb_grad_rows(const void *self, size_t m) { (void)m; return ((const TernaryEmbedding *)self)->vocab_size; }
static size_t emb_grad_row_width(const void *self, size_t m) { (void)m; return ((const TernaryEmbedding *)self)->embed_dim; }
static int emb_has_bias(const void *self, size_t m) { (void)m; (void)self; return 0; }
static int emb_propagates_grad_x(const void *self) { (void)self; return 0; }  /* los ids no se derivan */
static TernaryDType emb_input_dtype(const void *self) { (void)self; return TERNARY_DTYPE_ID; }

static void emb_forward(const void *self, const TernaryTensor *x, float *out, int32_t *out_raw_opt,
                         const TernaryCapabilities *caps) {
    (void)caps;
    ternary_embedding_forward((const TernaryEmbedding *)self, x->data_i, out, out_raw_opt);
}

static void emb_backward(const void *self, const TernaryTensor *x, const float *grad_out,
                          float *const *grad_w_out, float *grad_x_out, float *grad_scale_out,
                          float *const *grad_bias_out, const int32_t *z_raw_opt) {
    (void)grad_x_out; (void)grad_bias_out;
    ternary_embedding_backward((const TernaryEmbedding *)self, x->data_i, grad_out, grad_w_out[0],
                                grad_scale_out, z_raw_opt);
}

static void emb_apply_row_gradients(void *self, size_t m, size_t row, const float *grad_row) {
    (void)m;
    ternary_embedding_apply_row_gradients((TernaryEmbedding *)self, row, grad_row);
}
static void emb_apply_bias_gradients(void *self, size_t m, const float *g) {
    (void)m;
    (void)self; (void)g;  /* sin bias */
}
static void emb_apply_scale_gradient(void *self, size_t m, float grad_scale, float scale_lr) {
    (void)m;
    ternary_embedding_apply_scale_gradient((TernaryEmbedding *)self, grad_scale, scale_lr);
}
static void emb_destroy(void *self) { ternary_embedding_free((TernaryEmbedding *)self); }

static size_t emb_output_row_len(const void *self) { return ((const TernaryEmbedding *)self)->embed_dim; }

static const TernaryLayerVTable TERNARY_EMBEDDING_VTABLE = {
    "embedding",
    emb_input_len, emb_output_len, emb_num_matrices, emb_grad_rows, emb_grad_row_width,
    emb_has_bias, emb_propagates_grad_x, emb_input_dtype,
    emb_forward, emb_backward,
    emb_apply_row_gradients, emb_apply_bias_gradients, emb_apply_scale_gradient,
    emb_destroy,
    NULL, /* backward_accumulate: no soportado */
    NULL, /* backward_block: no soportado */
    NULL, /* apply_matrix_gradients: no soportado */
    NULL, /* forward_block: no soportado */
    NULL, /* state_bytes: sin estado compartible */
    NULL, /* bind_state */
    NULL, /* accepts_u8 */
    emb_output_row_len,
    emb_export, emb_import,
};

TernaryLayer ternary_layer_from_embedding(TernaryEmbedding *layer) {
    TernaryLayer l; l.impl = layer; l.vt = &TERNARY_EMBEDDING_VTABLE; return l;
}


/* ===================================================================
 * GQA (atencion causal multi-cabeza, RoPE opcional): entrada = seq_len*embed_dim
 * valores ternarios (un TernaryVec con todas las posiciones), salida =
 * seq_len*embed_dim floats (escala de Wo * z entero de Wo). Cuatro matrices:
 * m=0 Wq, m=1 Wk, m=2 Wv, m=3 Wo. SOLO la escala de Wo influye en la salida:
 * Q/K/V se usan por su signo y attn_cat se binariza antes de Wo (ver
 * tests/test_gqa_scale.c), asi que el gradiente de escala de Wq/Wk/Wv NO se
 * acumula ni se aplica (seria un gradiente sustituto sin efecto en la salida).
 * =================================================================== */
static size_t gqa_input_len(const void *self) {
    const TernaryGQA *g = (const TernaryGQA *)self;
    return g->seq_len * g->embed_dim;
}
static size_t gqa_output_len(const void *self) { return gqa_input_len(self); }
static size_t gqa_num_matrices(const void *self) { (void)self; return 4; }
static size_t gqa_grad_rows(const void *self, size_t m) {
    const TernaryGQA *g = (const TernaryGQA *)self;
    if (m == 0) return g->n_heads * g->head_dim;
    if (m == 3) return g->embed_dim;
    return g->n_kv_heads * g->head_dim;
}
static size_t gqa_grad_row_width(const void *self, size_t m) {
    const TernaryGQA *g = (const TernaryGQA *)self;
    return (m == 3) ? g->n_heads * g->head_dim : g->embed_dim;
}
static int gqa_has_bias(const void *self, size_t m) { (void)self; (void)m; return 0; }
static int gqa_propagates_grad_x(const void *self) { (void)self; return 1; }
static TernaryDType gqa_input_dtype(const void *self) { (void)self; return TERNARY_DTYPE_TERNARY; }
static size_t gqa_accepts_u8(const void *self) { return ((const TernaryGQA *)self)->embed_dim; }
static size_t gqa_output_row_len(const void *self) { return ((const TernaryGQA *)self)->embed_dim; }

static TernaryDense *gqa_matrix(const TernaryGQA *g, size_t m) {
    return (m == 0) ? g->wq : (m == 1) ? g->wk : (m == 2) ? g->wv : g->wo;
}

static void gqa_forward_adapter(const void *self, const TernaryTensor *x, float *out, int32_t *out_raw_opt,
                                const TernaryCapabilities *caps) {
    (void)caps;
    if (x->dtype == TERNARY_DTYPE_UINT8) { ternary_gqa_forward_u8((const TernaryGQA *)self, x->data_u8, out, out_raw_opt); return; }
    ternary_gqa_forward((const TernaryGQA *)self, x->data_t, out, out_raw_opt);
}

static void gqa_backward_adapter(const void *self, const TernaryTensor *x, const float *grad_out,
                                 float *const *grad_w_out, float *grad_x_out, float *grad_scale_out,
                                 float *const *grad_bias_out, const int32_t *z_raw_opt) {
    (void)grad_bias_out;
    float gs[4] = {0, 0, 0, 0};
    if (x->dtype == TERNARY_DTYPE_UINT8) {
        ternary_gqa_backward_u8((TernaryGQA *)self, x->data_u8, grad_out, grad_w_out, grad_x_out, gs);
        grad_scale_out[3] += gs[3];
        return;
    }
    ternary_gqa_backward((TernaryGQA *)self, x->data_t, grad_out, grad_w_out, grad_x_out, gs, z_raw_opt);
    grad_scale_out[3] += gs[3];   /* solo la escala de Wo; ver comentario de arriba */
}

static void gqa_apply_row_gradients(void *self, size_t m, size_t row, const float *grad_row) {
    const TernaryGQA *g = (const TernaryGQA *)self;
    ternary_dense_apply_row_gradients(gqa_matrix(g, m), row, grad_row);
}
static void gqa_apply_matrix_gradients(void *self, size_t m, const float *grad_w) {
    ternary_dense_apply_matrix_gradients(gqa_matrix((const TernaryGQA *)self, m), grad_w);   /* filas repartidas en hilos */
}
static void gqa_apply_bias_gradients(void *self, size_t m, const float *g) {
    (void)self; (void)m; (void)g;  /* sin bias */
}
static void gqa_apply_scale_gradient(void *self, size_t m, float grad_scale, float scale_lr) {
    if (m == 3) ternary_dense_apply_scale_gradient(((TernaryGQA *)self)->wo, grad_scale, scale_lr);
}
static void gqa_destroy(void *self) { ternary_gqa_free((TernaryGQA *)self); }

/* modo de RoPE: 0 = sin RoPE, 1 = RoPE + puntaje entero, 2 = RoPE con Q/K binarizados */
static int gqa_rope_mode(const TernaryGQA *g) { return g->qk_int ? 1 : (g->rope ? 2 : 0); }

static size_t gqa_state_bytes(const void *self) { return ternary_gqa_state_bytes((const TernaryGQA *)self); }
static void gqa_bind_state(void *self, void *base) { ternary_gqa_bind_state((TernaryGQA *)self, base); }

static const TernaryLayerVTable TERNARY_GQA_VTABLE = {
    "gqa",
    gqa_input_len, gqa_output_len, gqa_num_matrices, gqa_grad_rows, gqa_grad_row_width,
    gqa_has_bias, gqa_propagates_grad_x, gqa_input_dtype,
    gqa_forward_adapter, gqa_backward_adapter,
    gqa_apply_row_gradients, gqa_apply_bias_gradients, gqa_apply_scale_gradient,
    gqa_destroy,
    NULL, /* backward_accumulate: no soportado */
    NULL, /* backward_block: no soportado */
    gqa_apply_matrix_gradients,
    NULL, /* forward_block: no soportado */
    gqa_state_bytes,
    gqa_bind_state,
    gqa_accepts_u8,
    gqa_output_row_len,
    gqa_export, gqa_import,
};

TernaryLayer ternary_layer_from_gqa(TernaryGQA *layer) {
    TernaryLayer l; l.impl = layer; l.vt = &TERNARY_GQA_VTABLE; return l;
}

/* ===================================================================
 * RMSNorm (ver ternary_rmsnorm.h)
 *
 * Entrada FLOAT (el crudo de la capa anterior, sin cuantizar). Una sola matriz: la ganancia ternaria
 * 1 x dim, con su propia escala float. Las 'rows' filas de la entrada (rows*dim floats) se normalizan
 * por separado. No hay bias.
 * =================================================================== */
static size_t rms_input_len(const void *self) {
    const TernaryRMSNorm *n = (const TernaryRMSNorm *)self;
    return n->rows * n->dim;
}
static size_t rms_output_len(const void *self) { return rms_input_len(self); }
static size_t rms_num_matrices(const void *self) { (void)self; return 1; }
static size_t rms_grad_rows(const void *self, size_t m) { (void)self; (void)m; return 1; }
static size_t rms_grad_row_width(const void *self, size_t m) { (void)m; return ((const TernaryRMSNorm *)self)->dim; }
static int rms_has_bias(const void *self, size_t m) { (void)self; (void)m; return 0; }
static int rms_propagates_grad_x(const void *self) { (void)self; return 1; }
static TernaryDType rms_input_dtype(const void *self) { (void)self; return TERNARY_DTYPE_FLOAT; }

static void rms_forward_adapter(const void *self, const TernaryTensor *x, float *out, int32_t *out_raw_opt,
                                const TernaryCapabilities *caps) {
    (void)caps;
    const TernaryRMSNorm *n = (const TernaryRMSNorm *)self;
    ternary_rmsnorm_forward(n, x->data_f, out);
    if (out_raw_opt) for (size_t i = 0; i < n->rows * n->dim; i++) out_raw_opt[i] = 0;  /* no hay Z entero */
}

static void rms_backward_adapter(const void *self, const TernaryTensor *x, const float *grad_out,
                                 float *const *grad_w_out, float *grad_x_out, float *grad_scale_out,
                                 float *const *grad_bias_out, const int32_t *z_raw_opt) {
    (void)grad_bias_out; (void)z_raw_opt;
    ternary_rmsnorm_backward((const TernaryRMSNorm *)self, x->data_f, grad_out, grad_w_out[0], grad_x_out, &grad_scale_out[0]);
}

static void rms_apply_row_gradients(void *self, size_t m, size_t row, const float *grad_row) {
    (void)m;
    ternary_dense_apply_row_gradients(((TernaryRMSNorm *)self)->gain, row, grad_row);
}
static void rms_apply_bias_gradients(void *self, size_t m, const float *g) {
    (void)self; (void)m; (void)g;  /* sin bias */
}
static void rms_apply_scale_gradient(void *self, size_t m, float grad_scale, float scale_lr) {
    (void)m; (void)scale_lr;   /* learning rate propio y fijo: ver TERNARY_RMSNORM_SCALE_LR */
    ternary_dense_apply_scale_gradient(((TernaryRMSNorm *)self)->gain, grad_scale, TERNARY_RMSNORM_SCALE_LR);
}
static size_t rms_output_row_len(const void *self) { return ((const TernaryRMSNorm *)self)->dim; }
static void rms_destroy(void *self) { ternary_rmsnorm_free((TernaryRMSNorm *)self); }

static const TernaryLayerVTable TERNARY_RMSNORM_VTABLE = {
    "rmsnorm",
    rms_input_len, rms_output_len, rms_num_matrices, rms_grad_rows, rms_grad_row_width,
    rms_has_bias, rms_propagates_grad_x, rms_input_dtype,
    rms_forward_adapter, rms_backward_adapter,
    rms_apply_row_gradients, rms_apply_bias_gradients, rms_apply_scale_gradient,
    rms_destroy,
    NULL, /* backward_accumulate: no soportado */
    NULL, /* backward_block: no soportado */
    NULL, /* apply_matrix_gradients: no soportado */
    NULL, /* forward_block: no soportado */
    NULL, /* state_bytes: sin estado compartible */
    NULL, /* bind_state */
    NULL, /* accepts_u8 */
    rms_output_row_len,
    rms_export, rms_import,
};

TernaryLayer ternary_layer_from_rmsnorm(TernaryRMSNorm *layer) {
    TernaryLayer l; l.impl = layer; l.vt = &TERNARY_RMSNORM_VTABLE; return l;
}

/* ===================================================================
 * SwiGLU (ver ternary_swiglu.h): entrada = rows*dim ternarios, salida = rows*dim floats. Tres matrices:
 * m=0 W_gate, m=1 W_up, m=2 W_down, sin bias. Solo influyen en la salida la escala de W_down y (opcion 1,
 * compuerta silu) la de W_gate; la de W_up nunca (el umbral de la banda es relativo), asi que no se aplica.
 * Con gate_mode=2 (ReLU² sin compuerta) son DOS matrices: m=0 W_up, m=1 W_down (ver ternary_swiglu_matrix).
 * =================================================================== */
static size_t swiglu_input_len(const void *self) {
    const TernarySwiGLU *s = (const TernarySwiGLU *)self;
    return s->rows * s->dim;
}
static size_t swiglu_output_len(const void *self) { return swiglu_input_len(self); }
static size_t swiglu_num_matrices(const void *self) { return ternary_swiglu_num_matrices((const TernarySwiGLU *)self); }
static size_t swiglu_grad_rows(const void *self, size_t m) {
    const TernarySwiGLU *s = (const TernarySwiGLU *)self;
    return (m == ternary_swiglu_num_matrices(s) - 1) ? s->dim : s->hidden;   /* la ultima (down) es dim x hidden */
}
static size_t swiglu_grad_row_width(const void *self, size_t m) {
    const TernarySwiGLU *s = (const TernarySwiGLU *)self;
    return (m == ternary_swiglu_num_matrices(s) - 1) ? s->hidden : s->dim;
}
static int swiglu_has_bias(const void *self, size_t m) { (void)self; (void)m; return 0; }
static int swiglu_propagates_grad_x(const void *self) { (void)self; return 1; }
static TernaryDType swiglu_input_dtype(const void *self) { (void)self; return TERNARY_DTYPE_TERNARY; }
static size_t swiglu_accepts_u8(const void *self) { return ((const TernarySwiGLU *)self)->dim; }
static size_t swiglu_output_row_len(const void *self) { return ((const TernarySwiGLU *)self)->dim; }

static TernaryDense *swiglu_matrix(const TernarySwiGLU *s, size_t m) { return ternary_swiglu_matrix(s, m); }

static void swiglu_forward_adapter(const void *self, const TernaryTensor *x, float *out, int32_t *out_raw_opt,
                                   const TernaryCapabilities *caps) {
    (void)caps;
    if (x->dtype == TERNARY_DTYPE_UINT8) { ternary_swiglu_forward_u8((const TernarySwiGLU *)self, x->data_u8, out, out_raw_opt); return; }
    ternary_swiglu_forward((const TernarySwiGLU *)self, x->data_t, out, out_raw_opt);
}

static void swiglu_backward_adapter(const void *self, const TernaryTensor *x, const float *grad_out,
                                    float *const *grad_w_out, float *grad_x_out, float *grad_scale_out,
                                    float *const *grad_bias_out, const int32_t *z_raw_opt) {
    (void)grad_bias_out; (void)z_raw_opt;   /* el Z crudo ya esta en el scratch, calculado en el forward inmediato */
    if (x->dtype == TERNARY_DTYPE_UINT8) { ternary_swiglu_backward_u8((TernarySwiGLU *)self, x->data_u8, grad_out, grad_w_out, grad_x_out, grad_scale_out); return; }
    ternary_swiglu_backward((TernarySwiGLU *)self, x->data_t, grad_out, grad_w_out, grad_x_out, grad_scale_out);
}

static void swiglu_apply_matrix_gradients(void *self, size_t m, const float *grad_w) {
    ternary_dense_apply_matrix_gradients(swiglu_matrix((const TernarySwiGLU *)self, m), grad_w);   /* filas repartidas en hilos */
}
static void swiglu_apply_row_gradients(void *self, size_t m, size_t row, const float *grad_row) {
    ternary_dense_apply_row_gradients(swiglu_matrix((const TernarySwiGLU *)self, m), row, grad_row);
}
static void swiglu_apply_bias_gradients(void *self, size_t m, const float *g) {
    (void)self; (void)m; (void)g;  /* sin bias */
}
static void swiglu_apply_scale_gradient(void *self, size_t m, float grad_scale, float scale_lr) {
    TernarySwiGLU *s = (TernarySwiGLU *)self;
    if (m == ternary_swiglu_num_matrices(s) - 1) ternary_dense_apply_scale_gradient(s->wd, grad_scale, scale_lr);
    else if (m == 0 && s->gate_mode == TERNARY_SWIGLU_GATE_SILU) {
        ternary_dense_apply_scale_gradient(s->wg, grad_scale, scale_lr);
        ternary_swiglu_refresh_tables(s);
    }
}
static void swiglu_destroy(void *self) { ternary_swiglu_free((TernarySwiGLU *)self); }

static size_t swiglu_state_bytes(const void *self) { return ternary_swiglu_state_bytes((const TernarySwiGLU *)self); }
static void swiglu_bind_state(void *self, void *base) { ternary_swiglu_bind_state((TernarySwiGLU *)self, base); }

static const TernaryLayerVTable TERNARY_SWIGLU_VTABLE = {
    "swiglu",
    swiglu_input_len, swiglu_output_len, swiglu_num_matrices, swiglu_grad_rows, swiglu_grad_row_width,
    swiglu_has_bias, swiglu_propagates_grad_x, swiglu_input_dtype,
    swiglu_forward_adapter, swiglu_backward_adapter,
    swiglu_apply_row_gradients, swiglu_apply_bias_gradients, swiglu_apply_scale_gradient,
    swiglu_destroy,
    NULL, /* backward_accumulate: no soportado */
    NULL, /* backward_block: no soportado */
    swiglu_apply_matrix_gradients,
    NULL, /* forward_block: no soportado */
    swiglu_state_bytes,
    swiglu_bind_state,
    swiglu_accepts_u8,
    swiglu_output_row_len,
    swiglu_export, swiglu_import,
};

TernaryLayer ternary_layer_from_swiglu(TernarySwiGLU *layer) {
    TernaryLayer l; l.impl = layer; l.vt = &TERNARY_SWIGLU_VTABLE; return l;
}

/* ===================================================================
 * LMHead (ver ternary_head.h): entrada = rows*dim ternarios, salida = rows*vocab floats (logits). Una sola
 * matriz W (vocab x dim) compartida por todas las posiciones, sin bias; su escala SI influye en la salida.
 * =================================================================== */
static size_t head_input_len(const void *self) {
    const TernaryHead *h = (const TernaryHead *)self;
    return h->rows * h->dim;
}
static size_t head_output_len(const void *self) {
    const TernaryHead *h = (const TernaryHead *)self;
    return h->rows * h->vocab;
}
static size_t head_num_matrices(const void *self) { (void)self; return 1; }
static size_t head_grad_rows(const void *self, size_t m) { (void)m; return ((const TernaryHead *)self)->vocab; }
static size_t head_grad_row_width(const void *self, size_t m) { (void)m; return ((const TernaryHead *)self)->dim; }
static int head_has_bias(const void *self, size_t m) { (void)self; (void)m; return 0; }
static int head_propagates_grad_x(const void *self) { (void)self; return 1; }
static TernaryDType head_input_dtype(const void *self) { (void)self; return TERNARY_DTYPE_TERNARY; }
static size_t head_accepts_u8(const void *self) { return ((const TernaryHead *)self)->dim; }

static void head_forward_adapter(const void *self, const TernaryTensor *x, float *out, int32_t *out_raw_opt,
                                 const TernaryCapabilities *caps) {
    (void)caps;
    if (x->dtype == TERNARY_DTYPE_UINT8) { ternary_head_forward_u8((TernaryHead *)(uintptr_t)self, x->data_u8, out, out_raw_opt); return; }
    ternary_head_forward((TernaryHead *)(uintptr_t)self, x->data_t, out, out_raw_opt);
}
static void head_backward_adapter(const void *self, const TernaryTensor *x, const float *grad_out,
                                  float *const *grad_w_out, float *grad_x_out, float *grad_scale_out,
                                  float *const *grad_bias_out, const int32_t *z_raw_opt) {
    (void)grad_bias_out; (void)z_raw_opt;   /* el Z crudo ya esta en h->zcur (el out_raw del stack), calculado en el forward inmediato */
    if (x->dtype == TERNARY_DTYPE_UINT8) {
        ternary_head_backward_u8((TernaryHead *)self, x->data_u8, grad_out, grad_w_out[0], grad_x_out, grad_scale_out);
        return;
    }
    ternary_head_backward((TernaryHead *)self, x->data_t, grad_out, grad_w_out[0], grad_x_out, grad_scale_out);
}
static void head_apply_row_gradients(void *self, size_t m, size_t row, const float *grad_row) {
    (void)m;
    ternary_dense_apply_row_gradients(((TernaryHead *)self)->w, row, grad_row);
}
/* Con head_beta2 > 0 (regla R2 vigente) el gradiente se normaliza por elemento (TernaryElemNorm) EN SITIO, sobre el acumulador del stack
 * (no se reserva otro buffer), y R2 lo recibe ya normalizado; con 0 queda la R2 de siempre (RMS por fila). */
static void head_apply_matrix_gradients(void *self, size_t m, const float *grad_w) {
    (void)m;
    TernaryHead *h = (TernaryHead *)self;
    const TernaryUpdateRule r = ternary_update_rule_get();
    if (r.head_beta2 > 0.0f) {
        if (!h->vn) h->vn = ternary_elemnorm_alloc(h->vocab, h->dim);
        if (h->vn) { ternary_elemnorm_apply(h->vn, (float *)(uintptr_t)grad_w, r.head_beta2); ternary_dense_apply_matrix_gradients_pre(h->w, grad_w); return; }
    }
    ternary_dense_apply_matrix_gradients(h->w, grad_w);
}
static void head_apply_bias_gradients(void *self, size_t m, const float *g) {
    (void)self; (void)m; (void)g;  /* sin bias */
}
static void head_apply_scale_gradient(void *self, size_t m, float grad_scale, float scale_lr) {
    (void)m;
    ternary_dense_apply_scale_gradient(((TernaryHead *)self)->w, grad_scale, scale_lr);
}
static void head_destroy(void *self) { ternary_head_free((TernaryHead *)self); }

static const TernaryLayerVTable TERNARY_HEAD_VTABLE = {
    "lm_head",
    head_input_len, head_output_len, head_num_matrices, head_grad_rows, head_grad_row_width,
    head_has_bias, head_propagates_grad_x, head_input_dtype,
    head_forward_adapter, head_backward_adapter,
    head_apply_row_gradients, head_apply_bias_gradients, head_apply_scale_gradient,
    head_destroy,
    NULL, /* backward_accumulate: no soportado */
    NULL, /* backward_block: no soportado */
    head_apply_matrix_gradients,
    NULL, /* forward_block: no soportado */
    NULL, /* state_bytes: sin estado compartible */
    NULL, /* bind_state */
    head_accepts_u8,
    NULL, /* output_row_len */
    head_export, head_import,
};

TernaryLayer ternary_layer_from_head(TernaryHead *layer) {
    TernaryLayer l; l.impl = layer; l.vt = &TERNARY_HEAD_VTABLE; return l;
}

static size_t shape_product(const size_t *shape, size_t ndims) {
    size_t p = 1;
    for (size_t i = 0; i < ndims; i++) p *= shape[i];
    return p;
}

/* --- Dense: [units, bias] --- */
static const TernaryParamSpec DENSE_PARAMS[] = {
    { "units",      TERNARY_PARAM_INT,  1, 0.0,  1, 1.0, "number of outputs", 0, 0.0 },
    { "bias",       TERNARY_PARAM_BOOL, 0, 1.0,  0, 0.0, "trainable bias", 0, 0.0 },
};

static int dense_build(const size_t *in_shape, size_t in_ndims, const double *p,
                       TernaryRng *rng, TernaryLayer *out, char *err, size_t err_cap) {
    size_t in_dim = shape_product(in_shape, in_ndims);
    size_t units = (size_t)p[0];
    int bias = p[1] != 0.0;
    TernaryDense *d = ternary_dense_alloc(in_dim, units, bias);
    if (!d) { set_err(err, err_cap, "Dense: out of memory"); return 0; }
    /* Orden de sorteo fijo (la inicializacion es reproducible bit a bit):
     * por fila, sus pesos y despues su bias. */
    for (size_t o = 0; o < units; o++) {
        for (size_t i = 0; i < in_dim; i++)
            ternary_dense_set_weight(d, o, i, ternary_rng_sample_faninit(rng, in_dim));
        if (bias) ternary_vec_set(d->bias, o, ternary_rng_sample_faninit(rng, in_dim));
    }
    *out = ternary_layer_from_dense(d);
    return 1;
}

static size_t dense_describe(const void *impl, double *params_out, size_t *out_shape) {
    const TernaryDense *d = (const TernaryDense *)impl;
    params_out[0] = (double)d->output_dim;
    params_out[1] = d->bias != NULL ? 1.0 : 0.0;
    out_shape[0] = d->output_dim;
    return 1;
}

/* --- Conv2D: [filters, kernel_h, kernel_w, stride, bias] --- */
static const TernaryParamSpec CONV2D_PARAMS[] = {
    { "filters",    TERNARY_PARAM_INT,  1, 0.0,  1, 1.0, "output channels", 0, 0.0 },
    { "kernel",     TERNARY_PARAM_PAIR, 1, 0.0,  1, 1.0, "kernel size (int or (h, w))", 0, 0.0 },
    { "stride",     TERNARY_PARAM_INT,  0, 1.0,  1, 1.0, "", 0, 0.0 },
    { "bias",       TERNARY_PARAM_BOOL, 0, 1.0,  0, 0.0, "trainable bias", 0, 0.0 },
};

static int conv2d_build(const size_t *in_shape, size_t in_ndims, const double *p,
                        TernaryRng *rng, TernaryLayer *out, char *err, size_t err_cap) {
    if (in_ndims != 3) {
        set_err(err, err_cap, "Conv2D needs a (height, width, channels) input, got %zu dimension(s)",
                in_ndims);
        return 0;
    }
    size_t H = in_shape[0], W = in_shape[1], C = in_shape[2];
    size_t filters = (size_t)p[0], kh = (size_t)p[1], kw = (size_t)p[2], stride = (size_t)p[3];
    int bias = p[4] != 0.0;
    if (kh > H || kw > W) {
        set_err(err, err_cap, "Conv2D: kernel (%zu, %zu) is larger than the input (%zu, %zu)",
                kh, kw, H, W);
        return 0;
    }
    TernaryConv2D *c = ternary_conv2d_alloc(H, W, C, kh, kw, filters, stride, bias);
    if (!c) { set_err(err, err_cap, "Conv2D: out of memory"); return 0; }
    for (size_t o = 0; o < filters; o++) {
        for (size_t k = 0; k < c->kernel_size; k++)
            ternary_conv2d_set_weight(c, o, k, ternary_rng_sample_faninit(rng, c->kernel_size));
        if (bias) ternary_vec_set(c->bias, o, ternary_rng_sample_faninit(rng, c->kernel_size));
    }
    *out = ternary_layer_from_conv2d(c);
    return 1;
}

static size_t conv2d_describe(const void *impl, double *params_out, size_t *out_shape) {
    const TernaryConv2D *c = (const TernaryConv2D *)impl;
    params_out[0] = (double)c->out_channels;
    params_out[1] = (double)c->kernel_h;
    params_out[2] = (double)c->kernel_w;
    params_out[3] = (double)c->stride;
    params_out[4] = c->bias != NULL ? 1.0 : 0.0;
    out_shape[0] = c->out_height; out_shape[1] = c->out_width; out_shape[2] = c->out_channels;
    return 3;
}

/* --- Embedding: [vocab_size, embed_dim]; seq_len sale de la forma de entrada --- */
static const TernaryParamSpec EMBEDDING_PARAMS[] = {
    { "vocab_size", TERNARY_PARAM_INT, 1, 0.0,  1, 1.0, "number of distinct token ids", 0, 0.0 },
    { "embed_dim",  TERNARY_PARAM_INT, 1, 0.0,  1, 1.0, "size of each embedding vector", 0, 0.0 },
};

static int embedding_build(const size_t *in_shape, size_t in_ndims, const double *p,
                           TernaryRng *rng, TernaryLayer *out, char *err, size_t err_cap) {
    if (in_ndims != 1) {
        set_err(err, err_cap, "Embedding needs a (sequence_length,) input of token ids, got %zu dimension(s)",
                in_ndims);
        return 0;
    }
    size_t vocab = (size_t)p[0], dim = (size_t)p[1];
    TernaryEmbedding *e = ternary_embedding_alloc(vocab, dim, in_shape[0]);
    if (!e) { set_err(err, err_cap, "Embedding: out of memory"); return 0; }
    for (size_t v = 0; v < vocab; v++)
        for (size_t d = 0; d < dim; d++)
            ternary_embedding_set_weight(e, v, d, ternary_rng_sample_faninit(rng, dim));
    *out = ternary_layer_from_embedding(e);
    return 1;
}

static size_t embedding_describe(const void *impl, double *params_out, size_t *out_shape) {
    const TernaryEmbedding *e = (const TernaryEmbedding *)impl;
    params_out[0] = (double)e->vocab_size;
    params_out[1] = (double)e->embed_dim;
    out_shape[0] = e->seq_len; out_shape[1] = e->embed_dim;
    return 2;
}

/* --- GQA: [n_heads, n_kv_heads, rope, rope_base, score_shift, out_scale, window, sinks, smear];
 *     seq_len y embed_dim salen de la forma de entrada (seq_len, embed_dim) --- */
static const TernaryParamSpec GQA_PARAMS[] = {
    { "n_heads",     TERNARY_PARAM_INT,   1, 0.0,  1, 1.0, "number of query heads (must divide embed_dim)", 0, 0.0 },
    { "n_kv_heads",  TERNARY_PARAM_INT,   0, 0.0,  1, 0.0, "number of key/value heads (0 = same as n_heads; must divide n_heads)", 0, 0.0 },
    { "rope",        TERNARY_PARAM_INT,   0, 1.0,  1, 0.0, "position encoding: 0 none, 1 RoPE with integer scores (recommended), 2 RoPE with binarized Q/K", 0, 0.0 },
    { "rope_base",   TERNARY_PARAM_FLOAT, 0, 10.0, 1, 1.0, "RoPE frequency base (> 1); lower = faster rotation", 0, 0.0 },
    { "score_shift", TERNARY_PARAM_INT,   0, 2.0,  1, 0.0, "RoPE mode 1: right shift of the integer q.k score (larger = flatter attention)", 0, 0.0 },
    { "out_scale",   TERNARY_PARAM_FLOAT, 0, 1.0,  1, 0.02, "initial float scale of the output projection Wo (< 1 starts a residual branch small next to the stream; learned afterwards)", 0, 0.0 },
    { "window",      TERNARY_PARAM_INT,   0, 0.0,  1, 0.0, "sliding-window attention: each position attends to the last `window` positions (0 = full causal attention; needs rope=1). Time O(T*window), layer memory O(window)", 0, 0.0 },
    { "sinks",       TERNARY_PARAM_INT,   0, 0.0,  1, 0.0, "with a window: the first `sinks` positions (<= 48) stay visible to every position (attention sinks)", 0, 0.0 },
    { "smear",       TERNARY_PARAM_BOOL,  0, 0.0,  0, 0.0, "smeared keys: K'_j = (K_j + K_{j-1}) >> 1 before RoPE, so a single layer can match a token against the one that precedes each key (the induction-head pattern [A][B] .. [A] -> [B]); needs rope=1 and full attention (window=0)", 0, 0.0 },
};

static int gqa_build(const size_t *in_shape, size_t in_ndims, const double *p,
                     TernaryRng *rng, TernaryLayer *out, char *err, size_t err_cap) {
    if (in_ndims != 2) {
        set_err(err, err_cap, "GQA needs a (sequence_length, embed_dim) input, got %zu dimension(s)", in_ndims);
        return 0;
    }
    size_t seq = in_shape[0], E = in_shape[1];
    size_t nh = (size_t)p[0], nkv = (p[1] == 0.0) ? nh : (size_t)p[1];
    int rope = (int)p[2], shift = (int)p[4];
    float base = (float)p[3];
    if (E % nh != 0) { set_err(err, err_cap, "GQA: embed_dim (%zu) must be divisible by n_heads (%zu)", E, nh); return 0; }
    if (nh % nkv != 0) { set_err(err, err_cap, "GQA: n_heads (%zu) must be divisible by n_kv_heads (%zu)", nh, nkv); return 0; }
    if (rope > 2) { set_err(err, err_cap, "GQA: rope must be 0, 1 or 2, got %d", rope); return 0; }
    if (rope != 0 && (E / nh) % 2 != 0) { set_err(err, err_cap, "GQA: RoPE needs an even head_dim, got %zu", E / nh); return 0; }
    if (rope != 0 && !(base > 1.0f)) { set_err(err, err_cap, "GQA: rope_base must be > 1, got %g", (double)base); return 0; }
    if (shift > 16) { set_err(err, err_cap, "GQA: score_shift must be <= 16, got %d", shift); return 0; }
    const size_t window = (size_t)p[6], sinks = (size_t)p[7];
    if (window == 0 && sinks != 0) { set_err(err, err_cap, "GQA: sinks needs window > 0"); return 0; }
    if (window != 0 && rope != 1) { set_err(err, err_cap, "GQA: window needs rope=1 (RoPE with integer scores)"); return 0; }
    if (sinks > 48 || sinks > seq) { set_err(err, err_cap, "GQA: sinks must be <= min(48, sequence length), got %zu", sinks); return 0; }
    const int smear = p[8] != 0.0;
    if (smear && rope != 1) { set_err(err, err_cap, "GQA: smear needs rope=1 (RoPE with integer scores)"); return 0; }
    if (smear && window != 0) { set_err(err, err_cap, "GQA: smear needs full attention (window=0)"); return 0; }
    TernaryGQA *g = ternary_gqa_alloc_window(E, seq, nh, nkv, window, sinks);
    if (!g) { set_err(err, err_cap, "GQA: invalid shape or out of memory"); return 0; }
    if (rope == 1 && ternary_gqa_enable_rope_int(g, base, shift) != 0) { ternary_gqa_free(g); set_err(err, err_cap, "GQA: could not enable RoPE"); return 0; }
    if (smear && ternary_gqa_set_smear(g, 1) != 0) { ternary_gqa_free(g); set_err(err, err_cap, "GQA: could not enable smear"); return 0; }
    if (rope == 2 && ternary_gqa_enable_rope(g, base) != 0) { ternary_gqa_free(g); set_err(err, err_cap, "GQA: could not enable RoPE"); return 0; }
    for (size_t m = 0; m < 4; m++) {
        TernaryDense *d = gqa_matrix(g, m);
        for (size_t o = 0; o < d->output_dim; o++)
            for (size_t i = 0; i < d->input_dim; i++)
                ternary_dense_set_weight(d, o, i, ternary_rng_sample_faninit(rng, d->input_dim));
    }
    gqa_matrix(g, 3)->scale = (float)p[5];   /* Wo: escala inicial de la salida (se aprende despues) */
    *out = ternary_layer_from_gqa(g);
    return 1;
}

static size_t gqa_describe(const void *impl, double *params_out, size_t *out_shape) {
    const TernaryGQA *g = (const TernaryGQA *)impl;
    params_out[0] = (double)g->n_heads;
    params_out[1] = (double)g->n_kv_heads;
    params_out[2] = (double)gqa_rope_mode(g);
    params_out[3] = g->rope ? (double)g->rope_base : 10.0;
    params_out[4] = (double)g->qk_shift;
    params_out[5] = 1.0;   /* out_scale solo vale al construir: la escala aprendida ya se guarda con los pesos */
    params_out[6] = (double)g->window;
    params_out[7] = (double)g->sinks;
    params_out[8] = g->smear ? 1.0 : 0.0;
    out_shape[0] = g->seq_len; out_shape[1] = g->embed_dim;
    return 2;
}

/* --- RMSNorm: [eps]; la entrada (..., dim) se normaliza por filas sobre la ULTIMA dimension --- */
static const TernaryParamSpec RMSNORM_PARAMS[] = {
    { "eps",        TERNARY_PARAM_FLOAT, 0, 1e-5, 1, 0.0, "added to the mean square before the square root (> 0)", 0, 0.0 },
};

static int rmsnorm_build(const size_t *in_shape, size_t in_ndims, const double *p,
                         TernaryRng *rng, TernaryLayer *out, char *err, size_t err_cap) {
    (void)rng;   /* la ganancia arranca en +1 en todas las dimensiones: RMSNorm puro, sin sorteo */
    if (in_ndims < 1) { set_err(err, err_cap, "RMSNorm needs an input with at least one dimension"); return 0; }
    float eps = (float)p[0];
    if (!(eps > 0.0f)) { set_err(err, err_cap, "RMSNorm: eps must be > 0, got %g", p[0]); return 0; }
    size_t dim = in_shape[in_ndims - 1];
    size_t rows = shape_product(in_shape, in_ndims) / dim;
    TernaryRMSNorm *n = ternary_rmsnorm_alloc(rows, dim, eps);
    if (!n) { set_err(err, err_cap, "RMSNorm: invalid shape or out of memory"); return 0; }
    *out = ternary_layer_from_rmsnorm(n);
    return 1;
}

static size_t rmsnorm_describe(const void *impl, double *params_out, size_t *out_shape) {
    const TernaryRMSNorm *n = (const TernaryRMSNorm *)impl;
    params_out[0] = (double)n->eps;
    if (n->rows == 1) { out_shape[0] = n->dim; return 1; }
    out_shape[0] = n->rows; out_shape[1] = n->dim;
    return 2;
}


/* --- SwiGLU: [hidden, gate_mode, band, out_scale]; la entrada (..., dim) se procesa por filas sobre la ULTIMA dimension --- */
static const TernaryParamSpec SWIGLU_PARAMS[] = {
    { "hidden",     TERNARY_PARAM_INT,   1, 0.0,  1, 1.0, "width of the inner (gate/up) projections", 0, 0.0 },
    { "gate_mode",  TERNARY_PARAM_INT,   0, 0.0,  1, 0.0, "gate: 0 silu through an integer table (float product, default), 1 ternary sign(gate)*up (all integer), 2 ReLU2 without a gate: relu(up)^2 (BitNet style, only 2 matrices: up and down)", 1, 2.0 },
    { "band",       TERNARY_PARAM_FLOAT, 0, 0.5,  1, 0.0, "zero band of the inner quantization, relative to the row's mean |h| (0 = none)", 0, 0.0 },
    { "out_scale",  TERNARY_PARAM_FLOAT, 0, 1.0,  1, 0.02, "initial float scale of the down projection (< 1 starts a residual branch small next to the stream; learned afterwards)", 0, 0.0 },
};

static int swiglu_build(const size_t *in_shape, size_t in_ndims, const double *p,
                        TernaryRng *rng, TernaryLayer *out, char *err, size_t err_cap) {
    if (in_ndims < 1) { set_err(err, err_cap, "SwiGLU needs an input with at least one dimension"); return 0; }
    size_t hidden = (size_t)p[0];
    int gate_mode = (int)p[1];
    float band = (float)p[2];
    if (gate_mode > TERNARY_SWIGLU_GATE_RELU2) { set_err(err, err_cap, "SwiGLU: gate_mode must be 0, 1 or 2, got %d", gate_mode); return 0; }
    size_t dim = in_shape[in_ndims - 1];
    size_t rows = shape_product(in_shape, in_ndims) / dim;
    TernarySwiGLU *s = ternary_swiglu_alloc(rows, dim, hidden, gate_mode, band);
    if (!s) {
        set_err(err, err_cap, "SwiGLU: invalid shape or out of memory (dim and hidden must be <= %d)", TERNARY_SWIGLU_MAX_DIM);
        return 0;
    }
    const size_t nm = ternary_swiglu_num_matrices(s);
    for (size_t m = 0; m < nm; m++) {
        TernaryDense *d = swiglu_matrix(s, m);
        for (size_t o = 0; o < d->output_dim; o++)
            for (size_t i = 0; i < d->input_dim; i++)
                ternary_dense_set_weight(d, o, i, ternary_rng_sample_faninit(rng, d->input_dim));
    }
    swiglu_matrix(s, nm - 1)->scale = (float)p[3];   /* down: escala inicial de la salida (se aprende despues) */
    *out = ternary_layer_from_swiglu(s);
    return 1;
}

static size_t swiglu_describe(const void *impl, double *params_out, size_t *out_shape) {
    const TernarySwiGLU *s = (const TernarySwiGLU *)impl;
    params_out[0] = (double)s->hidden;
    params_out[1] = (double)s->gate_mode;
    params_out[2] = (double)s->band;
    params_out[3] = 1.0;   /* out_scale solo vale al construir: la escala aprendida ya se guarda con los pesos */
    if (s->rows == 1) { out_shape[0] = s->dim; return 1; }
    out_shape[0] = s->rows; out_shape[1] = s->dim;
    return 2;
}


/* --- LMHead: [vocab]; la entrada (..., dim) se proyecta por filas sobre la ULTIMA dimension --- */
static const TernaryParamSpec HEAD_PARAMS[] = {
    { "vocab",      TERNARY_PARAM_INT, 1, 0.0,  1, 1.0, "number of output logits per position (vocabulary size)", 0, 0.0 },
};

static int head_build(const size_t *in_shape, size_t in_ndims, const double *p,
                      TernaryRng *rng, TernaryLayer *out, char *err, size_t err_cap) {
    if (in_ndims < 1) { set_err(err, err_cap, "LMHead needs an input with at least one dimension"); return 0; }
    size_t vocab = (size_t)p[0];
    size_t dim = in_shape[in_ndims - 1];
    size_t rows = shape_product(in_shape, in_ndims) / dim;
    TernaryHead *h = ternary_head_alloc(rows, dim, vocab);
    if (!h) {
        set_err(err, err_cap, "LMHead: invalid shape or out of memory (dim must be <= %d, vocab <= %d)",
                TERNARY_HEAD_MAX_DIM, TERNARY_HEAD_MAX_VOCAB);
        return 0;
    }
    for (size_t o = 0; o < vocab; o++)
        for (size_t i = 0; i < dim; i++)
            ternary_dense_set_weight(h->w, o, i, ternary_rng_sample_faninit(rng, dim));
    *out = ternary_layer_from_head(h);
    return 1;
}

static size_t head_describe(const void *impl, double *params_out, size_t *out_shape) {
    const TernaryHead *h = (const TernaryHead *)impl;
    params_out[0] = (double)h->vocab;
    if (h->rows == 1) { out_shape[0] = h->vocab; return 1; }
    out_shape[0] = h->rows; out_shape[1] = h->vocab;
    return 2;
}

/* --- La tabla: UNA entrada por tipo de capa --- */
static const TernaryLayerType TERNARY_LAYER_TYPES[] = {
    { "dense", "Dense", "Fully connected ternary layer.",
      DENSE_PARAMS, sizeof(DENSE_PARAMS) / sizeof(DENSE_PARAMS[0]),
      &TERNARY_DENSE_VTABLE, dense_build, dense_describe },
    { "conv2d", "Conv2D",
      "2D ternary convolution over (height, width, channels) input. Must be the first layer.",
      CONV2D_PARAMS, sizeof(CONV2D_PARAMS) / sizeof(CONV2D_PARAMS[0]),
      &TERNARY_CONV2D_VTABLE, conv2d_build, conv2d_describe },
    { "embedding", "Embedding",
      "Lookup table of ternary vectors indexed by token id. Must be the first layer.",
      EMBEDDING_PARAMS, sizeof(EMBEDDING_PARAMS) / sizeof(EMBEDDING_PARAMS[0]),
      &TERNARY_EMBEDDING_VTABLE, embedding_build, embedding_describe },
    { "gqa", "GQA",
      "Causal multi-head grouped-query attention over (sequence_length, embed_dim) input, with optional RoPE.",
      GQA_PARAMS, sizeof(GQA_PARAMS) / sizeof(GQA_PARAMS[0]),
      &TERNARY_GQA_VTABLE, gqa_build, gqa_describe },
    { "rmsnorm", "RMSNorm",
      "Row-wise RMS normalization over the last dimension, with a ternary per-dimension gain and a learned float scale. Takes the raw float of the previous layer.",
      RMSNORM_PARAMS, sizeof(RMSNORM_PARAMS) / sizeof(RMSNORM_PARAMS[0]),
      &TERNARY_RMSNORM_VTABLE, rmsnorm_build, rmsnorm_describe },
    { "swiglu", "SwiGLU",
      "Feed-forward block over the last dimension: gate and up projections, a swappable gate (silu table or ternary sign), a zero-band ternary quantization and a down projection. No residual connection.",
      SWIGLU_PARAMS, sizeof(SWIGLU_PARAMS) / sizeof(SWIGLU_PARAMS[0]),
      &TERNARY_SWIGLU_VTABLE, swiglu_build, swiglu_describe },
    { "lm_head", "LMHead",
      "Language-model output head: one shared ternary (vocab x dim) projection applied to every position over the last dimension, with a learned float scale. Outputs raw logits (pair it with a softmax cross-entropy loss).",
      HEAD_PARAMS, sizeof(HEAD_PARAMS) / sizeof(HEAD_PARAMS[0]),
      &TERNARY_HEAD_VTABLE, head_build, head_describe },
};

size_t ternary_layer_type_count(void) {
    return sizeof(TERNARY_LAYER_TYPES) / sizeof(TERNARY_LAYER_TYPES[0]);
}

const TernaryLayerType *ternary_layer_type_at(size_t i) {
    return (i < ternary_layer_type_count()) ? &TERNARY_LAYER_TYPES[i] : NULL;
}

const TernaryLayerType *ternary_layer_type_by_name(const char *type_name) {
    if (!type_name) return NULL;
    for (size_t i = 0; i < ternary_layer_type_count(); i++)
        if (strcmp(TERNARY_LAYER_TYPES[i].type_name, type_name) == 0) return &TERNARY_LAYER_TYPES[i];
    return NULL;
}

size_t ternary_layer_type_num_slots(const TernaryLayerType *type) {
    size_t n = 0;
    for (size_t i = 0; i < type->num_params; i++)
        n += (type->params[i].kind == TERNARY_PARAM_PAIR) ? 2 : 1;
    return n;
}

int ternary_layer_type_build(const TernaryLayerType *type, const size_t *in_shape, size_t in_ndims,
                             const double *params, size_t num_slots, TernaryRng *rng,
                             TernaryLayer *out, char *err, size_t err_cap) {
    if (!type || !in_shape || in_ndims < 1 || in_ndims > TERNARY_MAX_SHAPE_DIMS || !params || !rng || !out) {
        set_err(err, err_cap, "invalid arguments");
        return 0;
    }
    const size_t full_slots = ternary_layer_type_num_slots(type);
    double filled[TERNARY_MAX_LAYER_SLOTS];
    /* Compatibilidad: GQA y SwiGLU sumaron un parametro NUEVO al final (out_scale). Quien las arma con la lista de la
     * version anterior (GQA 5 valores sin v_ternary, SwiGLU 3) recibe el valor por defecto del parametro nuevo en vez de un error.
     * Cualquier otra cantidad sigue siendo error. */
    size_t legacy_slots = 0;
    if (strcmp(type->type_name, "gqa") == 0) legacy_slots = 5;
    else if (strcmp(type->type_name, "swiglu") == 0) legacy_slots = 3;
    if (legacy_slots && params && num_slots >= legacy_slots && num_slots < full_slots &&
        (num_slots == legacy_slots || (legacy_slots == 5 && num_slots == 6))) {
        /* los parametros que faltan (out_scale; en GQA tambien window y sinks) toman su valor por defecto */
        for (size_t k = 0; k < num_slots; k++) filled[k] = params[k];
        size_t k = num_slots, pi = 0, sl = 0;
        for (; pi < type->num_params; pi++) {
            const size_t n = (type->params[pi].kind == TERNARY_PARAM_PAIR) ? 2 : 1;
            for (size_t j = 0; j < n; j++, sl++) if (sl >= k) filled[sl] = type->params[pi].default_value;
        }
        params = filled; num_slots = full_slots;
    }
    if (num_slots != full_slots) {
        set_err(err, err_cap, "%s expects %zu parameter value(s), got %zu", type->class_name,
                full_slots, num_slots);
        return 0;
    }
    size_t slot = 0;
    for (size_t i = 0; i < type->num_params; i++) {
        const TernaryParamSpec *ps = &type->params[i];
        size_t n = (ps->kind == TERNARY_PARAM_PAIR) ? 2 : 1;
        for (size_t j = 0; j < n; j++, slot++) {
            double v = params[slot];
            if (!isfinite(v)) {
                set_err(err, err_cap, "%s: %s must be a finite number", type->class_name, ps->name);
                return 0;
            }
            if (ps->kind != TERNARY_PARAM_FLOAT && v != floor(v)) {
                set_err(err, err_cap, "%s: %s must be an integer, got %g", type->class_name, ps->name, v);
                return 0;
            }
            if (ps->kind == TERNARY_PARAM_BOOL && v != 0.0 && v != 1.0) {
                set_err(err, err_cap, "%s: %s must be 0 or 1, got %g", type->class_name, ps->name, v);
                return 0;
            }
            if (ps->has_min && v < ps->min_value) {
                set_err(err, err_cap, "%s: %s must be >= %g, got %g", type->class_name, ps->name,
                        ps->min_value, v);
                return 0;
            }
            if (ps->has_max && v > ps->max_value) {
                set_err(err, err_cap, "%s: %s must be <= %g, got %g", type->class_name, ps->name,
                        ps->max_value, v);
                return 0;
            }
        }
    }
    return type->build(in_shape, in_ndims, params, rng, out, err, err_cap);
}

/* ===================================================================
 * Exportacion / importacion al formato model.ternary (ver ternary_tfile.h)
 *
 * Nombres de tensor: "<prefix>.<hoja>", con prefix = "layers.<i>" para la capa i de la red y
 * "<prefix del contenedor>.<j>" para la sub-capa j de una residual. Hojas:
 *   dense   : weight [out,in] TRIT5, scale [1] F32, bias [out] TRIT5 (si tiene)
 *   conv2d  : kernel [out_ch, kernel_size], scale, bias
 *   embedding: weight [vocab, dim], scale
 *   gqa     : q_proj / k_proj / v_proj / o_proj  (cada una weight + scale)
 *   rmsnorm : gain  (weight [1,dim] + scale)
 *   swiglu  : gate_proj (si hay compuerta) / up_proj / down_proj
 *   lm_head : weight [vocab, dim], scale
 * Con estado de entrenamiento (checkpoint), en training_state.ternary: <hoja de pesos>_counters I8, bias_counters I8 y la regla R2 en "r2" (ver ternaryai).
 * =================================================================== */
#define NAMEBUF 256

static void ierr(const TernaryImportCtx *cx, const char *fmt, ...) {
    if (!cx || !cx->err || cx->err_cap == 0 || cx->err[0]) return;      /* se conserva el primer error */
    va_list ap; va_start(ap, fmt); vsnprintf(cx->err, cx->err_cap, fmt, ap); va_end(ap);
}

static int tname(char *out, const char *prefix, const char *leaf) {
    int n = snprintf(out, NAMEBUF, "%s.%s", prefix, leaf);
    return n > 0 && n < NAMEBUF;
}

/* Una matriz de pesos (rows x cols) con su escala y bias opcional. wleaf: "weight"/"kernel". */
static int wmat_put(const TernaryExportCtx *cx, const char *prefix, const char *wleaf, const TernaryVec *W,
                    size_t rows, size_t cols, const float *scale, const TernaryVec *bias,
                    const int8_t *counter_w_t, const int8_t *counter_b) {
    char nm[NAMEBUF]; const size_t sw[2] = { rows, cols }, one = 1, sb = rows;
    if (!tname(nm, prefix, wleaf) || !ternary_tw_add_trits(cx->model, nm, sw, 2, W)) return 0;
    if (!tname(nm, prefix, "scale") || !ternary_tw_add(cx->model, nm, TTD_F32, &one, 1, scale)) return 0;
    if (bias) {
        if (!tname(nm, prefix, "bias") || !ternary_tw_add_trits(cx->model, nm, &sb, 1, bias)) return 0;
    }
    if (cx->state) {
        char leaf[64];
        snprintf(leaf, sizeof leaf, "%s_counters", wleaf);
        if (!tname(nm, prefix, leaf) || !ternary_tw_add(cx->state, nm, TTD_I8, sw, 2, counter_w_t)) return 0;
        if (bias && (!tname(nm, prefix, "bias_counters") || !ternary_tw_add(cx->state, nm, TTD_I8, &sb, 1, counter_b))) return 0;
    }
    return 1;
}

static int wmat_get(const TernaryImportCtx *cx, const char *prefix, const char *wleaf, TernaryVec *W,
                    size_t rows, size_t cols, float *scale, TernaryVec *bias, int8_t *counter_w_t, int8_t *counter_b) {
    char nm[NAMEBUF]; const size_t sw[2] = { rows, cols }, one = 1, sb = rows;
    const TernaryTInfo *t;
    if (!tname(nm, prefix, wleaf)) { ierr(cx, "nombre de tensor demasiado largo (%s)", prefix); return 0; }
    if (!(t = ternary_tr_expect(cx->model, nm, TTD_TRIT5, sw, 2, cx->err, cx->err_cap))) return 0;
    if (!ternary_tinfo_read_trits(t, W)) { ierr(cx, "tensor '%s': byte fuera de 0..242 (archivo corrupto)", nm); return 0; }
    if (!tname(nm, prefix, "scale") || !(t = ternary_tr_expect(cx->model, nm, TTD_F32, &one, 1, cx->err, cx->err_cap))) return 0;
    *scale = ternary_tinfo_f32(t, 0);
    if (*scale != *scale || *scale > 3.4028235e38f || *scale < -3.4028235e38f) { ierr(cx, "tensor '%s': escala no finita", nm); return 0; }
    if (bias) {
        if (!tname(nm, prefix, "bias") || !(t = ternary_tr_expect(cx->model, nm, TTD_TRIT5, &sb, 1, cx->err, cx->err_cap))) return 0;
        if (!ternary_tinfo_read_trits(t, bias)) { ierr(cx, "tensor '%s': byte fuera de 0..242 (archivo corrupto)", nm); return 0; }
    }
    if (cx->state) {
        char leaf[64];
        snprintf(leaf, sizeof leaf, "%s_counters", wleaf);
        if (!tname(nm, prefix, leaf) || !(t = ternary_tr_expect(cx->state, nm, TTD_I8, sw, 2, cx->err, cx->err_cap))) return 0;
        memcpy(counter_w_t, t->data, t->nbytes);
        if (bias) {
            if (!tname(nm, prefix, "bias_counters") || !(t = ternary_tr_expect(cx->state, nm, TTD_I8, &sb, 1, cx->err, cx->err_cap))) return 0;
            memcpy(counter_b, t->data, t->nbytes);
        }
    }
    return 1;
}

static int dense_put(const TernaryExportCtx *cx, const char *prefix, const TernaryDense *d) {
    return wmat_put(cx, prefix, "weight", d->W, d->output_dim, d->input_dim, &d->scale, d->bias, d->counter_W, d->counter_b);
}
static int dense_get(const TernaryImportCtx *cx, const char *prefix, TernaryDense *d) {
    return wmat_get(cx, prefix, "weight", d->W, d->output_dim, d->input_dim, &d->scale, d->bias, d->counter_W, d->counter_b);
}

/* Subdense sin bias de un compuesto: "<prefix>.<leaf>" (y sus hojas weight/scale). */
static int sub_put(const TernaryExportCtx *cx, const char *prefix, const char *leaf, const TernaryDense *d) {
    char p[NAMEBUF];
    return tname(p, prefix, leaf) && dense_put(cx, p, d);
}
static int sub_get(const TernaryImportCtx *cx, const char *prefix, const char *leaf, TernaryDense *d) {
    char p[NAMEBUF];
    if (!tname(p, prefix, leaf)) { ierr(cx, "nombre de tensor demasiado largo (%s)", prefix); return 0; }
    return dense_get(cx, p, d);
}

/* Lectura de enteros del config con rango. */
static int cfg_int(const TernaryImportCtx *cx, const TernaryJNode *cfg, const char *key, int64_t lo, int64_t hi, int64_t *out) {
    int64_t v;
    if (!ternary_jdom_get_int(cfg, key, &v)) { ierr(cx, "config: falta '%s' (o no es un entero)", key); return 0; }
    if (v < lo || v > hi) { ierr(cx, "config: '%s' = %lld fuera de rango [%lld, %lld]", key, (long long)v, (long long)lo, (long long)hi); return 0; }
    *out = v;
    return 1;
}
static int cfg_float(const TernaryImportCtx *cx, const TernaryJNode *cfg, const char *key, float *out) {
    float v;
    if (!ternary_jdom_get_float(cfg, key, &v)) { ierr(cx, "config: falta '%s' (o no es un numero)", key); return 0; }
    *out = v;
    return 1;
}
#define MAXDIM ((int64_t)1 << 31)

/* ---- dense ---- */
static int dense_export(const void *self, const TernaryExportCtx *cx, const char *prefix, TernaryJBuf *cfg) {
    const TernaryDense *d = (const TernaryDense *)self;
    ternary_jbuf_kint(cfg, "input_dim", (int64_t)d->input_dim);
    ternary_jbuf_kint(cfg, "output_dim", (int64_t)d->output_dim);
    ternary_jbuf_kbool(cfg, "has_bias", d->bias != NULL);
    return dense_put(cx, prefix, d);
}
static void *dense_import(const TernaryImportCtx *cx, const char *prefix, const TernaryJNode *cfg) {
    int64_t in, out; int bias = 0;
    if (!cfg_int(cx, cfg, "input_dim", 1, MAXDIM, &in) || !cfg_int(cx, cfg, "output_dim", 1, MAXDIM, &out)) return NULL;
    if (!ternary_jdom_get_bool(cfg, "has_bias", &bias)) { ierr(cx, "config: falta 'has_bias'"); return NULL; }
    TernaryDense *d = ternary_dense_alloc((size_t)in, (size_t)out, bias);
    if (!d) { ierr(cx, "dense: dimensiones invalidas o sin memoria"); return NULL; }
    if (!dense_get(cx, prefix, d)) { ternary_dense_free(d); return NULL; }
    return d;
}

/* ---- conv2d ---- */
static int conv2d_export(const void *self, const TernaryExportCtx *cx, const char *prefix, TernaryJBuf *cfg) {
    const TernaryConv2D *c = (const TernaryConv2D *)self;
    ternary_jbuf_kint(cfg, "in_height", (int64_t)c->in_height);
    ternary_jbuf_kint(cfg, "in_width", (int64_t)c->in_width);
    ternary_jbuf_kint(cfg, "in_channels", (int64_t)c->in_channels);
    ternary_jbuf_kint(cfg, "kernel_h", (int64_t)c->kernel_h);
    ternary_jbuf_kint(cfg, "kernel_w", (int64_t)c->kernel_w);
    ternary_jbuf_kint(cfg, "out_channels", (int64_t)c->out_channels);
    ternary_jbuf_kint(cfg, "stride", (int64_t)c->stride);
    ternary_jbuf_kbool(cfg, "has_bias", c->bias != NULL);
    return wmat_put(cx, prefix, "kernel", c->kernels, c->out_channels, c->kernel_size, &c->scale, c->bias,
                    c->counter_kernels, c->counter_bias);
}
static void *conv2d_import(const TernaryImportCtx *cx, const char *prefix, const TernaryJNode *cfg) {
    static const char *const K[7] = { "in_height", "in_width", "in_channels", "kernel_h", "kernel_w", "out_channels", "stride" };
    int64_t v[7]; int bias = 0;
    for (int i = 0; i < 7; i++) if (!cfg_int(cx, cfg, K[i], 1, MAXDIM, &v[i])) return NULL;
    if (!ternary_jdom_get_bool(cfg, "has_bias", &bias)) { ierr(cx, "config: falta 'has_bias'"); return NULL; }
    TernaryConv2D *c = ternary_conv2d_alloc((size_t)v[0], (size_t)v[1], (size_t)v[2], (size_t)v[3], (size_t)v[4],
                                            (size_t)v[5], (size_t)v[6], bias);
    if (!c) { ierr(cx, "conv2d: dimensiones invalidas o sin memoria"); return NULL; }
    if (!wmat_get(cx, prefix, "kernel", c->kernels, c->out_channels, c->kernel_size, &c->scale, c->bias,
                  c->counter_kernels, c->counter_bias)) { ternary_conv2d_free(c); return NULL; }
    return c;
}

/* ---- embedding ---- */
static int emb_export(const void *self, const TernaryExportCtx *cx, const char *prefix, TernaryJBuf *cfg) {
    const TernaryEmbedding *e = (const TernaryEmbedding *)self;
    ternary_jbuf_kint(cfg, "vocab_size", (int64_t)e->vocab_size);
    ternary_jbuf_kint(cfg, "embed_dim", (int64_t)e->embed_dim);
    ternary_jbuf_kint(cfg, "seq_len", (int64_t)e->seq_len);
    return wmat_put(cx, prefix, "weight", e->W, e->vocab_size, e->embed_dim, &e->scale, NULL, e->counter_W, NULL);
}
static void *emb_import(const TernaryImportCtx *cx, const char *prefix, const TernaryJNode *cfg) {
    int64_t vocab, dim, seq;
    if (!cfg_int(cx, cfg, "vocab_size", 1, MAXDIM, &vocab) || !cfg_int(cx, cfg, "embed_dim", 1, MAXDIM, &dim) ||
        !cfg_int(cx, cfg, "seq_len", 1, MAXDIM, &seq)) return NULL;
    TernaryEmbedding *e = ternary_embedding_alloc((size_t)vocab, (size_t)dim, (size_t)seq);
    if (!e) { ierr(cx, "embedding: dimensiones invalidas o sin memoria"); return NULL; }
    if (!wmat_get(cx, prefix, "weight", e->W, e->vocab_size, e->embed_dim, &e->scale, NULL, e->counter_W, NULL)) {
        ternary_embedding_free(e); return NULL;
    }
    return e;
}

/* ---- gqa ---- */
static int gqa_export(const void *self, const TernaryExportCtx *cx, const char *prefix, TernaryJBuf *cfg) {
    const TernaryGQA *g = (const TernaryGQA *)self;
    ternary_jbuf_kint(cfg, "embed_dim", (int64_t)g->embed_dim);
    ternary_jbuf_kint(cfg, "seq_len", (int64_t)g->seq_len);
    ternary_jbuf_kint(cfg, "n_heads", (int64_t)g->n_heads);
    ternary_jbuf_kint(cfg, "n_kv_heads", (int64_t)g->n_kv_heads);
    ternary_jbuf_kint(cfg, "rope_mode", gqa_rope_mode(g));    /* 0 sin RoPE, 1 RoPE + puntaje entero, 2 RoPE con Q/K binarizados */
    ternary_jbuf_kint(cfg, "qk_shift", g->qk_shift);
    ternary_jbuf_kfloat(cfg, "rope_base", g->rope ? g->rope_base : 10.0f);
    ternary_jbuf_kint(cfg, "window", (int64_t)g->window);
    ternary_jbuf_kint(cfg, "sinks", (int64_t)g->sinks);
    if (g->smear) ternary_jbuf_kbool(cfg, "smear", 1);          /* solo se escribe si esta activo: los modelos viejos no cambian */
    static const char *const LEAF[4] = { "q_proj", "k_proj", "v_proj", "o_proj" };
    for (size_t m = 0; m < 4; m++) if (!sub_put(cx, prefix, LEAF[m], gqa_matrix(g, m))) return 0;
    return 1;
}
static void *gqa_import(const TernaryImportCtx *cx, const char *prefix, const TernaryJNode *cfg) {
    int64_t embed, seq, heads, kv, rope_mode, shift, window, sinks; float base;
    if (!cfg_int(cx, cfg, "embed_dim", 1, MAXDIM, &embed) || !cfg_int(cx, cfg, "seq_len", 1, MAXDIM, &seq) ||
        !cfg_int(cx, cfg, "n_heads", 1, MAXDIM, &heads) || !cfg_int(cx, cfg, "n_kv_heads", 1, MAXDIM, &kv) ||
        !cfg_int(cx, cfg, "rope_mode", 0, 2, &rope_mode) || !cfg_int(cx, cfg, "qk_shift", -64, 64, &shift) ||
        !cfg_int(cx, cfg, "window", 0, MAXDIM, &window) || !cfg_int(cx, cfg, "sinks", 0, MAXDIM, &sinks) ||
        !cfg_float(cx, cfg, "rope_base", &base)) return NULL;
    int vt = 0;   /* modelos anteriores: V ternario ya no existe, hay que reentrenarlos (V ahora es de 8 bits) */
    if (ternary_jdom_get_bool(cfg, "v_ternary", &vt) && vt) { ierr(cx, "gqa: el modelo guardado usa V ternario, que se elimino; hay que reentrenarlo"); return NULL; }
    if (window && rope_mode != 1) { ierr(cx, "gqa: la ventana deslizante exige rope_mode 1"); return NULL; }
    TernaryGQA *g = ternary_gqa_alloc_window((size_t)embed, (size_t)seq, (size_t)heads, (size_t)kv, (size_t)window, (size_t)sinks);
    if (!g) { ierr(cx, "gqa: dimensiones invalidas o sin memoria"); return NULL; }
    if (rope_mode == 1 && ternary_gqa_enable_rope_int(g, base, (int)shift) != 0) { ternary_gqa_free(g); ierr(cx, "gqa: rope invalido"); return NULL; }
    if (rope_mode == 2 && ternary_gqa_enable_rope(g, base) != 0) { ternary_gqa_free(g); ierr(cx, "gqa: rope invalido"); return NULL; }
    int smear = 0;
    ternary_jdom_get_bool(cfg, "smear", &smear);               /* opcional: ausente = false (modelos anteriores a esta opcion) */
    if (smear && ternary_gqa_set_smear(g, 1) != 0) { ternary_gqa_free(g); ierr(cx, "gqa: smear exige rope_mode 1 y atencion completa"); return NULL; }
    static const char *const LEAF[4] = { "q_proj", "k_proj", "v_proj", "o_proj" };
    for (size_t m = 0; m < 4; m++)
        if (!sub_get(cx, prefix, LEAF[m], gqa_matrix(g, m))) { ternary_gqa_free(g); return NULL; }
    return g;
}

/* ---- rmsnorm ---- */
static int rms_export(const void *self, const TernaryExportCtx *cx, const char *prefix, TernaryJBuf *cfg) {
    const TernaryRMSNorm *n = (const TernaryRMSNorm *)self;
    ternary_jbuf_kint(cfg, "rows", (int64_t)n->rows);
    ternary_jbuf_kint(cfg, "dim", (int64_t)n->dim);
    ternary_jbuf_kfloat(cfg, "eps", n->eps);
    return sub_put(cx, prefix, "gain", n->gain);
}
static void *rms_import(const TernaryImportCtx *cx, const char *prefix, const TernaryJNode *cfg) {
    int64_t rows, dim; float eps;
    if (!cfg_int(cx, cfg, "rows", 1, MAXDIM, &rows) || !cfg_int(cx, cfg, "dim", 1, MAXDIM, &dim) ||
        !cfg_float(cx, cfg, "eps", &eps)) return NULL;
    TernaryRMSNorm *n = ternary_rmsnorm_alloc((size_t)rows, (size_t)dim, eps);
    if (!n) { ierr(cx, "rmsnorm: dimensiones invalidas o sin memoria"); return NULL; }
    if (!sub_get(cx, prefix, "gain", n->gain)) { ternary_rmsnorm_free(n); return NULL; }
    return n;
}

/* ---- swiglu ---- */
static int swiglu_export(const void *self, const TernaryExportCtx *cx, const char *prefix, TernaryJBuf *cfg) {
    const TernarySwiGLU *s = (const TernarySwiGLU *)self;
    ternary_jbuf_kint(cfg, "rows", (int64_t)s->rows);
    ternary_jbuf_kint(cfg, "dim", (int64_t)s->dim);
    ternary_jbuf_kint(cfg, "hidden", (int64_t)s->hidden);
    ternary_jbuf_kint(cfg, "gate_mode", s->gate_mode);       /* 0 silu, 1 sign, 2 relu^2 (sin compuerta) */
    ternary_jbuf_kfloat(cfg, "band", s->band);
    if (s->wg && !sub_put(cx, prefix, "gate_proj", s->wg)) return 0;
    return sub_put(cx, prefix, "up_proj", s->wu) && sub_put(cx, prefix, "down_proj", s->wd);
}
static void *swiglu_import(const TernaryImportCtx *cx, const char *prefix, const TernaryJNode *cfg) {
    int64_t rows, dim, hidden, gate; float band;
    if (!cfg_int(cx, cfg, "rows", 1, MAXDIM, &rows) || !cfg_int(cx, cfg, "dim", 1, MAXDIM, &dim) ||
        !cfg_int(cx, cfg, "hidden", 1, MAXDIM, &hidden) || !cfg_int(cx, cfg, "gate_mode", 0, 2, &gate) ||
        !cfg_float(cx, cfg, "band", &band)) return NULL;
    TernarySwiGLU *s = ternary_swiglu_alloc((size_t)rows, (size_t)dim, (size_t)hidden, (int)gate, band);
    if (!s) { ierr(cx, "swiglu: dimensiones, gate_mode o band invalidos (o sin memoria)"); return NULL; }
    if ((s->wg && !sub_get(cx, prefix, "gate_proj", s->wg)) || !sub_get(cx, prefix, "up_proj", s->wu) ||
        !sub_get(cx, prefix, "down_proj", s->wd)) { ternary_swiglu_free(s); return NULL; }
    ternary_swiglu_refresh_tables(s);   /* la escala de gate recien cargada cambia la tabla de silu */
    return s;
}

/* ---- lm_head ---- */
static int head_export(const void *self, const TernaryExportCtx *cx, const char *prefix, TernaryJBuf *cfg) {
    const TernaryHead *h = (const TernaryHead *)self;
    ternary_jbuf_kint(cfg, "rows", (int64_t)h->rows);
    ternary_jbuf_kint(cfg, "dim", (int64_t)h->dim);
    ternary_jbuf_kint(cfg, "vocab", (int64_t)h->vocab);
    if (!dense_put(cx, prefix, h->w)) return 0;
    if (cx->state && h->vn) {      /* checkpoint: estado de la normalizacion por elemento (codigos de 4 bits, escala por fila, pasos) */
        char nm[NAMEBUF]; const TernaryElemNorm *v = h->vn;
        const size_t sc[2] = { v->rows, (v->cols + 1) / 2 }, sr = v->rows, one = 1;
        if (!tname(nm, prefix, "vnorm_codes") || !ternary_tw_add(cx->state, nm, TTD_U8, sc, 2, v->code)) return 0;
        if (!tname(nm, prefix, "vnorm_rmax") || !ternary_tw_add(cx->state, nm, TTD_F32, &sr, 1, v->rmax)) return 0;
        if (!tname(nm, prefix, "vnorm_steps") || !ternary_tw_add(cx->state, nm, TTD_I32, &one, 1, &v->steps)) return 0;
    }
    return 1;
}
static void *head_import(const TernaryImportCtx *cx, const char *prefix, const TernaryJNode *cfg) {
    int64_t rows, dim, vocab;
    if (!cfg_int(cx, cfg, "rows", 1, MAXDIM, &rows) || !cfg_int(cx, cfg, "dim", 1, MAXDIM, &dim) ||
        !cfg_int(cx, cfg, "vocab", 1, MAXDIM, &vocab)) return NULL;
    TernaryHead *h = ternary_head_alloc((size_t)rows, (size_t)dim, (size_t)vocab);
    if (!h) { ierr(cx, "lm_head: dimensiones invalidas o sin memoria"); return NULL; }
    if (!dense_get(cx, prefix, h->w)) { ternary_head_free(h); return NULL; }
    char nm[NAMEBUF];
    if (cx->state && tname(nm, prefix, "vnorm_codes") && ternary_tr_find(cx->state, nm)) {   /* checkpoint con normalizacion por elemento */
        const size_t st = ((size_t)dim + 1) / 2, sc[2] = { (size_t)vocab, st }, sr = (size_t)vocab, one = 1;
        const TernaryTInfo *tc, *tr, *ts;
        if (!(tc = ternary_tr_expect(cx->state, nm, TTD_U8, sc, 2, cx->err, cx->err_cap)) ||
            !tname(nm, prefix, "vnorm_rmax") || !(tr = ternary_tr_expect(cx->state, nm, TTD_F32, &sr, 1, cx->err, cx->err_cap)) ||
            !tname(nm, prefix, "vnorm_steps") || !(ts = ternary_tr_expect(cx->state, nm, TTD_I32, &one, 1, cx->err, cx->err_cap))) { ternary_head_free(h); return NULL; }
        h->vn = ternary_elemnorm_alloc((size_t)vocab, (size_t)dim);
        if (!h->vn) { ierr(cx, "lm_head: sin memoria para el estado de normalizacion"); ternary_head_free(h); return NULL; }
        memcpy(h->vn->code, tc->data, tc->nbytes);
        for (size_t i = 0; i < (size_t)vocab; i++) {
            const float x = ternary_tinfo_f32(tr, i);
            if (!(x >= 0.0f) || x > 3.4028235e38f) { ierr(cx, "lm_head: escala de normalizacion no valida (archivo corrupto)"); ternary_head_free(h); return NULL; }
            h->vn->rmax[i] = x;
        }
        int32_t t32; memcpy(&t32, ts->data, sizeof t32);
        if (t32 < 0) { ierr(cx, "lm_head: pasos de normalizacion negativos (archivo corrupto)"); ternary_head_free(h); return NULL; }
        h->vn->steps = (uint32_t)t32;
    }
    return h;
}

/* ---- entradas genericas {"type","activation","config"} ---- */
int ternary_layer_export_entry(TernaryLayer l, const char *act_name, const TernaryExportCtx *cx, const char *prefix,
                               TernaryJBuf *out) {
    if (!l.vt->export_layer || strlen(prefix) > 160) return 0;
    ternary_jbuf_open(out, '{');
    ternary_jbuf_kstr(out, "name", prefix);
    ternary_jbuf_kstr(out, "type", l.vt->type_name);
    ternary_jbuf_kstr(out, "activation", act_name);
    ternary_jbuf_key(out, "config");
    ternary_jbuf_open(out, '{');
    int ok = l.vt->export_layer(l.impl, cx, prefix, out);
    ternary_jbuf_close(out, '}');
    ternary_jbuf_close(out, '}');
    return ok && !out->failed;
}

int ternary_layer_import_entry(const TernaryImportCtx *cx, const char *prefix, const TernaryJNode *entry,
                               TernaryLayer *layer_out, const TernaryActivation **act_out) {
    if (!entry || entry->type != TJ_OBJ) { ierr(cx, "%s: la entrada de capa no es un objeto", prefix); return 0; }
    const char *type = ternary_jdom_get_str(entry, "type"), *act = ternary_jdom_get_str(entry, "activation");
    const TernaryJNode *cfg = ternary_jdom_get(entry, "config");
    if (!type || !act || !cfg || cfg->type != TJ_OBJ) { ierr(cx, "%s: faltan 'type', 'activation' o 'config'", prefix); return 0; }
    const TernaryActivation *a = ternary_activation_by_name(act);
    if (!a) {
        ierr(cx, "%s: activacion desconocida '%s'%s", prefix, act,
             (!strcmp(act, "sign") || !strcmp(act, "band")) ? " (activacion eliminada: solo existe 'uint8'; hay que reentrenar)" : "");
        return 0;
    }
    int ok = 0;
    TernaryLayer l = ternary_layer_import_by_type(type, cx, prefix, cfg, &ok);
    if (!ok) { ierr(cx, "%s: tipo de capa desconocido o invalido ('%s')", prefix, type); return 0; }
    *layer_out = l; *act_out = a;
    return 1;
}

TernaryLayer ternary_layer_import_by_type(const char *type_name, const TernaryImportCtx *cx, const char *prefix,
                                          const TernaryJNode *cfg, int *ok) {
    TernaryLayer l = { NULL, NULL };
    const TernaryLayerVTable *vt = NULL;
    const TernaryLayerType *type = ternary_layer_type_by_name(type_name);
    if (type) vt = type->vt;
    else if (strcmp(type_name, "residual") == 0) vt = ternary_residual_vtable();   /* contenedor: no esta en la tabla publica */
    if (!vt || !vt->import_layer) { *ok = 0; return l; }
    void *impl = vt->import_layer(cx, prefix, cfg);
    if (!impl) { *ok = 0; return l; }
    l.impl = impl; l.vt = vt; *ok = 1;
    return l;
}
