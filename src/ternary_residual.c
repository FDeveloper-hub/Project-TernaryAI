#include "ternary_mem.h"
#include "ternary_threadpool.h"
#include "ternary_avx512.h"
#include "ternary_neon_policy.h"
#include "ternary_prof.h"
#include "ternary_residual.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct TernaryResidual {
    size_t n, cap;
    size_t len;                 /* largo de x (= de la salida); lo fija la primera sub-capa */
    TernaryLayer *sub;          /* [n] */
    TernaryActivation *act;     /* [n] */
    float **out;                /* [n] salida escalada cruda de cada sub-capa */
    int32_t **out_raw;          /* [n] Z crudo (para el backward de cada una) */
    TernaryU8 **act_u8;         /* [n] salida uint8 de j (NULL si la sub-capa siguiente la lee como FLOAT, y en la ultima) */
    TernaryTensor **view;       /* [n] vista de la salida de j que se le pasa a j+1 (NULL en la ultima) */
    float **gx;                 /* [n] dL/d(entrada) de cada sub-capa */
    size_t *off;                /* [n+1] indice de la primera matriz de cada sub-capa dentro de esta capa */
    int bound;                  /* el estado (buffers de las sub-capas y el de ellas) vive en un espacio de trabajo ajeno */
};

/* GQA, RMSNorm y SwiGLU no usan el Z crudo que se les pasa en el backward (lo tienen en su propio estado): no se
 * reserva el buffer rows*out_len de enteros para ellas. */
static int sub_wants_raw(const TernaryLayer *l) {
    const char *t = l->vt->type_name;
    return !(strcmp(t, "gqa") == 0 || strcmp(t, "rmsnorm") == 0 || strcmp(t, "swiglu") == 0);
}

TernaryResidual *ternary_residual_create(void) { return (TernaryResidual *)ternary_calloc64(1, sizeof(TernaryResidual)); }

static void slot_free(TernaryResidual *r, size_t j) {
    if (r->sub[j].impl) r->sub[j].vt->destroy(r->sub[j].impl);
    if (r->bound) { free(r->act_u8[j]); }      /* los arreglos viven en el espacio de trabajo ajeno */
    else {
        free(r->out[j]); free(r->out_raw[j]); free(r->gx[j]);
        if (r->act_u8[j]) ternary_u8_free(r->act_u8[j]);
    }
    if (r->view[j]) ternary_tensor_free(r->view[j]);
}

void ternary_residual_free(TernaryResidual *r) {
    if (!r) return;
    for (size_t j = 0; j < r->n; j++) slot_free(r, j);
    free(r->sub); free(r->act); free(r->out); free(r->out_raw); free(r->act_u8); free(r->view); free(r->gx); free(r->off);
    free(r);
}

static int grow(TernaryResidual *r) {
    if (r->n < r->cap) return 1;
    size_t nc = r->cap ? r->cap * 2 : 4;
    if (nc > TERNARY_RESIDUAL_MAX_LAYERS) nc = TERNARY_RESIDUAL_MAX_LAYERS;
    if (nc <= r->cap) return 0;
    void *p;
#define GROW(field, T, extra) \
    p = realloc(r->field, (nc + (extra)) * sizeof(T)); if (!p) return 0; r->field = (T *)p;
    GROW(sub, TernaryLayer, 0) GROW(act, TernaryActivation, 0) GROW(out, float *, 0) GROW(out_raw, int32_t *, 0)
    GROW(act_u8, TernaryU8 *, 0) GROW(view, TernaryTensor *, 0) GROW(gx, float *, 0) GROW(off, size_t, 1)
#undef GROW
    r->cap = nc;
    return 1;
}

int ternary_residual_add(TernaryResidual *r, TernaryLayer layer, TernaryActivation act) {
    if (!r || !layer.impl || !layer.vt || r->bound) return 0;
    const size_t in_len = layer.vt->input_len(layer.impl), out_len = layer.vt->output_len(layer.impl);
    const char *name = layer.vt->type_name;
    if (r->n >= TERNARY_RESIDUAL_MAX_LAYERS) {
        ternary_diag("ternary_residual_add: demasiadas sub-capas (maximo %d).\n", TERNARY_RESIDUAL_MAX_LAYERS);
        return 0;
    }
    if (!layer.vt->propagates_grad_x(layer.impl)) {
        ternary_diag("ternary_residual_add: la sub-capa '%s' no propaga gradiente hacia su entrada -- dentro de una "
                        "conexion residual x tambien recibe el gradiente de la rama, asi que todas deben hacerlo. No se agrega.\n", name);
        return 0;
    }
    if (layer.vt->input_dtype(layer.impl) == TERNARY_DTYPE_ID) {
        ternary_diag("ternary_residual_add: la sub-capa '%s' espera ids como entrada; dentro de una residual no tiene sentido. No se agrega.\n", name);
        return 0;
    }
    if (r->n == 0) {
        if (layer.vt->input_dtype(layer.impl) != TERNARY_DTYPE_FLOAT) {
            ternary_diag("ternary_residual_add: la primera sub-capa ('%s') debe tomar entrada FLOAT (el flujo residual es "
                            "float; p.ej. RMSNorm). No se agrega.\n", name);
            return 0;
        }
    } else {
        const size_t prev_out = r->sub[r->n - 1].vt->output_len(r->sub[r->n - 1].impl);
        if (prev_out != in_len) {
            ternary_diag("ternary_residual_add: la sub-capa '%s' espera entrada de largo %zu pero la anterior entrega %zu. No se agrega.\n",
                    name, in_len, prev_out);
            return 0;
        }
    }
    const int next_u8 = r->n > 0 && layer.vt->input_dtype(layer.impl) != TERNARY_DTYPE_FLOAT;
    size_t prev_rl = 0, prev_out_len = 0;
    if (next_u8) {
        /* la salida de la anterior pasa a uint8: esta sub-capa debe aceptarla (se comprueba antes de reservar nada) */
        const TernaryLayer *pl = &r->sub[r->n - 1];
        const size_t need = layer.vt->accepts_u8 ? layer.vt->accepts_u8(layer.impl) : 0;
        if (need == 0) {
            ternary_diag("ternary_residual_add: la sub-capa '%s' no acepta entrada uint8. No se agrega.\n", name);
            return 0;
        }
        prev_out_len = pl->vt->output_len(pl->impl);
        prev_rl = pl->vt->output_row_len ? pl->vt->output_row_len(pl->impl) : prev_out_len;
        if ((prev_rl != need && prev_out_len != need) || need == 0 || prev_out_len % need != 0) {
            ternary_diag("ternary_residual_add: la sub-capa '%s' espera filas uint8 de %zu valores (la anterior entrega %zu). No se agrega.\n",
                         name, need, prev_rl);
            return 0;
        }
    }
    if (!grow(r)) return 0;

    const size_t j = r->n;
    r->out[j] = (float *)ternary_malloc64(out_len * sizeof(float));
    r->out_raw[j] = sub_wants_raw(&layer) ? (int32_t *)ternary_malloc64(out_len * sizeof(int32_t)) : NULL;
    r->act_u8[j] = NULL;
    r->gx[j] = (float *)ternary_malloc64(in_len * sizeof(float));
    r->view[j] = NULL;
    TernaryU8 *pu8 = next_u8 ? ternary_u8_alloc(prev_out_len / layer.vt->accepts_u8(layer.impl), layer.vt->accepts_u8(layer.impl)) : NULL;
    if (!r->out[j] || (sub_wants_raw(&layer) && !r->out_raw[j]) || !r->gx[j] || (next_u8 && !pu8)) {
        free(r->out[j]); free(r->out_raw[j]); free(r->gx[j]);
        if (pu8) ternary_u8_free(pu8);
        return 0;      /* el llamador sigue siendo dueno de la capa */
    }
    if (j > 0) {
        const size_t dims1[1] = { in_len };
        r->view[j - 1] = next_u8 ? ternary_tensor_view_u8(dims1, 1, pu8) : ternary_tensor_view_float(dims1, 1, r->out[j - 1]);
        if (!r->view[j - 1]) {
            free(r->out[j]); free(r->out_raw[j]); free(r->gx[j]);
            if (pu8) ternary_u8_free(pu8);
            return 0;
        }
        r->act_u8[j - 1] = pu8;
    } else {
        r->len = in_len;
        r->off[0] = 0;
    }
    r->sub[j] = layer; r->act[j] = act;
    r->off[j + 1] = r->off[j] + layer.vt->num_matrices(layer.impl);
    r->n++;
    return 1;
}

size_t ternary_residual_num_layers(const TernaryResidual *r) { return r ? r->n : 0; }
TernaryLayer ternary_residual_layer(const TernaryResidual *r, size_t i) {
    TernaryLayer none = { NULL, NULL };
    return (r && i < r->n) ? r->sub[i] : none;
}
const TernaryActivation *ternary_residual_activation(const TernaryResidual *r, size_t i) {
    return (r && i < r->n) ? &r->act[i] : NULL;
}

/* ---- vtable ---- */
/* Sumas elemento a elemento repartidas en hilos (cada elemento depende solo de su posicion: no cambia ningun bit). */
typedef struct { float *o; const float *a, *b; } ResAdd;
#define RES_ADD_GRAIN 32768
static void res_add_range(size_t i0, size_t i1, void *c_) {
    ResAdd *c = (ResAdd *)c_;
    if (ternary_avx512_enabled()) ternary_avx512_add_f32(c->o + i0, c->a + i0, c->b + i0, i1 - i0);
    else if (TN_ON(ELEM)) ternary_neon_add_f32(c->o + i0, c->a + i0, c->b + i0, i1 - i0);
    else for (size_t i = i0; i < i1; i++) c->o[i] = c->a[i] + c->b[i];
}
static void res_acc_range(size_t i0, size_t i1, void *c_) {
    ResAdd *c = (ResAdd *)c_;
    if (ternary_avx512_enabled()) ternary_avx512_acc_add2(c->o + i0, c->a + i0, c->b + i0, i1 - i0);
    else if (TN_ON(ELEM)) ternary_neon_acc_add2(c->o + i0, c->a + i0, c->b + i0, i1 - i0);
    else for (size_t i = i0; i < i1; i++) c->o[i] += c->a[i] + c->b[i];
}

#define R(self) ((const TernaryResidual *)(self))

static size_t res_input_len(const void *self) { return R(self)->len; }
static size_t res_output_len(const void *self) { return R(self)->len; }
static size_t res_num_matrices(const void *self) { return R(self)->off[R(self)->n]; }
static int res_propagates_grad_x(const void *self) { (void)self; return 1; }
static TernaryDType res_input_dtype(const void *self) { (void)self; return TERNARY_DTYPE_FLOAT; }

/* matriz m de esta capa -> (sub-capa j, matriz local) */
static size_t locate(const TernaryResidual *r, size_t m, size_t *local) {
    size_t j = 0;
    while (j + 1 < r->n && m >= r->off[j + 1]) j++;
    *local = m - r->off[j];
    return j;
}
static size_t res_grad_rows(const void *self, size_t m) {
    size_t lm, j = locate(R(self), m, &lm);
    return R(self)->sub[j].vt->grad_rows(R(self)->sub[j].impl, lm);
}
static size_t res_grad_row_width(const void *self, size_t m) {
    size_t lm, j = locate(R(self), m, &lm);
    return R(self)->sub[j].vt->grad_row_width(R(self)->sub[j].impl, lm);
}
static int res_has_bias(const void *self, size_t m) {
    size_t lm, j = locate(R(self), m, &lm);
    return R(self)->sub[j].vt->has_bias(R(self)->sub[j].impl, lm);
}

static void res_forward(const void *self, const TernaryTensor *x, float *out, int32_t *out_raw_opt,
                        const TernaryCapabilities *caps) {
    const TernaryResidual *r = R(self);
    const TernaryTensor *cur = x;
    for (size_t j = 0; j < r->n; j++) {
        TPROF_BEGIN_I(r->sub[j].vt->type_name, j);
        r->sub[j].vt->forward(r->sub[j].impl, cur, r->out[j], r->out_raw[j], caps);
        TPROF_END();
        if (j + 1 < r->n) {     /* la activacion de la ultima no se usa: se suma su salida cruda */
            if (r->act_u8[j]) { TPROF_BEGIN_I("quantize_u8", j); r->act[j].apply_u8(r->out[j], r->act_u8[j]); TPROF_END(); }
            cur = r->view[j];
        }
    }
    const float *f = r->out[r->n - 1];
    TPROF_BEGIN("residual_add");
    { ResAdd c = { out, x->data_f, f }; ternary_parallel_ranges(ternary_get_compute_pool(), 0, r->len, RES_ADD_GRAIN, res_add_range, &c); }
    TPROF_END();
    /* No hay un "Z crudo" con sentido para x + f(x): cada sub-capa guarda el suyo para su propio backward. */
    if (out_raw_opt) memset(out_raw_opt, 0, r->len * sizeof(int32_t));
}

static void res_backward(const void *self, const TernaryTensor *x, const float *grad_out,
                         float *const *grad_w_out, float *grad_x_out, float *grad_scale_out,
                         float *const *grad_bias_out, const int32_t *z_raw_opt) {
    const TernaryResidual *r = R(self);
    (void)z_raw_opt;
    const float *g = grad_out;      /* la rama f recibe dL/dout tal cual */
    for (size_t j = r->n; j-- > 0; ) {
        const TernaryTensor *in = (j == 0) ? x : r->view[j - 1];
        TPROF_BEGIN_I(r->sub[j].vt->type_name, j);
        ternary_par_zero(r->gx[j], r->sub[j].vt->input_len(r->sub[j].impl) * sizeof(float));
        r->sub[j].vt->backward(r->sub[j].impl, in, g, grad_w_out + r->off[j], r->gx[j],
                               grad_scale_out + r->off[j], grad_bias_out + r->off[j], r->out_raw[j]);
        TPROF_END();
        g = r->gx[j];
    }
    TPROF_BEGIN("residual_add");
    { ResAdd c = { grad_x_out, grad_out, r->gx[0] }; ternary_parallel_ranges(ternary_get_compute_pool(), 0, r->len, RES_ADD_GRAIN, res_acc_range, &c); }   /* atajo + rama */
    TPROF_END();
}

/* Estado compartible: los buffers de cada sub-capa (salida, Z crudo, vector ternario de la activacion, gradiente de la
 * entrada) y, a continuacion, el estado compartible de la propia sub-capa. Con base == NULL solo cuenta. */
static size_t res_carve(TernaryResidual *r, unsigned char *base) {
    TernaryCarve c = { base, 0 };
    for (size_t j = 0; j < r->n; j++) {
        const size_t out_len = r->sub[j].vt->output_len(r->sub[j].impl), in_len = r->sub[j].vt->input_len(r->sub[j].impl);
        float *o = (float *)ternary_carve(&c, out_len * sizeof(float));
        int32_t *raw = sub_wants_raw(&r->sub[j]) ? (int32_t *)ternary_carve(&c, out_len * sizeof(int32_t)) : NULL;
        float *gx = (float *)ternary_carve(&c, in_len * sizeof(float));
        uint8_t *uq = NULL; float *usc = NULL; int32_t *uzp = NULL;
        if (r->act_u8[j]) {
            uq = (uint8_t *)ternary_carve(&c, out_len);
            usc = (float *)ternary_carve(&c, r->act_u8[j]->rows * sizeof(float));
            uzp = (int32_t *)ternary_carve(&c, r->act_u8[j]->rows * sizeof(int32_t));
        }
        const size_t sb = r->sub[j].vt->state_bytes ? r->sub[j].vt->state_bytes(r->sub[j].impl) : 0;
        unsigned char *sbase = sb ? (unsigned char *)ternary_carve(&c, sb) : NULL;
        if (base) {
            r->out[j] = o; r->out_raw[j] = raw; r->gx[j] = gx;
            if (r->act_u8[j]) { r->act_u8[j]->q = uq; r->act_u8[j]->scale = usc; r->act_u8[j]->zp = uzp; }
            if (r->view[j] && r->view[j]->dtype == TERNARY_DTYPE_FLOAT) r->view[j]->data_f = o;
            if (sb) r->sub[j].vt->bind_state(r->sub[j].impl, sbase);
        }
    }
    return c.off;
}
static size_t res_state_bytes(const void *self) { return res_carve((TernaryResidual *)self, NULL); }
static void res_bind_state(void *self, void *base) {
    TernaryResidual *r = (TernaryResidual *)self;
    if (!base) return;
    if (!r->bound) {
        for (size_t j = 0; j < r->n; j++) {
            free(r->out[j]); free(r->out_raw[j]); free(r->gx[j]);
            r->out[j] = NULL; r->out_raw[j] = NULL; r->gx[j] = NULL;
            if (r->act_u8[j]) { free(r->act_u8[j]->q); free(r->act_u8[j]->scale); free(r->act_u8[j]->zp); r->act_u8[j]->q = NULL; r->act_u8[j]->scale = NULL; r->act_u8[j]->zp = NULL; }
        }
        r->bound = 1;
    }
    res_carve(r, (unsigned char *)base);
}

static void res_apply_row_gradients(void *self, size_t m, size_t row, const float *grad_row) {
    size_t lm, j = locate(R(self), m, &lm);
    TPROF_BEGIN_I(R(self)->sub[j].vt->type_name, j);
    R(self)->sub[j].vt->apply_row_gradients(R(self)->sub[j].impl, lm, row, grad_row);
    TPROF_END();
}
static void res_apply_bias_gradients(void *self, size_t m, const float *gb) {
    size_t lm, j = locate(R(self), m, &lm);
    R(self)->sub[j].vt->apply_bias_gradients(R(self)->sub[j].impl, lm, gb);
}
static void res_apply_scale_gradient(void *self, size_t m, float grad_scale, float scale_lr) {
    size_t lm, j = locate(R(self), m, &lm);
    TPROF_BEGIN_I("apply_scale", j);
    R(self)->sub[j].vt->apply_scale_gradient(R(self)->sub[j].impl, lm, grad_scale, scale_lr);
    TPROF_END();
}
static void res_destroy(void *self) { ternary_residual_free((TernaryResidual *)self); }

/* ---- guardado / carga ---- */
static int res_export(const void *self, const TernaryExportCtx *cx, const char *prefix, TernaryJBuf *cfg) {
    const TernaryResidual *r = R(self);
    ternary_jbuf_key(cfg, "layers");
    ternary_jbuf_open(cfg, '[');
    for (size_t j = 0; j < r->n; j++) {
        char sub[200];
        if (snprintf(sub, sizeof sub, "%s.%zu", prefix, j) >= (int)sizeof sub) return 0;
        TernaryLayer l = { r->sub[j].impl, r->sub[j].vt };
        if (!ternary_layer_export_entry(l, r->act[j].name, cx, sub, cfg)) return 0;
    }
    ternary_jbuf_close(cfg, ']');
    return 1;
}

static void *res_import(const TernaryImportCtx *cx, const char *prefix, const TernaryJNode *cfg) {
    const TernaryJNode *arr = ternary_jdom_get(cfg, "layers");
    if (!arr || arr->type != TJ_ARR || arr->n == 0 || arr->n > TERNARY_RESIDUAL_MAX_LAYERS) {
        if (cx->err && cx->err_cap && !cx->err[0]) snprintf(cx->err, cx->err_cap, "%s: 'layers' de la residual falta o tiene un tamano invalido", prefix);
        return NULL;
    }
    TernaryResidual *r = ternary_residual_create();
    if (!r) return NULL;
    for (size_t j = 0; j < arr->n; j++) {
        char sub[200];
        TernaryLayer layer; const TernaryActivation *act;
        if (snprintf(sub, sizeof sub, "%s.%zu", prefix, j) >= (int)sizeof sub ||
            !ternary_layer_import_entry(cx, sub, &arr->kids[j], &layer, &act)) { ternary_residual_free(r); return NULL; }
        if (!ternary_residual_add(r, layer, *act)) {
            layer.vt->destroy(layer.impl); ternary_residual_free(r);
            if (cx->err && cx->err_cap && !cx->err[0]) snprintf(cx->err, cx->err_cap, "%s: las sub-capas no encajan entre si", sub);
            return NULL;
        }
    }
    if (r->sub[r->n - 1].vt->output_len(r->sub[r->n - 1].impl) != r->len) {
        ternary_residual_free(r);
        if (cx->err && cx->err_cap && !cx->err[0]) snprintf(cx->err, cx->err_cap, "%s: la ultima sub-capa no devuelve el largo de la entrada", prefix);
        return NULL;
    }
    return r;
}

static const TernaryLayerVTable TERNARY_RESIDUAL_VTABLE = {
    "residual",
    res_input_len, res_output_len, res_num_matrices, res_grad_rows, res_grad_row_width,
    res_has_bias, res_propagates_grad_x, res_input_dtype,
    res_forward, res_backward,
    res_apply_row_gradients, res_apply_bias_gradients, res_apply_scale_gradient,
    res_destroy,
    NULL, /* backward_accumulate: no soportado */
    NULL, /* backward_block: no soportado */
    NULL, /* apply_matrix_gradients: no soportado */
    NULL, /* forward_block: no soportado */
    res_state_bytes,
    res_bind_state,
    NULL, /* accepts_u8 */
    NULL, /* output_row_len */
    res_export, res_import,
};

const TernaryLayerVTable *ternary_residual_vtable(void) { return &TERNARY_RESIDUAL_VTABLE; }

TernaryLayer ternary_layer_from_residual(TernaryResidual *r) {
    TernaryLayer none = { NULL, NULL };
    if (!r || r->n == 0) return none;
    if (r->sub[r->n - 1].vt->output_len(r->sub[r->n - 1].impl) != r->len) {
        ternary_diag("ternary_layer_from_residual: la ultima sub-capa entrega %zu valores pero x tiene %zu -- "
                        "x + f(x) necesita el mismo largo.\n", r->sub[r->n - 1].vt->output_len(r->sub[r->n - 1].impl), r->len);
        return none;
    }
    TernaryLayer l; l.impl = r; l.vt = &TERNARY_RESIDUAL_VTABLE; return l;
}
