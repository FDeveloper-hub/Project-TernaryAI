#include "ternaryai.h"
#include "ternary_stack.h"
#include "ternary_tensor.h"
#include "ternary_dispatch.h"
#include "ternary_tokenizer.h"
#include "ternary_dataset.h"
#include "ternary_targets.h"
#include "ternary_loss.h"
#include "ternary_residual.h"
#include "ternary_prof.h"
#include "ternary_threadpool.h"
#include "ternary_swiglu.h"
#include "ternary_gqa.h"
#include <stdint.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ===================================================================
 * Capacidades de CPU compartidas: se detectan al crearse el primer
 * modelo y se liberan cuando se libera el ultimo. Solo familia CPU.
 * =================================================================== */
static pthread_mutex_t g_caps_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_compute_lock = PTHREAD_MUTEX_INITIALIZER;
static TernaryCapabilities g_caps;
static int g_caps_refs = 0;

static const TernaryCapabilities *caps_acquire(void) {
    pthread_mutex_lock(&g_caps_lock);
    if (g_caps_refs == 0) g_caps = ternary_init_cpu();
    g_caps_refs++;
    pthread_mutex_unlock(&g_caps_lock);
    return &g_caps;
}

static void caps_release(void) {
    pthread_mutex_lock(&g_caps_lock);
    if (--g_caps_refs == 0) ternary_shutdown(&g_caps);
    pthread_mutex_unlock(&g_caps_lock);
}

static void set_err(char *err, size_t cap, const char *fmt, ...) {
    if (!err || cap == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, cap, fmt, ap);
    va_end(ap);
}

/* Bloque residual abierto con ternaryai_model_begin_residual y aun no cerrado. */
typedef struct {
    TernaryResidual *r;
    const TernaryActivation *act;   /* activacion de la capa residual (la del stack / la del bloque padre) */
    TernaryRng rng_backup;          /* generador de pesos al abrir: cancelar o fallar lo deja como estaba */
    size_t shape[TERNARYAI_MAX_DIMS];   /* forma de x (la entrada del bloque, y su salida) */
    size_t nd;
} OpenResidual;

struct TernaryAIModel {
    TernaryStack *stack;
    OpenResidual open[TERNARYAI_MAX_RESIDUAL_DEPTH];
    int n_open;
    TernaryRng rng;
    const TernaryCapabilities *caps;
    size_t in_shape[TERNARYAI_MAX_DIMS];
    size_t in_ndims;
    size_t cur_shape[TERNARYAI_MAX_DIMS];   /* forma tras la ultima capa (la de entrada si no hay) */
    size_t cur_ndims;
    int restore_best_off;                    /* 1 = no volver al mejor estado en train_ids */
    /* "Retener el mejor": mientras esta activo, train_ids guarda el mejor estado ENTRE llamadas (los trozos de un
     * fit con verbose) y no restaura al final de cada una; ternaryai_model_restore_held_best lo hace una sola vez. */
    int hold_active;
    TernaryStackSnapshot hold_best;
    double hold_best_loss;
    /* Mini-lotes: 0 (o >= n) = lote completo (una actualizacion por epoca, como siempre). */
    size_t batch_size;
    int batch_shuffle;
    uint64_t batch_seed;
};

void ternaryai_model_set_batching(TernaryAIModel *m, size_t batch_size, int shuffle, uint64_t seed) {
    if (!m) return;
    m->batch_size = batch_size;
    m->batch_shuffle = shuffle ? 1 : 0;
    m->batch_seed = seed;
}

int ternaryai_model_set_update_rule(TernaryAIModel *m, double tau, double beta, char *err, size_t err_cap) {
    if (!m) { set_err(err, err_cap, "invalid arguments"); return 0; }
    if (!(tau > 0.0 && tau <= 1.0)) { set_err(err, err_cap, "tau must be in (0, 1], got %g", tau); return 0; }
    if (!(beta >= 0.0 && beta < 1.0)) { set_err(err, err_cap, "beta must be in [0, 1), got %g", beta); return 0; }
    ternary_stack_set_update_rule(m->stack, (float)tau, (float)beta);
    return 1;
}

int ternaryai_model_set_r2_params(TernaryAIModel *m, double inc, double zero_target, double reset, char *err, size_t err_cap) {
    if (!m) { set_err(err, err_cap, "invalid arguments"); return 0; }
    if (!(inc > 0.0 && isfinite(inc))) { set_err(err, err_cap, "inc must be > 0, got %g", inc); return 0; }
    if (!(zero_target < 1.0)) { set_err(err, err_cap, "zero_target must be < 1 (negative disables it), got %g", zero_target); return 0; }
    if (!(reset >= 0.0 && reset <= 1.0)) { set_err(err, err_cap, "reset must be in [0, 1], got %g", reset); return 0; }
    ternary_stack_set_r2_params(m->stack, (float)inc, (float)zero_target, (float)reset);
    return 1;
}

int ternaryai_model_get_update_rule(const TernaryAIModel *m, double *out5) {
    if (!m || !out5) return 0;
    const TernaryUpdateRule r = ternary_stack_get_update_rule(m->stack);
    out5[0] = r.tau; out5[1] = r.beta; out5[2] = r.inc; out5[3] = r.zero_target; out5[4] = r.reset;
    return 1;
}

int ternaryai_model_set_head_beta2(TernaryAIModel *m, double head_beta2, char *err, size_t err_cap) {
    if (!m) { set_err(err, err_cap, "invalid arguments"); return 0; }
    if (!(head_beta2 >= 0.0 && head_beta2 < 1.0)) { set_err(err, err_cap, "head_beta2 must be in [0, 1) (0 turns it off), got %g", head_beta2); return 0; }
    ternary_stack_set_head_beta2(m->stack, (float)head_beta2);
    return 1;
}
int ternaryai_model_get_head_beta2(const TernaryAIModel *m, double *out) {
    if (!m || !out) return 0;
    *out = ternary_stack_get_update_rule(m->stack).head_beta2;
    return 1;
}

void ternaryai_model_set_restore_best(TernaryAIModel *m, int on) { if (m) m->restore_best_off = on ? 0 : 1; }
void ternaryai_model_hold_best(TernaryAIModel *m, int on) {
    if (!m) return;
    ternary_stack_snapshot_free(&m->hold_best);
    m->hold_best.buf = NULL; m->hold_best.size = 0;
    m->hold_best_loss = 1e300;
    m->hold_active = on ? 1 : 0;
}

int ternaryai_model_restore_held_best(TernaryAIModel *m, double last_loss) {
    if (!m || !m->hold_active) return 0;
    int restored = 0;
    if (m->hold_best.buf && last_loss > m->hold_best_loss) { ternary_stack_restore(m->stack, &m->hold_best); restored = 1; }
    ternaryai_model_hold_best(m, 0);
    return restored;
}

int ternaryai_model_set_checkpoint(TernaryAIModel *m, int on) { return m && m->stack ? ternary_stack_set_checkpoint(m->stack, on) : -1; }

int ternaryai_abi_version(void) { return TERNARYAI_ABI_VERSION; }

/* --- registro --- */
static const TernaryLayerType *type_at(int t) {
    return (t >= 0) ? ternary_layer_type_at((size_t)t) : NULL;
}
static const TernaryParamSpec *param_at(int t, int p) {
    const TernaryLayerType *ty = type_at(t);
    return (ty && p >= 0 && (size_t)p < ty->num_params) ? &ty->params[p] : NULL;
}

int ternaryai_layer_type_count(void) { return (int)ternary_layer_type_count(); }
const char *ternaryai_layer_type_name(int t) { const TernaryLayerType *ty = type_at(t); return ty ? ty->type_name : NULL; }
const char *ternaryai_layer_type_class(int t) { const TernaryLayerType *ty = type_at(t); return ty ? ty->class_name : NULL; }
const char *ternaryai_layer_type_doc(int t) { const TernaryLayerType *ty = type_at(t); return ty ? ty->doc : NULL; }
int ternaryai_layer_param_count(int t) { const TernaryLayerType *ty = type_at(t); return ty ? (int)ty->num_params : 0; }
const char *ternaryai_layer_param_name(int t, int p) { const TernaryParamSpec *ps = param_at(t, p); return ps ? ps->name : NULL; }
int ternaryai_layer_param_kind(int t, int p) { const TernaryParamSpec *ps = param_at(t, p); return ps ? (int)ps->kind : -1; }
int ternaryai_layer_param_required(int t, int p) { const TernaryParamSpec *ps = param_at(t, p); return ps ? ps->required : 0; }
double ternaryai_layer_param_default(int t, int p) { const TernaryParamSpec *ps = param_at(t, p); return ps ? ps->default_value : 0.0; }
int ternaryai_layer_param_has_min(int t, int p) { const TernaryParamSpec *ps = param_at(t, p); return ps ? ps->has_min : 0; }
double ternaryai_layer_param_min(int t, int p) { const TernaryParamSpec *ps = param_at(t, p); return ps ? ps->min_value : 0.0; }
int ternaryai_layer_param_has_max(int t, int p) { const TernaryParamSpec *ps = param_at(t, p); return ps ? ps->has_max : 0; }
double ternaryai_layer_param_max(int t, int p) { const TernaryParamSpec *ps = param_at(t, p); return ps ? ps->max_value : 0.0; }
const char *ternaryai_layer_param_doc(int t, int p) { const TernaryParamSpec *ps = param_at(t, p); return ps ? ps->doc : NULL; }
int ternaryai_activation_count(void) { return (int)ternary_activation_count(); }
const char *ternaryai_activation_name(int i) {
    const TernaryActivation *a = (i >= 0) ? ternary_activation_at((size_t)i) : NULL;
    return a ? a->name : NULL;
}

/* --- modelo --- */
static int valid_shape(const size_t *dims, int ndims) {
    if (!dims || ndims < 1 || ndims > TERNARYAI_MAX_DIMS) return 0;
    for (int i = 0; i < ndims; i++) if (dims[i] == 0) return 0;
    return 1;
}

static void copy_shape(size_t *dst, const size_t *src, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = src[i];
}

TernaryAIModel *ternaryai_model_create(const size_t *in_shape, int in_ndims, unsigned int seed,
                                 char *err, size_t err_cap) {
    if (!valid_shape(in_shape, in_ndims)) {
        set_err(err, err_cap, "input shape must have 1 to %d positive dimensions", TERNARYAI_MAX_DIMS);
        return NULL;
    }
    TernaryAIModel *m = (TernaryAIModel *)calloc(1, sizeof(TernaryAIModel));
    TernaryStack *s = ternary_stack_create();
    if (!m || !s || !ternary_stack_set_input_shape(s, in_shape, (size_t)in_ndims)) {
        free(m); ternary_stack_free(s);
        set_err(err, err_cap, "out of memory");
        return NULL;
    }
    m->stack = s;
    ternary_rng_seed(&m->rng, seed);
    m->in_ndims = m->cur_ndims = (size_t)in_ndims;
    copy_shape(m->in_shape, in_shape, m->in_ndims);
    copy_shape(m->cur_shape, in_shape, m->in_ndims);
    m->caps = caps_acquire();
    return m;
}

void ternaryai_model_free(TernaryAIModel *m) {
    if (!m) return;
    while (m->n_open > 0) ternary_residual_free(m->open[--m->n_open].r);
    ternary_stack_snapshot_free(&m->hold_best);
    ternary_stack_free(m->stack);
    caps_release();
    free(m);
}

static void set_activation_err(char *err, size_t cap, const char *name) {
    if (strcmp(name, "sign") == 0 || strcmp(name, "band") == 0)
        set_err(err, cap, "activation '%s' was removed: the only activation is 'uint8' (use 'uint8')", name);
    else
        set_err(err, cap, "unknown activation '%s' (the only one is 'uint8')", name);
}

int ternaryai_model_add_layer(TernaryAIModel *m, const char *type_name, const double *params, int num_slots,
                           const char *activation, size_t *out_shape, int *out_ndims,
                           char *err, size_t err_cap) {
    if (!m || !type_name || !activation || !params || num_slots < 0) {
        set_err(err, err_cap, "invalid arguments");
        return 0;
    }
    const TernaryLayerType *type = ternary_layer_type_by_name(type_name);
    if (!type) { set_err(err, err_cap, "unknown layer type '%s'", type_name); return 0; }
    const TernaryActivation *act = ternary_activation_by_name(activation);
    if (!act) { set_activation_err(err, err_cap, activation); return 0; }

    /* Si algo falla, el modelo (incluido su generador de pesos) queda intacto. */
    TernaryRng rng_backup = m->rng;
    TernaryLayer layer;
    if (!ternary_layer_type_build(type, m->cur_shape, m->cur_ndims, params, (size_t)num_slots,
                                  &m->rng, &layer, err, err_cap)) {
        m->rng = rng_backup;
        return 0;
    }
    char diag[256];
    if (m->n_open > 0) {
        /* Dentro de un bloque residual: la capa va a su sub-cadena (ternary_residual_add valida las reglas). */
        ternary_diag_capture(diag, sizeof(diag));
        int ok = ternary_residual_add(m->open[m->n_open - 1].r, layer, *act);
        ternary_diag_capture(NULL, 0);
        if (!ok) {
            set_err(err, err_cap, "the %s layer cannot be added to the residual block: %s", type->class_name,
                    diag[0] ? diag : "internal error");
            layer.vt->destroy(layer.impl);
            m->rng = rng_backup;
            return 0;
        }
    } else {
        if (ternary_stack_num_layers(m->stack) > 0 && !layer.vt->propagates_grad_x(layer.impl)) {
            set_err(err, err_cap, "%s does not propagate gradient to its input, so it can only be the "
                                  "first layer", type->class_name);
            layer.vt->destroy(layer.impl);
            m->rng = rng_backup;
            return 0;
        }
        ternary_diag_capture(diag, sizeof(diag));
        int ok = ternary_stack_add(m->stack, layer, *act);
        ternary_diag_capture(NULL, 0);
        if (!ok) {
            set_err(err, err_cap, "could not add the %s layer to the model: %s", type->class_name,
                    diag[0] ? diag : "internal error");
            layer.vt->destroy(layer.impl);
            m->rng = rng_backup;
            return 0;
        }
    }
    double slots[TERNARY_MAX_LAYER_SLOTS];
    size_t nd = type->describe(layer.impl, slots, m->cur_shape);
    m->cur_ndims = nd;
    if (out_shape) copy_shape(out_shape, m->cur_shape, nd);
    if (out_ndims) *out_ndims = (int)nd;
    return 1;
}

/* --- bloques residuales: x + f(x) con f = una cadena de capas --- */
int ternaryai_model_begin_residual(TernaryAIModel *m, const char *activation, char *err, size_t err_cap) {
    if (!m || !activation) { set_err(err, err_cap, "invalid arguments"); return 0; }
    if (m->n_open >= TERNARYAI_MAX_RESIDUAL_DEPTH) {
        set_err(err, err_cap, "residual blocks can be nested at most %d deep", TERNARYAI_MAX_RESIDUAL_DEPTH);
        return 0;
    }
    const TernaryActivation *act = ternary_activation_by_name(activation);
    if (!act) { set_activation_err(err, err_cap, activation); return 0; }
    TernaryResidual *r = ternary_residual_create();
    if (!r) { set_err(err, err_cap, "out of memory"); return 0; }
    OpenResidual *o = &m->open[m->n_open++];
    o->r = r; o->act = act; o->rng_backup = m->rng;
    o->nd = m->cur_ndims;
    copy_shape(o->shape, m->cur_shape, o->nd);
    return 1;
}

/* Descarta el bloque abierto mas interno y deja el modelo (forma y generador de pesos) como antes de abrirlo. */
static void discard_open_residual(TernaryAIModel *m) {
    OpenResidual *o = &m->open[--m->n_open];
    ternary_residual_free(o->r);
    m->rng = o->rng_backup;
    m->cur_ndims = o->nd;
    copy_shape(m->cur_shape, o->shape, o->nd);
}

int ternaryai_model_cancel_residual(TernaryAIModel *m, char *err, size_t err_cap) {
    if (!m || m->n_open == 0) { set_err(err, err_cap, "there is no open residual block to cancel"); return 0; }
    discard_open_residual(m);
    return 1;
}

int ternaryai_model_open_residuals(const TernaryAIModel *m) { return m ? m->n_open : 0; }

int ternaryai_model_end_residual(TernaryAIModel *m, size_t *out_shape, int *out_ndims, char *err, size_t err_cap) {
    if (!m || m->n_open == 0) { set_err(err, err_cap, "there is no open residual block to close"); return 0; }
    OpenResidual *o = &m->open[m->n_open - 1];
    char diag[256];
    ternary_diag_capture(diag, sizeof(diag));
    TernaryLayer l = ternary_layer_from_residual(o->r);
    ternary_diag_capture(NULL, 0);
    if (!l.impl) {
        set_err(err, err_cap, "%s; the block was discarded", ternary_residual_num_layers(o->r) == 0
                ? "the residual block is empty" : (diag[0] ? diag : "its last layer must return the same size as its input"));
        discard_open_residual(m);
        return 0;
    }
    const TernaryActivation *act = o->act;
    int ok;
    ternary_diag_capture(diag, sizeof(diag));
    if (m->n_open >= 2) {
        ok = ternary_residual_add(m->open[m->n_open - 2].r, l, *act);
    } else {
        ok = ternary_stack_add(m->stack, l, *act);
    }
    ternary_diag_capture(NULL, 0);
    if (!ok) {
        /* l sigue siendo nuestra: destruirla libera la residual y todas sus sub-capas */
        set_err(err, err_cap, "the residual block cannot be added here: %s; the block was discarded",
                diag[0] ? diag : "internal error");
        m->open[m->n_open - 1].r = NULL;
        l.vt->destroy(l.impl);
        OpenResidual *d = &m->open[--m->n_open];
        m->rng = d->rng_backup;
        m->cur_ndims = d->nd;
        copy_shape(m->cur_shape, d->shape, d->nd);
        return 0;
    }
    /* ahora el bloque es de su duenio (el stack o el bloque padre): x + f(x) conserva la forma de x */
    m->n_open--;
    m->cur_ndims = o->nd;
    copy_shape(m->cur_shape, o->shape, o->nd);
    if (out_shape) copy_shape(out_shape, m->cur_shape, m->cur_ndims);
    if (out_ndims) *out_ndims = (int)m->cur_ndims;
    return 1;
}

/* Mientras haya un bloque abierto, el modelo todavia no esta completo. */
static int model_closed(const TernaryAIModel *m, char *err, size_t err_cap) {
    if (m->n_open == 0) return 1;
    set_err(err, err_cap, "a residual block is still open: close it with ternaryai_model_end_residual first");
    return 0;
}

int ternaryai_model_num_layers(const TernaryAIModel *m) { return m ? (int)ternary_stack_num_layers(m->stack) : 0; }

int ternaryai_model_input_shape(const TernaryAIModel *m, size_t *dims, int *ndims) {
    if (!m || !dims || !ndims) return 0;
    copy_shape(dims, m->in_shape, m->in_ndims);
    *ndims = (int)m->in_ndims;
    return 1;
}

/* Forma de salida de una capa dada la de su entrada: una residual conserva la forma de x. */
static size_t layer_out_shape(const TernaryLayer *l, const size_t *in_shape, size_t in_nd, size_t *out_shape) {
    if (strcmp(l->vt->type_name, "residual") == 0) { copy_shape(out_shape, in_shape, in_nd); return in_nd; }
    const TernaryLayerType *type = ternary_layer_type_by_name(l->vt->type_name);
    double slots[TERNARY_MAX_LAYER_SLOTS];
    return type ? type->describe(l->impl, slots, out_shape) : 0;
}

typedef struct {
    TernaryLayer layer;
    const TernaryActivation *act;
    size_t in_shape[TERNARYAI_MAX_DIMS];
    size_t in_nd;
} ResolvedLayer;

/* path[0] = capa del modelo, path[1] = sub-capa dentro de esa residual, ... Calcula tambien la forma de su entrada. */
static int resolve_path(const TernaryAIModel *m, const int *path, int depth, ResolvedLayer *out) {
    if (!m || !path || depth < 1) return 0;
    size_t shape[TERNARYAI_MAX_DIMS], next[TERNARYAI_MAX_DIMS];
    size_t nd = m->in_ndims;
    copy_shape(shape, m->in_shape, nd);
    TernaryResidual *cur = NULL;     /* NULL: nivel del modelo */
    for (int k = 0; k < depth; k++) {
        const int idx = path[k];
        const size_t count = cur ? ternary_residual_num_layers(cur) : ternary_stack_num_layers(m->stack);
        if (idx < 0 || (size_t)idx >= count) return 0;
        for (int i = 0; i < idx; i++) {
            TernaryLayer l = cur ? ternary_residual_layer(cur, (size_t)i) : ternary_stack_layer(m->stack, (size_t)i);
            size_t nn = layer_out_shape(&l, shape, nd, next);
            if (nn == 0) return 0;
            copy_shape(shape, next, nn); nd = nn;
        }
        TernaryLayer l = cur ? ternary_residual_layer(cur, (size_t)idx) : ternary_stack_layer(m->stack, (size_t)idx);
        if (k == depth - 1) {
            out->layer = l;
            out->act = cur ? ternary_residual_activation(cur, (size_t)idx) : ternary_stack_activation(m->stack, (size_t)idx);
            out->in_nd = nd;
            copy_shape(out->in_shape, shape, nd);
            return out->act != NULL;
        }
        if (strcmp(l.vt->type_name, "residual") != 0) return 0;   /* solo una residual tiene sub-capas */
        cur = (TernaryResidual *)l.impl;
    }
    return 0;
}

int ternaryai_model_layer_info_at(const TernaryAIModel *m, const int *path, int depth, char *type_name, double *params,
                                  int *num_slots, char *activation, size_t *out_shape, int *out_ndims) {
    ResolvedLayer r;
    if (!resolve_path(m, path, depth, &r)) return 0;
    snprintf(activation, TERNARYAI_NAME_CAP, "%s", r.act->name);
    if (strcmp(r.layer.vt->type_name, "residual") == 0) {
        snprintf(type_name, TERNARYAI_NAME_CAP, "residual");
        *num_slots = 0;
        copy_shape(out_shape, r.in_shape, r.in_nd);
        *out_ndims = (int)r.in_nd;
        return 1;
    }
    const TernaryLayerType *type = ternary_layer_type_by_name(r.layer.vt->type_name);
    if (!type) return 0;
    size_t nd = type->describe(r.layer.impl, params, out_shape);
    snprintf(type_name, TERNARYAI_NAME_CAP, "%s", type->type_name);
    *num_slots = (int)ternary_layer_type_num_slots(type);
    *out_ndims = (int)nd;
    return 1;
}

int ternaryai_model_layer_children(const TernaryAIModel *m, const int *path, int depth) {
    ResolvedLayer r;
    if (!resolve_path(m, path, depth, &r)) return -1;
    if (strcmp(r.layer.vt->type_name, "residual") != 0) return -1;
    return (int)ternary_residual_num_layers((const TernaryResidual *)r.layer.impl);
}

static int ffn_stats_walk(TernaryLayer l, uint64_t *zeros, uint64_t *total, int max, int reset, int idx) {
    if (strcmp(l.vt->type_name, "residual") == 0) {
        const TernaryResidual *r = (const TernaryResidual *)l.impl;
        for (size_t i = 0; i < ternary_residual_num_layers(r); i++)
            idx = ffn_stats_walk(ternary_residual_layer(r, i), zeros, total, max, reset, idx);
    } else if (strcmp(l.vt->type_name, "swiglu") == 0) {
        uint64_t z, t;
        ternary_swiglu_zero_stats((const TernarySwiGLU *)l.impl, &z, &t, reset);
        if (idx < max) { if (zeros) zeros[idx] = z; if (total) total[idx] = t; }
        idx++;
    }
    return idx;
}

static const TernaryGQA *gqa_walk(TernaryLayer l, int target, int *idx) {
    if (strcmp(l.vt->type_name, "residual") == 0) {
        const TernaryResidual *r = (const TernaryResidual *)l.impl;
        for (size_t i = 0; i < ternary_residual_num_layers(r); i++) {
            const TernaryGQA *g = gqa_walk(ternary_residual_layer(r, i), target, idx);
            if (g) return g;
        }
    } else if (strcmp(l.vt->type_name, "gqa") == 0) {
        if ((*idx)++ == target) return (const TernaryGQA *)l.impl;
    }
    return NULL;
}

int ternaryai_model_attention_probs(const TernaryAIModel *m, int gqa_index, int head, float *out, size_t cap) {
    if (!m) return -1;
    int idx = 0;
    if (gqa_index < 0) {                                   /* cuantas capas GQA tiene */
        for (size_t i = 0; i < ternary_stack_num_layers(m->stack); i++) (void)gqa_walk(ternary_stack_layer(m->stack, i), -1, &idx);
        return idx;
    }
    const TernaryGQA *g = NULL;
    for (size_t i = 0; i < ternary_stack_num_layers(m->stack) && !g; i++) g = gqa_walk(ternary_stack_layer(m->stack, i), gqa_index, &idx);
    if (!g || head < 0 || cap < g->seq_len * g->seq_len) return -1;
    return ternary_gqa_attention_probs(g, (size_t)head, out);
}

void ternaryai_set_threads(int n) { ternary_set_num_threads(n); }
int ternaryai_get_threads(void) { return ternary_get_num_threads(); }
void ternaryai_prof_enable(int on) { ternary_prof_enable(on); }
void ternaryai_prof_reset(void) { ternary_prof_reset(); }
size_t ternaryai_prof_dump_json(char *buf, size_t cap) { return ternary_prof_dump_json(buf, cap); }

int ternaryai_model_ffn_zero_stats(const TernaryAIModel *m, uint64_t *zeros, uint64_t *total, int max, int reset) {
    if (!m) return -1;
    int idx = 0;
    for (size_t i = 0; i < ternary_stack_num_layers(m->stack); i++)
        idx = ffn_stats_walk(ternary_stack_layer(m->stack, i), zeros, total, max, reset, idx);
    return idx;
}

int ternaryai_model_layer_info(const TernaryAIModel *m, int i, char *type_name, double *params,
                            int *num_slots, char *activation, size_t *out_shape, int *out_ndims) {
    return ternaryai_model_layer_info_at(m, &i, 1, type_name, params, num_slots, activation, out_shape, out_ndims);
}

/* --- entrenamiento e inferencia --- */
/* fit()/predict() de este nivel solo saben alimentar entradas ternarias
 * (int8 -1/0/1). Una red cuya primera capa espera ids (Embedding) se
 * puede armar y probar con ternary_stack_* directamente, pero todavia
 * no tiene entrada por esta API -- se rechaza en vez de dejar que la
 * capa lea un buffer de ids que nadie le paso. */
static int first_layer_takes_ternary(const TernaryAIModel *m) {
    TernaryLayer first = ternary_stack_layer(m->stack, 0);
    return first.vt->input_dtype(first.impl) == TERNARY_DTYPE_TERNARY;
}

static int check_x(const int8_t *x, size_t count) {
    for (size_t i = 0; i < count; i++)
        if (x[i] < -1 || x[i] > 1) return 0;
    return 1;
}

static TernaryVec **make_inputs(const int8_t *x, size_t n, size_t features) {
    TernaryVec **xs = (TernaryVec **)calloc(n, sizeof(TernaryVec *));
    if (!xs) return NULL;
    for (size_t s = 0; s < n; s++) {
        xs[s] = ternary_vec_alloc(features);
        if (!xs[s]) {
            for (size_t j = 0; j < s; j++) ternary_vec_free(xs[j]);
            free(xs);
            return NULL;
        }
        const int8_t *row = x + s * features;
        for (size_t i = 0; i < features; i++) ternary_vec_set(xs[s], i, row[i]);
    }
    return xs;
}

static void free_inputs(TernaryVec **xs, size_t n) {
    if (!xs) return;
    for (size_t s = 0; s < n; s++) ternary_vec_free(xs[s]);
    free(xs);
}

/* Orden de las muestras en la epoca global 'epoch': una permutacion determinista que depende SOLO de
 * (seed, epoch), no de como se reparta el fit en trozos (verbose) ni de llamadas anteriores. splitmix64 +
 * Fisher-Yates con rechazo (sin sesgo de modulo). */
static uint64_t splitmix64(uint64_t *st) {
    uint64_t z = (*st += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static void batch_order(size_t *idx, size_t n, uint64_t seed, size_t epoch) {
    uint64_t st = seed ^ ((uint64_t)(epoch + 1) * 0xD1B54A32D192ED03ULL);
    (void)splitmix64(&st);
    for (size_t i = 0; i < n; i++) idx[i] = i;
    for (size_t i = n; i > 1; i--) {
        uint64_t lim = UINT64_MAX - (UINT64_MAX % (uint64_t)i), r;
        do { r = splitmix64(&st); } while (r >= lim);
        size_t j = (size_t)(r % (uint64_t)i), t = idx[i - 1];
        idx[i - 1] = idx[j]; idx[j] = t;
    }
}
/* Tamano efectivo del mini-lote: 0 o >= n significa lote completo (n). */
static size_t effective_batch(const TernaryAIModel *m, size_t n) {
    return (m->batch_size == 0 || m->batch_size >= n) ? n : m->batch_size;
}

/* Entrenamiento por bloques de muestras (ver ternary_stack.h). Tamano del
 * bloque: TERNARYAI_BLOCK (0 = desactivado), 32 por defecto. */
#define TERNARYAI_DEFAULT_TRAIN_BLOCK 32
static size_t train_block_size(void) {
    const char *v = getenv("TERNARYAI_BLOCK");
    if (v && *v) { long b = strtol(v, NULL, 10); return b > 0 ? (size_t)b : 0; }
    return TERNARYAI_DEFAULT_TRAIN_BLOCK;
}
typedef struct { const float *y; size_t out_len; float loss_norm; double sq; const size_t *idx; } MseCtx;
static void mse_loss_grad(size_t sample, const float *out, float *grad, void *ctx_) {
    MseCtx *c = (MseCtx *)ctx_;
    const float *target = c->y + (c->idx ? c->idx[sample] : sample) * c->out_len;
    for (size_t j = 0; j < c->out_len; j++) {
        float diff = out[j] - target[j];
        grad[j] = 2.0f * c->loss_norm * diff;
        c->sq += (double)diff * (double)diff;
    }
}

/* Entrada uint8: la primera capa lee filas uint8 (una escala y un zero-point por fila de 'need' valores). Cada
 * muestra float se cuantiza UNA vez antes de entrenar (el modelo ve siempre la misma cuantizacion). */
static TernaryU8 **make_inputs_u8(const TernaryAIModel *m, const float *x, size_t n, size_t features,
                                  char *err, size_t err_cap) {
    TernaryLayer first = ternary_stack_layer(m->stack, 0);
    const size_t need = first.vt->accepts_u8 ? first.vt->accepts_u8(first.impl) : 0;
    if (first.vt->input_dtype(first.impl) != TERNARY_DTYPE_TERNARY || need == 0 || features % need != 0) {
        set_err(err, err_cap, "the first layer does not accept uint8 input (use the ternary int8 input instead)");
        return NULL;
    }
    for (size_t i = 0; i < n * features; i++)
        if (!isfinite(x[i])) { set_err(err, err_cap, "x must only contain finite numbers"); return NULL; }
    TernaryU8 **us = (TernaryU8 **)calloc(n, sizeof(TernaryU8 *));
    if (!us) { set_err(err, err_cap, "out of memory"); return NULL; }
    for (size_t s = 0; s < n; s++) {
        us[s] = ternary_u8_alloc(features / need, need);
        if (!us[s]) {
            for (size_t j = 0; j < s; j++) ternary_u8_free(us[j]);
            free(us); set_err(err, err_cap, "out of memory"); return NULL;
        }
        ternary_u8_quantize(us[s], x + s * features);
    }
    return us;
}
static void free_inputs_u8(TernaryU8 **us, size_t n) {
    if (!us) return;
    for (size_t s = 0; s < n; s++) ternary_u8_free(us[s]);
    free(us);
}

static int train_impl(TernaryAIModel *m, const int8_t *x, const float *xf, const float *y, size_t n, size_t first_epoch,
                      size_t count, const double *hyper, double *loss_out, char *err, size_t err_cap) {
    if (!m || (!x && !xf) || !y || !hyper || !loss_out || n == 0 || count == 0) {
        set_err(err, err_cap, "invalid arguments");
        return 0;
    }
    if (!model_closed(m, err, err_cap)) return 0;
    if (ternary_stack_num_layers(m->stack) == 0) { set_err(err, err_cap, "the model has no layers"); return 0; }
    if (!first_layer_takes_ternary(m)) {
        set_err(err, err_cap, "the first layer expects token ids: use ternaryai_model_train_ids / ternaryai_model_predict_ids");
        return 0;
    }
    for (int h = 0; h < TERNARYAI_NUM_HYPER; h++) {
        if (!isfinite(hyper[h]) || hyper[h] <= 0.0) {
            set_err(err, err_cap, "hyperparameter %d must be a positive number", h);
            return 0;
        }
    }
    size_t features = ternary_stack_input_len(m->stack);
    size_t out_len = ternary_stack_output_len(m->stack);
    if (x && !check_x(x, n * features)) { set_err(err, err_cap, "x must only contain -1, 0 and 1"); return 0; }

    TernaryVec **xs = x ? make_inputs(x, n, features) : NULL;
    TernaryU8 **us = NULL;
    if (xf) { us = make_inputs_u8(m, xf, n, features, err, err_cap); if (!us) return 0; }
    float *out = (float *)malloc(out_len * sizeof(float));
    float *grad = (float *)malloc(out_len * sizeof(float));
    if ((x && !xs) || !out || !grad) {
        free_inputs(xs, n); free_inputs_u8(us, n); free(out); free(grad);
        set_err(err, err_cap, "out of memory");
        return 0;
    }
    /* Vista tensor reutilizable sobre cada xs[s] -- ternary_stack_forward/
     * backward ahora piden un TernaryTensor (ver ternary_tensor.h); 'dims'
     * se reserva UNA vez aca, no por muestra (ver ternary_tensor_rebind_ternary). */
    size_t x_dims[1] = { features };
    TernaryTensor *x_view = us ? ternary_tensor_view_u8(x_dims, 1, us[0]) : ternary_tensor_view_ternary(x_dims, 1, NULL);
    if (!x_view) {
        free_inputs(xs, n); free_inputs_u8(us, n); free(out); free(grad);
        set_err(err, err_cap, "out of memory");
        return 0;
    }

    const float scale_lr = (float)hyper[TERNARYAI_H_SCALE_LR];

    /* Mini-lotes: B = n (o batch_size 0) es el lote completo de siempre, sin permutar (mismo orden de sumas,
     * mismos bits). Con B < n cada epoca recorre una permutacion de las muestras en pasos de B (el ultimo
     * puede ser mas chico); cada paso hace su propio zero/acumular/apply. El calendario (umbral, k) sigue
     * siendo POR EPOCA: todos los pasos de una epoca usan el mismo piso y k. */
    const size_t B = effective_batch(m, n);
    const int minib = B < n;
    size_t *order = (minib && m->batch_shuffle) ? (size_t *)malloc(n * sizeof(size_t)) : NULL;
    TernaryVec **bxs = minib ? (TernaryVec **)malloc(B * sizeof(TernaryVec *)) : NULL;
    size_t *bidx = minib ? (size_t *)malloc(B * sizeof(size_t)) : NULL;
    if (minib && ((m->batch_shuffle && !order) || !bxs || !bidx)) {
        free(order); free(bxs); free(bidx);
        ternary_tensor_free(x_view); free_inputs(xs, n); free_inputs_u8(us, n); free(out); free(grad);
        set_err(err, err_cap, "out of memory");
        return 0;
    }

    size_t block_S = us ? 0 : train_block_size();
    pthread_mutex_lock(&g_compute_lock);
    for (size_t e = 0; e < count; e++) {
        if (order) batch_order(order, n, m->batch_seed, first_epoch + e);

        double sq = 0.0;
        for (size_t b0 = 0; b0 < n; b0 += B) {
            const size_t bn = (n - b0 < B) ? (n - b0) : B;
            const float loss_norm = 1.0f / ((float)bn * (float)out_len);
            TernaryVec *const *bx = xs ? xs + b0 : NULL;      /* lote completo: xs tal cual, indices = posicion */
            const size_t *map = NULL;
            if (minib) {
                for (size_t j = 0; j < bn; j++) {
                    size_t sidx = order ? order[b0 + j] : b0 + j;
                    bidx[j] = sidx; bxs[j] = xs ? xs[sidx] : NULL;
                }
                bx = bxs; map = bidx;
            }
            ternary_stack_zero_gradients(m->stack);
            int done = 0;
            if (block_S > 0) {
                MseCtx mc = { y, out_len, loss_norm, 0.0, map };
                if (ternary_stack_train_epoch_blocked(m->stack, bx, bn, block_S, m->caps, mse_loss_grad, &mc)) {
                    sq += mc.sq;
                    done = 1;
                } else {
                    block_S = 0; /* no bloqueable (o sin memoria): camino de siempre */
                }
            }
            if (!done) {
                for (size_t j = 0; j < bn; j++) {
                    if (us) ternary_tensor_rebind_u8(x_view, us[map ? map[j] : b0 + j]);
                    else ternary_tensor_rebind_ternary(x_view, bx[j]);
                    ternary_stack_forward(m->stack, x_view, out, m->caps);
                    const float *target = y + (map ? map[j] : b0 + j) * out_len;
                    for (size_t q = 0; q < out_len; q++) {
                        float diff = out[q] - target[q];
                        grad[q] = 2.0f * loss_norm * diff;
                        sq += (double)diff * (double)diff;
                    }
                    ternary_stack_backward(m->stack, x_view, grad);
                }
            }
            ternary_stack_apply_gradients(m->stack, scale_lr);
        }
        loss_out[e] = sq / ((double)n * (double)out_len);
    }
    pthread_mutex_unlock(&g_compute_lock);

    free(order); free(bxs); free(bidx);
    ternary_tensor_free(x_view);
    free_inputs(xs, n); free_inputs_u8(us, n); free(out); free(grad);
    return 1;
}

int ternaryai_model_train(TernaryAIModel *m, const int8_t *x, const float *y, size_t n, size_t first_epoch,
                          size_t count, const double *hyper, double *loss_out, char *err, size_t err_cap) {
    if (!x) { set_err(err, err_cap, "invalid arguments"); return 0; }
    return train_impl(m, x, NULL, y, n, first_epoch, count, hyper, loss_out, err, err_cap);
}

int ternaryai_model_train_f32(TernaryAIModel *m, const float *x, const float *y, size_t n, size_t first_epoch,
                              size_t count, const double *hyper, double *loss_out, char *err, size_t err_cap) {
    if (!x) { set_err(err, err_cap, "invalid arguments"); return 0; }
    return train_impl(m, NULL, x, y, n, first_epoch, count, hyper, loss_out, err, err_cap);
}

int ternaryai_model_predict(TernaryAIModel *m, const int8_t *x, size_t n, float *out, char *err, size_t err_cap) {
    if (!m || !x || !out || n == 0) { set_err(err, err_cap, "invalid arguments"); return 0; }
    if (!model_closed(m, err, err_cap)) return 0;
    if (ternary_stack_num_layers(m->stack) == 0) { set_err(err, err_cap, "the model has no layers"); return 0; }
    if (!first_layer_takes_ternary(m)) {
        set_err(err, err_cap, "the first layer expects token ids: use ternaryai_model_train_ids / ternaryai_model_predict_ids");
        return 0;
    }
    size_t features = ternary_stack_input_len(m->stack);
    size_t out_len = ternary_stack_output_len(m->stack);
    if (!check_x(x, n * features)) { set_err(err, err_cap, "x must only contain -1, 0 and 1"); return 0; }
    if (train_block_size() > 0 && n > 1) {
        TernaryVec **bxs = make_inputs(x, n, features);
        if (bxs) {
            pthread_mutex_lock(&g_compute_lock);
            int done = ternary_stack_predict_blocked(m->stack, bxs, n, train_block_size(), m->caps, out);
            pthread_mutex_unlock(&g_compute_lock);
            free_inputs(bxs, n);
            if (done) return 1;
        }
    }
    TernaryVec *v = ternary_vec_alloc(features);
    if (!v) { set_err(err, err_cap, "out of memory"); return 0; }
    size_t x_dims[1] = { features };
    TernaryTensor *x_view = ternary_tensor_view_ternary(x_dims, 1, v);
    if (!x_view) { ternary_vec_free(v); set_err(err, err_cap, "out of memory"); return 0; }

    pthread_mutex_lock(&g_compute_lock);
    for (size_t s = 0; s < n; s++) {
        const int8_t *row = x + s * features;
        for (size_t i = 0; i < features; i++) ternary_vec_set(v, i, row[i]);
        ternary_stack_forward(m->stack, x_view, out + s * out_len, m->caps);
    }
    pthread_mutex_unlock(&g_compute_lock);
    ternary_tensor_free(x_view);
    ternary_vec_free(v);
    return 1;
}

int ternaryai_model_predict_f32(TernaryAIModel *m, const float *x, size_t n, float *out, char *err, size_t err_cap) {
    if (!m || !x || !out || n == 0) { set_err(err, err_cap, "invalid arguments"); return 0; }
    if (!model_closed(m, err, err_cap)) return 0;
    if (ternary_stack_num_layers(m->stack) == 0) { set_err(err, err_cap, "the model has no layers"); return 0; }
    if (!first_layer_takes_ternary(m)) {
        set_err(err, err_cap, "the first layer expects token ids: use ternaryai_model_train_ids / ternaryai_model_predict_ids");
        return 0;
    }
    const size_t features = ternary_stack_input_len(m->stack), out_len = ternary_stack_output_len(m->stack);
    TernaryU8 **us = make_inputs_u8(m, x, n, features, err, err_cap);
    if (!us) return 0;
    size_t x_dims[1] = { features };
    TernaryTensor *x_view = ternary_tensor_view_u8(x_dims, 1, us[0]);
    if (!x_view) { free_inputs_u8(us, n); set_err(err, err_cap, "out of memory"); return 0; }
    pthread_mutex_lock(&g_compute_lock);
    for (size_t s = 0; s < n; s++) {
        ternary_tensor_rebind_u8(x_view, us[s]);
        ternary_stack_forward(m->stack, x_view, out + s * out_len, m->caps);
    }
    pthread_mutex_unlock(&g_compute_lock);
    ternary_tensor_free(x_view);
    free_inputs_u8(us, n);
    return 1;
}

/* --- entrenamiento e inferencia con ids de token --- */
int ternaryai_model_takes_ids(const TernaryAIModel *m) {
    if (!m || ternary_stack_num_layers(m->stack) == 0) return 0;
    TernaryLayer first = ternary_stack_layer(m->stack, 0);
    return first.vt->input_dtype(first.impl) == TERNARY_DTYPE_ID;
}

static int check_hyper(const double *hyper, char *err, size_t err_cap) {
    for (int h = 0; h < TERNARYAI_NUM_HYPER; h++) {
        if (!isfinite(hyper[h]) || hyper[h] <= 0.0) {
            set_err(err, err_cap, "hyperparameter %d must be a positive number", h);
            return 0;
        }
    }
    return 1;
}

/* Un modelo de lenguaje: primera capa con ids, salida (seq, vocab) con
 * seq = largo de la entrada. Devuelve 1 y fija seq/vocab si lo es. */
static int lm_geometry(const TernaryAIModel *m, size_t *seq, size_t *vocab, char *err, size_t err_cap) {
    if (ternary_stack_num_layers(m->stack) == 0) { set_err(err, err_cap, "the model has no layers"); return 0; }
    TernaryLayer first = ternary_stack_layer(m->stack, 0);
    if (first.vt->input_dtype(first.impl) != TERNARY_DTYPE_ID) {
        set_err(err, err_cap, "the first layer does not take token ids: use ternaryai_model_train / ternaryai_model_predict");
        return 0;
    }
    size_t in_len = ternary_stack_input_len(m->stack), out_len = ternary_stack_output_len(m->stack);
    if (in_len == 0 || out_len % in_len != 0 || out_len / in_len < 2) {
        set_err(err, err_cap, "the model must output one vector of logits per input position "
                              "(output length %zu is not a multiple >= 2 of the input length %zu)", out_len, in_len);
        return 0;
    }
    *seq = in_len;
    *vocab = out_len / in_len;
    return 1;
}

/* Embedding: un id fuera de [0, vocab) se trata como relleno (vector cero), asi que
 * aqui solo se rechaza lo que ninguna capa podria interpretar. */
static int check_ids(const int32_t *ids, size_t count, size_t in_vocab, char *err, size_t err_cap) {
    for (size_t i = 0; i < count; i++) {
        if (ids[i] >= 0 && (size_t)ids[i] >= in_vocab) {
            set_err(err, err_cap, "ids[%zu] = %d is outside the embedding vocabulary (%zu)", i, (int)ids[i], in_vocab);
            return 0;
        }
    }
    return 1;
}

static size_t embedding_vocab(const TernaryAIModel *m) {
    TernaryLayer first = ternary_stack_layer(m->stack, 0);
    const TernaryLayerType *type = ternary_layer_type_by_name(first.vt->type_name);
    if (!type) return 0;
    double slots[TERNARY_MAX_LAYER_SLOTS];
    size_t shape[TERNARYAI_MAX_DIMS];
    type->describe(first.impl, slots, shape);
    return (strcmp(type->type_name, "embedding") == 0) ? (size_t)slots[0] : (size_t)-1;   /* -1: sin tope conocido */
}

int ternaryai_model_train_ids(TernaryAIModel *m, const int32_t *ids, const int32_t *targets, size_t n,
                              size_t first_epoch, size_t count, const double *hyper,
                              double *loss_out, double *acc_out, char *err, size_t err_cap) {
    if (!m || !ids || !targets || !hyper || !loss_out || n == 0 || count == 0) {
        set_err(err, err_cap, "invalid arguments");
        return 0;
    }
    if (!model_closed(m, err, err_cap)) return 0;
    size_t seq, vocab;
    if (!lm_geometry(m, &seq, &vocab, err, err_cap)) return 0;
    if (!check_hyper(hyper, err, err_cap)) return 0;
    TPROF_BEGIN("fit_validate");
    if (!check_ids(ids, n * seq, embedding_vocab(m), err, err_cap)) { TPROF_END(); return 0; }
    size_t valid_total = 0;
    for (size_t i = 0; i < n * seq; i++) {
        if (targets[i] >= 0 && (size_t)targets[i] >= vocab) {
            set_err(err, err_cap, "targets[%zu] = %d is outside the output vocabulary (%zu); use a negative value to ignore a position",
                    i, (int)targets[i], vocab);
            TPROF_END();
            return 0;
        }
        if (targets[i] >= 0) valid_total++;
    }
    TPROF_END();
    if (valid_total == 0) { set_err(err, err_cap, "every target is negative (ignored): nothing to learn from"); return 0; }

    /* Con la cabeza lm_head al final no se materializan los logits rows*vocab (ni su gradiente). */
    const int fused = ternary_stack_xent_fusable(m->stack);
    float *out = fused ? NULL : (float *)malloc(seq * vocab * sizeof(float));
    float *grad = fused ? NULL : (float *)malloc(seq * vocab * sizeof(float));
    size_t dims[TERNARYAI_MAX_DIMS];
    size_t nd = ternary_stack_input_shape(m->stack, dims);
    TernaryTensor *x_view = ternary_tensor_view_id(dims, nd, NULL);
    if ((!fused && (!out || !grad)) || !x_view) {
        free(out); free(grad); ternary_tensor_free(x_view);
        set_err(err, err_cap, "out of memory");
        return 0;
    }
    const float scale_lr = (float)hyper[TERNARYAI_H_SCALE_LR];

    /* Mini-lotes (ver el comentario en ternaryai_model_train): B = n es el lote completo de siempre (sin
     * permutar: mismos bits). Con B < n cada epoca recorre una permutacion de las muestras en pasos de B; el
     * gradiente de cada paso se promedia sobre SU lote (1/bn), y cada paso aplica sus votos. La perdida y
     * la accuracy de la epoca son la media sobre todas las muestras tal como se fueron viendo. */
    const size_t B = effective_batch(m, n);
    const int minib = B < n;
    size_t *order = (minib && m->batch_shuffle) ? (size_t *)malloc(n * sizeof(size_t)) : NULL;
    if (minib && m->batch_shuffle && !order) {
        free(out); free(grad); ternary_tensor_free(x_view);
        set_err(err, err_cap, "out of memory");
        return 0;
    }

    /* Por defecto, al terminar vuelve al estado de menor perdida de entrenamiento (evita el colapso tardio
       del valle del umbral). Desactivable con ternaryai_model_set_restore_best(m,0) o TERNARYAI_RESTORE_BEST=0. */
    const char *rb_env = getenv("TERNARYAI_RESTORE_BEST");
    const int restore_best = !m->restore_best_off && !(rb_env && rb_env[0] == '0');
    /* Con "retener" activo el mejor estado vive en el modelo y sobrevive entre llamadas (ver hold_best). */
    const int hold = restore_best && m->hold_active;
    TernaryStackSnapshot local_best = { NULL, 0 };
    TernaryStackSnapshot *bestp = hold ? &m->hold_best : &local_best;
    TernaryStackSnapshot cand = { NULL, 0 };   /* mini-lotes: estado al INICIO de la epoca en curso */
    double best_loss = hold ? m->hold_best_loss : 1e300;
    pthread_mutex_lock(&g_compute_lock);
    for (size_t e = 0; e < count; e++) {
        TPROF_BEGIN("epoch");
        TPROF_BEGIN("batch_order");
        if (order) batch_order(order, n, m->batch_seed, first_epoch + e);
        TPROF_END();
        /* Con mini-lotes la perdida de la epoca se mide a lo largo de varios estados distintos; el estado al que
         * se le atribuye es el del inicio de la epoca (con B = n coincide con el de siempre: el que produjo la
         * perdida, antes de aplicar). Se guarda siempre y solo pasa a ser "el mejor" si la perdida mejora. */
        TPROF_BEGIN("snapshot_start_of_epoch");
        if (minib && restore_best) (void)ternary_stack_snapshot(m->stack, &cand);
        TPROF_END();
        double loss = 0.0;
        size_t valid = 0, correct = 0;
        for (size_t b0 = 0; b0 < n; b0 += B) {
            const size_t bn = (n - b0 < B) ? (n - b0) : B;
            const float inv_b = 1.0f / (float)bn;
            ternary_stack_zero_gradients(m->stack);
            for (size_t j = 0; j < bn; j++) {
                const size_t s = order ? order[b0 + j] : b0 + j;
                int32_t *row = (int32_t *)(uintptr_t)(ids + s * seq);   /* la capa solo lee */
                ternary_tensor_rebind_id(x_view, row);
                size_t nv = 0, nc = 0;
                if (fused) {
                    TPROF_BEGIN("sample");
                    loss += ternary_stack_train_xent(m->stack, x_view, m->caps, targets + s * seq, inv_b, &nv, &nc);
                    TPROF_END();
                    valid += nv; correct += nc;
                    continue;
                }
                ternary_stack_forward(m->stack, x_view, out, m->caps);
                /* media por muestra (sobre sus posiciones validas) y luego media sobre la tanda,
                 * igual que tests/test_residual.c */
                double l = ternary_xent(out, seq, vocab, targets + s * seq, grad, &nv, &nc);
                loss += l; valid += nv; correct += nc;
                for (size_t i = 0; i < seq * vocab; i++) grad[i] *= inv_b;
                ternary_stack_backward(m->stack, x_view, grad);
            }
            if (!minib && restore_best && loss / (double)n < best_loss) {   /* estado que produjo esta perdida (antes de aplicar) */
                TPROF_BEGIN("snapshot_best");
                if (ternary_stack_snapshot(m->stack, bestp)) best_loss = loss / (double)n;
                TPROF_END();
            }
            ternary_stack_apply_gradients(m->stack, scale_lr);
        }
        loss_out[e] = loss / (double)n;
        TPROF_BEGIN("snapshot_swap_best");
        if (minib && restore_best && cand.buf && loss_out[e] < best_loss) {
            ternary_stack_snapshot_free(bestp);
            *bestp = cand; cand.buf = NULL; cand.size = 0;      /* el estado de inicio de esta epoca pasa a ser el mejor */
            best_loss = loss_out[e];
        }
        TPROF_END();
        if (acc_out) acc_out[e] = valid ? (double)correct / (double)valid : 0.0;
        TPROF_END();   /* epoch */
    }
    ternary_stack_snapshot_free(&cand);
    free(order);
    if (hold) {
        m->hold_best_loss = best_loss;   /* se restaura una sola vez al terminar el fit */
    } else {
        if (restore_best && bestp->buf && loss_out[count - 1] > best_loss) ternary_stack_restore(m->stack, bestp);
        ternary_stack_snapshot_free(bestp);
    }
    pthread_mutex_unlock(&g_compute_lock);

    ternary_tensor_free(x_view);
    free(out); free(grad);
    return 1;
}

int ternaryai_model_predict_ids(TernaryAIModel *m, const int32_t *ids, size_t n, float *out,
                                char *err, size_t err_cap) {
    if (!m || !ids || !out || n == 0) { set_err(err, err_cap, "invalid arguments"); return 0; }
    if (!model_closed(m, err, err_cap)) return 0;
    size_t seq, vocab;
    if (!lm_geometry(m, &seq, &vocab, err, err_cap)) return 0;
    if (!check_ids(ids, n * seq, embedding_vocab(m), err, err_cap)) return 0;
    size_t dims[TERNARYAI_MAX_DIMS];
    size_t nd = ternary_stack_input_shape(m->stack, dims);
    TernaryTensor *x_view = ternary_tensor_view_id(dims, nd, NULL);
    if (!x_view) { set_err(err, err_cap, "out of memory"); return 0; }
    pthread_mutex_lock(&g_compute_lock);
    for (size_t s = 0; s < n; s++) {
        ternary_tensor_rebind_id(x_view, (int32_t *)(uintptr_t)(ids + s * seq));
        ternary_stack_forward(m->stack, x_view, out + s * seq * vocab, m->caps);
    }
    pthread_mutex_unlock(&g_compute_lock);
    ternary_tensor_free(x_view);
    return 1;
}

/* --- persistencia --- */
int ternaryai_model_save(const TernaryAIModel *m, const char *path, int full_checkpoint, char *err, size_t err_cap) {
    if (!m || !path) { set_err(err, err_cap, "invalid arguments"); return 0; }
    if (!model_closed(m, err, err_cap)) return 0;
    if (ternary_stack_num_layers(m->stack) == 0) { set_err(err, err_cap, "the model has no layers"); return 0; }
    if (!ternary_stack_save(m->stack, path, full_checkpoint ? 1 : 0)) {
        set_err(err, err_cap, "could not write the model folder '%s' (the parent folder must exist and be writable)", path);
        return 0;
    }
    return 1;
}

TernaryAIModel *ternaryai_model_load(const char *path, char *err, size_t err_cap) {
    if (!path) { set_err(err, err_cap, "invalid arguments"); return NULL; }
    char why[320] = "";
    TernaryStack *s = ternary_stack_load_ex(path, why, sizeof why);
    if (!s || ternary_stack_num_layers(s) == 0) {
        ternary_stack_free(s);
        set_err(err, err_cap, "cannot read '%s': %s", path, why[0] ? why : "missing, corrupt, or an unknown layer/activation type");
        return NULL;
    }
    TernaryAIModel *m = (TernaryAIModel *)calloc(1, sizeof(TernaryAIModel));
    if (!m) { ternary_stack_free(s); set_err(err, err_cap, "out of memory"); return NULL; }
    m->stack = s;
    ternary_rng_seed(&m->rng, 0);
    m->in_ndims = ternary_stack_input_shape(s, m->in_shape);

    size_t shape[TERNARYAI_MAX_DIMS], next[TERNARYAI_MAX_DIMS];
    size_t nd = m->in_ndims;
    copy_shape(shape, m->in_shape, nd);
    for (size_t i = 0; i < ternary_stack_num_layers(s) && nd > 0; i++) {
        TernaryLayer l = ternary_stack_layer(s, i);
        nd = layer_out_shape(&l, shape, nd, next);
        if (nd) copy_shape(shape, next, nd);
    }
    m->cur_ndims = nd;
    copy_shape(m->cur_shape, shape, nd);
    if (m->cur_ndims == 0) {
        ternary_stack_free(s); free(m);
        set_err(err, err_cap, "cannot describe the layers of '%s'", path);
        return NULL;
    }
    m->caps = caps_acquire();
    return m;
}

/* ===================================================================
 * Tokenizador BPE por bytes -- API plana sobre ternary_tokenizer.h.
 * Wrapper delgado: el handle publico solo guarda el puntero interno
 * para no exponer el struct de ternary_tokenizer.c directamente.
 * =================================================================== */
struct TernaryAITokenizer {
    TernaryTokenizer *tok;
};

TernaryAITokenizer *ternaryai_tokenizer_create(size_t vocab_size, char *err, size_t err_cap) {
    TernaryTokenizer *tok = ternary_tokenizer_create(vocab_size, err, err_cap);
    if (!tok) return NULL;
    TernaryAITokenizer *h = malloc(sizeof(TernaryAITokenizer));
    if (!h) { ternary_tokenizer_free(tok); set_err(err, err_cap, "sin memoria"); return NULL; }
    h->tok = tok;
    return h;
}

int ternaryai_tokenizer_train(TernaryAITokenizer *t, const uint8_t *corpus, size_t corpus_len,
                               char *err, size_t err_cap) {
    if (!t) { set_err(err, err_cap, "tokenizador nulo"); return 0; }
    return ternary_tokenizer_train(t->tok, corpus, corpus_len, err, err_cap);
}

size_t ternaryai_tokenizer_vocab_size(const TernaryAITokenizer *t) {
    return t ? ternary_tokenizer_vocab_size(t->tok) : 0;
}

size_t ternaryai_tokenizer_num_merges(const TernaryAITokenizer *t) {
    return t ? ternary_tokenizer_num_merges(t->tok) : 0;
}

int ternaryai_tokenizer_encode(const TernaryAITokenizer *t, const uint8_t *text, size_t text_len,
                                int32_t *out_ids, size_t out_cap, size_t *out_len,
                                char *err, size_t err_cap) {
    if (!t) { set_err(err, err_cap, "tokenizador nulo"); return 0; }
    return ternary_tokenizer_encode(t->tok, text, text_len, out_ids, out_cap, out_len, err, err_cap);
}

int ternaryai_tokenizer_decode(const TernaryAITokenizer *t, const int32_t *ids, size_t num_ids,
                                uint8_t *out_bytes, size_t out_cap, size_t *out_len,
                                char *err, size_t err_cap) {
    if (!t) { set_err(err, err_cap, "tokenizador nulo"); return 0; }
    return ternary_tokenizer_decode(t->tok, ids, num_ids, out_bytes, out_cap, out_len, err, err_cap);
}

int ternaryai_tokenizer_add_special(TernaryAITokenizer *t, const uint8_t *text, size_t len, char *err, size_t err_cap) {
    if (!t) { set_err(err, err_cap, "tokenizador nulo"); return 0; }
    return ternary_tokenizer_add_special(t->tok, text, len, err, err_cap);
}

size_t ternaryai_tokenizer_num_special(const TernaryAITokenizer *t) {
    return t ? ternary_tokenizer_num_special(t->tok) : 0;
}

int ternaryai_tokenizer_special_text(const TernaryAITokenizer *t, size_t index, uint8_t *out, size_t out_cap,
                                     size_t *out_len, char *err, size_t err_cap) {
    if (!t) { set_err(err, err_cap, "tokenizador nulo"); return 0; }
    return ternary_tokenizer_special_text(t->tok, index, out, out_cap, out_len, err, err_cap);
}

int32_t ternaryai_tokenizer_special_id(const TernaryAITokenizer *t, const uint8_t *text, size_t len) {
    return t ? ternary_tokenizer_special_id(t->tok, text, len) : -1;
}

int ternaryai_tokenizer_encode_ex(const TernaryAITokenizer *t, const uint8_t *text, size_t text_len,
                                  int parse_special, int32_t *out_ids, size_t out_cap, size_t *out_len,
                                  char *err, size_t err_cap) {
    if (!t) { set_err(err, err_cap, "tokenizador nulo"); return 0; }
    return ternary_tokenizer_encode_ex(t->tok, text, text_len, parse_special, out_ids, out_cap, out_len, err, err_cap);
}

int ternaryai_tokenizer_decode_ex(const TernaryAITokenizer *t, const int32_t *ids, size_t num_ids,
                                  int skip_special, uint8_t *out_bytes, size_t out_cap, size_t *out_len,
                                  char *err, size_t err_cap) {
    if (!t) { set_err(err, err_cap, "tokenizador nulo"); return 0; }
    return ternary_tokenizer_decode_ex(t->tok, ids, num_ids, skip_special, out_bytes, out_cap, out_len, err, err_cap);
}

int ternaryai_tokenizer_save(const TernaryAITokenizer *t, const char *path, char *err, size_t err_cap) {
    if (!t) { set_err(err, err_cap, "tokenizador nulo"); return 0; }
    return ternary_tokenizer_save(t->tok, path, err, err_cap);
}

TernaryAITokenizer *ternaryai_tokenizer_load(const char *path, char *err, size_t err_cap) {
    TernaryTokenizer *tok = ternary_tokenizer_load(path, err, err_cap);
    if (!tok) return NULL;
    TernaryAITokenizer *h = malloc(sizeof(TernaryAITokenizer));
    if (!h) { ternary_tokenizer_free(tok); set_err(err, err_cap, "sin memoria"); return NULL; }
    h->tok = tok;
    return h;
}

void ternaryai_tokenizer_free(TernaryAITokenizer *t) {
    if (!t) return;
    ternary_tokenizer_free(t->tok);
    free(t);
}

/* ===================================================================
 * Carga de datasets de entrenamiento -- API plana sobre ternary_dataset.h.
 * Wrapper de una linea: no hay estado propio que guardar (a diferencia
 * del tokenizador), asi que no hace falta un handle opaco.
 * =================================================================== */
int ternaryai_dataset_load(const char *path, int format, int strict,
                            uint8_t *out_buf, size_t out_cap, size_t *out_len, size_t *out_count,
                            size_t *out_skipped, char *err, size_t err_cap) {
    return ternary_dataset_load(path, format, strict, out_buf, out_cap, out_len, out_count,
                                 out_skipped, err, err_cap);
}

int ternaryai_dataset_targets(const int32_t *ids, const size_t *offsets, size_t n_examples,
                              int32_t special_lo, int32_t special_hi, const int8_t *kind, int unmarked_train,
                              int32_t *y, char *err, size_t err_cap) {
#define TGT_FAIL(msg) do { if (err && err_cap) { strncpy(err, msg, err_cap - 1); err[err_cap - 1] = 0; } return 0; } while (0)
    if (!offsets || (n_examples && (!ids || !y))) TGT_FAIL("argumentos nulos");
    for (size_t e = 0; e < n_examples; e++) if (offsets[e + 1] < offsets[e]) TGT_FAIL("offsets decrecientes");
    if (special_lo < special_hi) {
        if (!kind || special_lo < 0) TGT_FAIL("rango de especiales invalido");
        for (int32_t k = 0; k < special_hi - special_lo; k++) if (kind[k] < 0 || kind[k] > 3) TGT_FAIL("kind fuera de 0..3");
    }
    TernaryTargetsCfg cfg = { special_lo, special_hi, kind, unmarked_train ? 1 : 0 };
    return ternary_targets_build_batch(ids, offsets, n_examples, &cfg, y);
#undef TGT_FAIL
}

int ternaryai_dataset_load_ex(const char *path, int format, int strict, int flags,
                              uint8_t *out_buf, size_t out_cap, size_t *out_len, size_t *out_count,
                              size_t *out_skipped, char *err, size_t err_cap) {
    return ternary_dataset_load_ex(path, format, strict, flags, out_buf, out_cap, out_len, out_count, out_skipped, err, err_cap);
}
