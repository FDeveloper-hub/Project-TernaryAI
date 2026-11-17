#include "ternary_mem.h"
#include "ternary_threadpool.h"
#define _POSIX_C_SOURCE 200809L   /* open_memstream / fmemopen para las instantaneas en memoria */
#include "ternary_prof.h"
#include "ternary_stack.h"
#include "ternary_head.h"
#include <errno.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
    TernaryLayer layer;
    TernaryActivation act;
    int gshared;           /* grad_x_buf es uno de los dos buffers compartidos del stack (no se libera aqui) */
    int ws_user;           /* su estado vive en el espacio de trabajo compartido (checkpoint) */

    float *out;            /* longitud output_len(), salida escalada cruda */
    int32_t *out_raw;      /* longitud output_len(), Z sin escalar (para backward) */
    TernaryU8 *act_u8;     /* salida cuantizada a uint8 (una escala/zero-point por fila de output_row_len): solo si la SIGUIENTE capa la
                            * lee como TERNARY (la ultima capa y las que alimentan a una capa FLOAT no cuantizan) */

    /* Vista reutilizable de la salida de ESTA capa para pasarsela a la
     * SIGUIENTE como su 'x' (ver ternary_layer.h: input_dtype()). Se
     * crea UNA vez, al agregar la capa siguiente (recien ahi se sabe
     * que dtype hace falta), y en cada forward() solo se reapunta a
     * act_u8 (la siguiente lee TERNARY: activacion uint8) o a 'out' (dtype FLOAT, sin cuantizar) --
     * nunca se vuelve a pedir memoria en el camino caliente. NULL en la
     * ULTIMA capa (no hay siguiente a la que alimentar). */
    TernaryTensor *out_tensor;

    /* Gradientes: un juego por MATRIZ de la capa (ver num_matrices en
     * ternary_layer.h). Las capas de una sola matriz (Dense, Conv2D,
     * Embedding) tienen nm == 1 y se comportan exactamente como antes. */
    size_t nm;
    float **grad_w_scratch; /* [nm], cada uno rows*row_width, se SOBREESCRIBE cada backward() */
    float **grad_w_acc;     /* [nm], cada uno rows*row_width, se ACUMULA durante la epoca */
    float **grad_bias_acc;  /* [nm], cada uno rows, o NULL si esa matriz no tiene bias */
    float *grad_scale_acc;  /* [nm] */
    float *grad_x_buf;     /* longitud input_len(), NULL si !propagates_grad_x */
} TernaryStackEntry;

struct TernaryStack {
    TernaryStackEntry *entries;
    size_t n, cap;
    size_t in_shape[TERNARY_MAX_SHAPE_DIMS];
    size_t in_ndims;       /* 0 = sin forma fijada (se usa {input_len}) */

    /* Checkpoint por bloque (ternary_stack_set_checkpoint) */
    int ckpt, ckpt_env_checked;
    unsigned char *ws_raw, *ws;       /* espacio de trabajo compartido por las capas con estado grande */
    float *gbuf[2];                   /* gradiente de la entrada: dos buffers que alternan entre todas las capas */
    size_t ws_owner;                  /* capa cuyo estado esta hoy en el espacio de trabajo (n = ninguna) */
    TernaryCapabilities caps;         /* los del ultimo forward: el backward los usa para recalcular */
    TernaryUpdateRule rule;           /* regla de actualizacion de pesos (R2 por defecto, ver ternary_dense.h) */
};

void ternary_stack_set_update_rule(TernaryStack *s, float tau, float beta) {
    if (!s) return;
    s->rule.tau = tau; s->rule.beta = beta;
}
void ternary_stack_set_r2_params(TernaryStack *s, float inc, float zero_target, float reset) {
    if (!s) return;
    s->rule.inc = inc; s->rule.zero_target = zero_target; s->rule.reset = reset;
}
void ternary_stack_set_head_beta2(TernaryStack *s, float head_beta2) {
    if (s) s->rule.head_beta2 = head_beta2;
}
TernaryUpdateRule ternary_stack_get_update_rule(const TernaryStack *s) {
    TernaryUpdateRule z = TERNARY_R2_RULE_DEFAULT;
    return s ? s->rule : z;
}

TernaryStack *ternary_stack_create(void) {
    TernaryStack *s = (TernaryStack *)ternary_calloc64(1, sizeof(TernaryStack));
    if (s) { const TernaryUpdateRule d = TERNARY_R2_RULE_DEFAULT; s->rule = d; }   /* default: regla R2 */
    return s;
}

static void grads_free(TernaryStackEntry *e) {
    for (size_t m = 0; m < e->nm; m++) {
        if (e->grad_w_scratch) free(e->grad_w_scratch[m]);
        if (e->grad_w_acc) free(e->grad_w_acc[m]);
        if (e->grad_bias_acc) free(e->grad_bias_acc[m]);
    }
    free(e->grad_w_scratch); free(e->grad_w_acc); free(e->grad_bias_acc); free(e->grad_scale_acc);
    if (!e->gshared) free(e->grad_x_buf);
}

static void entry_free(TernaryStackEntry *e) {
    if (!e->layer.impl) return;
    e->layer.vt->destroy(e->layer.impl);
    free(e->out); free(e->out_raw);
    if (e->act_u8) ternary_u8_free(e->act_u8);
    if (e->out_tensor) ternary_tensor_free(e->out_tensor);
    grads_free(e);
    memset(e, 0, sizeof(*e));
}

void ternary_stack_free(TernaryStack *s) {
    if (!s) return;
    for (size_t i = 0; i < s->n; i++) entry_free(&s->entries[i]);
    free(s->ws_raw); free(s->gbuf[0]); free(s->gbuf[1]);
    free(s->entries);
    free(s);
}

int ternary_stack_add(TernaryStack *s, TernaryLayer layer, TernaryActivation act) {
    if (!s || !layer.impl || !layer.vt) return 0;
    if (s->ckpt) { ternary_diag("ternary_stack_add: con checkpoint activo ya no se pueden agregar capas.\n"); return 0; }

    size_t in_len = layer.vt->input_len(layer.impl);
    if (s->n == 0 && s->in_ndims > 0) {
        size_t prod = 1;
        for (size_t i = 0; i < s->in_ndims; i++) prod *= s->in_shape[i];
        if (prod != in_len) {
            ternary_diag("ternary_stack_add: la primera capa espera entrada de largo %zu pero la "
                            "forma de entrada fijada tiene %zu elementos.\n", in_len, prod);
            return 0;
        }
    }
    if (s->n > 0) {
        size_t prev_out = s->entries[s->n - 1].layer.vt->output_len(s->entries[s->n - 1].layer.impl);
        if (prev_out != in_len) {
            ternary_diag("ternary_stack_add: la capa nueva espera entrada de largo %zu pero "
                            "la capa anterior entrega %zu -- arquitectura invalida, no se agrega.\n",
                    in_len, prev_out);
            return 0;
        }
        if (!layer.vt->propagates_grad_x(layer.impl)) {
            ternary_diag("ternary_stack_add: la capa '%s' no propaga gradiente hacia su entrada "
                            "y no es la primera capa de la red -- las capas anteriores se quedarian "
                            "sin gradiente para siempre. No se agrega.\n", layer.vt->type_name);
            return 0;
        }
        if (layer.vt->input_dtype(layer.impl) == TERNARY_DTYPE_TERNARY) {
            /* la activacion entre capas es uint8: la capa nueva debe leerla, con filas del largo que la anterior entrega */
            const TernaryStackEntry *pe = &s->entries[s->n - 1];
            const size_t need = layer.vt->accepts_u8 ? layer.vt->accepts_u8(layer.impl) : 0;
            const size_t prow = pe->layer.vt->output_row_len ? pe->layer.vt->output_row_len(pe->layer.impl) : prev_out;
            if (need == 0) {
                ternary_diag("ternary_stack_add: la capa '%s' no acepta entrada uint8 (la activacion entre capas es uint8). No se agrega.\n", layer.vt->type_name);
                return 0;
            }
            if (need != prow && need != prev_out) {
                ternary_diag("ternary_stack_add: la capa '%s' espera filas uint8 de %zu valores (la anterior entrega filas de %zu). No se agrega.\n",
                             layer.vt->type_name, need, prow);
                return 0;
            }
        }
        if (layer.vt->input_dtype(layer.impl) == TERNARY_DTYPE_ID) {
            ternary_diag("ternary_stack_add: la capa '%s' espera ids como entrada (dtype ID), pero "
                            "eso solo tiene sentido como PRIMERA capa de la red -- ninguna capa produce "
                            "ids como salida. No se agrega.\n", layer.vt->type_name);
            return 0;
        }
    }

    if (s->n == s->cap) {
        size_t new_cap = s->cap ? s->cap * 2 : 4;
        TernaryStackEntry *ne = (TernaryStackEntry *)realloc(s->entries, new_cap * sizeof(TernaryStackEntry));
        if (!ne) return 0;
        s->entries = ne; s->cap = new_cap;
    }

    TernaryStackEntry *e = &s->entries[s->n];
    memset(e, 0, sizeof(*e));
    e->layer = layer;
    e->act = act;

    size_t out_len = layer.vt->output_len(layer.impl);
    size_t nm = layer.vt->num_matrices(layer.impl);
    int propagates = layer.vt->propagates_grad_x(layer.impl);

    /* La cabeza de salida (rows*vocab) reserva out/out_raw recien al primer forward normal: el entrenamiento
     * fusionado (ternary_stack_train_xent) nunca los necesita. */
    const int lazy_out = strcmp(layer.vt->type_name, "lm_head") == 0;
    e->out = lazy_out ? NULL : (float *)ternary_malloc64(out_len * sizeof(float));
    const int no_raw = strcmp(layer.vt->type_name, "rmsnorm") == 0;   /* RMSNorm no tiene Z entero (el backward no lo lee) */
    e->out_raw = (lazy_out || no_raw) ? NULL : (int32_t *)ternary_malloc64(out_len * sizeof(int32_t));
    e->grad_x_buf = propagates ? (float *)ternary_malloc64(in_len * sizeof(float)) : NULL;
    e->nm = nm;
    e->grad_w_scratch = (float **)ternary_calloc64(nm ? nm : 1, sizeof(float *));
    e->grad_w_acc = (float **)ternary_calloc64(nm ? nm : 1, sizeof(float *));
    e->grad_bias_acc = (float **)ternary_calloc64(nm ? nm : 1, sizeof(float *));
    e->grad_scale_acc = (float *)ternary_calloc64(nm ? nm : 1, sizeof(float));

    int ok = nm >= 1 && (lazy_out || (e->out && (e->out_raw || no_raw))) && (!propagates || e->grad_x_buf) &&
             e->grad_w_scratch && e->grad_w_acc && e->grad_bias_acc && e->grad_scale_acc;
    for (size_t m = 0; ok && m < nm; m++) {
        size_t rows = layer.vt->grad_rows(layer.impl, m);
        size_t row_w = layer.vt->grad_row_width(layer.impl, m);
        e->grad_w_scratch[m] = (float *)ternary_malloc64(rows * row_w * sizeof(float));
        e->grad_w_acc[m] = (float *)ternary_malloc64(rows * row_w * sizeof(float));
        if (!e->grad_w_scratch[m] || !e->grad_w_acc[m]) ok = 0;
        if (layer.vt->has_bias(layer.impl, m)) {
            e->grad_bias_acc[m] = (float *)ternary_calloc64(rows, sizeof(float));
            if (!e->grad_bias_acc[m]) ok = 0;
        }
    }
    if (!ok) {
        e->layer.impl = NULL;       /* el llamador sigue siendo dueno de la capa */
        e->layer.vt = NULL;
        free(e->out); free(e->out_raw); grads_free(e);
        memset(e, 0, sizeof(*e));
        return 0;
    }

    if (s->n > 0) {
        /* Recien ahora se sabe que dtype de entrada quiere ESTA capa
         * nueva -- crear la vista de salida de la capa ANTERIOR con ese
         * dtype (ver el comentario grande junto a 'out_tensor' arriba). */
        TernaryStackEntry *prev = &s->entries[s->n - 1];
        size_t prev_out_len = prev->layer.vt->output_len(prev->layer.impl);
        size_t dims1[1] = { prev_out_len };
        TernaryDType want = layer.vt->input_dtype(layer.impl);
        if (want == TERNARY_DTYPE_FLOAT) {
            prev->out_tensor = ternary_tensor_view_float(dims1, 1, prev->out);
        } else {
            const size_t prow = layer.vt->accepts_u8(layer.impl);      /* filas del largo que la capa nueva lee */
            prev->act_u8 = (prow > 0 && prev_out_len % prow == 0) ? ternary_u8_alloc(prev_out_len / prow, prow) : NULL;
            if (prev->act_u8) prev->out_tensor = ternary_tensor_view_u8(dims1, 1, prev->act_u8);
        }
        if (!prev->out_tensor) { if (prev->act_u8) { ternary_u8_free(prev->act_u8); prev->act_u8 = NULL; } free(e->out); free(e->out_raw); grads_free(e); memset(e, 0, sizeof(*e)); return 0; }
    }
    s->n++;
    return 1;
}

size_t ternary_stack_num_layers(const TernaryStack *s) { return s ? s->n : 0; }

int ternary_stack_set_input_shape(TernaryStack *s, const size_t *dims, size_t ndims) {
    if (!s || !dims || ndims < 1 || ndims > TERNARY_MAX_SHAPE_DIMS) return 0;
    size_t prod = 1;
    for (size_t i = 0; i < ndims; i++) {
        if (dims[i] == 0) return 0;
        prod *= dims[i];
    }
    if (s->n > 0 && prod != ternary_stack_input_len(s)) return 0;
    for (size_t i = 0; i < ndims; i++) s->in_shape[i] = dims[i];
    s->in_ndims = ndims;
    return 1;
}

size_t ternary_stack_input_shape(const TernaryStack *s, size_t *dims) {
    if (!s || s->n == 0) return 0;
    if (s->in_ndims > 0) {
        for (size_t i = 0; i < s->in_ndims; i++) dims[i] = s->in_shape[i];
        return s->in_ndims;
    }
    dims[0] = ternary_stack_input_len(s);
    return 1;
}

TernaryLayer ternary_stack_layer(const TernaryStack *s, size_t i) {
    TernaryLayer none = { NULL, NULL };
    return (s && i < s->n) ? s->entries[i].layer : none;
}

const TernaryActivation *ternary_stack_activation(const TernaryStack *s, size_t i) {
    return (s && i < s->n) ? &s->entries[i].act : NULL;
}

size_t ternary_stack_input_len(const TernaryStack *s) {
    if (!s || s->n == 0) return 0;
    return s->entries[0].layer.vt->input_len(s->entries[0].layer.impl);
}

size_t ternary_stack_output_len(const TernaryStack *s) {
    if (!s || s->n == 0) return 0;
    const TernaryStackEntry *last = &s->entries[s->n - 1];
    return last->layer.vt->output_len(last->layer.impl);
}

int ternary_stack_set_checkpoint(TernaryStack *s, int on) {
    if (!s || s->n == 0) return -1;
    if (!on) return s->ckpt ? -1 : 0;          /* no se deshace */
    if (s->ckpt) return 0;
    size_t need = 0, gneed = 0;
    for (size_t i = 0; i < s->n; i++) {
        const TernaryLayerVTable *vt = s->entries[i].layer.vt; void *impl = s->entries[i].layer.impl;
        if (vt->state_bytes && vt->bind_state) { const size_t b = vt->state_bytes(impl); if (b > need) need = b; }
        if (s->entries[i].grad_x_buf) { const size_t l = vt->input_len(impl); if (l > gneed) gneed = l; }
    }
    float *g0 = gneed ? (float *)ternary_malloc64(gneed * sizeof(float)) : NULL, *g1 = gneed ? (float *)ternary_malloc64(gneed * sizeof(float)) : NULL;
    unsigned char *raw = need ? (unsigned char *)ternary_calloc64(need + 64, 1) : NULL;   /* paginas sin tocar hasta usarlas */
    if ((gneed && (!g0 || !g1)) || (need && !raw)) { free(g0); free(g1); free(raw); return -1; }
    s->gbuf[0] = g0; s->gbuf[1] = g1; s->ws_raw = raw;
    s->ws = raw ? (unsigned char *)(((uintptr_t)raw + 63u) & ~(uintptr_t)63u) : NULL;
    for (size_t i = 0; i < s->n; i++) {
        TernaryStackEntry *e = &s->entries[i];
        if (e->layer.vt->state_bytes && e->layer.vt->bind_state && e->layer.vt->state_bytes(e->layer.impl) > 0) {
            e->layer.vt->bind_state(e->layer.impl, s->ws);
            e->ws_user = 1;
            if (strcmp(e->layer.vt->type_name, "residual") == 0) { free(e->out_raw); e->out_raw = NULL; }   /* sin Z crudo con sentido */
        }
        if (e->grad_x_buf) { free(e->grad_x_buf); e->grad_x_buf = s->gbuf[i & 1]; e->gshared = 1; }
    }
    s->ws_owner = s->n;
    s->ckpt = 1;
    return 0;
}

static int entry_ensure_out(TernaryStackEntry *e) {
    const size_t n = e->layer.vt->output_len(e->layer.impl);
    if (!e->out) e->out = (float *)ternary_malloc64(n * sizeof(float));
    if (!e->out_raw && strcmp(e->layer.vt->type_name, "residual") != 0 && strcmp(e->layer.vt->type_name, "rmsnorm") != 0) e->out_raw = (int32_t *)ternary_malloc64(n * sizeof(int32_t));
    return e->out != NULL;
}

static void stack_check_env(TernaryStack *s) {
    if (!s->ckpt_env_checked) {
        s->ckpt_env_checked = 1;
        const char *env = getenv("TERNARYAI_CHECKPOINT");
        if (env && env[0] && env[0] != '0') ternary_stack_set_checkpoint(s, 1);
    }
}

/* Forward de las primeras 'count' capas. */
static void stack_forward_upto(TernaryStack *s, const TernaryTensor *x, size_t count, const TernaryCapabilities *caps) {
    if (s->ckpt) { if (caps) s->caps = *caps; else memset(&s->caps, 0, sizeof(s->caps)); }
    const TernaryTensor *cur_in = x;
    TPROF_BEGIN("forward");
    for (size_t i = 0; i < count; i++) {
        TernaryStackEntry *e = &s->entries[i];
        if (!entry_ensure_out(e)) { TPROF_END(); return; }
        TPROF_BEGIN_I(e->layer.vt->type_name, i);
        e->layer.vt->forward(e->layer.impl, cur_in, e->out, e->out_raw, caps);
        TPROF_END();
        /* La activacion de ESTA capa siempre se calcula (aunque la
         * siguiente pida FLOAT y no la use) -- ternary_stack_last_activated()
         * depende de que exista siempre, igual que antes de este cambio. */
        if (e->act_u8) { TPROF_BEGIN_I("quantize_u8", i); e->act.apply_u8(e->out, e->act_u8); TPROF_END(); }
        if (e->out_tensor) cur_in = e->out_tensor;  /* NULL solo en la ULTIMA capa */
        if (e->ws_user) s->ws_owner = i;
    }
    TPROF_END();
}

void ternary_stack_forward(TernaryStack *s, const TernaryTensor *x, float *out_final,
                            const TernaryCapabilities *caps) {
    stack_check_env(s);
    stack_forward_upto(s, x, s->n, caps);
    size_t out_len = ternary_stack_output_len(s);
    memcpy(out_final, s->entries[s->n - 1].out, out_len * sizeof(float));
}


void ternary_stack_zero_gradients(TernaryStack *s) {
    TPROF_BEGIN("zero_gradients");
    for (size_t i = 0; i < s->n; i++) {
        TernaryStackEntry *e = &s->entries[i];
        for (size_t m = 0; m < e->nm; m++) {
            size_t rows = e->layer.vt->grad_rows(e->layer.impl, m);
            size_t row_w = e->layer.vt->grad_row_width(e->layer.impl, m);
            ternary_par_zero(e->grad_w_acc[m], rows * row_w * sizeof(float));
            if (e->grad_bias_acc[m]) memset(e->grad_bias_acc[m], 0, rows * sizeof(float));
            e->grad_scale_acc[m] = 0.0f;
        }
    }
    TPROF_END();
}

static void stack_backward_from(TernaryStack *s, const TernaryTensor *x, size_t top, const float *grad_out) {
    TPROF_BEGIN("backward");
    for (size_t idx = top; idx-- > 0; ) {
        TernaryStackEntry *e = &s->entries[idx];
        const TernaryTensor *layer_input = (idx == 0) ? x : s->entries[idx - 1].out_tensor;

        if (e->ws_user && s->ws_owner != idx) {      /* checkpoint: su estado ya no esta, se recalcula el forward */
            TPROF_BEGIN_I("recompute_forward", idx);
            e->layer.vt->forward(e->layer.impl, layer_input, e->out, e->out_raw, &s->caps);
            TPROF_END();
            s->ws_owner = idx;
        }
        TPROF_BEGIN_I(e->layer.vt->type_name, idx);
        if (e->grad_x_buf) ternary_par_zero(e->grad_x_buf, e->layer.vt->input_len(e->layer.impl) * sizeof(float));

        if (e->layer.vt->backward_accumulate) {
            /* Fusionado: el gradiente de pesos se acumula directo en el
             * acumulador de la epoca (mismos bits: el producto es exacto y
             * la suma por muestra conserva el orden). */
            e->layer.vt->backward_accumulate(e->layer.impl, layer_input, grad_out,
                                              e->grad_w_acc, e->grad_x_buf,
                                              e->grad_scale_acc, e->grad_bias_acc, e->out_raw);
        } else {
            e->layer.vt->backward(e->layer.impl, layer_input, grad_out,
                                   e->grad_w_scratch, e->grad_x_buf,
                                   e->grad_scale_acc, e->grad_bias_acc, e->out_raw);
            for (size_t m = 0; m < e->nm; m++) {
                size_t n_w = e->layer.vt->grad_rows(e->layer.impl, m) * e->layer.vt->grad_row_width(e->layer.impl, m);
                float *acc = e->grad_w_acc[m];
                const float *src = e->grad_w_scratch[m];
                for (size_t k = 0; k < n_w; k++) acc[k] += src[k];
            }
        }

        TPROF_END();
        grad_out = e->grad_x_buf; /* NULL si esta capa no propaga -- solo valido si idx==0 (nadie mas la usa) */
    }
    TPROF_END();
}

void ternary_stack_backward(TernaryStack *s, const TernaryTensor *x, const float *grad_final) {
    stack_backward_from(s, x, s->n, grad_final);
}

int ternary_stack_xent_fusable(const TernaryStack *s) {
    if (!s || s->n == 0) return 0;
    const char *env = getenv("TERNARYAI_NO_FUSED_XENT");
    if (env && env[0] && env[0] != '0') return 0;
    const TernaryStackEntry *e = &s->entries[s->n - 1];
    return strcmp(e->layer.vt->type_name, "lm_head") == 0 && e->grad_w_acc && e->grad_w_scratch && e->grad_x_buf;
}

double ternary_stack_train_xent(TernaryStack *s, const TernaryTensor *x, const TernaryCapabilities *caps,
                                const int32_t *targets, float inv_n, size_t *n_valid, size_t *n_correct) {
    stack_check_env(s);
    const size_t top = s->n - 1;
    TernaryStackEntry *e = &s->entries[top];
    TernaryHead *h = (TernaryHead *)e->layer.impl;
    TPROF_BEGIN("train_xent");
    stack_forward_upto(s, x, top, caps);
    const TernaryTensor *hin = (top == 0) ? x : s->entries[top - 1].out_tensor;
    TPROF_BEGIN_I("head_xent", top);
    if (e->grad_x_buf) ternary_par_zero(e->grad_x_buf, e->layer.vt->input_len(e->layer.impl) * sizeof(float));
    double l = hin->dtype == TERNARY_DTYPE_UINT8
        ? ternary_head_train_xent_u8(h, hin->data_u8, targets, inv_n, e->grad_w_scratch[0], e->grad_x_buf,
                                     e->grad_scale_acc, n_valid, n_correct)
        : ternary_head_train_xent(h, hin->data_t, targets, inv_n, e->grad_w_scratch[0], e->grad_x_buf,
                                  e->grad_scale_acc, n_valid, n_correct);
    if (l != l) { TPROF_END(); TPROF_END(); return l; }    /* NAN: target fuera de rango, sin tocar acumuladores */
    TPROF_END();
    TPROF_BEGIN_I("head_grad_acc", top);
    const size_t n_w = e->layer.vt->grad_rows(e->layer.impl, 0) * e->layer.vt->grad_row_width(e->layer.impl, 0);
    float *acc = e->grad_w_acc[0]; const float *src = e->grad_w_scratch[0];
    for (size_t k = 0; k < n_w; k++) acc[k] += src[k];
    TPROF_END();
    stack_backward_from(s, x, top, e->grad_x_buf);
    TPROF_END();
    return l;
}

int ternary_stack_blockable(const TernaryStack *s) {
    if (!s || s->n != 1) return 0;   /* entre capas la activacion es uint8: el camino por bloques ternario solo sirve con UNA capa */
    for (size_t i = 0; i < s->n; i++) {
        const TernaryStackEntry *e = &s->entries[i];
        if (e->act_u8 || !e->layer.vt->backward_block || e->nm != 1 ||
            e->layer.vt->input_dtype(e->layer.impl) != TERNARY_DTYPE_TERNARY) return 0;
    }
    return 1;
}

int ternary_stack_train_epoch_blocked(TernaryStack *s, TernaryVec *const *xs, size_t n, size_t S,
                                      const TernaryCapabilities *caps,
                                      TernaryLossGradFn loss_fn, void *ctx) {
    if (!ternary_stack_blockable(s) || S == 0 || n == 0) return 0;
    if (S > n) S = n;
    const size_t L = s->n;
    int ok = 1;

    /* Almacenamiento por (capa, ranura): Z crudo, activacion cuantizada
     * (entrada de la capa siguiente), vista tensor de esa activacion, y el
     * gradiente que baja (grad_x de la capa l+1 == grad_out de la capa l). */
    int32_t ***raw = (int32_t ***)ternary_calloc64(L, sizeof(*raw));
    float ***bout = (float ***)ternary_calloc64(L, sizeof(*bout));
    TernaryVec ***act = (TernaryVec ***)ternary_calloc64(L, sizeof(*act));
    TernaryTensor ***view = (TernaryTensor ***)ternary_calloc64(L + 1, sizeof(*view)); /* view[l] = entrada de la capa l */
    float ***gdown = (float ***)ternary_calloc64(L + 1, sizeof(*gdown));              /* gdown[l] = grad_out de la capa l */
    float *scratch_in = NULL;
    if (!raw || !bout || !act || !view || !gdown) ok = 0;
    for (size_t l = 0; ok && l < L; l++) {
        TernaryStackEntry *e = &s->entries[l];
        size_t ol = e->layer.vt->output_len(e->layer.impl);
        size_t il = e->layer.vt->input_len(e->layer.impl);
        raw[l] = (int32_t **)ternary_calloc64(S, sizeof(int32_t *));
        bout[l] = (float **)ternary_calloc64(S, sizeof(float *));
        act[l] = (TernaryVec **)ternary_calloc64(S, sizeof(TernaryVec *));
        view[l] = (TernaryTensor **)ternary_calloc64(S, sizeof(TernaryTensor *));
        gdown[l] = (float **)ternary_calloc64(S, sizeof(float *));
        if (!raw[l] || !bout[l] || !act[l] || !view[l] || !gdown[l]) { ok = 0; break; }
        for (size_t j = 0; ok && j < S; j++) {
            raw[l][j] = (int32_t *)ternary_malloc64(ol * sizeof(int32_t));
            bout[l][j] = (float *)ternary_malloc64(ol * sizeof(float));
            if (l + 1 < L) act[l][j] = ternary_vec_alloc(ol);
            /* grad_out de la capa l: la ultima lo recibe de la perdida; las
             * demas, del grad_x de la capa l+1 (buffer de ESTA ranura,
             * del largo de la entrada de l+1 == salida de l). */
            gdown[l][j] = (float *)ternary_malloc64(ol * sizeof(float));
            size_t dims[1] = { il };
            if (l == 0) view[0][j] = ternary_tensor_view_ternary(dims, 1, NULL);
            else        view[l][j] = ternary_tensor_view_ternary(dims, 1, act[l - 1][j]);
            if (!raw[l][j] || !bout[l][j] || (l + 1 < L && !act[l][j]) || !gdown[l][j] || !view[l][j]) ok = 0;
        }
    }
    (void)scratch_in;

    if (ok) {
        for (size_t b0 = 0; b0 < n; b0 += S) {
            size_t m = (n - b0 < S) ? (n - b0) : S;

            /* Forward del bloque capa por capa (cada fila de W se lee una vez para
             * las m muestras), luego perdida muestra por muestra y EN ORDEN. */
            for (size_t j = 0; j < m; j++) ternary_tensor_rebind_ternary(view[0][j], xs[b0 + j]);
            for (size_t l = 0; l < L; l++) {
                TernaryStackEntry *e = &s->entries[l];
                size_t ol = e->layer.vt->output_len(e->layer.impl);
                if (e->layer.vt->forward_block) {
                    const TernaryTensor *xin[m ? m : 1];
                    float *fo[m ? m : 1]; int32_t *fr[m ? m : 1];
                    for (size_t j = 0; j < m; j++) { xin[j] = view[l][j]; fo[j] = bout[l][j]; fr[j] = raw[l][j]; }
                    e->layer.vt->forward_block(e->layer.impl, m, xin, fo, fr, caps);
                } else {
                    for (size_t j = 0; j < m; j++)
                        e->layer.vt->forward(e->layer.impl, view[l][j], bout[l][j], raw[l][j], caps);
                }
                if (l + 1 < L)
                    /* sin activacion ternaria: el camino por bloques solo vale para pilas de UNA capa (ver ternary_stack_blockable) */ (void)ol;
            }
            for (size_t j = 0; j < m; j++) loss_fn(b0 + j, bout[L - 1][j], gdown[L - 1][j], ctx);

            /* Backward por capas, de la ultima a la primera, sobre el bloque. */
            for (size_t l = L; l-- > 0; ) {
                TernaryStackEntry *e = &s->entries[l];
                size_t il = e->layer.vt->input_len(e->layer.impl);
                const float *gout[m ? m : 1]; const int32_t *zr[m ? m : 1];
                const TernaryTensor *xin[m ? m : 1];
                float *gx[m ? m : 1];
                for (size_t j = 0; j < m; j++) {
                    gout[j] = gdown[l][j]; zr[j] = raw[l][j]; xin[j] = view[l][j];
                    gx[j] = (l > 0) ? gdown[l - 1][j] : NULL;
                    if (l > 0) memset(gx[j], 0, il * sizeof(float));
                }
                e->layer.vt->backward_block(e->layer.impl, m, xin, gout, zr,
                                             e->grad_w_acc, l > 0 ? gx : NULL,
                                             e->grad_scale_acc, e->grad_bias_acc);
            }
        }
    }

    for (size_t l = 0; l < L; l++) {
        for (size_t j = 0; j < S; j++) {
            if (raw && raw[l]) free(raw[l][j]);
            if (bout && bout[l]) free(bout[l][j]);
            if (act && act[l] && act[l][j]) ternary_vec_free(act[l][j]);
            if (view && view[l] && view[l][j]) ternary_tensor_free(view[l][j]);
            if (gdown && gdown[l]) free(gdown[l][j]);
        }
        if (raw) free(raw[l]);
        if (bout) free(bout[l]);
        if (act) free(act[l]);
        if (view) free(view[l]);
        if (gdown) free(gdown[l]);
    }
    free(raw); free(bout); free(act); free(view); free(gdown);
    return ok;
}

int ternary_stack_predict_blocked(TernaryStack *s, TernaryVec *const *xs, size_t n, size_t S,
                                  const TernaryCapabilities *caps, float *out) {
    if (!s || s->n != 1 || S == 0 || n == 0) return 0;
    const size_t L = s->n;
    for (size_t l = 0; l < L; l++) {
        const TernaryStackEntry *e = &s->entries[l];
        if (e->act_u8 || !e->layer.vt->forward_block || e->layer.vt->input_dtype(e->layer.impl) != TERNARY_DTYPE_TERNARY) return 0;
    }
    if (S > n) S = n;
    const size_t out_len = ternary_stack_output_len(s);
    int ok = 1;
    int32_t ***raw = (int32_t ***)ternary_calloc64(L, sizeof(*raw));
    float ***bout = (float ***)ternary_calloc64(L, sizeof(*bout));
    TernaryVec ***act = (TernaryVec ***)ternary_calloc64(L, sizeof(*act));
    TernaryTensor ***view = (TernaryTensor ***)ternary_calloc64(L, sizeof(*view));
    if (!raw || !bout || !act || !view) ok = 0;
    for (size_t l = 0; ok && l < L; l++) {
        const TernaryStackEntry *e = &s->entries[l];
        size_t ol = e->layer.vt->output_len(e->layer.impl);
        size_t il = e->layer.vt->input_len(e->layer.impl);
        raw[l] = (int32_t **)ternary_calloc64(S, sizeof(int32_t *));
        bout[l] = (float **)ternary_calloc64(S, sizeof(float *));
        act[l] = (TernaryVec **)ternary_calloc64(S, sizeof(TernaryVec *));
        view[l] = (TernaryTensor **)ternary_calloc64(S, sizeof(TernaryTensor *));
        if (!raw[l] || !bout[l] || !act[l] || !view[l]) { ok = 0; break; }
        for (size_t j = 0; ok && j < S; j++) {
            raw[l][j] = (int32_t *)ternary_malloc64(ol * sizeof(int32_t));
            if (l + 1 < L) { bout[l][j] = (float *)ternary_malloc64(ol * sizeof(float)); act[l][j] = ternary_vec_alloc(ol); }
            size_t dims[1] = { il };
            view[l][j] = ternary_tensor_view_ternary(dims, 1, l == 0 ? NULL : act[l - 1][j]);
            if (!raw[l][j] || (l + 1 < L && (!bout[l][j] || !act[l][j])) || !view[l][j]) ok = 0;
        }
    }
    if (ok) {
        for (size_t b0 = 0; b0 < n; b0 += S) {
            size_t m = (n - b0 < S) ? (n - b0) : S;
            for (size_t j = 0; j < m; j++) ternary_tensor_rebind_ternary(view[0][j], xs[b0 + j]);
            for (size_t l = 0; l < L; l++) {
                TernaryStackEntry *e = &s->entries[l];
                size_t ol = e->layer.vt->output_len(e->layer.impl);
                const TernaryTensor *xin[m];
                float *fo[m]; int32_t *fr[m];
                for (size_t j = 0; j < m; j++) {
                    xin[j] = view[l][j];
                    fo[j] = (l + 1 == L) ? out + (b0 + j) * out_len : bout[l][j];
                    fr[j] = raw[l][j];
                }
                e->layer.vt->forward_block(e->layer.impl, m, xin, fo, fr, caps);
                if (l + 1 < L)
                    /* sin activacion ternaria: el camino por bloques solo vale para pilas de UNA capa (ver ternary_stack_blockable) */ (void)ol;
            }
        }
    }
    for (size_t l = 0; l < L; l++) {
        for (size_t j = 0; j < S; j++) {
            if (raw && raw[l]) free(raw[l][j]);
            if (bout && bout[l]) free(bout[l][j]);
            if (act && act[l] && act[l][j]) ternary_vec_free(act[l][j]);
            if (view && view[l] && view[l][j]) ternary_tensor_free(view[l][j]);
        }
        if (raw) free(raw[l]);
        if (bout) free(bout[l]);
        if (act) free(act[l]);
        if (view) free(view[l]);
    }
    free(raw); free(bout); free(act); free(view);
    return ok;
}

void ternary_stack_apply_gradients(TernaryStack *s, float scale_lr) {
    const TernaryUpdateRule prev_rule = ternary_update_rule_get();   /* la regla del stack rige solo mientras dura esta llamada */
    ternary_update_rule_set(&s->rule);
    TPROF_BEGIN("apply_gradients");
    for (size_t i = 0; i < s->n; i++) {
        TernaryStackEntry *e = &s->entries[i];
        TPROF_BEGIN_I(e->layer.vt->type_name, i);
        for (size_t m = 0; m < e->nm; m++) {
            size_t rows = e->layer.vt->grad_rows(e->layer.impl, m);
            size_t row_w = e->layer.vt->grad_row_width(e->layer.impl, m);
            if (e->layer.vt->apply_matrix_gradients) {
                e->layer.vt->apply_matrix_gradients(e->layer.impl, m, e->grad_w_acc[m]);
            } else {
                for (size_t r = 0; r < rows; r++) {
                    e->layer.vt->apply_row_gradients(e->layer.impl, m, r, e->grad_w_acc[m] + r * row_w);
                }
            }
            if (e->grad_bias_acc[m]) {
                e->layer.vt->apply_bias_gradients(e->layer.impl, m, e->grad_bias_acc[m]);
            }
            e->layer.vt->apply_scale_gradient(e->layer.impl, m, e->grad_scale_acc[m], scale_lr);
        }
        TPROF_END();
    }
    TPROF_END();
    ternary_update_rule_set(&prev_rule);
}

/* ===================================================================
 * Guardado / carga
 * =================================================================== */


/* ===========================================================================================================
 * Guardado / carga: carpeta con config.json + model.ternary (+ training_state.ternary en un checkpoint)
 * =========================================================================================================== */
#define CFG_NAME   "config.json"
#define MODEL_NAME "model.ternary"
#define STATE_NAME "training_state.ternary"

static int path_join(char *out, size_t cap, const char *dir, const char *name) {
    int n = snprintf(out, cap, "%s/%s", dir, name);
    return n > 0 && (size_t)n < cap;
}

/* Arma el texto JSON (compacto) de config.json y llena los escritores. 1 en exito. */
static int stack_export(const TernaryStack *s, TernaryTWriter *model, TernaryTWriter *state, TernaryJBuf *cfg) {
    TernaryExportCtx cx = { model, state };
    ternary_jbuf_open(cfg, '{');
    ternary_jbuf_kstr(cfg, "format", TERNARY_TFILE_FORMAT);
    ternary_jbuf_kint(cfg, "format_version", TERNARY_CONFIG_VERSION);
    ternary_jbuf_kbool(cfg, "checkpoint", state != NULL);
    if (state) {   /* solo el checkpoint de entrenamiento lleva la regla R2; el export de inferencia no */
        ternary_jbuf_key(cfg, "r2");
        ternary_jbuf_open(cfg, '{');
        ternary_jbuf_kfloat(cfg, "tau", s->rule.tau);
        ternary_jbuf_kfloat(cfg, "beta", s->rule.beta);
        ternary_jbuf_kfloat(cfg, "inc", s->rule.inc);
        ternary_jbuf_kfloat(cfg, "zero_target", s->rule.zero_target);
        ternary_jbuf_kfloat(cfg, "reset", s->rule.reset);
        ternary_jbuf_kfloat(cfg, "head_beta2", s->rule.head_beta2);
        ternary_jbuf_close(cfg, '}');
    }
    {
        size_t dims[TERNARY_MAX_SHAPE_DIMS];
        size_t nd = ternary_stack_input_shape(s, dims);
        ternary_jbuf_key(cfg, "input_shape");
        ternary_jbuf_open(cfg, '[');
        for (size_t i = 0; i < nd; i++) ternary_jbuf_int(cfg, (int64_t)dims[i]);
        ternary_jbuf_close(cfg, ']');
    }
    ternary_jbuf_key(cfg, "layers");
    ternary_jbuf_open(cfg, '[');
    for (size_t i = 0; i < s->n; i++) {
        char prefix[32];
        snprintf(prefix, sizeof prefix, "layers.%zu", i);
        if (!ternary_layer_export_entry(s->entries[i].layer, s->entries[i].act.name, &cx, prefix, cfg)) return 0;
    }
    ternary_jbuf_close(cfg, ']');
    ternary_jbuf_close(cfg, '}');
    return !cfg->failed;
}

static void set_msg(char *err, size_t cap, const char *fmt, ...) {
    if (!err || !cap) return;
    va_list ap; va_start(ap, fmt); vsnprintf(err, cap, fmt, ap); va_end(ap);
}

/* Reconstruye la red desde el texto de config.json y los lectores. NULL + mensaje en err si algo no cuadra. */
static TernaryStack *stack_build(const char *cfg_text, size_t cfg_len, const TernaryTReader *model,
                                 const TernaryTReader *state, char *err, size_t err_cap) {
    char jerr[160] = "";
    TernaryJNode *root = ternary_jdom_parse(cfg_text, cfg_len, jerr, sizeof jerr);
    if (!root) { set_msg(err, err_cap, "config.json invalido: %s", jerr); return NULL; }
    TernaryStack *s = NULL;
    const char *fmt = ternary_jdom_get_str(root, "format");
    int64_t ver = 0;
    const TernaryJNode *layers = ternary_jdom_get(root, "layers");
    if (root->type != TJ_OBJ || !fmt || strcmp(fmt, TERNARY_TFILE_FORMAT) != 0) { set_msg(err, err_cap, "config.json: 'format' no es '%s'", TERNARY_TFILE_FORMAT); goto done; }
    if (!ternary_jdom_get_int(root, "format_version", &ver) || ver != TERNARY_CONFIG_VERSION) {
        set_msg(err, err_cap, "config.json: format_version %lld no soportada (esta libreria lee la %d)", (long long)ver, TERNARY_CONFIG_VERSION); goto done;
    }
    if (!layers || layers->type != TJ_ARR || layers->n == 0) { set_msg(err, err_cap, "config.json: 'layers' falta o esta vacio"); goto done; }
    s = ternary_stack_create();
    if (!s) { set_msg(err, err_cap, "sin memoria"); goto done; }
    if (err && err_cap) err[0] = '\0';
    if (state) {   /* checkpoint de entrenamiento: restaura la regla R2 (si falta, quedan los valores por defecto) */
        const TernaryJNode *r2 = ternary_jdom_get(root, "r2");
        if (r2 && r2->type == TJ_OBJ) {
            TernaryUpdateRule r = s->rule; float v;
            if (ternary_jdom_get_float(r2, "tau", &v)) r.tau = v;
            if (ternary_jdom_get_float(r2, "beta", &v)) r.beta = v;
            if (ternary_jdom_get_float(r2, "inc", &v)) r.inc = v;
            if (ternary_jdom_get_float(r2, "zero_target", &v)) r.zero_target = v;
            if (ternary_jdom_get_float(r2, "reset", &v)) r.reset = v;
            if (ternary_jdom_get_float(r2, "head_beta2", &v)) r.head_beta2 = v;   /* checkpoints anteriores no lo traen: queda el valor por defecto */
            if (!(r.tau > 0.0f && r.tau <= 1.0f) || !(r.beta >= 0.0f && r.beta < 1.0f) || !(r.inc > 0.0f) ||
                !(r.reset >= 0.0f && r.reset <= 1.0f) || !(r.zero_target < 1.0f) || !(r.head_beta2 >= 0.0f && r.head_beta2 < 1.0f)) {
                set_msg(err, err_cap, "config.json: parametros 'r2' fuera de rango");
                ternary_stack_free(s); s = NULL; goto done;
            }
            s->rule = r;
        }
    }
    TernaryImportCtx cx = { model, state, err, err_cap };
    for (size_t i = 0; i < layers->n; i++) {
        char prefix[32];
        snprintf(prefix, sizeof prefix, "layers.%zu", i);
        TernaryLayer layer; const TernaryActivation *act;
        if (!ternary_layer_import_entry(&cx, prefix, &layers->kids[i], &layer, &act)) { ternary_stack_free(s); s = NULL; goto done; }
        if (!ternary_stack_add(s, layer, *act)) {
            layer.vt->destroy(layer.impl);
            set_msg(err, err_cap, "%s: la capa no encaja con la anterior (dimensiones o tipo de dato)", prefix);
            ternary_stack_free(s); s = NULL; goto done;
        }
    }
    {
        const TernaryJNode *shp = ternary_jdom_get(root, "input_shape");
        if (shp && shp->type == TJ_ARR && shp->n > 0) {
            size_t dims[TERNARY_MAX_SHAPE_DIMS];
            int ok = shp->n <= TERNARY_MAX_SHAPE_DIMS;
            for (size_t i = 0; ok && i < shp->n; i++) {
                ok = shp->kids[i].is_int && shp->kids[i].i > 0 && shp->kids[i].i <= ((int64_t)1 << 31);
                if (ok) dims[i] = (size_t)shp->kids[i].i;
            }
            if (!ok || !ternary_stack_set_input_shape(s, dims, shp->n)) {
                set_msg(err, err_cap, "config.json: 'input_shape' no coincide con la primera capa");
                ternary_stack_free(s); s = NULL; goto done;
            }
        }
    }
done:
    ternary_jdom_free(root);
    return s;
}

static int is_dir(const char *path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}
static int file_exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

static int write_text_file(const char *path, const char *data, size_t len) {
    size_t pl = strlen(path);
    char *tmp = (char *)ternary_malloc64(pl + 5);
    if (!tmp) return 0;
    memcpy(tmp, path, pl); memcpy(tmp + pl, ".tmp", 5);
    FILE *f = fopen(tmp, "wb");
    if (!f) { free(tmp); return 0; }
    int ok = fwrite(data, 1, len, f) == len;
    if (fclose(f) != 0) ok = 0;
    if (ok && rename(tmp, path) != 0) ok = 0;
    if (!ok) remove(tmp);
    free(tmp);
    return ok;
}

static int read_text_file(const char *path, char **out, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return 0; }
    long sz = ftell(f);
    if (sz < 0 || sz > (1L << 26) || fseek(f, 0, SEEK_SET) != 0) { fclose(f); return 0; }
    char *buf = (char *)ternary_malloc64((size_t)sz + 1);
    if (!buf || fread(buf, 1, (size_t)sz, f) != (size_t)sz) { free(buf); fclose(f); return 0; }
    fclose(f);
    buf[sz] = '\0';
    *out = buf; *out_len = (size_t)sz;
    return 1;
}

int ternary_stack_save(const TernaryStack *s, const char *dir, int full_checkpoint) {
    if (!s || s->n == 0 || !dir || !dir[0]) return 0;
    if (mkdir(dir, 0777) != 0 && !(errno == EEXIST && is_dir(dir))) return 0;
    char pm[1024], ps[1024], pc[1024];
    if (!path_join(pm, sizeof pm, dir, MODEL_NAME) || !path_join(ps, sizeof ps, dir, STATE_NAME) ||
        !path_join(pc, sizeof pc, dir, CFG_NAME)) return 0;
    TernaryTWriter *model = ternary_tw_create(), *state = full_checkpoint ? ternary_tw_create() : NULL;
    TernaryJBuf cfg; ternary_jbuf_init(&cfg);
    int ok = model && (!full_checkpoint || state) && stack_export(s, model, state, &cfg);
    char *pretty = NULL; size_t pretty_len = 0;
    if (ok) { pretty = ternary_json_pretty(cfg.buf, cfg.len, &pretty_len); ok = pretty != NULL; }
    ok = ok && ternary_tw_write_file(model, pm);
    if (ok && state) ok = ternary_tw_write_file(state, ps);
    if (ok && !state && file_exists(ps)) remove(ps);   /* un export no debe dejar el estado de entrenamiento de un checkpoint anterior */
    ok = ok && write_text_file(pc, pretty, pretty_len);
    free(pretty);
    ternary_jbuf_free(&cfg);
    ternary_tw_free(model); ternary_tw_free(state);
    return ok;
}

TernaryStack *ternary_stack_load_ex(const char *dir, char *err, size_t err_cap) {
    if (err && err_cap) err[0] = '\0';
    char pm[1024], ps[1024], pc[1024];
    if (!dir || !path_join(pm, sizeof pm, dir, MODEL_NAME) || !path_join(ps, sizeof ps, dir, STATE_NAME) ||
        !path_join(pc, sizeof pc, dir, CFG_NAME)) { set_msg(err, err_cap, "ruta invalida"); return NULL; }
    if (!is_dir(dir)) { set_msg(err, err_cap, "'%s' no es una carpeta de modelo (se esperaba una carpeta con %s, %s y opcionalmente %s)", dir, CFG_NAME, MODEL_NAME, STATE_NAME); return NULL; }
    char *cfg = NULL; size_t cfg_len = 0;
    if (!read_text_file(pc, &cfg, &cfg_len)) { set_msg(err, err_cap, "no se puede leer '%s'", pc); return NULL; }
    TernaryStack *s = NULL;
    TernaryTReader *model = ternary_tr_open(pm, err, err_cap), *state = NULL;
    if (model) {
        int ok = 1;
        if (file_exists(ps)) { state = ternary_tr_open(ps, err, err_cap); ok = state != NULL; }
        if (ok) s = stack_build(cfg, cfg_len, model, state, err, err_cap);
    }
    ternary_tr_close(model); ternary_tr_close(state);
    free(cfg);
    return s;
}

TernaryStack *ternary_stack_load(const char *dir) {
    char err[320];
    TernaryStack *s = ternary_stack_load_ex(dir, err, sizeof err);
    if (!s) ternary_diag("ternary_stack_load: %s\n", err[0] ? err : "no se pudo cargar el modelo");
    return s;
}

/* ---- instantaneas en memoria (checkpoint completo, con votos) ------------------------------------------------
 * Misma representacion que los archivos: [3 x uint64 LE: largo de config, de model, de state][config][model][state]. */
static void put_u64(uint8_t *p, uint64_t v) { for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i)); }
static uint64_t get_u64(const uint8_t *p) { uint64_t v = 0; for (int i = 0; i < 8; i++) v |= (uint64_t)p[i] << (8 * i); return v; }

int ternary_stack_snapshot(const TernaryStack *s, TernaryStackSnapshot *snap) {
#if defined(_WIN32)
    (void)s; (void)snap; return 0;
#else
    if (!s || !snap || s->n == 0) return 0;
    TernaryTWriter *model = ternary_tw_create(), *state = ternary_tw_create();
    TernaryJBuf cfg; ternary_jbuf_init(&cfg);
    uint8_t *mb = NULL, *sb = NULL; size_t ml = 0, sl = 0;
    int ok = model && state && stack_export(s, model, state, &cfg) &&
             ternary_tw_serialize(model, &mb, &ml) && ternary_tw_serialize(state, &sb, &sl);
    char *buf = NULL;
    if (ok) {
        buf = (char *)ternary_malloc64(24 + cfg.len + ml + sl);
        ok = buf != NULL;
    }
    if (ok) {
        put_u64((uint8_t *)buf, cfg.len); put_u64((uint8_t *)buf + 8, ml); put_u64((uint8_t *)buf + 16, sl);
        memcpy(buf + 24, cfg.buf, cfg.len); memcpy(buf + 24 + cfg.len, mb, ml); memcpy(buf + 24 + cfg.len + ml, sb, sl);
        free(snap->buf);
        snap->buf = buf; snap->size = 24 + cfg.len + ml + sl;
    }
    free(mb); free(sb);
    ternary_jbuf_free(&cfg);
    ternary_tw_free(model); ternary_tw_free(state);
    return ok;
#endif
}

int ternary_stack_restore(TernaryStack *s, const TernaryStackSnapshot *snap) {
#if defined(_WIN32)
    (void)s; (void)snap; return 0;
#else
    if (!s || !snap || !snap->buf || snap->size < 24) return 0;
    const uint8_t *b = (const uint8_t *)snap->buf;
    uint64_t cl = get_u64(b), ml = get_u64(b + 8), sl = get_u64(b + 16);
    if (cl > snap->size || ml > snap->size || sl > snap->size || 24 + cl + ml + sl != snap->size) return 0;
    char err[320] = "";
    TernaryTReader *model = ternary_tr_from_memory(b + 24 + cl, (size_t)ml, err, sizeof err);
    TernaryTReader *state = ternary_tr_from_memory(b + 24 + cl + ml, (size_t)sl, err, sizeof err);
    TernaryStack *t = (model && state) ? stack_build((const char *)b + 24, (size_t)cl, model, state, err, sizeof err) : NULL;
    ternary_tr_close(model); ternary_tr_close(state);
    if (!t) return 0;
    if (t->n != s->n) { ternary_stack_free(t); return 0; }
    /* Se intercambian las capas (mismas formas: los buffers de cada entrada siguen valiendo) y 't' destruye las viejas. */
    for (size_t i = 0; i < s->n; i++) {
        TernaryLayer tmp = s->entries[i].layer;
        s->entries[i].layer = t->entries[i].layer;
        t->entries[i].layer = tmp;
        if (s->entries[i].ws_user) s->entries[i].layer.vt->bind_state(s->entries[i].layer.impl, s->ws);   /* checkpoint: la capa nueva vuelve al espacio compartido */
    }
    ternary_stack_free(t);
    if (s->ckpt) s->ws_owner = s->n;   /* el estado del espacio de trabajo ya no es de nadie: el backward recalcula */
    return 1;
#endif
}

void ternary_stack_snapshot_free(TernaryStackSnapshot *snap) {
    if (!snap) return;
    free(snap->buf); snap->buf = NULL; snap->size = 0;
}
