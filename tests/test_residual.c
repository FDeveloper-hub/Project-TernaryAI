/*
 * Conexion residual como capa envoltorio (src/ternary_residual.c): out = x + f(x), con f una cadena de
 * sub-capas, vista como UNA sola capa por el TernaryStack. Verifica:
 *  1. reglas de la cadena (se rechazan, con la capa sin tocar): primera sub-capa con entrada ternaria,
 *     sub-capa que no propaga gradiente, largos que no calzan, ultima sub-capa con otro largo que x, cadena vacia;
 *  2. forward IDENTICO a x + (cadena calculada a mano con las mismas sub-capas);
 *  3. identidad: con la rama anulada (escala de salida 0) la salida es x bit a bit y dL/dx == dL/dout;
 *  4. backward IDENTICO a la cadena derivada a mano (gradientes de TODAS las matrices internas, escalas y x),
 *     y expone las matrices de las sub-capas en orden al stack;
 *  5. dentro de un TernaryStack, en un mini modelo de lenguaje completo
 *        Embedding(ids) -> [x + GQA(banda(RMSNorm x))] -> [x + SwiGLU(banda(RMSNorm x))] -> RMSNorm -> lm_head
 *     checkpoint retomable exacto (20+20 == 40, comparado byte a byte), export identico, archivo truncado -> NULL;
 *  6. APRENDIZAJE con entropia cruzada: (a) siguiente token fijo (bigrama), (b) copiar el token ANTERIOR (necesita
 *     atencion: sin la rama de GQA no se puede resolver).
 */
#include "ternary_stack.h"
#include "tmp_path.h"
#include "ternary_layer.h"
#include "ternary_residual.h"
#include "ternary_activation.h"
#include "ternary_loss.h"
#include "ternary_rng.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg) do { if (!(cond)) { fprintf(stderr, "FALLO: %s (linea %d)\n", msg, __LINE__); return 1; } } while (0)

#define SEQ 6
#define DIM 16
#define HID 24
#define N (SEQ * DIM)

static uint64_t rs = 88172645463325252ull;
static double urand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (double)(rs >> 11) / 9007199254740992.0; }
static double nrand(void) { double u = urand() + 1e-12, v = urand(); return sqrt(-2.0 * log(u)) * cos(6.283185307179586 * v); }

static TernaryCapabilities g_cn;

static TernaryLayer build(const char *type, const size_t *in, size_t nd, const double *p, size_t np, TernaryRng *rng) {
    TernaryLayer l; char err[256];
    if (!ternary_layer_type_build(ternary_layer_type_by_name(type), in, nd, p, np, rng, &l, err, sizeof(err))) {
        fprintf(stderr, "build %s: %s\n", type, err); l.impl = NULL; l.vt = NULL;
    }
    return l;
}
static TernaryLayer mk_rms(TernaryRng *rng) { size_t in[2] = { SEQ, DIM }; double p[1] = { 1e-5 }; return build("rmsnorm", in, 2, p, 1, rng); }
static TernaryLayer mk_gqa(TernaryRng *rng) {
    size_t in[2] = { SEQ, DIM }; double p[5] = { 2, 0, 1, 10.0, 2 };
    return build("gqa", in, 2, p, 5, rng);
}
static TernaryLayer mk_swiglu(TernaryRng *rng) { size_t in[2] = { SEQ, DIM }; double p[3] = { HID, 0, 0.5 }; return build("swiglu", in, 2, p, 3, rng); }

/* x + f(x) con f = [RMSNorm -> banda -> sub] (sub = GQA o SwiGLU) */
static TernaryResidual *mk_block(int attention, unsigned seed) {
    TernaryRng rng; ternary_rng_seed(&rng, seed);
    TernaryResidual *r = ternary_residual_create();
    if (!r) return NULL;
    if (!ternary_residual_add(r, mk_rms(&rng), TERNARY_ACTIVATION_UINT8)) return NULL;
    if (!ternary_residual_add(r, attention ? mk_gqa(&rng) : mk_swiglu(&rng), TERNARY_ACTIVATION_UINT8)) return NULL;
    return r;
}

static void random_x(float *x, size_t n) { for (size_t i = 0; i < n; i++) x[i] = (float)nrand(); }

static int test_validation(void) {
    TernaryRng rng; ternary_rng_seed(&rng, 3);
    {   /* primera sub-capa con entrada TERNARIA (GQA): rechazada, y la capa queda del llamador */
        TernaryResidual *r = ternary_residual_create();
        TernaryLayer g = mk_gqa(&rng);
        CHECK(g.impl && !ternary_residual_add(r, g, TERNARY_ACTIVATION_UINT8), "la primera sub-capa debe tomar FLOAT");
        CHECK(ternary_residual_num_layers(r) == 0, "no se agrego nada");
        g.vt->destroy(g.impl);
        TernaryLayer none = ternary_layer_from_residual(r);
        CHECK(none.impl == NULL, "cadena vacia -> capa nula");
        ternary_residual_free(r);
    }
    {   /* sub-capa que no propaga gradiente (Embedding) */
        TernaryResidual *r = ternary_residual_create();
        CHECK(ternary_residual_add(r, mk_rms(&rng), TERNARY_ACTIVATION_UINT8), "rmsnorm");
        size_t in0[1] = { SEQ }; double pe[2] = { 8, DIM };
        TernaryLayer e = build("embedding", in0, 1, pe, 2, &rng);
        CHECK(e.impl && !ternary_residual_add(r, e, TERNARY_ACTIVATION_UINT8), "una sub-capa que no propaga gradiente se rechaza");
        e.vt->destroy(e.impl);
        ternary_residual_free(r);
    }
    {   /* largos que no calzan: SwiGLU sobre otra forma */
        TernaryResidual *r = ternary_residual_create();
        CHECK(ternary_residual_add(r, mk_rms(&rng), TERNARY_ACTIVATION_UINT8), "rmsnorm");
        size_t in[2] = { SEQ, DIM + 8 }; double p[3] = { HID, 0, 0.5 };
        TernaryLayer s = build("swiglu", in, 2, p, 3, &rng);
        CHECK(s.impl && !ternary_residual_add(r, s, TERNARY_ACTIVATION_UINT8), "largo de entrada que no calza se rechaza");
        s.vt->destroy(s.impl);
        ternary_residual_free(r);
    }
    {   /* la ultima sub-capa entrega otro largo que x: el chequeo final */
        TernaryResidual *r = ternary_residual_create();
        CHECK(ternary_residual_add(r, mk_rms(&rng), TERNARY_ACTIVATION_UINT8), "rmsnorm");
        size_t in[2] = { SEQ, DIM }; double p[1] = { 5 };
        TernaryLayer h = build("lm_head", in, 2, p, 1, &rng);
        CHECK(ternary_residual_add(r, h, TERNARY_ACTIVATION_UINT8), "lm_head encadena bien con RMSNorm");
        CHECK(ternary_layer_from_residual(r).impl == NULL, "x + f(x) con distinto largo -> capa nula");
        ternary_residual_free(r);
    }
    {   /* valida: matrices de las sub-capas expuestas en orden */
        TernaryResidual *r = mk_block(1, 7);
        CHECK(r, "bloque de atencion");
        TernaryLayer l = ternary_layer_from_residual(r);
        CHECK(l.impl && strcmp(l.vt->type_name, "residual") == 0, "capa residual valida");
        TernaryLayer rms = ternary_residual_layer(r, 0), gqa = ternary_residual_layer(r, 1);
        const size_t n0 = rms.vt->num_matrices(rms.impl), n1 = gqa.vt->num_matrices(gqa.impl);
        CHECK(l.vt->num_matrices(l.impl) == n0 + n1, "num_matrices = suma de las sub-capas");
        for (size_t m = 0; m < n0 + n1; m++) {
            TernaryLayer sub = (m < n0) ? rms : gqa; size_t lm = (m < n0) ? m : m - n0;
            CHECK(l.vt->grad_rows(l.impl, m) == sub.vt->grad_rows(sub.impl, lm) &&
                  l.vt->grad_row_width(l.impl, m) == sub.vt->grad_row_width(sub.impl, lm) &&
                  l.vt->has_bias(l.impl, m) == sub.vt->has_bias(sub.impl, lm), "forma de la matriz m mapeada a la sub-capa");
        }
        CHECK(l.vt->input_len(l.impl) == N && l.vt->output_len(l.impl) == N, "largos x y salida");
        CHECK(l.vt->input_dtype(l.impl) == TERNARY_DTYPE_FLOAT && l.vt->propagates_grad_x(l.impl), "entrada float, propaga gradiente");
        CHECK(strcmp(ternary_residual_activation(r, 0)->name, "uint8") == 0, "activacion de la sub-capa 0");
        l.vt->destroy(l.impl);
    }
    return 0;
}

/* Cadena a mano para el bloque FFN, con las MISMAS sub-capas: devuelve f(x) en f. Usa buffers propios. */
typedef struct { float y1[N], f[N]; int32_t raw1[N], raw2[N]; TernaryU8 *t; } Manual;
static void manual_forward(TernaryResidual *r, const float *x, Manual *m) {
    TernaryLayer a = ternary_residual_layer(r, 0), b = ternary_residual_layer(r, 1);
    size_t d[1] = { N };
    TernaryTensor *xt = ternary_tensor_view_float(d, 1, (float *)x);
    a.vt->forward(a.impl, xt, m->y1, m->raw1, &g_cn);
    ternary_tensor_free(xt);
    TERNARY_ACTIVATION_UINT8.apply_u8(m->y1, m->t);
    TernaryTensor *tt = ternary_tensor_view_u8(d, 1, m->t);
    b.vt->forward(b.impl, tt, m->f, m->raw2, &g_cn);
    ternary_tensor_free(tt);
}

static int test_forward_and_identity(void) {
    memset(&g_cn, 0, sizeof(g_cn)); g_cn.num_cores = 1; g_cn.dot_impl_name = "scalar";
    TernaryResidual *r = mk_block(0, 11);
    CHECK(r, "bloque ffn");
    TernaryLayer l = ternary_layer_from_residual(r);
    CHECK(l.impl, "capa");
    float x[N], out[N];
    random_x(x, N);
    size_t d[1] = { N };
    TernaryTensor *xt = ternary_tensor_view_float(d, 1, x);
    int32_t raw[N];
    l.vt->forward(l.impl, xt, out, raw, &g_cn);

    Manual *m = (Manual *)calloc(1, sizeof(Manual));
    m->t = ternary_u8_alloc(SEQ, DIM);
    manual_forward(r, x, m);
    int nonzero_f = 0;
    for (size_t i = 0; i < N; i++) {
        CHECK(out[i] == x[i] + m->f[i], "forward: out distinto de x + f(x) calculado a mano");
        if (m->f[i] != 0.0f) nonzero_f++;
    }
    CHECK(nonzero_f > (int)N / 2, "la rama debe aportar algo (no todo cero)");

    /* identidad: rama anulada */
    ((TernarySwiGLU *)ternary_residual_layer(r, 1).impl)->wd->scale = 0.0f;
    l.vt->forward(l.impl, xt, out, raw, &g_cn);
    CHECK(memcmp(out, x, sizeof(x)) == 0, "rama anulada: la salida debe ser x bit a bit");
    float go[N], gx[N]; memset(gx, 0, sizeof(gx));
    random_x(go, N);
    size_t nm = l.vt->num_matrices(l.impl);
    float **gw = (float **)calloc(nm, sizeof(float *)); float *gb[8] = { 0 }; float gs[8] = { 0 };
    for (size_t k = 0; k < nm; k++) gw[k] = (float *)malloc(l.vt->grad_rows(l.impl, k) * l.vt->grad_row_width(l.impl, k) * sizeof(float));
    l.vt->backward(l.impl, xt, go, gw, gx, gs, gb, raw);
    CHECK(memcmp(gx, go, sizeof(go)) == 0, "rama anulada: dL/dx debe ser exactamente dL/dout");

    for (size_t k = 0; k < nm; k++) free(gw[k]);
    free(gw); ternary_u8_free(m->t); free(m); ternary_tensor_free(xt);
    l.vt->destroy(l.impl);
    return 0;
}

static int test_backward_matches_manual(int attention) {
    TernaryResidual *r = mk_block(attention, 21 + (unsigned)attention);
    CHECK(r, "bloque");
    TernaryLayer l = ternary_layer_from_residual(r);
    CHECK(l.impl, "capa");
    TernaryLayer a = ternary_residual_layer(r, 0), b = ternary_residual_layer(r, 1);
    float x[N], out[N], go[N];
    random_x(x, N); random_x(go, N);
    size_t d[1] = { N };
    TernaryTensor *xt = ternary_tensor_view_float(d, 1, x);
    int32_t raw[N];
    l.vt->forward(l.impl, xt, out, raw, &g_cn);

    /* backward del envoltorio */
    const size_t na = a.vt->num_matrices(a.impl), nb = b.vt->num_matrices(b.impl), nm = na + nb;
    CHECK(l.vt->num_matrices(l.impl) == nm, "num_matrices");
    float **gw = (float **)calloc(nm, sizeof(float *)), **gb = (float **)calloc(nm, sizeof(float *));
    float gs[16] = { 0 }, gx[N]; memset(gx, 0, sizeof(gx));
    for (size_t k = 0; k < nm; k++) {
        gw[k] = (float *)calloc(l.vt->grad_rows(l.impl, k) * l.vt->grad_row_width(l.impl, k), sizeof(float));
        if (l.vt->has_bias(l.impl, k)) gb[k] = (float *)calloc(l.vt->grad_rows(l.impl, k), sizeof(float));
    }
    l.vt->backward(l.impl, xt, go, gw, gx, gs, gb, raw);

    /* cadena derivada a mano sobre las MISMAS sub-capas: forward propio (buffers propios) y backward en orden inverso */
    Manual *mn = (Manual *)calloc(1, sizeof(Manual));
    mn->t = ternary_u8_alloc(SEQ, DIM);
    TernaryTensor *tt = NULL;
    TernaryTensor *x2 = ternary_tensor_view_float(d, 1, x);
    a.vt->forward(a.impl, x2, mn->y1, mn->raw1, &g_cn);
    TERNARY_ACTIVATION_UINT8.apply_u8(mn->y1, mn->t);
    tt = ternary_tensor_view_u8(d, 1, mn->t);
    b.vt->forward(b.impl, tt, mn->f, mn->raw2, &g_cn);

    float **mw = (float **)calloc(nm, sizeof(float *)), **mb = (float **)calloc(nm, sizeof(float *));
    float ms[16] = { 0 }, g1[N], g0[N];
    for (size_t k = 0; k < nm; k++) {
        mw[k] = (float *)calloc(l.vt->grad_rows(l.impl, k) * l.vt->grad_row_width(l.impl, k), sizeof(float));
        if (l.vt->has_bias(l.impl, k)) mb[k] = (float *)calloc(l.vt->grad_rows(l.impl, k), sizeof(float));
    }
    memset(g1, 0, sizeof(g1)); memset(g0, 0, sizeof(g0));
    b.vt->backward(b.impl, tt, go, mw + na, g1, ms + na, mb + na, mn->raw2);
    a.vt->backward(a.impl, x2, g1, mw, g0, ms, mb, mn->raw1);
    float want_gx[N];
    for (size_t i = 0; i < N; i++) want_gx[i] = go[i] + g0[i];

    CHECK(memcmp(gx, want_gx, sizeof(gx)) == 0, "backward: dL/dx distinto de atajo + rama");
    double maxref = 0;
    for (size_t k = 0; k < nm; k++) {
        const size_t cnt = l.vt->grad_rows(l.impl, k) * l.vt->grad_row_width(l.impl, k);
        CHECK(memcmp(gw[k], mw[k], cnt * sizeof(float)) == 0, "backward: gradiente de una matriz interna distinto de la cadena a mano");
        for (size_t q = 0; q < cnt; q++) maxref = fmax(maxref, fabs((double)mw[k][q]));
        if (gb[k]) CHECK(memcmp(gb[k], mb[k], l.vt->grad_rows(l.impl, k) * sizeof(float)) == 0, "backward: gradiente de bias");
        CHECK(gs[k] == ms[k], "backward: gradiente de escala distinto de la cadena a mano");
    }
    CHECK(maxref > 1e-3, "los gradientes internos no deben ser ~0");

    for (size_t k = 0; k < nm; k++) { free(gw[k]); free(gb[k]); free(mw[k]); free(mb[k]); }
    free(gw); free(gb); free(mw); free(mb);
    ternary_u8_free(mn->t); free(mn); ternary_tensor_free(tt); ternary_tensor_free(x2); ternary_tensor_free(xt);
    l.vt->destroy(l.impl);
    return 0;
}

/* ---------------- mini modelo de lenguaje con residuales ---------------- */
#define LV 12       /* vocabulario */
static const float SCALE_LR = 0.01f;
#define BATCH 8

static TernaryStack *new_lm(unsigned seed) {
    TernaryRng rng; ternary_rng_seed(&rng, seed);
    size_t in0[1] = { SEQ }, in2[2] = { SEQ, DIM };
    double pe[2] = { LV, DIM }, pr[1] = { 1e-5 }, ph[1] = { LV };
    TernaryStack *s = ternary_stack_create();
    TernaryLayer emb = build("embedding", in0, 1, pe, 2, &rng);
    if (!emb.impl || !ternary_stack_add(s, emb, TERNARY_ACTIVATION_UINT8)) return NULL;
    for (int attention = 1; attention >= 0; attention--) {
        TernaryResidual *r = ternary_residual_create();
        TernaryLayer rms = mk_rms(&rng), sub = attention ? mk_gqa(&rng) : mk_swiglu(&rng);
        if (!ternary_residual_add(r, rms, TERNARY_ACTIVATION_UINT8) || !ternary_residual_add(r, sub, TERNARY_ACTIVATION_UINT8)) return NULL;
        TernaryLayer l = ternary_layer_from_residual(r);
        if (!l.impl || !ternary_stack_add(s, l, TERNARY_ACTIVATION_UINT8)) return NULL;
    }
    TernaryLayer fin = build("rmsnorm", in2, 2, pr, 1, &rng);
    TernaryLayer head = build("lm_head", in2, 2, ph, 1, &rng);
    if (!fin.impl || !head.impl) return NULL;
    if (!ternary_stack_add(s, fin, TERNARY_ACTIVATION_UINT8) || !ternary_stack_add(s, head, TERNARY_ACTIVATION_UINT8)) return NULL;
    return s;
}

static int32_t g_next[LV];
static int g_task;   /* 0: bigrama fijo; 1: copiar el token anterior */

static void make_sample(int32_t *ids, int32_t *tg) {
    for (int p = 0; p < SEQ; p++) ids[p] = (int32_t)(urand() * LV);
    for (int p = 0; p < SEQ; p++) {
        if (g_task == 0) tg[p] = g_next[ids[p]];
        else tg[p] = (p == 0) ? -1 : ids[p - 1];
    }
}

static double train_epoch(TernaryStack *s) {
    int32_t ids[SEQ], tg[SEQ]; float out[SEQ * LV], go[SEQ * LV];
    size_t xd[1] = { SEQ };
    double loss = 0.0;
    ternary_stack_zero_gradients(s);
    for (int b = 0; b < BATCH; b++) {
        make_sample(ids, tg);
        TernaryTensor *xt = ternary_tensor_view_id(xd, 1, ids);
        ternary_stack_forward(s, xt, out, &g_cn);
        loss += ternary_xent(out, SEQ, LV, tg, go, NULL, NULL);
        for (int i = 0; i < SEQ * LV; i++) go[i] /= (float)BATCH;
        ternary_stack_backward(s, xt, go);
        ternary_tensor_free(xt);
    }
    ternary_stack_apply_gradients(s, SCALE_LR);
    return loss / BATCH;
}

static void eval(TernaryStack *s, int n, double *loss_out, double *acc_out) {
    const uint64_t keep = rs;
    rs = 0xD1B54A32D192ED03ull;
    int32_t ids[SEQ], tg[SEQ]; float out[SEQ * LV];
    size_t xd[1] = { SEQ };
    double loss = 0.0; size_t hits = 0, total = 0;
    for (int i = 0; i < n; i++) {
        make_sample(ids, tg);
        TernaryTensor *xt = ternary_tensor_view_id(xd, 1, ids);
        ternary_stack_forward(s, xt, out, &g_cn);
        size_t nv, nc;
        loss += ternary_xent(out, SEQ, LV, tg, NULL, &nv, &nc);
        hits += nc; total += nv;
        ternary_tensor_free(xt);
    }
    rs = keep;
    *loss_out = loss / n; *acc_out = (double)hits / (double)total;
}

#define files_equal model_dirs_equal

static int test_lm_persistence(void) {
    g_task = 0;
    rs = 0xA24BAED4963EE407ull;
    for (int v = 0; v < LV; v++) g_next[v] = (int32_t)(urand() * LV);
    const char *ck_a = tmp_path("test_res_a.bin"), *ck_b = tmp_path("test_res_b.bin"), *ck_c = tmp_path("test_res_c.bin"), *ex = tmp_path("test_res_ex.bin");
    const uint64_t rs0 = 123456789012345ull;
    rs = rs0;
    TernaryStack *a = new_lm(9);
    CHECK(a, "modelo a");
    CHECK(ternary_stack_num_layers(a) == 5 && strcmp(ternary_stack_layer(a, 1).vt->type_name, "residual") == 0, "5 capas, la 2a es residual");
    for (int e = 0; e < 40; e++) train_epoch(a);
    rs = rs0;
    TernaryStack *b = new_lm(9);
    for (int e = 0; e < 20; e++) train_epoch(b);
    CHECK(ternary_stack_save(b, ck_b, 1), "guardar checkpoint");
    TernaryStack *c = ternary_stack_load(ck_b);
    CHECK(c, "cargar checkpoint con capas residuales");
    CHECK(ternary_stack_num_layers(c) == 5 && strcmp(ternary_stack_layer(c, 2).vt->type_name, "residual") == 0, "estructura recuperada");
    CHECK(ternary_stack_save(c, ck_c, 1) && files_equal(ck_b, ck_c), "guardar -> cargar -> guardar: archivos identicos");
    for (int e = 0; e < 20; e++) train_epoch(c);
    CHECK(ternary_stack_save(a, ck_a, 1) && ternary_stack_save(c, ck_c, 1) && files_equal(ck_a, ck_c),
          "checkpoint: 20+20 distinto de 40 de un tiron (comparado byte a byte)");

    /* export: inferencia identica */
    CHECK(ternary_stack_save(a, ex, 0), "guardar export");
    TernaryStack *e2 = ternary_stack_load(ex);
    CHECK(e2, "cargar export");
    int32_t ids[SEQ], tg[SEQ]; float o1[SEQ * LV], o2[SEQ * LV]; size_t xd[1] = { SEQ };
    for (int k = 0; k < 10; k++) {
        make_sample(ids, tg);
        TernaryTensor *xt = ternary_tensor_view_id(xd, 1, ids);
        ternary_stack_forward(a, xt, o1, &g_cn);
        ternary_stack_forward(e2, xt, o2, &g_cn);
        ternary_tensor_free(xt);
        CHECK(memcmp(o1, o2, sizeof(o1)) == 0, "export: inferencia distinta");
    }

    /* model.ternary o training_state.ternary truncados: la carga devuelve NULL, sin colgarse ni cargar a medias */
    {
        const char *trunc = tmp_path("test_res_trunc");
        for (int which = 1; which <= 2; which++) {
            long sz = 0; char *buf = read_whole_file(model_file_path(ck_a, MODEL_DIR_FILES[which]), &sz);
            CHECK(buf != NULL, "leer");
            for (int cut = 0; buf && cut < 3; cut++) {
                long keep = (cut == 0) ? sz / 2 : (cut == 1) ? sz / 4 : sz - 1;
                model_dir_remove(trunc);
                model_dir_copy(ck_a, trunc);
                FILE *f = fopen(model_file_path(trunc, MODEL_DIR_FILES[which]), "wb"); fwrite(buf, 1, (size_t)keep, f); fclose(f);
                TernaryStack *t = ternary_stack_load(trunc);
                CHECK(t == NULL, "un archivo truncado no debe cargar");
                if (t) ternary_stack_free(t);
            }
            free(buf);
        }
        model_dir_remove(trunc);
    }
    ternary_stack_free(a); ternary_stack_free(b); ternary_stack_free(c); ternary_stack_free(e2);
    model_dir_remove(ck_a); model_dir_remove(ck_b); model_dir_remove(ck_c); model_dir_remove(ex);
    return 0;
}

/* Criterios ABSOLUTOS por tarea, no "subir X puntos":
 *  - bigrama (tarea 0): lo resuelve un modelo sin atencion; se exige una perdida claramente baja y buena precision.
 *  - copiar el token anterior (tarea 1): SIN atencion el techo es el azar (el objetivo no depende del token actual),
 *    asi que superar 2x el azar con una perdida menor que la de predecir uniforme (ln V) demuestra que la rama de GQA
 *    aprende A TRAVES del residual. No se exige resolver la tarea: converge lento (ver README, "GQA: pruebas mas
 *    realistas"; con 3000 epocas y SCALE_LR=0.001 llega a ~31% y sigue subiendo). */
/* Una corrida: devuelve perdida/precision finales (semilla de pesos 'seed'). Las mismas semillas de datos siempre. */
static int lm_run(int task, unsigned seed, int epochs, double *l0, double *a0, double *l1, double *a1) {
    g_task = task;
    rs = 0xA24BAED4963EE407ull;
    for (int v = 0; v < LV; v++) g_next[v] = (int32_t)(urand() * LV);
    rs = 0x9E3779B97F4A7C15ull;
    TernaryStack *s = new_lm(seed);
    CHECK(s, "modelo");
    eval(s, 64, l0, a0);
    for (int e = 0; e < epochs; e++) train_epoch(s);
    eval(s, 64, l1, a1);
    ternary_stack_free(s);
    return 0;
}

static int test_lm_learning(int task, int epochs, double min_acc, double max_loss) {
    double l0, a0, l1, a1;
    if (lm_run(task, 23 + (unsigned)task, epochs, &l0, &a0, &l1, &a1)) return 1;
    printf("APRENDIZAJE %s (%d epocas): perdida %.3f -> %.3f | precision %.1f%% -> %.1f%% (azar %.1f%%)\n",
           task == 0 ? "LM bigrama fijo       " : "LM copiar token anterior", epochs, l0, l1, 100.0 * a0, 100.0 * a1, 100.0 / LV);
    CHECK(l1 < max_loss, "la perdida final no es lo bastante baja");
    CHECK(a1 > min_acc, "la precision final no supera el umbral");
    return 0;
}

/* "Copiar el token anterior" converge lento y es MUY ruidosa entre semillas (ver README): con una sola semilla
 * el resultado dependia del orden de las sumas float. Con las 7 semillas de abajo, 800 epocas: camino exacto (tablas de
 * Dense apagadas, bit a bit el codigo de antes) 5/7 sobre 2x el azar, precision media 29.6 %; con tablas 6/7, media
 * 28.1 %; en un barrido aparte de 7 semillas distintas las dos variantes dieron 5/7 y medias de 27.8 y 26.8 %. Se pide: la mayoria de las semillas (>= 4/7) supera 2x
 * el azar con perdida menor que ln V -- demuestra que la rama de GQA aprende a traves del residual. */
static int test_lm_copy_prev_statistical(void) {
    static const unsigned seeds[7] = { 24, 47, 59, 71, 83, 97, 101 };
    int ok = 0; double sum = 0.0;
    for (int k = 0; k < 7; k++) {
        double l0, a0, l1, a1;
        if (lm_run(1, seeds[k], 800, &l0, &a0, &l1, &a1)) return 1;
        sum += a1;
        if (a1 > 2.0 / LV && l1 < log((double)LV)) ok++;
    }
    printf("APRENDIZAJE LM copiar token anterior (800 epocas, 7 semillas): %d/7 sobre 2x el azar (se piden 4), precision media %.1f%% (azar %.1f%%)\n",
           ok, 100.0 * sum / 7.0, 100.0 / LV);
    CHECK(ok >= 4, "copiar token anterior: menos de 4/7 semillas superan 2x el azar con perdida < ln V");
    return 0;
}

int main(void) {
    memset(&g_cn, 0, sizeof(g_cn)); g_cn.num_cores = 1; g_cn.dot_impl_name = "scalar";
    if (test_validation()) return 1;
    printf("OK: reglas de la cadena (se rechaza lo invalido sin tocar la capa), matrices de las sub-capas en orden\n");
    if (test_forward_and_identity()) return 1;
    printf("OK: forward == x + cadena a mano; rama anulada -> salida == x y dL/dx == dL/dout (bit a bit)\n");
    if (test_backward_matches_manual(0) || test_backward_matches_manual(1)) return 1;
    printf("OK: backward identico a la cadena derivada a mano (SwiGLU y GQA): todas las matrices internas, escalas y x\n");
    if (test_lm_persistence()) return 1;
    printf("OK: mini LM con residuales: checkpoint 20+20 == 40 byte a byte, guardar->cargar->guardar identico, export identico, truncado -> NULL\n");
    if (test_lm_learning(0, 400, 0.30, 2.0)) return 1;
    if (test_lm_copy_prev_statistical()) return 1;
    printf("OK: test_residual\n");
    return 0;
}
