#define _POSIX_C_SOURCE 200809L
/* Benchmark NEON contra escalar (mismas formas que usan los modelos tipicos de la libreria). Imprime ns por llamada y la
 * aceleracion = t_escalar / t_neon. Para medir de verdad hay que ejecutarlo en un dispositivo ARM real, compilado con las mismas
 * banderas que la libreria mas NEON (p. ej. -O3 -ffp-contract=off -mfpu=neon-vfpv4 en ARMv7); bajo qemu solo comprueba que corre.
 *   make tests/bench_neon && ./tests/bench_neon [filtro]
 * Los bucles escalares son los de la libreria (los del 'else' del despacho), compilados con las mismas banderas. */
#include "ternary_neon.h"
#include "ternary_fexp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t rs = 88172645463325252ull;
static uint64_t rnd64(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static void *A(size_t n) { void *p = calloc(n ? n : 1, 1); if (!p) abort(); return p; }
static float *FA(size_t n) { float *p = (float *)A(n * 4); for (size_t i = 0; i < n; i++) p[i] = (float)((int)(rnd64() % 2001) - 1000) * 0.0137f; return p; }
static int32_t *IA(size_t n, int32_t lim) { int32_t *p = (int32_t *)A(n * 4); for (size_t i = 0; i < n; i++) p[i] = (int32_t)(rnd64() % (2 * (uint64_t)lim + 1)) - lim; return p; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (double)t.tv_sec * 1e9 + (double)t.tv_nsec; }
static const char *filt = "";
static volatile float sink_f; static volatile double sink_d;

typedef void (*fn_t)(void *);
static double time_fn(fn_t f, void *ctx) {
    long it = 1; double dt = 0;
    for (;;) { const double t0 = now(); for (long i = 0; i < it; i++) f(ctx); dt = now() - t0; if (dt > 6e7 || it > (1L << 28)) break; it *= 2; }
    double best = dt / (double)it;
    for (int r = 0; r < 4; r++) { const double t0 = now(); for (long i = 0; i < it; i++) f(ctx); const double d = (now() - t0) / (double)it; if (d < best) best = d; }
    return best;
}
static long cnt_K = -1; static const char *cnt_side = "n";
static void report(const char *name, fn_t fn_neon, fn_t fn_sc, void *ctx) {
    if (*filt && !strstr(name, filt)) return;
    if (cnt_K >= 0) { fn_t f = cnt_side[0] == 'n' ? fn_neon : fn_sc; for (long i = 0; i < cnt_K; i++) f(ctx); return; }   /* modo conteo de instrucciones (qemu) */
    const double tn = time_fn(fn_neon, ctx), ts = time_fn(fn_sc, ctx);
    printf("%-26s neon %9.1f ns   escalar %9.1f ns   x%.2f%s\n", name, tn, ts, ts / tn, ts / tn < 1.0 ? "   <-- MAS LENTO" : "");
}
#define NOINL __attribute__((noinline))

/* ---------------------------------------------------------------------------------------------------------------- */
typedef struct { size_t G, H, n, ld, scs, vs, es; int shift; int32_t lim; int32_t *kT, *qb, *sc, *m, *kr; const int32_t **q; float *tab; uint32_t K0;
                 float *v, *e, *out, *inv, *ga, *o, *vT, *gq, *gk, *gv, *p, *ds, *qf; int32_t *qr; } AC;

static NOINL void sc_scores_i32(void *c_) { AC *c = c_;
    for (size_t j = 0; j < c->n; j++) { int64_t dot = 0; for (size_t d = 0; d < c->H; d++) dot += (int64_t)c->qb[d] * (int64_t)c->kr[j * c->H + d];
        dot >>= c->shift; if (dot > c->lim) dot = c->lim; else if (dot < -c->lim) dot = -c->lim; c->sc[j] = (int32_t)dot; } }
static void nn_scores_i32(void *c_) { AC *c = c_; ternary_neon_scores_i32(c->qb, c->kr, c->H, c->H, c->n, c->shift, c->lim, c->sc); }

static NOINL void sc_attn_scores(void *c_) { AC *c = c_;
    for (size_t b = 0; b < c->G; b++) { int32_t mx = 0;
        for (size_t j = 0; j < c->n; j++) { int64_t dot = 0; for (size_t d = 0; d < c->H; d++) dot += (int64_t)c->q[b][d] * (int64_t)c->kT[d * c->ld + j];
            dot >>= c->shift; if (dot > c->lim) dot = c->lim; else if (dot < -c->lim) dot = -c->lim; c->sc[b * c->scs + j] = (int32_t)dot; if (j == 0 || (int32_t)dot > mx) mx = (int32_t)dot; }
        c->m[b] = mx; } }
static void nn_attn_scores(void *c_) { AC *c = c_; ternary_neon_attn_scores(c->q, c->G, c->kT, c->ld, c->H, c->n, c->shift, c->lim, c->sc, c->scs, c->m); }

static NOINL void sc_attn_pv(void *c_) { AC *c = c_;
    for (size_t b = 0; b < c->G; b++) { float part[16] = {0};
        for (size_t j = 0; j < c->n; j++) { const uint32_t k = (uint32_t)(c->m[b] - c->sc[b * c->scs + j]); const float e = c->tab[k < c->K0 ? k : c->K0]; c->e[b * c->es + j] = e; part[j & 15] += e; }
        const float inv = 1.0f / ternary_sum16(part); c->inv[b] = inv;
        float *acc = c->out + b * c->H; memset(acc, 0, c->H * 4);
        for (size_t j = 0; j < c->n; j++) { const float e = c->e[b * c->es + j]; const float *vj = c->v + j * c->vs; for (size_t d = 0; d < c->H; d++) acc[d] += e * vj[d]; }
        for (size_t d = 0; d < c->H; d++) acc[d] *= inv; } }
static void nn_attn_pv(void *c_) { AC *c = c_; ternary_neon_attn_pv(c->G, c->sc, c->scs, c->m, c->tab, c->K0, c->n, c->v, c->vs, c->H, c->e, c->es, c->out, c->inv); }

static NOINL void sc_attn_bwd(void *c_) { AC *c = c_;
    for (size_t b = 0; b < c->G; b++) {
        const float *gab = c->ga + b * c->H, *ob = c->o + b * c->H; float D = 0.0f; for (size_t d = 0; d < c->H; d++) D += gab[d] * ob[d];
        for (size_t j = 0; j < c->n; j++) {
            float dot = 0.0f; for (size_t d = 0; d < c->H; d++) dot += gab[d] * c->vT[d * c->ld + j];
            const uint32_t k = (uint32_t)(c->m[b] - c->sc[b * c->scs + j]); const float p = c->tab[k < c->K0 ? k : c->K0] * c->inv[b];
            const float ds = ((1.0f * p) * (dot - D)) * 0.001f;
            for (size_t d = 0; d < c->H; d++) c->gv[j * c->H + d] += p * gab[d];
            for (size_t d = 0; d < c->H; d++) c->gk[j * c->H + d] += ds * (float)c->qr[b * c->H + d];
            for (size_t d = 0; d < c->H; d++) c->gq[b * c->H + d] += ds * (float)c->kr[j * c->H + d]; } } }
static void nn_attn_bwd(void *c_) { AC *c = c_; ternary_neon_attn_bwd(c->G, c->n, c->H, c->sc, c->scs, c->m, c->inv, c->tab, c->K0, c->ga, c->o, c->vT, c->ld, 1.0f, 0.001f, c->qr, (const int32_t *)c->kr, c->H, c->gq, c->gk, c->gv, c->H, c->p, c->ds, c->es, c->qf); }

typedef struct { size_t n, V; float *a, *b, *c; int32_t *z, *z2; float *tab; size_t len; int32_t *sc; float it; } EC;
static NOINL void sc_softmax_fwd(void *c_) { EC *c = c_; int32_t m = c->sc[0]; for (size_t i = 1; i < c->n; i++) if (c->sc[i] > m) m = c->sc[i];
    float part[16] = {0}; for (size_t i = 0; i < c->n; i++) { int64_t k = (int64_t)m - (int64_t)c->sc[i]; if (k >= (int64_t)c->len) k = (int64_t)c->len - 1; const float e = c->tab[k]; c->a[i] = e; part[i & 15] += e; }
    const float inv = 1.0f / ternary_sum16(part); for (size_t i = 0; i < c->n; i++) c->a[i] *= inv; }
static void nn_softmax_fwd(void *c_) { EC *c = c_; ternary_neon_softmax_fwd(c->sc, c->n, c->tab, c->len, c->a); }
static NOINL void sc_softmax_bwd(void *c_) { EC *c = c_; float part[16] = {0}; for (size_t j = 0; j < c->n; j++) part[j & 15] += c->b[j] * c->a[j]; const float dot = ternary_sum16(part);
    for (size_t i = 0; i < c->n; i++) c->c[i] = c->it * c->a[i] * (c->b[i] - dot); }
static void nn_softmax_bwd(void *c_) { EC *c = c_; ternary_neon_softmax_bwd(c->a, c->b, c->n, c->it, c->c); }

typedef struct { size_t D; uint16_t *gp, *gm; int *g; float *x, *go, *o, *gg, *gx; double gs; float eps, scale; } RC;
static NOINL void sc_rms_fwd(void *c_) { RC *c = c_; const double ss = ternary_canon_sumsq(c->x, c->D); const float f = c->scale * (float)(1.0 / sqrt(ss / (double)c->D + (double)c->eps));
    for (size_t j = 0; j < c->D; j++) c->o[j] = (c->g[j] == 0) ? 0.0f : (c->g[j] > 0 ? f * c->x[j] : -f * c->x[j]); }
static void nn_rms_fwd(void *c_) { RC *c = c_; ternary_neon_rmsnorm_forward(1, c->D, c->eps, c->scale, c->gp, c->gm, c->x, c->o); }
static NOINL void sc_rms_bwd(void *c_) { RC *c = c_; const double ss = ternary_canon_sumsq(c->x, c->D); const float r = (float)(1.0 / sqrt(ss / (double)c->D + (double)c->eps)); double dl[16] = {0};
    for (size_t j = 0; j < c->D; j++) { dl[j & 15] += (double)c->g[j] * (double)c->go[j] * (double)c->x[j]; c->gg[j] += c->go[j] * c->scale * c->x[j] * r; }
    const double dot = ternary_canon_reduce_d16(dl); c->gs += dot * (double)r; const float cc = (float)(dot * (double)r * (double)r / (double)c->D);
    for (size_t k = 0; k < c->D; k++) c->gx[k] += c->scale * r * ((float)c->g[k] * c->go[k] - c->x[k] * cc); }
static void nn_rms_bwd(void *c_) { RC *c = c_; ternary_neon_rmsnorm_backward(1, c->D, c->eps, c->scale, c->gp, c->gm, c->x, c->go, c->gg, c->gx, &c->gs); }

typedef struct { size_t H; int32_t D; int32_t *zg, *zu; float *silu, *dsilu, *gt, *gu, *ga, *h, *fa, *fu; float su; int mode; } SC;
static NOINL void sc_sw_prod(void *c_) { SC *c = c_; for (size_t j = 0; j < c->H; j++) {
    if (c->mode == 0) { int64_t k = (int64_t)c->zg[j] + c->D; if (k < 0) k = 0; else if (k > 2 * c->D) k = 2 * c->D; c->h[j] = c->silu[k] * (c->su * (float)c->zu[j]); }
    else { const float a = c->su * (float)c->zu[j]; c->h[j] = a > 0.0f ? a * a : 0.0f; } } }
static void nn_sw_prod(void *c_) { SC *c = c_; ternary_neon_swiglu_prod_z(c->mode, c->H, c->zg, c->zu, c->su, c->silu, c->D, c->h); }
static NOINL void sc_sw_bwd(void *c_) { SC *c = c_; for (size_t j = 0; j < c->H; j++) { const float u = c->su * (float)c->zu[j];
    if (c->mode == 2) c->gu[j] = u > 0.0f ? c->gt[j] * 2.0f * u : 0.0f;
    else { int64_t k = (int64_t)c->zg[j] + c->D; if (k < 0) k = 0; else if (k > 2 * c->D) k = 2 * c->D; c->gu[j] = c->gt[j] * c->silu[k]; c->ga[j] = c->gt[j] * u * c->dsilu[k]; } } }
static void nn_sw_bwd(void *c_) { SC *c = c_; ternary_neon_swiglu_bwd_z(c->mode, c->H, c->zg, c->zu, c->gt, c->su, c->silu, c->dsilu, c->D, c->gu, c->ga); }
static inline void silu_f(float a, float *sv, float *dsv) { const float y = -fabsf(a) * 1.44269504088896341f; const float e = (y < TERNARY_FEXP_CUTOFF) ? 0.0f : ternary_fexp2_core(y);
    const float sg = (a >= 0.0f) ? 1.0f / (1.0f + e) : e / (1.0f + e); *sv = a * sg; *dsv = sg * (1.0f + a * (1.0f - sg)); }
static NOINL void sc_sw_fwd_u8(void *c_) { SC *c = c_; for (size_t j = 0; j < c->H; j++) { if (c->mode == 2) { const float a = c->fu[j]; c->h[j] = a > 0.0f ? a * a : 0.0f; }
    else { float sv, dsv; silu_f(c->fa[j], &sv, &dsv); c->h[j] = sv * c->fu[j]; } } }
static void nn_sw_fwd_u8(void *c_) { SC *c = c_; ternary_neon_swiglu_fwd_u8(c->mode, c->H, c->fa, c->fu, c->zg, c->h); }
static NOINL void sc_sw_bwd_u8(void *c_) { SC *c = c_; for (size_t j = 0; j < c->H; j++) { const float u = 0.01f * (float)c->zu[j];
    if (c->mode == 2) c->gu[j] = u > 0.0f ? c->gt[j] * 2.0f * u : 0.0f;
    else { float sv, dsv; silu_f(0.01f * (float)c->zg[j], &sv, &dsv); c->gu[j] = c->gt[j] * sv; c->ga[j] = c->gt[j] * u * dsv; } } }
static void nn_sw_bwd_u8(void *c_) { SC *c = c_; ternary_neon_swiglu_bwd_u8(c->mode, c->H, c->zg, c->zu, c->gt, 0.01f, 0.01f, c->gu, c->ga); }

static NOINL void sc_xent(void *c_) { EC *c = c_; size_t arg = 0; float m = c->a[0]; for (size_t v = 1; v < c->V; v++) if (c->a[v] > m) { m = c->a[v]; arg = v; }
    double dl[16] = {0}; for (size_t v = 0; v < c->V; v++) dl[v & 15] += (double)ternary_expf_nonpos(c->a[v] - m);
    const double sum = ternary_canon_reduce_d16(dl); const double inv = 1.0 / (sum * 8.0); for (size_t v = 0; v < c->V; v++) c->c[v] = (float)((double)ternary_expf_nonpos(c->a[v] - m) * inv);
    c->c[3] -= 0.125f; sink_d = log(sum) - (double)(c->a[3] - m); sink_f = (float)arg; }
static void nn_xent(void *c_) { EC *c = c_; int ok; double lo; ternary_neon_xent_row(c->a, c->V, 3, 8, c->c, &ok, &lo); sink_d = lo; }
static NOINL void sc_dotgz(void *c_) { EC *c = c_; double d[16] = {0}; for (size_t j = 0; j < c->V; j++) d[j & 15] += (double)c->a[j] * (double)c->z[j]; sink_d = ternary_canon_reduce_d16(d); }
static void nn_dotgz(void *c_) { EC *c = c_; sink_d = ternary_neon_dot_gz(c->a, c->z, c->V, 0, 0.0); }

/* ---- bloque de elementales y de la ruta uint8 (se llenan al implementarlos) ---- */
#include "bench_neon_elem.h"

int main(int argc, char **argv) {
    if (argc > 1) filt = argv[1];
    if (argc > 4 && !strcmp(argv[2], "--count")) { cnt_K = atol(argv[3]); cnt_side = argv[4]; }
    if (!ternary_neon_enabled()) { printf("NEON no disponible\n"); return 0; }
    AC a; memset(&a, 0, sizeof a);
    a.G = 2; a.H = 16; a.n = 128; a.ld = 128 + 16; a.scs = 128 + 16; a.vs = a.H; a.es = a.scs; a.shift = 16 + 2; a.lim = 128 * 16;
    a.kT = IA(a.H * a.ld, 1 << 12); a.qb = IA(a.G * a.H, 1 << 12); a.sc = IA(a.G * a.scs, a.lim); a.m = IA(3 * a.G, 1); a.kr = IA(a.n * a.H + 16, 1 << 12);
    a.q = (const int32_t **)A(a.G * sizeof(void *)); for (size_t b = 0; b < a.G; b++) a.q[b] = a.qb + b * a.H;
    a.K0 = 199; a.tab = (float *)A(200 * 4); for (uint32_t k = 0; k < 200; k++) a.tab[k] = expf(-0.03f * (float)k);
    for (size_t b = 0; b < a.G; b++) { int32_t mx = a.sc[b * a.scs]; for (size_t j = 0; j < a.n; j++) if (a.sc[b * a.scs + j] > mx) mx = a.sc[b * a.scs + j]; a.m[b] = mx; }
    a.v = FA(a.n * a.H + 16); a.e = FA(a.G * a.es); a.out = FA(a.G * a.H); a.inv = FA(a.G); for (size_t b = 0; b < a.G; b++) a.inv[b] = 0.05f;
    a.ga = FA(a.G * a.H); a.o = FA(a.G * a.H); a.vT = FA(a.H * a.ld); a.gq = FA(a.G * a.H); a.gk = FA(a.n * a.H); a.gv = FA(a.n * a.H);
    a.p = FA(a.G * a.es + 32); a.ds = FA(a.G * a.es + 32); a.qf = FA(a.G * a.H + 16); a.qr = IA(a.G * a.H, 1 << 12);
    for (size_t i = 0; i < a.G * a.H; i++) { a.gq[i] = 0; }
    printf("== atencion (G=2, H=16, n=128) ==\n");
    report("scores_i32 (H16,n128)", nn_scores_i32, sc_scores_i32, &a);
    report("attn_scores", nn_attn_scores, sc_attn_scores, &a);
    report("attn_pv", nn_attn_pv, sc_attn_pv, &a);
    report("attn_bwd", nn_attn_bwd, sc_attn_bwd, &a);
    EC e; memset(&e, 0, sizeof e); e.n = 128; e.V = 1024; e.len = 200; e.tab = a.tab; e.sc = IA(1024, 300); e.a = FA(1024); e.b = FA(1024); e.c = FA(1024); e.z = IA(1024, 1 << 16); e.it = 0.5f;
    for (size_t i = 0; i < 1024; i++) e.a[i] = fabsf(e.a[i]) * 0.1f;
    printf("== softmax / entropia (n=128, V=1024) ==\n");
    report("softmax_fwd", nn_softmax_fwd, sc_softmax_fwd, &e);
    report("softmax_bwd", nn_softmax_bwd, sc_softmax_bwd, &e);
    for (size_t i = 0; i < 1024; i++) e.a[i] = (float)((int)(rnd64() % 4000) - 2000) * 0.01f;
    report("xent_row (V1024)", nn_xent, sc_xent, &e);
    report("dot_gz (n1024)", nn_dotgz, sc_dotgz, &e);
    RC r; memset(&r, 0, sizeof r); r.D = 128; r.eps = 1e-5f; r.scale = 0.7f; r.gp = (uint16_t *)A(2 * 8 * 2); r.gm = r.gp + 8; r.g = (int *)A(128 * sizeof(int));
    for (size_t j = 0; j < 128; j++) { r.g[j] = (int)(rnd64() % 3) - 1; if (r.g[j] > 0) r.gp[j >> 4] |= (uint16_t)(1u << (j & 15)); if (r.g[j] < 0) r.gm[j >> 4] |= (uint16_t)(1u << (j & 15)); }
    r.x = FA(128); r.go = FA(128); r.o = FA(128); r.gg = FA(128); r.gx = FA(128);
    printf("== RMSNorm (D=128) ==\n");
    report("rmsnorm_fwd", nn_rms_fwd, sc_rms_fwd, &r);
    report("rmsnorm_bwd", nn_rms_bwd, sc_rms_bwd, &r);
    SC s; memset(&s, 0, sizeof s); s.H = 256; s.D = 128; s.zg = IA(256, 128); s.zu = IA(256, 128); s.silu = FA(257); s.dsilu = FA(257); s.gt = FA(256); s.gu = FA(256); s.ga = FA(256); s.h = FA(256); s.fa = FA(256); s.fu = FA(256); s.su = 0.01f;
    printf("== SwiGLU (H=256) ==\n");
    for (int mode = 0; mode <= 2; mode += 2) {
        s.mode = mode; const char *mn = mode == 0 ? "silu" : "relu2"; char nm[48];
        snprintf(nm, sizeof nm, "swiglu_prod_z %s", mn); report(nm, nn_sw_prod, sc_sw_prod, &s);
        snprintf(nm, sizeof nm, "swiglu_bwd_z %s", mn); report(nm, nn_sw_bwd, sc_sw_bwd, &s);
        snprintf(nm, sizeof nm, "swiglu_fwd_u8 %s", mn); report(nm, nn_sw_fwd_u8, sc_sw_fwd_u8, &s);
        snprintf(nm, sizeof nm, "swiglu_bwd_u8 %s", mn); report(nm, nn_sw_bwd_u8, sc_sw_bwd_u8, &s);
    }
    bench_elem_main();
    return 0;
}
