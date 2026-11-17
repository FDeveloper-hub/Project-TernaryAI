/* Nucleos NEON de 128 bits: RMSNorm, SwiGLU y entropia cruzada: rmsnorm_forward, rmsnorm_backward, swiglu_prod_z, swiglu_bwd_z, swiglu_fwd_u8, swiglu_bwd_u8, xent_row, dot_gz.
 * Interfaz y reglas: include/ternary_neon.h. Los esqueletos para compilaciones sin NEON estan en ternary_neon_stubs.c.
 *
 * Igualdad de bits con el escalar de la libreria: producto y suma float en instrucciones separadas (sin vfma/vmla), las
 * sumas en double las hace el VFP escalar con 16 acumuladores independientes (carril = indice & 15) y la reduccion final
 * es ternary_canon_reduce_d16 (la misma que usa el escalar). Division y raiz: VFP escalar por elemento. Todas las
 * funciones son reentrantes (solo pila y las salidas recibidas). Diferencias aceptadas: denormales a cero (NEON) y la
 * carga util de un NaN (NEON produce el NaN por defecto). */
#include "ternary_neon.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "ternary_fexp.h"

#if defined(__ARM_NEON) && defined(__GNUC__)
#include <arm_neon.h>

#define NINL static inline __attribute__((always_inline))

/* ---- cargas y escrituras de 4 carriles con cola parcial (nunca se toca mas alla de n) ---- */
NINL float32x4_t ldf(const float *p, size_t rem) {
    if (rem >= 4) return vld1q_f32(p);
    float t[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (size_t i = 0; i < rem; i++) t[i] = p[i];
    return vld1q_f32(t);
}
NINL int32x4_t ldi(const int32_t *p, size_t rem) {
    if (rem >= 4) return vld1q_s32(p);
    int32_t t[4] = {0, 0, 0, 0};
    for (size_t i = 0; i < rem; i++) t[i] = p[i];
    return vld1q_s32(t);
}
NINL void stf(float *p, float32x4_t v, size_t rem) {
    if (rem >= 4) { vst1q_f32(p, v); return; }
    float t[4];
    vst1q_f32(t, v);
    for (size_t i = 0; i < rem; i++) p[i] = t[i];
}

NINL uint32x4_t u32(uint32_t v) { return vdupq_n_u32(v); }
NINL float32x4_t f32(float v) { return vdupq_n_f32(v); }
NINL float32x4_t bitsf(uint32x4_t v) { return vreinterpretq_f32_u32(v); }
NINL uint32x4_t fbits(float32x4_t v) { return vreinterpretq_u32_f32(v); }

/* Bit j (0..3) de un nibble -> 0xFFFFFFFF en el carril j. */
NINL uint32x4_t mask4(uint32_t nib) {
    static const uint32_t sel[4] = {1, 2, 4, 8};
    return vtstq_u32(vdupq_n_u32(nib & 15u), vld1q_u32(sel));
}

/* a / b carril a carril con la division IEEE escalar del VFP (NEON de ARMv7 no la tiene). */
NINL float32x4_t div4(float32x4_t a, float32x4_t b) {
    float x[4], y[4];
    vst1q_f32(x, a); vst1q_f32(y, b);
    x[0] = x[0] / y[0]; x[1] = x[1] / y[1]; x[2] = x[2] / y[2]; x[3] = x[3] / y[3];
    return vld1q_f32(x);
}

/* tab[k[l]] por carril (sin gather en NEON); los indices ya estan recortados a la tabla. */
NINL float32x4_t gather4(const float *tab, int32x4_t k) {
    float32x4_t v = vdupq_n_f32(0.0f);
    v = vld1q_lane_f32(tab + vgetq_lane_s32(k, 0), v, 0);
    v = vld1q_lane_f32(tab + vgetq_lane_s32(k, 1), v, 1);
    v = vld1q_lane_f32(tab + vgetq_lane_s32(k, 2), v, 2);
    v = vld1q_lane_f32(tab + vgetq_lane_s32(k, 3), v, 3);
    return v;
}
/* Indice de tabla: clamp(zg + D, 0, 2D). La suma saturada da el mismo resultado que la de 64 bits del escalar (D > 0). */
NINL int32x4_t tab_index(int32x4_t zg, int32_t D) {
    int32x4_t k = vqaddq_s32(zg, vdupq_n_s32(D));
    k = vmaxq_s32(k, vdupq_n_s32(0));
    return vminq_s32(k, vdupq_n_s32(2 * D));
}
/* (float)sign(z): +1, -1 o 0. */
NINL float32x4_t sign4(int32x4_t z) {
    const uint32x4_t pos = vcgtq_s32(z, vdupq_n_s32(0)), neg = vcltq_s32(z, vdupq_n_s32(0));
    return bitsf(vorrq_u32(vandq_u32(pos, u32(0x3F800000u)), vandq_u32(neg, u32(0xBF800000u))));
}

/* ---- exponencial compartida (ternary_fexp2_core) ----
 * rintf(y) sin vrndn: r = (y + m) - m, m = copysign(2^23, y). Vale porque aqui y esta en [CUTOFF, 0] o es NaN; las
 * lineas con y < CUTOFF se anulan despues con la mascara (su valor intermedio no importa, vcvt satura sin fallar). */
NINL float32x4_t fexp2_v(float32x4_t y) {
    const float32x4_t m = bitsf(vorrq_u32(vandq_u32(fbits(y), u32(0x80000000u)), u32(0x4B000000u)));
    const float32x4_t fl = vsubq_f32(vaddq_f32(y, m), m);
    const float32x4_t f = vsubq_f32(y, fl);
    float32x4_t p = f32(TERNARY_FEXP_P0);
    p = vmulq_f32(p, f); p = vaddq_f32(p, f32(TERNARY_FEXP_P1));
    p = vmulq_f32(p, f); p = vaddq_f32(p, f32(TERNARY_FEXP_P2));
    p = vmulq_f32(p, f); p = vaddq_f32(p, f32(TERNARY_FEXP_P3));
    p = vmulq_f32(p, f); p = vaddq_f32(p, f32(TERNARY_FEXP_P4));
    p = vmulq_f32(p, f); p = vaddq_f32(p, f32(TERNARY_FEXP_P5));
    float32x4_t r = vmulq_f32(f, p); r = vaddq_f32(r, f32(1.0f));
    const int32x4_t e = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(fl), vdupq_n_s32(127)), 23);
    return vmulq_f32(r, vreinterpretq_f32_s32(e));
}
/* (y < CUTOFF) ? 0 : 2^y */
NINL float32x4_t fexp2_cut(float32x4_t y) {
    const uint32x4_t ok = vmvnq_u32(vcltq_f32(y, f32(TERNARY_FEXP_CUTOFF)));    /* !(y < CUTOFF): NaN pasa, como en el escalar */
    return bitsf(vandq_u32(ok, fbits(fexp2_v(y))));
}
/* ternary_expf_nonpos */
NINL float32x4_t expf_nonpos_v(float32x4_t x) { return fexp2_cut(vmulq_f32(x, f32(TERNARY_LOG2E_F))); }

/* silu(a) y su derivada (silu_f de ternary_swiglu.c). */
NINL void silu_v(float32x4_t a, float32x4_t *sv, float32x4_t *dsv) {
    const float32x4_t one = f32(1.0f);
    const float32x4_t y = vmulq_f32(vnegq_f32(vabsq_f32(a)), f32(1.44269504088896341f));
    const float32x4_t e = fexp2_cut(y);
    const float32x4_t den = vaddq_f32(one, e);
    const float32x4_t num = vbslq_f32(vcgeq_f32(a, f32(0.0f)), one, e);
    const float32x4_t sg = div4(num, den);
    *sv = vmulq_f32(a, sg);
    *dsv = vmulq_f32(sg, vaddq_f32(one, vmulq_f32(a, vsubq_f32(one, sg))));
}

/* ---- sumas double de 16 carriles: d[l] += (double)a[l] * (double)b[l], l < cnt (VFP escalar) ---- */
NINL void acc_prod(double *d, const float *a, const float *b, size_t cnt) {
    for (size_t l = 0; l < cnt; l++) d[l] += (double)a[l] * (double)b[l];
}
NINL double sumsq_row(const float *x, size_t n) {
    double d[16] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    size_t j = 0;
    for (; j + 16 <= n; j += 16) acc_prod(d, x + j, x + j, 16);
    if (j < n) acc_prod(d, x + j, x + j, n - j);
    return ternary_canon_reduce_d16(d);
}
static inline float inv_rms_from_ss(double ss, size_t D, float eps) {
    return (float)(1.0 / sqrt(ss / (double)D + (double)eps));
}

/* ======================================================================================================
 * RMSNorm
 * ====================================================================================================== */
void ternary_neon_rmsnorm_forward(size_t rows, size_t D, float eps, float scale, const uint16_t *gp,
                                  const uint16_t *gm, const float *x, float *out) {
    for (size_t i = 0; i < rows; i++) {
        const float *xi = x + i * D;
        float *oi = out + i * D;
        const float f = scale * inv_rms_from_ss(sumsq_row(xi, D), D, eps);
        const float32x4_t vf = f32(f);
        for (size_t j = 0; j < D; j += 4) {
            const size_t rem = D - j, t = j >> 4, sh = j & 15;
            const uint32x4_t pm = mask4((uint32_t)(gp[t] >> sh)), nm = mask4((uint32_t)(gm[t] >> sh));
            const float32x4_t v = vmulq_f32(vf, ldf(xi + j, rem));
            const uint32x4_t r = veorq_u32(vandq_u32(fbits(v), vorrq_u32(pm, nm)), vandq_u32(nm, u32(0x80000000u)));
            stf(oi + j, bitsf(r), rem);
        }
    }
}

void ternary_neon_rmsnorm_backward(size_t rows, size_t D, float eps, float scale, const uint16_t *gp,
                                   const uint16_t *gm, const float *x, const float *go,
                                   float *grad_gain_out, float *grad_x_out, double *gscale) {
    const float32x4_t vs = f32(scale);
    for (size_t i = 0; i < rows; i++) {
        const float *xi = x + i * D, *gi = go + i * D;
        const float r = inv_rms_from_ss(sumsq_row(xi, D), D, eps);
        const float32x4_t vr = f32(r);
        double d[16] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        /* Por tandas de 16 elementos: la parte float en NEON, las sumas double en el VFP. */
        for (size_t j0 = 0; j0 < D; j0 += 16) {
            const size_t cnt = (D - j0 < 16) ? D - j0 : 16, t = j0 >> 4;
            float sgo[16], xs[16];
            for (size_t c = 0; c < 16; c += 4) {
                if (c >= cnt) break;
                const size_t rem = cnt - c;
                const uint32x4_t pm = mask4((uint32_t)(gp[t] >> c)), nm = mask4((uint32_t)(gm[t] >> c));
                const float32x4_t xv = ldf(xi + j0 + c, rem), gv = ldf(gi + j0 + c, rem);
                /* grad_gain[j] += ((go*s)*x)*r */
                const float32x4_t gg = vmulq_f32(vmulq_f32(vmulq_f32(gv, vs), xv), vr);
                float32x4_t acc = ldf(grad_gain_out + j0 + c, rem);
                stf(grad_gain_out + j0 + c, vaddq_f32(acc, gg), rem);
                /* g*go es exacto en float (+-go, o 0*go) y (double)g*(double)go tambien: mismo valor que el escalar */
                const float32x4_t gf = bitsf(vorrq_u32(vandq_u32(pm, u32(0x3F800000u)), vandq_u32(nm, u32(0xBF800000u))));
                vst1q_f32(sgo + c, vmulq_f32(gf, gv));
                vst1q_f32(xs + c, xv);
            }
            if (cnt == 16) acc_prod(d, sgo, xs, 16); else acc_prod(d, sgo, xs, cnt);
        }
        const double dot = ternary_canon_reduce_d16(d);
        *gscale += dot * (double)r;
        if (grad_x_out) {
            float *gx = grad_x_out + i * D;
            const float c = (float)(dot * (double)r * (double)r / (double)D);
            const float sr = scale * r;
            const float32x4_t vc = f32(c), vsr = f32(sr);
            for (size_t j = 0; j < D; j += 4) {
                const size_t rem = D - j, t = j >> 4, sh = j & 15;
                const uint32x4_t pm = mask4((uint32_t)(gp[t] >> sh)), nm = mask4((uint32_t)(gm[t] >> sh));
                const float32x4_t xv = ldf(xi + j, rem), gv = ldf(gi + j, rem);
                const float32x4_t g = bitsf(vorrq_u32(vandq_u32(pm, u32(0x3F800000u)), vandq_u32(nm, u32(0xBF800000u))));
                const float32x4_t inner = vsubq_f32(vmulq_f32(g, gv), vmulq_f32(xv, vc));
                stf(gx + j, vaddq_f32(ldf(gx + j, rem), vmulq_f32(vsr, inner)), rem);
            }
        }
    }
}

/* ======================================================================================================
 * SwiGLU elemento a elemento (modo 0 silu, 1 sign, 2 ReLU2)
 * ====================================================================================================== */
/* a > 0 ? a*a : 0 */
NINL float32x4_t relu2_v(float32x4_t a) {
    return bitsf(vandq_u32(vcgtq_f32(a, f32(0.0f)), fbits(vmulq_f32(a, a))));
}

void ternary_neon_swiglu_prod_z(int mode, size_t H, const int32_t *zg, const int32_t *zu, float su,
                                const float *silu_tab, int32_t D, float *h) {
    const float32x4_t vsu = f32(su);
    for (size_t j = 0; j < H; j += 4) {
        const size_t rem = H - j;
        const int32x4_t u = ldi(zu + j, rem);
        float32x4_t r;
        if (mode == 0) {
            const int32x4_t k = tab_index(ldi(zg + j, rem), D);
            r = vmulq_f32(gather4(silu_tab, k), vmulq_f32(vsu, vcvtq_f32_s32(u)));
        } else if (mode == 2) {
            r = relu2_v(vmulq_f32(vsu, vcvtq_f32_s32(u)));
        } else {
            const int32x4_t g = ldi(zg + j, rem);
            int32x4_t v = vreinterpretq_s32_u32(vandq_u32(vtstq_s32(g, g), vreinterpretq_u32_s32(u)));
            v = vbslq_s32(vcltq_s32(g, vdupq_n_s32(0)), vnegq_s32(v), v);
            r = vcvtq_f32_s32(v);
        }
        stf(h + j, r, rem);
    }
}

void ternary_neon_swiglu_bwd_z(int mode, size_t H, const int32_t *zg, const int32_t *zu, const float *gt, float su,
                               const float *silu_tab, const float *dsilu_tab, int32_t D, float *gu, float *ga) {
    const float32x4_t vsu = f32(su);
    for (size_t j = 0; j < H; j += 4) {
        const size_t rem = H - j;
        const float32x4_t g = ldf(gt + j, rem);
        const float32x4_t u = vmulq_f32(vsu, vcvtq_f32_s32(ldi(zu + j, rem)));
        if (mode == 2) {
            stf(gu + j, bitsf(vandq_u32(vcgtq_f32(u, f32(0.0f)), fbits(vmulq_f32(vmulq_f32(g, f32(2.0f)), u)))), rem);
        } else if (mode == 0) {
            const int32x4_t k = tab_index(ldi(zg + j, rem), D);
            stf(gu + j, vmulq_f32(g, gather4(silu_tab, k)), rem);
            stf(ga + j, vmulq_f32(vmulq_f32(g, u), gather4(dsilu_tab, k)), rem);
        } else {
            stf(gu + j, vmulq_f32(g, sign4(ldi(zg + j, rem))), rem);
            stf(ga + j, vmulq_f32(g, u), rem);
        }
    }
}

void ternary_neon_swiglu_fwd_u8(int mode, size_t H, const float *fa, const float *fu, const int32_t *zg, float *h) {
    for (size_t j = 0; j < H; j += 4) {
        const size_t rem = H - j;
        float32x4_t r;
        if (mode == 2) {
            r = relu2_v(ldf(fu + j, rem));
        } else if (mode == 0) {
            float32x4_t sv, dsv;
            silu_v(ldf(fa + j, rem), &sv, &dsv);
            r = vmulq_f32(sv, ldf(fu + j, rem));
        } else {
            r = vmulq_f32(sign4(ldi(zg + j, rem)), ldf(fu + j, rem));
        }
        stf(h + j, r, rem);
    }
}

void ternary_neon_swiglu_bwd_u8(int mode, size_t H, const int32_t *zg, const int32_t *zu, const float *gt,
                                float kg, float ku, float *gu, float *ga) {
    const float32x4_t vkg = f32(kg), vku = f32(ku);
    for (size_t j = 0; j < H; j += 4) {
        const size_t rem = H - j;
        const float32x4_t g = ldf(gt + j, rem);
        const float32x4_t u = vmulq_f32(vku, vcvtq_f32_s32(ldi(zu + j, rem)));
        if (mode == 2) {
            stf(gu + j, bitsf(vandq_u32(vcgtq_f32(u, f32(0.0f)), fbits(vmulq_f32(vmulq_f32(g, f32(2.0f)), u)))), rem);
        } else if (mode == 0) {
            const float32x4_t a = vmulq_f32(vkg, vcvtq_f32_s32(ldi(zg + j, rem)));
            float32x4_t sv, dsv;
            silu_v(a, &sv, &dsv);
            stf(gu + j, vmulq_f32(g, sv), rem);
            stf(ga + j, vmulq_f32(vmulq_f32(g, u), dsv), rem);
        } else {
            stf(gu + j, vmulq_f32(g, sign4(ldi(zg + j, rem))), rem);
            stf(ga + j, vmulq_f32(g, u), rem);
        }
    }
}

/* ======================================================================================================
 * Entropia cruzada (una fila) y suma g*z de la escala de la cabeza
 * ====================================================================================================== */
/* e[l] = expf_nonpos(l[l] - m) para l < cnt <= 16 (la cola se rellena con m: no se lee mas alla). */
NINL void exp_tile(const float *l, size_t cnt, float m, float *e) {
    const float32x4_t vm = f32(m);
    for (size_t c = 0; c < 16; c += 4) {
        if (c >= cnt) break;
        vst1q_f32(e + c, expf_nonpos_v(vsubq_f32(cnt - c >= 4 ? vld1q_f32(l + c) : ({ float t[4] = {m, m, m, m}; for (size_t i = 0; i < cnt - c; i++) t[i] = l[c + i]; vld1q_f32(t); }), vm)));
    }
}

NINL int ranges_disjoint(const float *a, const float *b, size_t n) {
    const uintptr_t pa = (uintptr_t)a, pb = (uintptr_t)b, len = (uintptr_t)(n * sizeof(float));
    return pa + len <= pb || pb + len <= pa;
}

void ternary_neon_xent_row(const float *l, size_t V, size_t t, size_t n, float *g, int *ok_out, double *loss_out) {
    /* Maximo y primer argmax con la semantica del escalar (`if (v > m)`: los NaN se ignoran, salvo l[0] NaN). */
    float m = l[0];
    size_t arg = 0;
    if (m == m) {
        float32x4_t mv = f32(-INFINITY);
        const float32x4_t ninf = f32(-INFINITY);
        size_t v = 0;
        for (; v + 4 <= V; v += 4) {
            const float32x4_t x = vld1q_f32(l + v);
            mv = vmaxq_f32(mv, vbslq_f32(vceqq_f32(x, x), x, ninf));
        }
        float32x2_t mm = vmax_f32(vget_low_f32(mv), vget_high_f32(mv));
        float mx = vget_lane_f32(vpmax_f32(mm, mm), 0);
        for (; v < V; v++) if (l[v] > mx) mx = l[v];
        for (v = 0; v < V; v++) if (l[v] == mx) { arg = v; break; }
        m = l[arg];                         /* el escalar conserva el primer elemento igual al maximo (±0) */
    }
    *ok_out = (arg == t);
    const int store_e = g && ranges_disjoint(g, l, V);
    double d[16] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (size_t v = 0; v < V; v += 16) {
        const size_t cnt = (V - v < 16) ? V - v : 16;
        float e[16];
        exp_tile(l + v, cnt, m, e);
        if (cnt == 16) { for (int k = 0; k < 16; k++) d[k] += (double)e[k]; }
        else { for (size_t k = 0; k < cnt; k++) d[k] += (double)e[k]; }
        if (store_e) memcpy(g + v, e, cnt * sizeof(float));
    }
    const double sum = ternary_canon_reduce_d16(d);
    *loss_out = log(sum) - (double)(l[t] - m);
    if (g) {
        const double inv = 1.0 / (sum * (double)n);
        for (size_t v = 0; v < V; v += 16) {
            const size_t cnt = (V - v < 16) ? V - v : 16;
            float e[16];
            if (store_e) memcpy(e, g + v, cnt * sizeof(float)); else exp_tile(l + v, cnt, m, e);
            if (cnt == 16) { for (int k = 0; k < 16; k++) g[v + k] = (float)((double)e[k] * inv); }
            else { for (size_t k = 0; k < cnt; k++) g[v + k] = (float)((double)e[k] * inv); }
        }
        g[t] -= (float)(1.0 / (double)n);
    }
}

NINL void acc_gz(double *d, const float *g, const int32_t *z, size_t cnt, int use_sx, double sx) {
    if (use_sx) { for (size_t l = 0; l < cnt; l++) d[l] += ((double)g[l] * sx) * (double)z[l]; }
    else        { for (size_t l = 0; l < cnt; l++) d[l] += (double)g[l] * (double)z[l]; }
}

double ternary_neon_dot_gz(const float *g, const int32_t *z, size_t n, int use_sx, double sx) {
    double d[16] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    size_t j = 0;
    if (use_sx) { for (; j + 16 <= n; j += 16) acc_gz(d, g + j, z + j, 16, 1, sx); }
    else        { for (; j + 16 <= n; j += 16) acc_gz(d, g + j, z + j, 16, 0, sx); }
    if (j < n) acc_gz(d, g + j, z + j, n - j, use_sx, sx);
    return ternary_canon_reduce_d16(d);
}

#endif
