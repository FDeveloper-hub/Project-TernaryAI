#include "ternary_neon.h"
#include <stdlib.h>

/* GENERADO a partir de include/ternary_neon.h: esqueletos para las compilaciones sin NEON (x86, ARM sin -mfpu=neon).
 * ternary_neon_enabled() da 0 y nada mas se llama nunca. */
#if !(defined(__ARM_NEON) && defined(__GNUC__))

int ternary_neon_enabled(void) { return 0; }
size_t ternary_neon_find_range(const int32_t *ids, size_t from, size_t n, int32_t lo, int32_t hi) { (void)ids; (void)from; (void)n; (void)lo; (void)hi; abort(); }
void ternary_neon_gradw(float *base, size_t ld, size_t outd, size_t n, size_t S, const float *gT, const uint16_t *xp, const uint16_t *xn) { (void)base; (void)ld; (void)outd; (void)n; (void)S; (void)gT; (void)xp; (void)xn; abort(); }
void ternary_neon_gradx(float *const *gx, size_t S, size_t outd, size_t n, const float *gs, const uint16_t *wp, const uint16_t *wn) { (void)gx; (void)S; (void)outd; (void)n; (void)gs; (void)wp; (void)wn; abort(); }
void ternary_neon_pack_thr(const float *x, size_t n, float thr, uint64_t *mask, uint64_t *sign) { (void)x; (void)n; (void)thr; (void)mask; (void)sign; abort(); }
float ternary_neon_sum_abs(const float *x, size_t n) { (void)x; (void)n; abort(); }
void ternary_neon_rmsnorm_forward(size_t rows, size_t D, float eps, float scale, const uint16_t *gp, const uint16_t *gm, const float *x, float *out) { (void)rows; (void)D; (void)eps; (void)scale; (void)gp; (void)gm; (void)x; (void)out; abort(); }
void ternary_neon_rmsnorm_backward(size_t rows, size_t D, float eps, float scale, const uint16_t *gp, const uint16_t *gm, const float *x, const float *go, float *grad_gain_out, float *grad_x_out, double *gscale) { (void)rows; (void)D; (void)eps; (void)scale; (void)gp; (void)gm; (void)x; (void)go; (void)grad_gain_out; (void)grad_x_out; (void)gscale; abort(); }
void ternary_neon_bwd1_row(size_t in, const uint16_t *xp, const uint16_t *xn, const uint16_t *wp, const uint16_t *wn, float g, float gs, float *grad_row, float *gx, int accumulate) { (void)in; (void)xp; (void)xn; (void)wp; (void)wn; (void)g; (void)gs; (void)grad_row; (void)gx; (void)accumulate; abort(); }
void ternary_neon_unpack_row(const uint64_t *mask, const uint64_t *sign, size_t nbits_total, size_t bit0, size_t n, float scale, float *out, int32_t *raw) { (void)mask; (void)sign; (void)nbits_total; (void)bit0; (void)n; (void)scale; (void)out; (void)raw; abort(); }
void ternary_neon_scores_i32(const int32_t *a, const int32_t *kb, size_t kstride, size_t H, size_t count, int shift, int32_t lim, int32_t *out) { (void)a; (void)kb; (void)kstride; (void)H; (void)count; (void)shift; (void)lim; (void)out; abort(); }
void ternary_neon_qk_grad(float *gq, const int32_t *qv, const int32_t *kb, float *gk, size_t kstride, size_t H, size_t count, const float *ds) { (void)gq; (void)qv; (void)kb; (void)gk; (void)kstride; (void)H; (void)count; (void)ds; abort(); }
void ternary_neon_outer_add(float *dst, size_t stride, size_t H, size_t count, const float *p, const float *ga) { (void)dst; (void)stride; (void)H; (void)count; (void)p; (void)ga; abort(); }
void ternary_neon_attn_scores(const int32_t *const *q, size_t G, const int32_t *kT, size_t ld, size_t H, size_t n, int shift, int32_t lim, int32_t *sc, size_t scs, int32_t *m_out) { (void)q; (void)G; (void)kT; (void)ld; (void)H; (void)n; (void)shift; (void)lim; (void)sc; (void)scs; (void)m_out; abort(); }
void ternary_neon_attn_pv(size_t G, const int32_t *sc, size_t scs, const int32_t *m, const float *tab, uint32_t K0, size_t n, const float *v, size_t vs, size_t H, float *e, size_t es, float *out, float *inv_out) { (void)G; (void)sc; (void)scs; (void)m; (void)tab; (void)K0; (void)n; (void)v; (void)vs; (void)H; (void)e; (void)es; (void)out; (void)inv_out; abort(); }
void ternary_neon_attn_bwd(size_t G, size_t n, size_t H, const int32_t *sc, size_t scs, const int32_t *m, const float *inv, const float *tab, uint32_t K0, const float *ga, const float *o, const float *vT, size_t ld, float it, float gam, const int32_t *qr, const int32_t *kr, size_t ks, float *gq, float *gk, float *gv, size_t gs, float *p, float *ds, size_t es, float *qf) { (void)G; (void)n; (void)H; (void)sc; (void)scs; (void)m; (void)inv; (void)tab; (void)K0; (void)ga; (void)o; (void)vT; (void)ld; (void)it; (void)gam; (void)qr; (void)kr; (void)ks; (void)gq; (void)gk; (void)gv; (void)gs; (void)p; (void)ds; (void)es; (void)qf; abort(); }
void ternary_neon_attn_pv2(size_t G, const int32_t *sc, size_t scs, const int32_t *m, const float *tab, uint32_t K0, size_t nA, size_t nB, size_t spA, const float *vA, const float *vB, size_t vs, size_t H, float *e, size_t es, float *out, float *inv_out) { (void)G; (void)sc; (void)scs; (void)m; (void)tab; (void)K0; (void)nA; (void)nB; (void)spA; (void)vA; (void)vB; (void)vs; (void)H; (void)e; (void)es; (void)out; (void)inv_out; abort(); }
void ternary_neon_attn_bwd2(size_t G, size_t nA, size_t nB, size_t spA, size_t H, const int32_t *sc, size_t scs, const int32_t *m, const float *inv, const float *tab, uint32_t K0, const float *ga, const float *o, const float *vTA, const float *vTB, size_t ld, float it, float gam, const int32_t *qr, const int32_t *krA, const int32_t *krB, size_t ks, float *gq, float *gkA, float *gkB, float *gvA, float *gvB, size_t gs, float *p, float *ds, size_t es, float *qf) { (void)G; (void)nA; (void)nB; (void)spA; (void)H; (void)sc; (void)scs; (void)m; (void)inv; (void)tab; (void)K0; (void)ga; (void)o; (void)vTA; (void)vTB; (void)ld; (void)it; (void)gam; (void)qr; (void)krA; (void)krB; (void)ks; (void)gq; (void)gkA; (void)gkB; (void)gvA; (void)gvB; (void)gs; (void)p; (void)ds; (void)es; (void)qf; abort(); }
int32_t ternary_neon_dot_u8(const uint64_t *wm, const uint64_t *ws, size_t n, const uint8_t *q, int32_t zp) { (void)wm; (void)ws; (void)n; (void)q; (void)zp; abort(); }
void ternary_neon_u8_to_f32(const uint8_t *q, size_t n, float sx, int32_t zp, float *out) { (void)q; (void)n; (void)sx; (void)zp; (void)out; abort(); }
void ternary_neon_outer_acc(float *gw, size_t n, float g, const float *xf) { (void)gw; (void)n; (void)g; (void)xf; abort(); }
void ternary_neon_gradx_row(float *gx, size_t n, const uint64_t *wm, const uint64_t *ws, float gs) { (void)gx; (void)n; (void)wm; (void)ws; (void)gs; abort(); }
void ternary_neon_swiglu_prod_z(int mode, size_t H, const int32_t *zg, const int32_t *zu, float su, const float *silu_tab, int32_t D, float *h) { (void)mode; (void)H; (void)zg; (void)zu; (void)su; (void)silu_tab; (void)D; (void)h; abort(); }
void ternary_neon_swiglu_bwd_z(int mode, size_t H, const int32_t *zg, const int32_t *zu, const float *gt, float su, const float *silu_tab, const float *dsilu_tab, int32_t D, float *gu, float *ga) { (void)mode; (void)H; (void)zg; (void)zu; (void)gt; (void)su; (void)silu_tab; (void)dsilu_tab; (void)D; (void)gu; (void)ga; abort(); }
void ternary_neon_swiglu_fwd_u8(int mode, size_t H, const float *fa, const float *fu, const int32_t *zg, float *h) { (void)mode; (void)H; (void)fa; (void)fu; (void)zg; (void)h; abort(); }
void ternary_neon_swiglu_bwd_u8(int mode, size_t H, const int32_t *zg, const int32_t *zu, const float *gt, float kg, float ku, float *gu, float *ga) { (void)mode; (void)H; (void)zg; (void)zu; (void)gt; (void)kg; (void)ku; (void)gu; (void)ga; abort(); }
void ternary_neon_xent_row(const float *l, size_t V, size_t t, size_t n, float *g, int *ok_out, double *loss_out) { (void)l; (void)V; (void)t; (void)n; (void)g; (void)ok_out; (void)loss_out; abort(); }
double ternary_neon_dot_gz(const float *g, const int32_t *z, size_t n, int use_sx, double sx) { (void)g; (void)z; (void)n; (void)use_sx; (void)sx; abort(); }
void ternary_neon_scale_f32(float *x, size_t n, float s) { (void)x; (void)n; (void)s; abort(); }
void ternary_neon_elemnorm_row(uint8_t *cr, float *rmax, float *g, float *v, size_t W, float beta2, float omb, float bc, float eps) { (void)cr; (void)rmax; (void)g; (void)v; (void)W; (void)beta2; (void)omb; (void)bc; (void)eps; abort(); }
void ternary_neon_r2_row(uint64_t *mask, uint64_t *sign, int8_t *cnt, size_t base, const float *g, size_t n, float beta, float inc, int ti, int tin, int tout, int ri, int pre) { (void)mask; (void)sign; (void)cnt; (void)base; (void)g; (void)n; (void)beta; (void)inc; (void)ti; (void)tin; (void)tout; (void)ri; (void)pre; abort(); }
void ternary_neon_u8_quantize_row(const float *x, size_t n, uint8_t *q, float *scale_out, int32_t *zp_out) { (void)x; (void)n; (void)q; (void)scale_out; (void)zp_out; abort(); }
void ternary_neon_add_f32(float *out, const float *a, const float *b, size_t n) { (void)out; (void)a; (void)b; (void)n; abort(); }
void ternary_neon_acc_add2(float *dst, const float *a, const float *b, size_t n) { (void)dst; (void)a; (void)b; (void)n; abort(); }
void ternary_neon_softmax_fwd(const int32_t *scores, size_t n_valid, const float *table, size_t table_len, float *out) { (void)scores; (void)n_valid; (void)table; (void)table_len; (void)out; abort(); }
void ternary_neon_softmax_bwd(const float *y, const float *g, size_t n, float inv_temp, float *out_ds) { (void)y; (void)g; (void)n; (void)inv_temp; (void)out_ds; abort(); }
void ternary_neon_rope_rotate_int(const int32_t *z, size_t heads, size_t H, size_t P, const int16_t *c, const int16_t *s, int32_t *out) { (void)z; (void)heads; (void)H; (void)P; (void)c; (void)s; (void)out; abort(); }
void ternary_neon_rope_unrotate_grad(float *grad, size_t heads, size_t H, size_t P, const int16_t *c, const int16_t *s, float inv) { (void)grad; (void)heads; (void)H; (void)P; (void)c; (void)s; (void)inv; abort(); }
void ternary_neon_shift_round_i32(const int32_t *z, size_t n, int shift, int32_t *out) { (void)z; (void)n; (void)shift; (void)out; abort(); }
void ternary_neon_sign_f32(const int32_t *z, size_t n, float *out, size_t stride) { (void)z; (void)n; (void)out; (void)stride; abort(); }
void ternary_neon_scale_i32_f32(const int32_t *z, size_t n, float k, float *out, size_t stride) { (void)z; (void)n; (void)k; (void)out; (void)stride; abort(); }
void ternary_neon_scatter_i32(const int32_t *src, size_t n, int32_t *dst, size_t stride) { (void)src; (void)n; (void)dst; (void)stride; abort(); }
void ternary_neon_smear_k(const int32_t *in, int32_t *out, int32_t *prev, size_t n, int first) { (void)in; (void)out; (void)prev; (void)n; (void)first; abort(); }
void ternary_neon_smear_grad_rows(float *g, size_t p0, size_t p1, size_t T, size_t w, const float *next_of_last) { (void)g; (void)p0; (void)p1; (void)T; (void)w; (void)next_of_last; abort(); }
void ternary_neon_expand_trits(const uint64_t *mask, const uint64_t *sign, size_t n, float *out) { (void)mask; (void)sign; (void)n; (void)out; abort(); }
void ternary_neon_pack_sign_i32(const int32_t *z, size_t n, uint64_t *mask, uint64_t *sign, size_t bit0) { (void)z; (void)n; (void)mask; (void)sign; (void)bit0; abort(); }

#endif
