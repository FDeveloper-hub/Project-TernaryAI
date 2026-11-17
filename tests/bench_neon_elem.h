/* Elementales, R2, cuantizacion y normalizacion por elemento (n = 256: fila tipica de pesos/activaciones). Se incluye desde bench_neon.c. */
typedef struct { size_t n; float *a, *b, *o; int32_t *z, *zo; uint8_t *q; float sc; int32_t zp; uint64_t *mask, *sign; int8_t *cnt; uint8_t *cr; float rmax, *v; } XC;

static NOINL void sc_add(void *c_) { XC *c = c_; for (size_t i = 0; i < c->n; i++) c->o[i] = c->a[i] + c->b[i]; }
static void nn_add(void *c_) { XC *c = c_; ternary_neon_add_f32(c->o, c->a, c->b, c->n); }
static NOINL void sc_acc2(void *c_) { XC *c = c_; for (size_t i = 0; i < c->n; i++) c->o[i] += (c->a[i] + c->b[i]); }
static void nn_acc2(void *c_) { XC *c = c_; ternary_neon_acc_add2(c->o, c->a, c->b, c->n); }
static NOINL void sc_scale(void *c_) { XC *c = c_; for (size_t i = 0; i < c->n; i++) c->o[i] *= 0.999f; }
static void nn_scale(void *c_) { XC *c = c_; ternary_neon_scale_f32(c->o, c->n, 0.999f); }
static NOINL void sc_shift(void *c_) { XC *c = c_; for (size_t i = 0; i < c->n; i++) c->zo[i] = (int32_t)((uint32_t)c->z[i] + (1u << 7)) >> 8; }
static void nn_shift(void *c_) { XC *c = c_; ternary_neon_shift_round_i32(c->z, c->n, 8, c->zo); }
static NOINL void sc_scale_i(void *c_) { XC *c = c_; for (size_t i = 0; i < c->n; i++) c->o[i] = 0.01f * (float)c->z[i]; }
static void nn_scale_i(void *c_) { XC *c = c_; ternary_neon_scale_i32_f32(c->z, c->n, 0.01f, c->o, 1); }
static NOINL void sc_smear(void *c_) { XC *c = c_; for (size_t i = 0; i < c->n; i++) { const int32_t a = c->z[i]; c->zo[i] = (int32_t)((uint32_t)a + (uint32_t)c->zo[i]) >> 1; } }
static void nn_smear(void *c_) { XC *c = c_; ternary_neon_smear_k(c->z, c->zo, c->zo, c->n, 0); }
static NOINL void sc_quant(void *c_) { XC *c = c_; const float *xr = c->a; float lo = 0.0f, hi = 0.0f;
    for (size_t i = 0; i < c->n; i++) { const float v = xr[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
    float sc = (hi - lo) / 255.0f; int zp = (int)rintf(-lo / sc); if (zp < 0) zp = 0; if (zp > 255) zp = 255; const float inv = 1.0f / sc;
    for (size_t i = 0; i < c->n; i++) { float qf = rintf(xr[i] * inv) + (float)zp; if (qf < 0.0f) qf = 0.0f; if (qf > 255.0f) qf = 255.0f; c->q[i] = (uint8_t)qf; } c->sc = sc; c->zp = zp; }
static void nn_quant(void *c_) { XC *c = c_; ternary_neon_u8_quantize_row(c->a, c->n, c->q, &c->sc, &c->zp); }
static NOINL void sc_r2(void *c_) { XC *c = c_; const float cb = 0.999f, cg = 0.03f * 127.0f; const int ti = 90, tin = 60, tout = 120, ri = 45;
    const double ms = ternary_canon_sumsq(c->a, c->n) / (double)c->n; const float inv = 1.0f / ((float)sqrt(ms) + 1e-12f);
    for (size_t i = 0; i < c->n; i++) {
        float gh = c->a[i] * inv; gh = gh > 3.0f ? 3.0f : (gh < -3.0f ? -3.0f : gh);
        float x = cb * (float)c->cnt[i] - cg * gh; if (!(x == x)) x = 0.0f;
        int v = (int)rintf(x); v = v > 127 ? 127 : (v < -127 ? -127 : v);
        const size_t word = i >> 6; const uint64_t m = (uint64_t)1 << (i & 63);
        const int cur = (c->mask[word] & m) ? ((c->sign[word] & m) ? -1 : 1) : 0, dir = v > 0 ? 1 : -1;
        const int th = cur == 0 ? tin : (dir == -cur ? tout : ti);
        if (v >= th || v <= -th) { int nuevo = cur + dir; if (nuevo > 1) nuevo = 1; else if (nuevo < -1) nuevo = -1;
            if (nuevo == 0) { c->mask[word] &= ~m; c->sign[word] &= ~m; } else { c->mask[word] |= m; if (nuevo < 0) c->sign[word] |= m; else c->sign[word] &= ~m; } v = -dir * ri; }
        c->cnt[i] = (int8_t)v; } }
static void nn_r2(void *c_) { XC *c = c_; ternary_neon_r2_row(c->mask, c->sign, c->cnt, 0, c->a, c->n, 0.999f, 0.03f, 90, 60, 120, 45, 0); }
static inline int en_code_b(float x) { if (!(x >= 0x1p-16f)) return 15; int e; const float m = frexpf(x, &e); int r = -e + (m < 0.70710678f ? 1 : 0); return r < 0 ? 0 : (r > 15 ? 15 : r); }
static NOINL void sc_elemnorm(void *c_) { XC *c = c_; const float beta2 = 0.99f, omb = 0.01f, bc = 0.5f, eps = 1e-6f; const size_t W = c->n; const float rm = c->rmax; float mx = 0.0f;
    memcpy(c->o, c->a, W * 4);
    for (size_t k = 0; k < W; k++) { const int cc = (k & 1) ? (c->cr[k >> 1] >> 4) : (c->cr[k >> 1] & 15); const float vh = ldexpf(rm, -cc); const float x = c->o[k] * c->o[k]; const float vn = beta2 * vh + omb * x; c->v[k] = vn; if (vn > mx) mx = vn; }
    for (size_t k = 0; k < W; k++) { const int cc = mx > 0.0f ? en_code_b(c->v[k] / mx) : 15;
        if (k & 1) c->cr[k >> 1] = (uint8_t)((c->cr[k >> 1] & 0x0F) | (cc << 4)); else c->cr[k >> 1] = (uint8_t)((c->cr[k >> 1] & 0xF0) | cc);
        const float vq = ldexpf(mx, -cc); c->o[k] = c->o[k] / (sqrtf(vq / bc) + eps); } c->rmax = mx; }
static void nn_elemnorm(void *c_) { XC *c = c_; memcpy(c->o, c->a, c->n * 4); ternary_neon_elemnorm_row(c->cr, &c->rmax, c->o, c->v, c->n, 0.99f, 0.01f, 0.5f, 1e-6f); }

static void bench_elem_main(void) {
    XC x; memset(&x, 0, sizeof x); x.n = 256;
    x.a = FA(256 + 16); x.b = FA(256 + 16); x.o = FA(256 + 16); x.z = IA(256 + 16, 1 << 16); x.zo = IA(256 + 16, 1 << 16); x.q = (uint8_t *)A(256 + 16); x.v = FA(256 + 16);
    x.mask = (uint64_t *)A(8 * 8); x.sign = (uint64_t *)A(8 * 8); x.cnt = (int8_t *)A(256 + 16); x.cr = (uint8_t *)A(256); x.rmax = 0.5f;
    for (int i = 0; i < 4; i++) { x.mask[i] = rnd64(); x.sign[i] = rnd64() & x.mask[i]; }
    for (int i = 0; i < 256; i++) x.cnt[i] = (int8_t)((int)(rnd64() % 101) - 50);
    for (int i = 0; i < 128; i++) x.cr[i] = (uint8_t)rnd64();
    for (int i = 0; i < 256; i++) x.a[i] = fabsf(x.a[i]) + 0.01f;      /* g > 0: v > 0 */
    printf("== elementales / R2 / cuantizacion (n=256) ==\n");
    report("add_f32", nn_add, sc_add, &x);
    report("acc_add2", nn_acc2, sc_acc2, &x);
    report("scale_f32", nn_scale, sc_scale, &x);
    report("shift_round_i32", nn_shift, sc_shift, &x);
    report("scale_i32_f32", nn_scale_i, sc_scale_i, &x);
    report("smear_k", nn_smear, sc_smear, &x);
    report("u8_quantize_row", nn_quant, sc_quant, &x);
    report("r2_row", nn_r2, sc_r2, &x);
    report("elemnorm_row", nn_elemnorm, sc_elemnorm, &x);
}
