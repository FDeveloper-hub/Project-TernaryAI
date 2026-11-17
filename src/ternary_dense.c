#include "ternary_mem.h"
#include "ternary_dense.h"
#include "ternary_kernels.h"
#include "ternary_avx512.h"
#include "ternary_neon.h"
#include "ternary_neon_policy.h"
#include "ternary_fexp.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/*
 * Extrae 'n' bits desde el bit 'offset' de un arreglo de palabras src (de
 * largo src_words) hacia dst (arreglo YA RESERVADO de ceil(n/64) palabras).
 * Bits mas alla del final de src se tratan como 0. Es la operacion que
 * reemplaza "leer la fila tal cual" cuando las filas viven todas
 * concatenadas en una sola matriz y no empiezan en un limite de palabra.
 * Trabaja palabra por palabra (bulk), nunca bit por bit.
 */
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

/*
 * Si input_dim es multiplo de 64, cada fila de W empieza justo en un limite
 * de palabra de 64 bits: la fila 'o' ES el tramo contiguo
 * W->mask[o*rw .. o*rw+rw), sin bits sobrantes. En ese caso NO hace falta
 * copiarla a un buffer (extract_bits): se pasa un puntero directo. La
 * extraccion era una parte grande del forward de CPU completo
 * (medido por fases durante el desarrollo). Es solo lectura -- seguro entre hilos.
 */
static inline int rows_word_aligned(const TernaryDense *layer) {
    return (layer->input_dim % 64) == 0;
}
static inline TernaryVec row_direct_view(const TernaryDense *layer, size_t row) {
    size_t rw = layer->input_dim / 64;
    TernaryVec v;
    v.mask = layer->W->mask + row * rw;
    v.sign = layer->W->sign + row * rw;
    v.n = layer->input_dim;
    v.words = rw;
    return v;
}

static void extract_row(const TernaryDense *layer, size_t row, TernaryVec *dst) {
    size_t offset = row * layer->input_dim;
    extract_bits(layer->W->mask, layer->W->words, offset, layer->input_dim, dst->mask);
    extract_bits(layer->W->sign, layer->W->words, offset, layer->input_dim, dst->sign);
}

TernaryDense *ternary_dense_alloc(size_t input_dim, size_t output_dim,
                                   int use_bias) {
    TernaryDense *layer = (TernaryDense *)ternary_malloc64(sizeof(TernaryDense));
    if (!layer) return NULL;
    layer->input_dim = input_dim;
    layer->output_dim = output_dim;
    layer->scale = 1.0f;

    layer->W = ternary_vec_alloc(input_dim * output_dim);
    layer->counter_W = (int8_t *)ternary_calloc64(input_dim * output_dim, sizeof(int8_t));
    layer->row_scratch = ternary_vec_alloc(input_dim);

    if (use_bias) {
        layer->bias = ternary_vec_alloc(output_dim);
        layer->counter_b = (int8_t *)ternary_calloc64(output_dim, sizeof(int8_t));
    } else {
        layer->bias = NULL;
        layer->counter_b = NULL;
    }
    return layer;
}

void ternary_dense_free(TernaryDense *layer) {
    if (!layer) return;
    ternary_vec_free(layer->W);
    free(layer->counter_W);
    ternary_vec_free(layer->row_scratch);
    if (layer->bias) ternary_vec_free(layer->bias);
    free(layer->counter_b);
    free(layer);
}

int ternary_dense_get_weight(const TernaryDense *layer, size_t o, size_t i) {
    return ternary_vec_get(layer->W, o * layer->input_dim + i);
}

void ternary_dense_set_weight(TernaryDense *layer, size_t o, size_t i, int value) {
    ternary_vec_set(layer->W, o * layer->input_dim + i, value);
}

/*
 * Producto punto de una fila NO alineada a palabra sin copiarla: el
 * desplazamiento de bits (funnel shift de dos palabras contiguas) se hace al
 * vuelo dentro del bucle de popcount, en vez de extract_bits() a un buffer y
 * luego releerlo. Mismo entero que extract_row + ternary_dot (solo x86-64 con
 * POPCNT; en otras arquitecturas, o con TERNARYAI_NO_FUSED_DOT=1, se usa el
 * camino de siempre).
 */
#if defined(__x86_64__) && defined(__GNUC__)
#define TERNARY_HAVE_FUSED_DOT 1
__attribute__((target("popcnt")))
static int32_t dot_unaligned_popcnt(const TernaryVec *W, size_t offset, size_t n, const TernaryVec *x) {
    const size_t nw = (n + 63) / 64;
    const size_t w0 = offset >> 6;
    const unsigned b = (unsigned)(offset & 63);
    const size_t src_words = W->words;
    const uint64_t *wm = W->mask, *ws = W->sign, *xm = x->mask, *xs = x->sign;
    int32_t total_active = 0, total_same = 0;
    const uint64_t keep = (n % 64) ? (((uint64_t)1 << (n % 64)) - 1) : ~(uint64_t)0;
    for (size_t k = 0; k < nw; k++) {
        uint64_t m, sg;
        int last = (k + 1 == nw);
        if (b == 0) {
            m = wm[w0 + k]; sg = ws[w0 + k];
        } else {
            uint64_t mlo = wm[w0 + k], slo = ws[w0 + k];
            uint64_t mhi = 0, shi = 0;
            if (!last || w0 + k + 1 < src_words) { mhi = wm[w0 + k + 1]; shi = ws[w0 + k + 1]; }
            m = (mlo >> b) | (mhi << (64 - b));
            sg = (slo >> b) | (shi << (64 - b));
        }
        if (last) { m &= keep; sg &= keep; }
        uint64_t active = m & xm[k];
        uint64_t same = active & ~(sg ^ xs[k]);
        total_active += __builtin_popcountll(active);
        total_same += __builtin_popcountll(same);
    }
    return 2 * total_same - total_active;
}

static int fused_dot_enabled(void) {
    static int v = -1;
    if (v < 0) {
        __builtin_cpu_init();
        v = __builtin_cpu_supports("popcnt") && !getenv("TERNARYAI_NO_FUSED_DOT");
    }
    return v;
}
#endif

void ternary_dense_forward(const TernaryDense *layer, const TernaryVec *x,
                            float *out, int32_t *out_raw_opt) {
    const int aligned = rows_word_aligned(layer);
    for (size_t o = 0; o < layer->output_dim; o++) {
        int32_t z;
        if (aligned) {
            TernaryVec row = row_direct_view(layer, o);
            z = ternary_dot(&row, x);
        }
#ifdef TERNARY_HAVE_FUSED_DOT
        else if (fused_dot_enabled()) {
            z = dot_unaligned_popcnt(layer->W, o * layer->input_dim, layer->input_dim, x);
        }
#endif
        else {
            extract_row(layer, o, layer->row_scratch);
            z = ternary_dot(layer->row_scratch, x);
        }
        if (layer->bias) z += ternary_vec_get(layer->bias, o);
        if (out_raw_opt) out_raw_opt[o] = z;
        out[o] = layer->scale * (float)z;
    }
}

void ternary_dense_forward_auto(const TernaryDense *layer, const TernaryVec *x,
                                 float *out, int32_t *out_raw_opt,
                                 const TernaryCapabilities *caps) {
    /* Camino escalar (unico). ternary_dot() se resuelve en ternary_core.c.
     *
     * ANTES: esto siempre llamaba a ternary_dense_forward() serial, el
     * pool de hilos (ternary_threadpool.c, verificado por separado desde
     * antes) nunca se usaba en el camino de produccion -- ver README.
     * ternary_dense_forward_parallel() ya trae su PROPIO umbral interno
     * (total_work<20000 -> serial) y cae sola a
     * ternary_dense_forward() si pool==NULL o tiene 1 solo hilo, asi que
     * llamarla incondicionalmente aca nunca es mas lento que el camino
     * de antes -- en el peor caso, es exactamente el camino de antes. */
    if (caps->cpu_pool) {
        ternary_dense_forward_parallel(layer, x, out, out_raw_opt, caps->cpu_pool);
    } else {
        ternary_dense_forward(layer, x, out, out_raw_opt);
    }
}

/* ---- forward por bloques ------------------------------------------------ */
typedef struct {
    const TernaryDense *layer;
    size_t S;
    const TernaryVec *const *xs;
    float *const *outs;
    int32_t *const *raws;
} ForwardBlockCtx;

static void forward_block_row(size_t o, void *ctx_) {
    const ForwardBlockCtx *c = (const ForwardBlockCtx *)ctx_;
    const TernaryDense *layer = c->layer;
    const size_t in = layer->input_dim;
    if (rows_word_aligned(layer)) {
        TernaryVec row = row_direct_view(layer, o);
        for (size_t s = 0; s < c->S; s++) {
            int32_t z = ternary_dot(&row, c->xs[s]);
            if (layer->bias) z += ternary_vec_get(layer->bias, o);
            c->raws[s][o] = z;
            c->outs[s][o] = layer->scale * (float)z;
        }
        return;
    }
#ifdef TERNARY_HAVE_FUSED_DOT
    if (fused_dot_enabled()) {
        for (size_t s = 0; s < c->S; s++) {
            int32_t z = dot_unaligned_popcnt(layer->W, o * in, in, c->xs[s]);
            if (layer->bias) z += ternary_vec_get(layer->bias, o);
            c->raws[s][o] = z;
            c->outs[s][o] = layer->scale * (float)z;
        }
        return;
    }
#endif
    enum { MAX_ROW_WORDS = 512 };
    uint64_t mask_buf[MAX_ROW_WORDS], sign_buf[MAX_ROW_WORDS];
    TernaryVec local = { mask_buf, sign_buf, in, (in + 63) / 64 };
    extract_bits(layer->W->mask, layer->W->words, o * in, in, local.mask);
    extract_bits(layer->W->sign, layer->W->words, o * in, in, local.sign);
    for (size_t s = 0; s < c->S; s++) {
        int32_t z = ternary_dot(&local, c->xs[s]);
        if (layer->bias) z += ternary_vec_get(layer->bias, o);
        c->raws[s][o] = z;
        c->outs[s][o] = layer->scale * (float)z;
    }
}

void ternary_dense_forward_block(const TernaryDense *layer, size_t S,
                                 const TernaryVec *const *xs, float *const *outs,
                                 int32_t *const *raws, const TernaryCapabilities *caps) {
    int fallback = 0;
#ifdef TERNARY_HAVE_FUSED_DOT
    int can_fused = fused_dot_enabled();
#else
    int can_fused = 0;
#endif
    if (!rows_word_aligned(layer) && !can_fused && (layer->input_dim + 63) / 64 > 512) fallback = 1;
    if (fallback) {
        for (size_t s = 0; s < S; s++) ternary_dense_forward_auto(layer, xs[s], outs[s], raws[s], caps);
        return;
    }
    ForwardBlockCtx ctx = { layer, S, xs, outs, raws };
    if (caps && caps->cpu_pool && layer->input_dim * layer->output_dim * S >= 20000 &&
        ternary_threadpool_num_threads(caps->cpu_pool) > 1) {
        ternary_parallel_for(caps->cpu_pool, 0, layer->output_dim, forward_block_row, &ctx);
    } else {
        for (size_t o = 0; o < layer->output_dim; o++) forward_block_row(o, &ctx);
    }
}

typedef struct {
    const TernaryDense *layer;
    const TernaryVec *x;
    float *out;
    int32_t *out_raw;
    /* Cada hilo necesita SU PROPIO scratch de extraccion de fila -- si
     * compartieran uno solo, dos hilos escribiendo filas distintas a la
     * vez se pisarian entre si (condicion de carrera real, no teorica). */
    TernaryVec **scratch_per_thread;
} ForwardCtx;

static void forward_one_row(size_t o, void *ctx_) {
    ForwardCtx *ctx = (ForwardCtx *)ctx_;
    const TernaryDense *layer = ctx->layer;
    /* Buffer de pila por-llamada (no compartido entre hilos -- si dos
     * hilos escriben filas distintas en el MISMO scratch, se pisan entre
     * si; cada llamada arma el suyo). 512 palabras = filas de hasta
     * 32768 pesos, generoso para el uso mobile de este proyecto -- pero
     * se verifica en tiempo de ejecucion en vez de romperse en silencio
     * si alguna capa futura lo excede. */
    if (rows_word_aligned(layer)) {
        /* Sin extraccion: ni buffer de pila ni limite de tamano de fila. */
        TernaryVec row = row_direct_view(layer, o);
        int32_t z = ternary_dot(&row, ctx->x);
        if (layer->bias) z += ternary_vec_get(layer->bias, o);
        if (ctx->out_raw) ctx->out_raw[o] = z;
        ctx->out[o] = layer->scale * (float)z;
        return;
    }
#ifdef TERNARY_HAVE_FUSED_DOT
    if (fused_dot_enabled()) {
        int32_t z = dot_unaligned_popcnt(layer->W, o * layer->input_dim, layer->input_dim, ctx->x);
        if (layer->bias) z += ternary_vec_get(layer->bias, o);
        if (ctx->out_raw) ctx->out_raw[o] = z;
        ctx->out[o] = layer->scale * (float)z;
        return;
    }
#endif
    enum { MAX_ROW_WORDS = 512 };
    if ((layer->input_dim + 63) / 64 > MAX_ROW_WORDS) {
        fprintf(stderr, "ternary_dense_forward_parallel: input_dim=%zu excede el buffer "
                        "de fila del path paralelo (max %d pesos). Usar ternary_dense_forward "
                        "(serial) para esta capa, o subir MAX_ROW_WORDS.\n",
                layer->input_dim, MAX_ROW_WORDS * 64);
        abort();
    }
    uint64_t mask_buf[MAX_ROW_WORDS], sign_buf[MAX_ROW_WORDS];
    TernaryVec local = { mask_buf, sign_buf, layer->input_dim, (layer->input_dim + 63) / 64 };
    size_t offset = o * layer->input_dim;
    extract_bits(layer->W->mask, layer->W->words, offset, layer->input_dim, local.mask);
    extract_bits(layer->W->sign, layer->W->words, offset, layer->input_dim, local.sign);

    int32_t z = ternary_dot(&local, ctx->x);
    if (layer->bias) z += ternary_vec_get(layer->bias, o);
    if (ctx->out_raw) ctx->out_raw[o] = z;
    ctx->out[o] = layer->scale * (float)z;
}

void ternary_dense_forward_parallel(const TernaryDense *layer, const TernaryVec *x,
                                     float *out, int32_t *out_raw_opt, TernaryThreadPool *pool) {
    size_t total_work = layer->input_dim * layer->output_dim;
    if (total_work < 20000 || ternary_threadpool_num_threads(pool) <= 1) {
        ternary_dense_forward(layer, x, out, out_raw_opt);
        return;
    }
    ForwardCtx ctx = { layer, x, out, out_raw_opt, NULL };
    ternary_parallel_for(pool, 0, layer->output_dim, forward_one_row, &ctx);
}

/* Tablas por byte: LUT_M[b][k] = bit k de b (0/1), LUT_S[b][k] = 1-2*bit.
 * w = M*S es exacto y se vectoriza como 2 cargas + 1 multiplicacion por 8
 * floats. 16 KB en total (caben en L1). Se llenan una vez; la carrera
 * entre hilos es benigna (todos escriben los mismos valores). */
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "expand_trits asume little-endian"
#endif
static float lut_m[256][8] __attribute__((aligned(32)));
static float lut_s[256][8] __attribute__((aligned(32)));
static uint16_t lut_m16[256][8] __attribute__((aligned(16)));
static uint16_t lut_s16[256][8] __attribute__((aligned(16))); /* 1 + bit de signo: digito base 3 = m*(1+s) */
static volatile int lut_ready = 0;
static void lut_init(void) {
    if (lut_ready) return;
    for (int b = 0; b < 256; b++)
        for (int k = 0; k < 8; k++) {
            int bit = (b >> k) & 1;
            lut_m[b][k] = (float)bit;
            lut_s[b][k] = (float)(1 - 2 * bit);
            lut_m16[b][k] = (uint16_t)bit;
            lut_s16[b][k] = (uint16_t)(1 + bit);
        }
    __sync_synchronize();
    lut_ready = 1;
}

/* Expande n trits (planos mask/sign, bit 0 = primer elemento) a floats
 * {-1,0,1}. */
static void expand_trits(const uint64_t *mask, const uint64_t *sign, size_t n,
                         float *out) {
    size_t nbytes = n / 8;
    const unsigned char *mb = (const unsigned char *)mask; /* little-endian */
    const unsigned char *sb = (const unsigned char *)sign;
    for (size_t j = 0; j < nbytes; j++) {
        const float *restrict m = lut_m[mb[j]];
        const float *restrict sg = lut_s[sb[j]];
        float *restrict o = out + j * 8;
        for (int k = 0; k < 8; k++) o[k] = m[k] * sg[k] + 0.0f; /* +0.0f: normaliza -0 a +0 (igual que (float)int) */
    }
    for (size_t i = nbytes * 8; i < n; i++) {
        int mbit = (int)((mb[i >> 3] >> (i & 7)) & 1u);
        int sbit = (int)((sb[i >> 3] >> (i & 7)) & 1u);
        out[i] = (float)(mbit * (1 - 2 * sbit));
    }
}

/*
 * Backward optimizado SIN cambiar un solo bit del resultado:
 *  - x y cada fila de W se expanden UNA vez a float {-1,0,1} (bulk por
 *    palabra) en vez de extraer bit a bit con funciones no inlineadas.
 *  - Los productos por -1/0/1 son exactos, asi que vectorizar/contraer
 *    elementwise no altera bits; el orden de acumulacion de grad_x
 *    (filas ascendentes) se conserva, y (g*scale)*w == g*scale*w (mismo
 *    orden de evaluacion que antes).
 * Los buffers son locales a la llamada (seguro entre hilos).
 */
static int avx512_bwd_enabled(void);
static void dense_backward_impl(const TernaryDense *layer, const TernaryVec *x,
                                const float *grad_out, float *grad_w_out,
                                float *grad_x_out, float *grad_scale_out,
                                float *grad_bias_out,
                                const int32_t *z_raw_opt, int accumulate) {
    const size_t in = layer->input_dim;
    const size_t rw = (in + 63) / 64;
    const int aligned = rows_word_aligned(layer);

    lut_init();
    const int avx = avx512_bwd_enabled();
    const size_t nt16 = (in + 15) / 16;
    /* Camino AVX-512: buffers pequenos en la pila (esta funcion se llama una vez por muestra: nada de malloc). */
    enum { STK_T = 256, STK_W = 64 };
    uint16_t pl_stk[4 * STK_T];
    uint64_t wbuf_stk[2 * STK_W];
    const int stk = avx && nt16 <= STK_T && rw <= STK_W;
    float *buf = (avx || !in) ? NULL : (float *)ternary_malloc64(2 * in * sizeof(float));
    uint64_t *wbuf = (aligned || stk) ? NULL : (uint64_t *)ternary_malloc64(2 * rw * sizeof(uint64_t));
    uint16_t *pl = !avx ? NULL : (stk ? pl_stk : (uint16_t *)ternary_malloc64(4 * (nt16 ? nt16 : 1) * sizeof(uint16_t)));
    if (!aligned && stk) wbuf = wbuf_stk;
    if ((!avx && in && !buf) || (!aligned && !wbuf) || (avx && !pl)) {
        free(buf); if (wbuf != wbuf_stk) free(wbuf); if (pl != pl_stk) free(pl);
        ternary_diag("dense_backward: sin memoria\n");
        return;
    }
    float *xf = buf;
    float *wf = buf ? buf + in : NULL;
    uint16_t *xpl = pl, *xmi = pl ? pl + nt16 : NULL, *wpl = pl ? pl + 2 * nt16 : NULL, *wmi = pl ? pl + 3 * nt16 : NULL;
    if (pl) ternary_avx512_split_planes(xpl, xmi, in, x->mask, x->sign);
    else expand_trits(x->mask, x->sign, in, xf);

    for (size_t o = 0; o < layer->output_dim; o++) {
        int32_t z_raw;
        if (z_raw_opt) {
            z_raw = z_raw_opt[o];
        } else {
            extract_row(layer, o, layer->row_scratch);
            z_raw = ternary_dot(layer->row_scratch, x);
            if (layer->bias) z_raw += ternary_vec_get(layer->bias, o);
        }

        float g_scaled = grad_out[o]; /* == "G_Z" en la version Python */
        *grad_scale_out += g_scaled * (float)z_raw;
        /* G_b: el bias ve una entrada implicita de 1 (ver historial). */
        if (grad_bias_out) grad_bias_out[o] += g_scaled;

        if (pl) {
            if (aligned) {
                ternary_avx512_split_planes(wpl, wmi, in, layer->W->mask + o * rw, layer->W->sign + o * rw);
            } else {
                size_t offset = o * in;
                extract_bits(layer->W->mask, layer->W->words, offset, in, wbuf);
                extract_bits(layer->W->sign, layer->W->words, offset, in, wbuf + rw);
                ternary_avx512_split_planes(wpl, wmi, in, wbuf, wbuf + rw);
            }
            ternary_avx512_bwd1_row(in, xpl, xmi, wpl, wmi, g_scaled, g_scaled * layer->scale,
                                    grad_w_out + o * in, grad_x_out, accumulate);
            continue;
        }
        if (aligned) {
            size_t off = o * rw;
            expand_trits(layer->W->mask + off, layer->W->sign + off, in, wf);
        } else {
            size_t offset = o * in;
            extract_bits(layer->W->mask, layer->W->words, offset, in, wbuf);
            extract_bits(layer->W->sign, layer->W->words, offset, in, wbuf + rw);
            expand_trits(wbuf, wbuf + rw, in, wf);
        }

        float *restrict grad_row = grad_w_out + o * in;
        const float *restrict xr = xf;
        const float *restrict wr = wf;
        float *restrict gx = grad_x_out;
        const float gs = g_scaled * layer->scale;
        if (accumulate) {
            for (size_t i = 0; i < in; i++) grad_row[i] += g_scaled * xr[i];
        } else {
            for (size_t i = 0; i < in; i++) grad_row[i] = g_scaled * xr[i];
        }
        for (size_t i = 0; i < in; i++) gx[i] += gs * wr[i];
    }
    free(buf);
    if (wbuf != wbuf_stk) free(wbuf);
    if (pl != pl_stk) free(pl);
}


/* ---- grad_w por TABLAS (suma de subconjuntos) ----------------------------
 * grad_w[o][i] += sum_s g[s][o] * x[s][i], con x en {-1,0,1}. En vez de S
 * multiplicar-sumar por peso, se agrupan las muestras de 6 en 6: para cada
 * fila 'o' y grupo se tabulan las 3^6=729 combinaciones posibles de (+g,-g,0)
 * y cada peso suma NG entradas de tabla elegidas por un indice base 3 que
 * depende SOLO de la entrada (se arma una vez por bloque y lo comparten todas
 * las filas). Cambia el orden de las sumas en float (error relativo ~3e-7,
 * medido), por eso esta ruta ya no es identica bit a bit a la de muestra a
 * muestra: se desactiva con TERNARYAI_GRADW_TABLES=0 (vuelve a multiplicar-
 * sumar, identica a la anterior). */
#define GW_K 6
#define GW_TS 729 /* 3^6 */
/* Valor por defecto: tablas ACTIVAS. TERNARYAI_*_TABLES=1/0 fuerza una u otra. */
#define TERNARY_TABLES_DEFAULT 1
static int tables_env(const char *name) {
    const char *e = getenv(name); /* una vez por bloque: costo despreciable */
    if (e && e[0] == '0') return 0;
    if (e && e[0] != '\0') return 1;
    return TERNARY_TABLES_DEFAULT;
}
/* Nucleos AVX-512 con mascaras (ternary_avx512.c): se usan si la CPU los tiene; TERNARYAI_AVX512_BWD=0 los apaga
 * solo para el backward de Dense (TERNARYAI_NO_AVX512=1 apaga todo AVX-512). */
static int avx512_bwd_enabled(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("TERNARYAI_AVX512_BWD"); v = ternary_avx512_enabled() && !(e && e[0] == '0'); }
    return v;
}
static int gradw_tables_enabled(void) { return tables_env("TERNARYAI_GRADW_TABLES"); }

/* grad_x por tablas (mismo esquema, filas de 6 en 6): TERNARYAI_GRADX_TABLES=0 lo desactiva. */
static int gradx_tables_enabled(void) { return tables_env("TERNARYAI_GRADX_TABLES"); }

/* Kernel de grad_x sin tablas: "blocked" (por defecto; W expandida por bloques de filas y reutilizada desde L1) o "rows"
 * (el anterior, recorre W por filas). Mismos bits; el benchmark del dispositivo decide: TERNARYAI_GRADX_KERNEL=rows|blocked. */
static int gradx_blocked_enabled(void) {
    const char *e = getenv("TERNARYAI_GRADX_KERNEL");
    return !(e && e[0] == 'r');
}

/* Backward de un bloque para el rango de columnas [c0, c0+w) de la entrada (c0 multiplo de 64). Cada
 * columna es independiente de las demas (grad_w[o][i] y grad_x[s][i] solo dependen de la columna i), y el
 * orden de suma de cada elemento es el mismo que sin dividir: partir las columnas entre hilos da EXACTAMENTE
 * los mismos bits que un solo hilo, tablas incluidas. do_sb: este rango tambien suma escala y bias. */
static void backward_block_cols(const TernaryDense *layer, size_t S,
                                const TernaryVec *const *xs, const float *const *grad_out,
                                const int32_t *const *z_raw,
                                float *grad_w_out, float *const *grad_x,
                                float *grad_scale_out, float *grad_bias_out,
                                const float *gT, const float *gsT,
                                size_t c0, size_t w, int do_sb) {
    enum { CH = 256 };
    const size_t full = layer->input_dim, outd = layer->output_dim;
    const size_t in = w;                       /* ancho del rango de columnas */
    const size_t frw = (full + 63) / 64;       /* palabras por fila de W completa */
    const size_t rw = (in + 63) / 64;
    const size_t cw = c0 >> 6;                 /* palabra de la primera columna */
    const int aligned = rows_word_aligned(layer);
    const int use_tab = gradw_tables_enabled();

    /* 1) escala y bias: orden (muestra, fila), igual que muestra a muestra. */
    if (do_sb) for (size_t s = 0; s < S; s++) {
        for (size_t o = 0; o < outd; o++) {
            float g = grad_out[s][o];
            *grad_scale_out += g * (float)z_raw[s][o];
            if (grad_bias_out) grad_bias_out[o] += g;
        }
    }

    if (avx512_bwd_enabled()) {
        /* Camino AVX-512: sin tablas ni expansion a float. gT[o*S+s] = g y gsT[s*outd+o] = g*escala (ver el llamador). */
        const size_t nt16 = (in + 15) / 16;
        uint16_t *xp = (uint16_t *)ternary_malloc64(S * nt16 * 2), *xn = (uint16_t *)ternary_malloc64(S * nt16 * 2);
        uint16_t *wp = grad_x ? (uint16_t *)ternary_malloc64(outd * nt16 * 2) : NULL, *wn = grad_x ? (uint16_t *)ternary_malloc64(outd * nt16 * 2) : NULL;
        uint64_t *wbuf2 = (grad_x && !aligned) ? (uint64_t *)ternary_malloc64((2 * rw + 2) * sizeof(uint64_t)) : NULL;
        if (!xp || !xn || (grad_x && (!wp || !wn)) || (grad_x && !aligned && !wbuf2)) {
            free(xp); free(xn); free(wp); free(wn); free(wbuf2);
            ternary_diag("dense_backward_block: sin memoria\n");
            return;
        }
        for (size_t s = 0; s < S; s++)
            ternary_avx512_split_planes(xp + s * nt16, xn + s * nt16, in, xs[s]->mask + cw, xs[s]->sign + cw);
        ternary_avx512_gradw(grad_w_out + c0, full, outd, in, S, gT, xp, xn);
        if (grad_x) {
            for (size_t o = 0; o < outd; o++) {
                if (aligned) {
                    ternary_avx512_split_planes(wp + o * nt16, wn + o * nt16, in, layer->W->mask + o * frw + cw, layer->W->sign + o * frw + cw);
                } else {
                    extract_bits(layer->W->mask, layer->W->words, o * full + c0, in, wbuf2);
                    extract_bits(layer->W->sign, layer->W->words, o * full + c0, in, wbuf2 + rw);
                    ternary_avx512_split_planes(wp + o * nt16, wn + o * nt16, in, wbuf2, wbuf2 + rw);
                }
            }
            float *gxp[256];
            for (size_t sb0 = 0; sb0 < S; sb0 += 256) {
                const size_t sn = (S - sb0 < 256) ? (S - sb0) : 256;
                for (size_t k = 0; k < sn; k++) gxp[k] = grad_x[sb0 + k] + c0;
                ternary_avx512_gradx(gxp, sn, outd, in, gsT + sb0 * outd, wp, wn);
            }
        }
        free(xp); free(xn); free(wp); free(wn); free(wbuf2);
        return;
    }

    lut_init();
    const int use_tab_x = grad_x && gradx_tables_enabled();
    /* Nucleos por tiles (ternary_kernels.h), identicos bit a bit al bucle de siempre: grad_w cuando no hay tablas
     * (gT lo arma el llamador), grad_x cuando no hay tablas y las filas de W estan alineadas a palabra (el byte de
     * mask/sign de 8 columnas se lee directo). */
    const int fast_w = gT != NULL;
    const int fast_x = grad_x && !use_tab_x && aligned && gsT != NULL;
    const size_t NG = (S + GW_K - 1) / GW_K;
    const size_t NGO = (outd + GW_K - 1) / GW_K;
    float *wf = (grad_x && !use_tab_x && !fast_x) ? (float *)ternary_malloc64((in ? in : 1) * sizeof(float)) : NULL;
    /* grad_w por tiles: el tile de x (S x CT) se expande a un bloque CONTIGUO (ld = CT: 512 B por fila, sin que las
     * filas caigan en los mismos sets del L1 de 4 vias del A53); resulto bastante mas rapido que expandir todo el
     * ancho y leer con ld = ancho del hilo. Mismos bits. xf (ancho completo) solo para el camino sin kernels. */
    const int need_xf = !use_tab && !fast_w;
    float *xf = need_xf ? (float *)ternary_malloc64(S * (in ? in : 1) * sizeof(float)) : NULL;
    float *xt = fast_w ? (float *)ternary_malloc64(S * TERNARY_KERNEL_CT * sizeof(float)) : NULL;
    const int blocked = fast_x && gradx_blocked_enabled();
    float *wsc = blocked ? (float *)ternary_malloc64(TERNARY_GRADX_OB * 8 * sizeof(float)) : NULL;   /* bloque de W expandido */
    uint16_t *idx = use_tab ? (uint16_t *)ternary_calloc64(NG * (in ? in : 1), sizeof(uint16_t)) : NULL;
    uint16_t *idxw = use_tab_x ? (uint16_t *)ternary_calloc64(NGO * (in ? in : 1), sizeof(uint16_t)) : NULL;
    float *tab = (use_tab || use_tab_x) ? (float *)ternary_malloc64((NG > 1 ? NG : 1) * GW_TS * sizeof(float)) : NULL;
    uint64_t *wbuf = (aligned || !grad_x) ? NULL : (uint64_t *)ternary_malloc64(2 * rw * sizeof(uint64_t));
    float *gs = (float *)ternary_malloc64(S * sizeof(float));
    if ((grad_x && !use_tab_x && !fast_x && !wf) || !gs || (grad_x && !aligned && !wbuf) ||
        (use_tab ? (!idx || !tab) : (need_xf ? !xf : !xt)) || (blocked && !wsc) || (use_tab_x && (!idxw || !tab))) {
        free(wf); free(xf); free(xt); free(wsc); free(idx); free(idxw); free(tab); free(wbuf); free(gs);
        ternary_diag("dense_backward_block: sin memoria\n");
        return;
    }
    if (use_tab) {
        static const uint16_t pow3[GW_K] = { 1, 3, 9, 27, 81, 243 };
        for (size_t s = 0; s < S; s++) {
            uint16_t *dst = idx + (s / GW_K) * in;
            const uint16_t p = pow3[s % GW_K];
            const uint64_t *xm = xs[s]->mask + cw, *xsg = xs[s]->sign + cw;
            for (size_t i = 0; i < in; i++) {
                unsigned m = (unsigned)((xm[i >> 6] >> (i & 63)) & 1u);
                unsigned sg = (unsigned)((xsg[i >> 6] >> (i & 63)) & 1u);
                dst[i] = (uint16_t)(dst[i] + p * (m + (m & sg))); /* +1 -> 1 ; -1 -> 2 ; 0 -> 0 */
            }
        }
    } else if (need_xf) {
        for (size_t s = 0; s < S; s++)
            tk_expand_trits(xs[s]->mask + cw, xs[s]->sign + cw, in, xf + s * in,
                            (const float (*)[8])lut_m, (const float (*)[8])lut_s);
    }

    for (size_t o = 0; o < outd; o++) {
        if (grad_x && !use_tab_x && !fast_x) {
            if (aligned) {
                size_t off = o * frw + cw;
                expand_trits(layer->W->mask + off, layer->W->sign + off, in, wf);
            } else {
                size_t offset = o * full + c0;
                extract_bits(layer->W->mask, layer->W->words, offset, in, wbuf);
                extract_bits(layer->W->sign, layer->W->words, offset, in, wbuf + rw);
                expand_trits(wbuf, wbuf + rw, in, wf);
            }
        }
        for (size_t s = 0; s < S; s++) gs[s] = grad_out[s][o] * layer->scale;

        float *row = grad_w_out + o * full + c0;
        if (use_tab) {
            for (size_t gr = 0; gr < NG; gr++) {
                float *T = tab + gr * GW_TS;
                T[0] = 0.0f;
                size_t size = 1;
                for (size_t j = 0; j < GW_K; j++) {
                    size_t sm = gr * GW_K + j;
                    float v = sm < S ? grad_out[sm][o] : 0.0f;
                    for (size_t d = 0; d < size; d++) { T[size + d] = T[d] + v; T[2 * size + d] = T[d] - v; }
                    size *= 3;
                }
            }
            for (size_t i = 0; i < in; i++) {
                float t = 0.0f;
                for (size_t gr = 0; gr < NG; gr++) t += tab[gr * GW_TS + idx[gr * in + i]];
                row[i] += t;
            }
        } else if (!fast_w) {
            for (size_t c = 0; c < in; c += CH) {
                size_t n = (in - c < CH) ? (in - c) : CH;
                float a[CH];
                memcpy(a, row + c, n * sizeof(float));
                for (size_t s = 0; s < S; s++) {
                    const float g = grad_out[s][o];
                    const float *restrict xr = xf + s * in + c;
                    for (size_t k = 0; k < n; k++) a[k] += g * xr[k];
                }
                memcpy(row + c, a, n * sizeof(float));
            }
        }
        if (grad_x && !use_tab_x && !fast_x) {
            const float *restrict wr = wf;
            for (size_t s = 0; s < S; s++) {
                float *restrict gx = grad_x[s] + c0;
                const float gsv = gs[s];
                for (size_t k = 0; k < in; k++) gx[k] += gsv * wr[k];
            }
        }
    }

    if (fast_w || fast_x) {
        /* Por tiles de columnas: el tile de x expandido (S x CT) queda en L1 mientras se recorren todas las filas. */
        const unsigned char *wmb = (const unsigned char *)(layer->W->mask + cw);
        const unsigned char *wsb = (const unsigned char *)(layer->W->sign + cw);
        const size_t wstride = frw * 8;
        for (size_t ct = 0; ct < in; ct += TERNARY_KERNEL_CT) {
            const size_t n = (in - ct < TERNARY_KERNEL_CT) ? (in - ct) : TERNARY_KERNEL_CT;
            if (fast_w) {
                for (size_t s = 0; s < S; s++)   /* ct es multiplo de 128 bits: empieza en palabra */
                    tk_expand_trits(xs[s]->mask + cw + (ct >> 6), xs[s]->sign + cw + (ct >> 6), n, xt + s * TERNARY_KERNEL_CT,
                                    (const float (*)[8])lut_m, (const float (*)[8])lut_s);
                for (size_t o = 0; o < outd; o++)
                    tk_gradw_row(grad_w_out + o * full + c0 + ct, n, S, gT + o * S, xt, TERNARY_KERNEL_CT);
            }
            if (fast_x) {
                const size_t n8 = n / 8;
                float *gxp[64]; /* punteros de fila por bloques de hasta 64 muestras */
                for (size_t sb0 = 0; sb0 < S; sb0 += 64) {
                    const size_t sn = (S - sb0 < 64) ? (S - sb0) : 64;
                    for (size_t t = 0; t < sn; t++) gxp[t] = grad_x[sb0 + t] + c0;
                    if (blocked)
                        tk_gradx_cols(gxp, ct, n8, sn, outd, gsT + (sb0 >> 2) * outd * 4, wmb + ct / 8, wsb + ct / 8, wstride,
                                      (const float (*)[8])lut_m, (const float (*)[8])lut_s, wsc);
                    else
                        tk_gradx_cols_rows(gxp, ct, n8, sn, outd, gsT + (sb0 >> 2) * outd * 4, wmb + ct / 8, wsb + ct / 8, wstride,
                                           (const float (*)[8])lut_m, (const float (*)[8])lut_s);
                }
                for (size_t i = ct + n8 * 8; i < ct + n; i++)   /* columnas sobrantes (< 8) */
                    for (size_t s = 0; s < S; s++) {
                        float a = grad_x[s][c0 + i];
                        for (size_t o = 0; o < outd; o++) {
                            const uint64_t *wm = layer->W->mask, *wsg = layer->W->sign;
                            int m = (int)((wm[o * frw + cw + (i >> 6)] >> (i & 63)) & 1u);
                            int sg = (int)((wsg[o * frw + cw + (i >> 6)] >> (i & 63)) & 1u);
                            a += gsT[(s >> 2) * outd * 4 + o * 4 + (s & 3)] * (float)(m * (1 - 2 * sg));
                        }
                        grad_x[s][c0 + i] = a;
                    }
            }
        }
    }

    if (use_tab_x) {
        /* grad_x[s][i] += sum_o gs[s][o]*w[o][i] por tablas: filas de 6 en 6, el
         * indice base 3 sale de W (compartido por todas las muestras). */
        static const uint16_t pow3[GW_K] = { 1, 3, 9, 27, 81, 243 };
        const uint64_t *wm = layer->W->mask, *wsg = layer->W->sign;
        for (size_t o = 0; o < outd; o++) {
            uint16_t *restrict d = idxw + (o / GW_K) * in;
            const uint16_t p = pow3[o % GW_K];
            const size_t bit0 = o * full + c0;
            if ((bit0 & 63) == 0) {
                const unsigned char *mb = (const unsigned char *)(wm + (bit0 >> 6));
                const unsigned char *sb = (const unsigned char *)(wsg + (bit0 >> 6));
                size_t nbytes = in / 8;
                for (size_t j = 0; j < nbytes; j++) {
                    const uint16_t *restrict lm = lut_m16[mb[j]], *restrict ls = lut_s16[sb[j]];
                    uint16_t *restrict dd = d + j * 8;
                    for (int k = 0; k < 8; k++) dd[k] = (uint16_t)(dd[k] + p * (uint16_t)(lm[k] * ls[k]));
                }
                for (size_t i = nbytes * 8; i < in; i++) {
                    unsigned m = (mb[i >> 3] >> (i & 7)) & 1u, sg = (sb[i >> 3] >> (i & 7)) & 1u;
                    d[i] = (uint16_t)(d[i] + p * (m + (m & sg)));
                }
            } else {
                extract_bits(wm, layer->W->words, bit0, in, wbuf);
                extract_bits(wsg, layer->W->words, bit0, in, wbuf + rw);
                const unsigned char *mb = (const unsigned char *)wbuf;
                const unsigned char *sb = (const unsigned char *)(wbuf + rw);
                size_t nbytes = in / 8;
                for (size_t j = 0; j < nbytes; j++) {
                    const uint16_t *restrict lm = lut_m16[mb[j]], *restrict ls = lut_s16[sb[j]];
                    uint16_t *restrict dd = d + j * 8;
                    for (int k = 0; k < 8; k++) dd[k] = (uint16_t)(dd[k] + p * (uint16_t)(lm[k] * ls[k]));
                }
                for (size_t i = nbytes * 8; i < in; i++) {
                    unsigned m = (mb[i >> 3] >> (i & 7)) & 1u, sg = (sb[i >> 3] >> (i & 7)) & 1u;
                    d[i] = (uint16_t)(d[i] + p * (m + (m & sg)));
                }
            }
        }
        float *T = tab; /* una sola tabla de 729 */
        for (size_t gr = 0; gr < NGO; gr++) {
            const uint16_t *restrict id = idxw + gr * in;
            for (size_t s = 0; s < S; s++) {
                T[0] = 0.0f;
                size_t size = 1;
                for (size_t j = 0; j < GW_K; j++) {
                    size_t o = gr * GW_K + j;
                    float v = o < outd ? grad_out[s][o] * layer->scale : 0.0f;
                    for (size_t d = 0; d < size; d++) { T[size + d] = T[d] + v; T[2 * size + d] = T[d] - v; }
                    size *= 3;
                }
                float *restrict gx = grad_x[s] + c0;
                for (size_t i = 0; i < in; i++) gx[i] += T[id[i]];
            }
        }
    }
    free(wf); free(xf); free(xt); free(wsc); free(idx); free(idxw); free(tab); free(wbuf); free(gs);
}

typedef struct {
    const TernaryDense *layer; size_t S;
    const TernaryVec *const *xs; const float *const *grad_out; const int32_t *const *z_raw;
    float *grad_w_out; float *const *grad_x; float *grad_scale_out; float *grad_bias_out;
    const float *gT, *gsT;
    size_t chunks, full;
} BwdBlockJob;

static void bwd_block_task(size_t idx, void *ctx) {
    const BwdBlockJob *j = (const BwdBlockJob *)ctx;
    /* columnas repartidas en multiplos de 64: la primera tarea toma el resto sobrante para no quedar corta */
    const size_t words = (j->full + 63) / 64;
    const size_t w0 = words * idx / j->chunks, w1 = words * (idx + 1) / j->chunks;
    const size_t c0 = w0 * 64, c1 = (w1 * 64 < j->full) ? w1 * 64 : j->full;
    if (c1 <= c0) return;
    backward_block_cols(j->layer, j->S, j->xs, j->grad_out, j->z_raw, j->grad_w_out, j->grad_x,
                        j->grad_scale_out, j->grad_bias_out, j->gT, j->gsT, c0, c1 - c0, idx == 0);
}

void ternary_dense_backward_block(const TernaryDense *layer, size_t S,
                                  const TernaryVec *const *xs, const float *const *grad_out,
                                  const int32_t *const *z_raw,
                                  float *grad_w_out, float *const *grad_x,
                                  float *grad_scale_out, float *grad_bias_out) {
    TernaryThreadPool *pool = ternary_get_compute_pool();
    const int nt = pool ? ternary_threadpool_num_threads(pool) : 1;
    const size_t full = layer->input_dim, outd = layer->output_dim;

    /* Gradientes por fila de salida en layout [o][s] para los nucleos por tiles (una sola vez por llamada, los
     * comparten todos los hilos): gT = g, gsT = g * escala (el mismo producto que antes). */
    const int avx = avx512_bwd_enabled();
    const int want_w = avx || !gradw_tables_enabled();
    const int want_x = grad_x && (avx || (!gradx_tables_enabled() && rows_word_aligned(layer)));
    float *gT = want_w ? (float *)ternary_malloc64(outd * S * sizeof(float)) : NULL;
    /* gsT en orden de grupo de 4 muestras (ver tk_gradx_cols); calloc: los huecos de un ultimo grupo incompleto no se leen */
    float *gsT = want_x ? (float *)ternary_calloc64(((S + 3) / 4) * outd * 4 > S * outd ? ((S + 3) / 4) * outd * 4 : S * outd, sizeof(float)) : NULL;
    if ((want_w && !gT) || (want_x && !gsT)) {
        free(gT); free(gsT);
        ternary_diag("dense_backward_block: sin memoria\n");
        return;
    }
    for (size_t s = 0; s < S; s++)
        for (size_t o = 0; o < outd; o++) {
            const float g = grad_out[s][o];
            if (gT) gT[o * S + s] = g;
            if (gsT) {
                if (avx) gsT[s * outd + o] = g * layer->scale;                       /* layout [s][o] para AVX-512 */
                else gsT[(s >> 2) * outd * 4 + o * 4 + (s & 3)] = g * layer->scale;
            }
        }

    size_t chunks = (size_t)nt;
    if (chunks > full / 64) chunks = full / 64;               /* al menos 64 columnas por hilo */
    /* trabajo ~ S*salidas*columnas: por debajo de este umbral coordinar hilos cuesta mas de lo que gana */
    if (chunks < 2 || (double)S * (double)layer->output_dim * (double)full < 2.0e5) {
        backward_block_cols(layer, S, xs, grad_out, z_raw, grad_w_out, grad_x, grad_scale_out, grad_bias_out,
                            gT, gsT, 0, full, 1);
    } else {
        lut_init();
        BwdBlockJob job = { layer, S, xs, grad_out, z_raw, grad_w_out, grad_x, grad_scale_out, grad_bias_out,
                            gT, gsT, chunks, full };
        ternary_parallel_for(pool, 0, chunks, bwd_block_task, &job);
    }
    free(gT); free(gsT);
}

void ternary_dense_backward(const TernaryDense *layer, const TernaryVec *x,
                             const float *grad_out, float *grad_w_out,
                             float *grad_x_out, float *grad_scale_out,
                             float *grad_bias_out,
                             const int32_t *z_raw_opt) {
    dense_backward_impl(layer, x, grad_out, grad_w_out, grad_x_out, grad_scale_out, grad_bias_out, z_raw_opt, 0);
}

void ternary_dense_backward_accumulate(const TernaryDense *layer, const TernaryVec *x,
                                        const float *grad_out, float *grad_w_out,
                                        float *grad_x_out, float *grad_scale_out,
                                        float *grad_bias_out,
                                        const int32_t *z_raw_opt) {
    dense_backward_impl(layer, x, grad_out, grad_w_out, grad_x_out, grad_scale_out, grad_bias_out, z_raw_opt, 1);
}

/* ---- Regla R2; ver la explicacion en ternary_dense.h ---- */
static _Thread_local TernaryUpdateRule g_rule = TERNARY_R2_RULE_DEFAULT;   /* por defecto, los de R2 */
void ternary_update_rule_set(const TernaryUpdateRule *r) {
    if (r) g_rule = *r; else { const TernaryUpdateRule d = TERNARY_R2_RULE_DEFAULT; g_rule = d; }
}
TernaryUpdateRule ternary_update_rule_get(void) { return g_rule; }

void ternary_r2_set_params(float inc, float zero_target, float reset) { g_rule.inc = inc; g_rule.zero_target = zero_target; g_rule.reset = reset; }
void ternary_r2_get_params(float *inc, float *zero_target, float *reset) { *inc = g_rule.inc; *zero_target = g_rule.zero_target; *reset = g_rule.reset; }

/* Regla R2, version escalar (ver ternary_dense.h). */
size_t ternary_r2_count_zeros(const TernaryVec *w, size_t base, size_t n) {
    size_t nz = 0, i = 0;
    while (i < n) {
        const size_t idx = base + i, b = idx & 63, take = (64 - b) < (n - i) ? (64 - b) : (n - i);
        uint64_t word = w->mask[idx >> 6] >> b; if (take < 64) word &= ((uint64_t)1 << take) - 1;
        nz += take - (size_t)__builtin_popcountll(word); i += take;
    }
    return nz;
}
void ternary_r2_thresholds(size_t nz, size_t n, const TernaryUpdateRule *r, int *ti, int *tin, int *tout, int *ri, float *inc) {
    const float tau = r->tau;
    float fin = 1.0f, fout = 1.0f;
    if (r->zero_target >= 0.0f) {
        const float d = ((float)nz / (float)n - r->zero_target) * 2.0f;
        float a = 1.0f - d, b = 1.0f + d;
        a = a < 0.4f ? 0.4f : (a > 1.6f ? 1.6f : a); b = b < 0.6f ? 0.6f : (b > 2.5f ? 2.5f : b);
        fin = a; fout = b;
    }
    int t0 = (int)lrintf(tau * 127.0f), t1 = (int)lrintf(tau * fin * 127.0f), t2 = (int)lrintf(tau * fout * 127.0f);
    if (t0 < 1) t0 = 1; else if (t0 > 127) t0 = 127;
    if (t1 < 1) t1 = 1; else if (t1 > 127) t1 = 127;
    if (t2 < 1) t2 = 1; else if (t2 > 127) t2 = 127;
    *ti = t0; *tin = t1; *tout = t2; *ri = (int)lrintf(r->reset * (float)t0); *inc = r->inc;
}

/* Regla R2, version escalar (ver ternary_dense.h). */
static void ternary_r2_row_scalar(TernaryVec *w, int8_t *cnt, size_t base, const float *g, size_t n, const TernaryUpdateRule *r, int pre) {
    float inv = 1.0f;       /* pre: el gradiente ya viene normalizado (normalizacion por elemento de la cabeza) */
    if (!pre) { const double ms = ternary_canon_sumsq(g, n) / (double)n; inv = 1.0f / ((float)sqrt(ms) + 1e-12f); }
    const float cb = r->beta;
    int ti, tin, tout, ri; float inc; ternary_r2_thresholds(ternary_r2_count_zeros(w, base, n), n, r, &ti, &tin, &tout, &ri, &inc);
    const float cg = inc * 127.0f;
    for (size_t i = 0; i < n; i++) {
        float gh = g[i] * inv;
        gh = gh > 3.0f ? 3.0f : (gh < -3.0f ? -3.0f : gh);
        float x = cb * (float)cnt[base + i] - cg * gh;
        if (!(x == x)) x = 0.0f;
        int v = (int)rintf(x); v = v > 127 ? 127 : (v < -127 ? -127 : v);
        const size_t idx = base + i, word = idx >> 6; const uint64_t m = (uint64_t)1 << (idx & 63);
        const int cur = (w->mask[word] & m) ? ((w->sign[word] & m) ? -1 : 1) : 0, dir = v > 0 ? 1 : -1;
        const int th = cur == 0 ? tin : (dir == -cur ? tout : ti);
        if (v >= th || v <= -th) {
            int nuevo = cur + dir; if (nuevo > 1) nuevo = 1; else if (nuevo < -1) nuevo = -1;
            if (nuevo == 0) { w->mask[word] &= ~m; w->sign[word] &= ~m; }
            else { w->mask[word] |= m; if (nuevo < 0) w->sign[word] |= m; else w->sign[word] &= ~m; }
            v = -dir * ri;
        }
        cnt[base + i] = (int8_t)v;
    }
}

void ternary_r2_update_row_ex(TernaryVec *w, int8_t *cnt, size_t base, const float *g, size_t n, const TernaryUpdateRule *r, int pre) {
    if (ternary_avx512_enabled()) {
        int ti, tin, tout, ri; float inc; ternary_r2_thresholds(ternary_r2_count_zeros(w, base, n), n, r, &ti, &tin, &tout, &ri, &inc);
        ternary_avx512_r2_row(w->mask, w->sign, cnt + base, base, g, n, r->beta, inc, ti, tin, tout, ri, pre);
    } else if (TN_ON(R2)) {
        int ti, tin, tout, ri; float inc; ternary_r2_thresholds(ternary_r2_count_zeros(w, base, n), n, r, &ti, &tin, &tout, &ri, &inc);
        ternary_neon_r2_row(w->mask, w->sign, cnt + base, base, g, n, r->beta, inc, ti, tin, tout, ri, pre);
    } else ternary_r2_row_scalar(w, cnt, base, g, n, r, pre);
}
void ternary_r2_update_row(TernaryVec *w, int8_t *cnt, size_t base, const float *g, size_t n, const TernaryUpdateRule *r) {
    ternary_r2_update_row_ex(w, cnt, base, g, n, r, 0);
}

/* ---- Normalizacion por elemento (ver ternary_dense.h) ---- */
#define EN_MT 0.70710678f   /* 2^-0.5 en float: frontera del redondeo logaritmico entre dos niveles */
TernaryElemNorm *ternary_elemnorm_alloc(size_t rows, size_t cols) {
    if (!rows || !cols) return NULL;
    TernaryElemNorm *n = (TernaryElemNorm *)ternary_calloc64(1, sizeof(*n));
    if (!n) return NULL;
    n->rows = rows; n->cols = cols;
    n->code = (uint8_t *)ternary_calloc64(rows * ((cols + 1) / 2), 1);
    n->rmax = (float *)ternary_calloc64(rows, sizeof(float));
    n->tmp = (float *)ternary_calloc64(cols, sizeof(float));
    if (!n->code || !n->rmax || !n->tmp) { ternary_elemnorm_free(n); return NULL; }
    return n;
}
void ternary_elemnorm_free(TernaryElemNorm *n) {
    if (!n) return;
    free(n->code); free(n->rmax); free(n->tmp); free(n);
}
size_t ternary_elemnorm_state_bytes(const TernaryElemNorm *n) { return n ? n->rows * ((n->cols + 1) / 2) + n->rows * sizeof(float) : 0; }

/* Codigo logaritmico de x = v/rmax en [0,1]: r = clamp(round(-log2 x), 0, 15), SIN log2: x = m*2^e con m en [0.5,1) ->
 * -log2 x = -e + (-log2 m), con -log2 m en (0,1]: redondea a -e + 1 si m < 2^-0.5 y a -e si no. Lo que cae por debajo de 2^-16 (cero y
 * denormales incluidos) es el nivel 15. Solo operaciones exactas: AVX-512 (getexp/getmant) da lo mismo. */
static inline int en_code(float x) {
    if (!(x >= 0x1p-16f)) return 15;
    int e; const float m = frexpf(x, &e);
    int r = -e + (m < EN_MT ? 1 : 0);
    return r < 0 ? 0 : (r > 15 ? 15 : r);
}

void ternary_elemnorm_apply_scalar(TernaryElemNorm *n, float *g, float beta2) {
    const size_t R = n->rows, W = n->cols, st = (W + 1) / 2, tot = R * W;
    n->steps++;
    const float bc = (float)(1.0 - pow((double)beta2, (double)n->steps));
    const double gm = ternary_canon_sumsq(g, tot) / (double)tot;
    const float eps = (float)(1e-3 * sqrt(gm) + 1e-20), omb = 1.0f - beta2;
    float *const v = n->tmp;
    for (size_t r = 0; r < R; r++) {
        uint8_t *cr = n->code + r * st; float *gr = g + r * W; const float rm = n->rmax[r]; float mx = 0.0f;
        for (size_t k = 0; k < W; k++) {
            const int c = (k & 1) ? (cr[k >> 1] >> 4) : (cr[k >> 1] & 15);
            const float vh = ldexpf(rm, -c);                    /* 0 si la fila no tiene historia */
            const float x = gr[k] * gr[k];
            const float vn = beta2 * vh + omb * x;
            v[k] = vn; if (vn > mx) mx = vn;
        }
        n->rmax[r] = mx;
        for (size_t k = 0; k < W; k++) {
            const int c = mx > 0.0f ? en_code(v[k] / mx) : 15;
            if (k & 1) cr[k >> 1] = (uint8_t)((cr[k >> 1] & 0x0F) | (c << 4)); else cr[k >> 1] = (uint8_t)((cr[k >> 1] & 0xF0) | c);
            const float vq = ldexpf(mx, -c);                    /* v cuantizado: el que se usa y se guarda */
            gr[k] = gr[k] / (sqrtf(vq / bc) + eps);
        }
    }
}
void ternary_elemnorm_apply(TernaryElemNorm *n, float *g, float beta2) {
    if (!TN_VEC(ELEMNORM)) { ternary_elemnorm_apply_scalar(n, g, beta2); return; }
    const size_t R = n->rows, W = n->cols, st = (W + 1) / 2, tot = R * W;
    n->steps++;
    const float bc = (float)(1.0 - pow((double)beta2, (double)n->steps));
    const double gm = ternary_canon_sumsq(g, tot) / (double)tot;
    const float eps = (float)(1e-3 * sqrt(gm) + 1e-20), omb = 1.0f - beta2;
    if (ternary_avx512_enabled()) for (size_t r = 0; r < R; r++) ternary_avx512_elemnorm_row(n->code + r * st, n->rmax + r, g + r * W, n->tmp, W, beta2, omb, bc, eps);
    else for (size_t r = 0; r < R; r++) ternary_neon_elemnorm_row(n->code + r * st, n->rmax + r, g + r * W, n->tmp, W, beta2, omb, bc, eps);
}

/* Filas [r0, r1) de la matriz, una a una con R2. */
static void apply_rows_range(TernaryDense *layer, const float *grad_w, size_t r0, size_t r1, const TernaryUpdateRule *rule, int pre) {
    const size_t n = layer->input_dim;
    for (size_t r = r0; r < r1; r++) ternary_r2_update_row_ex(layer->W, layer->counter_W, r * n, grad_w + r * n, n, rule, pre);
}

void ternary_dense_apply_row_gradients(TernaryDense *layer, size_t row, const float *grad_row) {
    const size_t n = layer->input_dim;
    ternary_r2_update_row(layer->W, layer->counter_W, row * n, grad_row, n, &g_rule);
}

typedef struct {
    TernaryDense *layer; const float *grad_w;
    size_t rows, step, chunks;
    TernaryUpdateRule rule;     /* la regla del hilo que llama (los hilos del pool no comparten su variable por hilo) */
    int pre;                    /* gradiente ya normalizado */
} ApplyJob;

static void apply_task(size_t idx, void *ctx) {
    const ApplyJob *j = (const ApplyJob *)ctx;
    /* limites en multiplos de 'step' filas: cada hilo escribe palabras de W que ningun otro toca */
    const size_t units = (j->rows + j->step - 1) / j->step;
    const size_t u0 = units * idx / j->chunks, u1 = units * (idx + 1) / j->chunks;
    size_t r0 = u0 * j->step, r1 = u1 * j->step;
    if (r1 > j->rows) r1 = j->rows;
    if (r0 < r1) apply_rows_range(j->layer, j->grad_w, r0, r1, &j->rule, j->pre);
}

static size_t gcd_size(size_t a, size_t b) { while (b) { size_t t = a % b; a = b; b = t; } return a; }

/* Todas las filas de una vez. Las filas son independientes (R2 normaliza por fila), asi que con un pool de hilos se
 * reparten en bloques de filas: el mismo resultado bit a bit. Los limites de bloque caen donde fila*ancho es multiplo
 * de 64 bits, para que dos hilos nunca escriban la misma palabra de mask/sign de W. */
static void apply_matrix_impl(TernaryDense *layer, const float *grad_w, int pre) {
    const size_t n = layer->input_dim, rows = layer->output_dim;
    TernaryThreadPool *pool = ternary_get_compute_pool();
    const int nt = pool ? ternary_threadpool_num_threads(pool) : 1;
    const size_t step = 64 / gcd_size(64, n ? n : 1);   /* filas entre limites alineados a palabra */
    if (nt > 1 && (double)rows * (double)n >= 1.0e5 && rows >= 16 * step) {
        size_t chunks = (size_t)nt;
        if (chunks > rows / (8 * step)) chunks = rows / (8 * step);   /* al menos 8 filas por tarea */
        if (chunks >= 2) {
            ApplyJob job = { layer, grad_w, rows, step, chunks, g_rule, pre };
            ternary_parallel_for(pool, 0, chunks, apply_task, &job);
            return;
        }
    }
    apply_rows_range(layer, grad_w, 0, rows, &g_rule, pre);
}
void ternary_dense_apply_matrix_gradients(TernaryDense *layer, const float *grad_w) { apply_matrix_impl(layer, grad_w, 0); }
void ternary_dense_apply_matrix_gradients_pre(TernaryDense *layer, const float *grad_w) { apply_matrix_impl(layer, grad_w, 1); }

/* El bias completo es una sola "fila" de output_dim pesos, con su propio contador. */
void ternary_dense_apply_bias_gradients(TernaryDense *layer, const float *grad_bias) {
    if (!layer->bias) return;
    ternary_r2_update_row(layer->bias, layer->counter_b, 0, grad_bias, layer->output_dim, &g_rule);
}

void ternary_dense_apply_scale_gradient(TernaryDense *layer, float grad_scale, float scale_lr) {
    float new_scale = layer->scale - scale_lr * grad_scale;
    if (new_scale < 0.02f) new_scale = 0.02f; /* mismo piso que la version Python */
    layer->scale = new_scale;
}


/* ======================= entrada uint8 (escalar) ======================= */
/* ---- Seguro para hilos: si las filas no caen alineadas a palabra, la fila se extrae en un buffer del LLAMADOR (no en layer->row_scratch,
 * que es compartido). Con filas alineadas se leen directo los pesos (solo lectura). */
typedef struct { uint64_t stk[2 * 128]; uint64_t *m, *s, *heap; } RowBuf;
static void rb_init(RowBuf *b, const TernaryDense *layer) {
    b->m = b->s = b->heap = NULL;
    if (rows_word_aligned(layer)) return;
    const size_t rw = (layer->input_dim + 63) / 64;
    if (rw <= 128) { b->m = b->stk; b->s = b->stk + 128; return; }
    b->heap = (uint64_t *)ternary_malloc64(2 * rw * sizeof(uint64_t));
    if (b->heap) { b->m = b->heap; b->s = b->heap + rw; }
}
static void rb_free(RowBuf *b) { free(b->heap); b->heap = NULL; }
static inline const uint64_t *u8_row_ts(const TernaryDense *layer, size_t o, const RowBuf *b, const uint64_t **ws_out) {
    if (!b->m) {                                         /* alineado: directo */
        const size_t rw = layer->input_dim / 64;
        *ws_out = layer->W->sign + o * rw;
        return layer->W->mask + o * rw;
    }
    const size_t offset = o * layer->input_dim;
    extract_bits(layer->W->mask, layer->W->words, offset, layer->input_dim, b->m);
    extract_bits(layer->W->sign, layer->W->words, offset, layer->input_dim, b->s);
    *ws_out = b->s;
    return b->m;
}

void ternary_dense_forward_u8_row(const TernaryDense *layer, const uint8_t *q, float sx, int32_t zp,
                                  float *out, int32_t *out_raw_opt) {
    const float k = layer->scale * sx;
    const int avx = ternary_avx512_enabled(), neon = !avx && ternary_neon_enabled();
    RowBuf rb; rb_init(&rb, layer);
    for (size_t o = 0; o < layer->output_dim; o++) {
        const uint64_t *ws;
        const uint64_t *wm = u8_row_ts(layer, o, &rb, &ws);
        int32_t z = avx ? ternary_avx512_dot_u8(wm, ws, layer->input_dim, q, zp)
                  : neon ? ternary_neon_dot_u8(wm, ws, layer->input_dim, q, zp)
                  : ternary_dot_u8_scalar(wm, ws, layer->input_dim, q, zp);
        if (layer->bias) z += ternary_vec_get(layer->bias, o);
        if (out_raw_opt) out_raw_opt[o] = z;
        out[o] = k * (float)z;
    }
    rb_free(&rb);
}

void ternary_dense_forward_u8(const TernaryDense *layer, const TernaryU8 *x, float *out, int32_t *out_raw_opt) {
    ternary_dense_forward_u8_row(layer, x->q, x->scale[0], x->zp[0], out, out_raw_opt);
}

void ternary_dense_backward_u8_row(const TernaryDense *layer, const uint8_t *q, float sx, int32_t zp,
                                   const float *grad_out, float *grad_w_out, float *grad_x_out,
                                   float *grad_scale_out, float *grad_bias_out, const int32_t *z_raw_opt) {
    const size_t in = layer->input_dim;
    float *xf = (float *)ternary_malloc64(in * sizeof(float));
    if (!xf) { ternary_diag("dense_backward_u8: sin memoria\n"); return; }
    const int avx = ternary_avx512_enabled(), neon = !avx && ternary_neon_enabled();
    if (avx) ternary_avx512_u8_to_f32(q, in, sx, zp, xf);
    else if (neon) ternary_neon_u8_to_f32(q, in, sx, zp, xf);
    else for (size_t i = 0; i < in; i++) xf[i] = sx * (float)((int32_t)q[i] - zp);
    RowBuf rb; rb_init(&rb, layer);
    for (size_t o = 0; o < layer->output_dim; o++) {
        /* Salto de ceros: un gradiente de salida exactamente 0 (p. ej. la unidad ReLU² apagada) no aporta nada a
         * gw, a grad_x, a la escala ni al sesgo (todo es g * algo), asi que se salta la fila entera. Mismos bits
         * que sin saltar: las sumas solo dejan de recibir +-0. */
        const float g = grad_out[o];
        if (g == 0.0f) continue;
        const uint64_t *ws;
        const uint64_t *wm = u8_row_ts(layer, o, &rb, &ws);
        int32_t z;
        if (z_raw_opt) z = z_raw_opt[o];
        else {
            z = avx ? ternary_avx512_dot_u8(wm, ws, in, q, zp)
              : neon ? ternary_neon_dot_u8(wm, ws, in, q, zp)
              : ternary_dot_u8_scalar(wm, ws, in, q, zp);
            if (layer->bias) z += ternary_vec_get(layer->bias, o);
        }
        *grad_scale_out += g * (sx * (float)z);
        if (grad_bias_out) grad_bias_out[o] += g * sx;
        float *gw = grad_w_out + o * in;
        const float gs = g * layer->scale;
        if (avx) {
            ternary_avx512_outer_acc(gw, in, g, xf);
            if (grad_x_out) ternary_avx512_gradx_row(grad_x_out, in, wm, ws, gs);
            continue;
        }
        if (neon) {
            ternary_neon_outer_acc(gw, in, g, xf);
            if (grad_x_out) ternary_neon_gradx_row(grad_x_out, in, wm, ws, gs);
            continue;
        }
        for (size_t i = 0; i < in; i++) {
            const uint64_t bit = (uint64_t)1 << (i & 63);
            const size_t k = i >> 6;
            gw[i] += g * xf[i];
            if (grad_x_out && (wm[k] & bit)) grad_x_out[i] += (ws[k] & bit) ? -gs : gs;
        }
    }
    rb_free(&rb);
    free(xf);
}

/* ---- Piezas del backward uint8 para repartir en hilos (mismas cuentas que ternary_dense_backward_u8_row, por separado) ---- */
void ternary_dense_u8_to_f32(const uint8_t *q, size_t n, float sx, int32_t zp, float *out) {
    if (ternary_avx512_enabled()) ternary_avx512_u8_to_f32(q, n, sx, zp, out);
    else if (ternary_neon_enabled()) ternary_neon_u8_to_f32(q, n, sx, zp, out);
    else for (size_t i = 0; i < n; i++) out[i] = sx * (float)((int32_t)q[i] - zp);
}

/* grad_x[i] += sum_o (+-)g[o]*escala sobre los pesos activos de cada fila (o ascendente; salta g == 0). Una posicion. */
void ternary_dense_u8_gradx(const TernaryDense *layer, const float *grad_out, float *grad_x_out) {
    const size_t in = layer->input_dim;
    const int avx = ternary_avx512_enabled(), neon = !avx && ternary_neon_enabled();
    RowBuf rb; rb_init(&rb, layer);
    for (size_t o = 0; o < layer->output_dim; o++) {
        const float g = grad_out[o];
        if (g == 0.0f) continue;
        const uint64_t *ws;
        const uint64_t *wm = u8_row_ts(layer, o, &rb, &ws);
        const float gs = g * layer->scale;
        if (avx) { ternary_avx512_gradx_row(grad_x_out, in, wm, ws, gs); continue; }
        if (neon) { ternary_neon_gradx_row(grad_x_out, in, wm, ws, gs); continue; }
        for (size_t i = 0; i < in; i++) {
            const uint64_t bit = (uint64_t)1 << (i & 63);
            const size_t k = i >> 6;
            if (wm[k] & bit) grad_x_out[i] += (ws[k] & bit) ? -gs : gs;
        }
    }
    rb_free(&rb);
}

/* ---- Forward entero por lotes de posiciones (AVX-512 + VNNI) ----
 * Los pesos se expanden UNA vez por llamada del llamador a int8 en bloques de 4 entradas (TernaryFwdW, de solo lectura y compartible
 * entre hilos) y ternary_vnni_forward produce 16 salidas por instruccion sin reduccion horizontal. Es entero exacto: mismos bits que
 * llamar ternary_dense_forward_u8_row posicion por posicion. */
struct TernaryFwdW { int8_t *wt; int32_t *wsum, *bias; size_t Opad, O, in; };

TernaryFwdW *ternary_dense_fwdw_create(const TernaryDense *layer) {
    if (!ternary_avx512_vnni_enabled()) return NULL;
    TernaryFwdW *fw = (TernaryFwdW *)calloc(1, sizeof *fw);
    if (!fw) return NULL;
    const size_t in = layer->input_dim, O = layer->output_dim, in4 = (in + 3) / 4;
    fw->in = in; fw->O = O; fw->Opad = (O + 15) & ~(size_t)15;
    fw->wt = (int8_t *)ternary_calloc64(1, in4 * fw->Opad * 4);
    fw->wsum = (int32_t *)ternary_calloc64(1, fw->Opad * sizeof(int32_t));
    fw->bias = layer->bias ? (int32_t *)ternary_calloc64(1, fw->Opad * sizeof(int32_t)) : NULL;
    int8_t *row = (int8_t *)ternary_calloc64(1, in4 * 4 + 64);
    if (!fw->wt || !fw->wsum || !row || (layer->bias && !fw->bias)) { free(fw->wt); free(fw->wsum); free(fw->bias); free(row); free(fw); return NULL; }
    RowBuf rb; rb_init(&rb, layer);
    for (size_t o = 0; o < O; o++) {
        const uint64_t *ws;
        const uint64_t *wm = u8_row_ts(layer, o, &rb, &ws);
        ternary_avx512_expand_i8(wm, ws, in, row);
        int32_t sum = 0;
        for (size_t i = 0; i < in; i++) sum += row[i];
        fw->wsum[o] = sum;
        if (fw->bias) fw->bias[o] = (int32_t)ternary_vec_get(layer->bias, o);
        for (size_t k = 0; k < in4; k++) memcpy(fw->wt + (k * fw->Opad + o) * 4, row + 4 * k, 4);
    }
    rb_free(&rb);
    free(row);
    return fw;
}

void ternary_dense_fwdw_free(TernaryFwdW *fw) {
    if (!fw) return;
    free(fw->wt); free(fw->wsum); free(fw->bias); free(fw);
}

void ternary_dense_forward_u8_batch(const TernaryDense *layer, const TernaryFwdW *fw, size_t P, const uint8_t *q, size_t qs,
                                    const float *sx, const int32_t *zp, float *out, size_t os, int32_t *raw, size_t rs) {
    if (!fw) {
        float *tmp = out ? NULL : (float *)ternary_malloc64(layer->output_dim * sizeof(float));   /* out == NULL: solo interesa raw */
        for (size_t p = 0; p < P; p++)
            ternary_dense_forward_u8_row(layer, q + p * qs, sx[p], zp[p], out ? out + p * os : tmp, raw ? raw + p * rs : NULL);
        free(tmp);
        return;
    }
    ternary_vnni_forward(fw->wt, fw->wsum, fw->bias, fw->Opad, fw->O, fw->in, P, q, qs, sx, zp, layer->scale, out, os, raw, rs);
}

/* ---- grad_x por lotes de posiciones (AVX-512) ----
 * El camino por posicion (ternary_dense_u8_gradx) relee las mascaras de pesos y lee/escribe grad_x en cada fila de salida; aqui
 * los pesos se parten UNA vez en planos +1/-1 (TernaryPlanes, de solo lectura y compartibles entre hilos) y el kernel por
 * bloques de posiciones x tiles mantiene grad_x en registros recorriendo o ascendente: mismas sumas y mismo orden. */
struct TernaryPlanes { uint16_t *p, *n; size_t nt16, outd, in; };

TernaryPlanes *ternary_dense_planes_create(const TernaryDense *layer) {
    if (!ternary_avx512_enabled()) return NULL;
    TernaryPlanes *pl = (TernaryPlanes *)calloc(1, sizeof *pl);
    if (!pl) return NULL;
    pl->in = layer->input_dim; pl->outd = layer->output_dim; pl->nt16 = (pl->in + 15) / 16;
    pl->p = (uint16_t *)ternary_malloc64(pl->outd * pl->nt16 * sizeof(uint16_t));
    pl->n = (uint16_t *)ternary_malloc64(pl->outd * pl->nt16 * sizeof(uint16_t));
    if (!pl->p || !pl->n) { free(pl->p); free(pl->n); free(pl); return NULL; }
    RowBuf rb; rb_init(&rb, layer);
    for (size_t o = 0; o < pl->outd; o++) {
        const uint64_t *ws;
        const uint64_t *wm = u8_row_ts(layer, o, &rb, &ws);
        ternary_avx512_split_planes(pl->p + o * pl->nt16, pl->n + o * pl->nt16, pl->in, wm, ws);
    }
    rb_free(&rb);
    return pl;
}

void ternary_dense_planes_free(TernaryPlanes *pl) {
    if (!pl) return;
    free(pl->p); free(pl->n); free(pl);
}

#define GX_BATCH 16
void ternary_dense_u8_gradx_batch(const TernaryDense *layer, const TernaryPlanes *pl, size_t P, const float *grad_out,
                                  size_t go_stride, float *grad_x_out, size_t gx_stride) {
    if (!pl || go_stride != layer->output_dim) {
        for (size_t i = 0; i < P; i++) ternary_dense_u8_gradx(layer, grad_out + i * go_stride, grad_x_out + i * gx_stride);
        return;
    }
    float *gxp[GX_BATCH];
    for (size_t i0 = 0; i0 < P; i0 += GX_BATCH) {
        const size_t S = (P - i0 < GX_BATCH) ? (P - i0) : GX_BATCH;
        for (size_t i = 0; i < S; i++) gxp[i] = grad_x_out + (i0 + i) * gx_stride;
        ternary_avx512_gradx_scaled(gxp, S, layer->output_dim, layer->input_dim, grad_out + i0 * go_stride, layer->scale, pl->p, pl->n);
    }
}

/* Filas [o0, o1) de los pesos sobre P posiciones (p ascendente): gw[o][:] += g[p][o] * xf[p][:]. gw es la matriz completa
 * (output_dim x input_dim); cada hilo escribe solo sus filas. Si scale_terms != NULL guarda en scale_terms[p*terms_stride + o]
 * el termino g*(sx[p]*z[p][o]) (0 si g == 0) de la escala del gradiente: el llamador los suma despues en el MISMO orden que
 * ternary_dense_backward_u8_row (p ascendente, o ascendente), asi la suma de punto flotante no depende del reparto. */
void ternary_dense_u8_gradw(const TernaryDense *layer, size_t o0, size_t o1, size_t P,
                            const float *gout, size_t gout_stride, const float *xf, size_t xf_stride,
                            const float *sx, const int32_t *z, size_t z_stride, float *gw,
                            float *scale_terms, size_t terms_stride) {
    const size_t in = layer->input_dim;
    const int avx = ternary_avx512_enabled();
    if (avx) {   /* terminos de la escala aparte; el producto exterior con los acumuladores de cada fila en registros (mismos bits) */
        if (scale_terms) ternary_avx512_scale_terms(scale_terms, terms_stride, gout, gout_stride, sx, z, z_stride, P, o0, o1);
        ternary_avx512_outer_acc_rows(gw, in, o0, o1, P, gout, gout_stride, xf, xf_stride);
        return;
    }
    for (size_t p = 0; p < P; p++) {
        const float *g = gout + p * gout_stride, *x = xf + p * xf_stride;
        for (size_t o = o0; o < o1; o++) {
            const float gv = g[o];
            if (gv == 0.0f) { if (scale_terms) scale_terms[p * terms_stride + o] = 0.0f; continue; }
            if (scale_terms) scale_terms[p * terms_stride + o] = gv * (sx[p] * (float)z[p * z_stride + o]);
            float *row = gw + o * in;
            for (size_t i = 0; i < in; i++) row[i] = fmaf(gv, x[i], row[i]);   /* fusionado: igual bit a bit que el nucleo AVX-512 */
        }
    }
}

void ternary_dense_backward_u8(const TernaryDense *layer, const TernaryU8 *x, const float *grad_out,
                               float *grad_w_out, float *grad_x_out, float *grad_scale_out,
                               float *grad_bias_out, const int32_t *z_raw_opt) {
    ternary_dense_backward_u8_row(layer, x->q, x->scale[0], x->zp[0], grad_out, grad_w_out, grad_x_out,
                                  grad_scale_out, grad_bias_out, z_raw_opt);
}
