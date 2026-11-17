#include "ternary_mem.h"
#include <stdio.h>
#include <stdarg.h>
#include "ternary_core.h"
#include "ternary_dispatch.h"
#include <stdlib.h>
#include <string.h>

TernaryVec *ternary_vec_alloc(size_t n) {
    TernaryVec *v = (TernaryVec *)ternary_malloc64(sizeof(TernaryVec));
    if (!v) return NULL;
    v->n = n;
    v->words = (n + 63) / 64;
    v->mask = (uint64_t *)ternary_calloc64(v->words, sizeof(uint64_t));
    v->sign = (uint64_t *)ternary_calloc64(v->words, sizeof(uint64_t));
    if (!v->mask || !v->sign) {
        free(v->mask);
        free(v->sign);
        free(v);
        return NULL;
    }
    return v;
}

void ternary_vec_free(TernaryVec *v) {
    if (!v) return;
    free(v->mask);
    free(v->sign);
    free(v);
}

void ternary_vec_set(TernaryVec *v, size_t i, int value) {
    size_t word = i >> 6;      /* i / 64 */
    size_t bit  = i & 63;      /* i % 64 */
    uint64_t m = (uint64_t)1 << bit;

    if (value == 0) {
        v->mask[word] &= ~m;
        v->sign[word] &= ~m;   /* sin basura: bit de signo tambien se limpia */
    } else {
        v->mask[word] |= m;
        if (value < 0) v->sign[word] |= m;
        else           v->sign[word] &= ~m;
    }
}

int ternary_vec_get(const TernaryVec *v, size_t i) {
    size_t word = i >> 6;
    size_t bit  = i & 63;
    uint64_t m = (uint64_t)1 << bit;
    if (!(v->mask[word] & m)) return 0;
    return (v->sign[word] & m) ? -1 : 1;
}

size_t ternary_vec_bytes(const TernaryVec *v) {
    /* 2 arreglos de 'words' palabras de 8 bytes cada una. */
    return v->words * sizeof(uint64_t) * 2;
}

int32_t ternary_dot_scalar(const TernaryVec *w, const TernaryVec *x) {
    int32_t total_active = 0;
    int32_t total_same_sign = 0;

    for (size_t k = 0; k < w->words; k++) {
        uint64_t active = w->mask[k] & x->mask[k];
        uint64_t xnor_signs = ~(w->sign[k] ^ x->sign[k]);
        uint64_t same_sign_active = active & xnor_signs;

        total_active    += __builtin_popcountll(active);
        total_same_sign += __builtin_popcountll(same_sign_active);
    }
    return 2 * total_same_sign - total_active;
}

#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
/* Misma cuenta entera (resultado identico), pero compilada con la
 * instruccion POPCNT: sin ella gcc llama a __popcountdi2 (software). Se
 * activa en tiempo de ejecucion solo si la CPU la tiene, asi el binario
 * sigue corriendo en cualquier x86-64 sin tocar los flags del Makefile. */
__attribute__((target("popcnt")))
static int32_t ternary_dot_popcnt(const TernaryVec *w, const TernaryVec *x) {
    int32_t total_active = 0;
    int32_t total_same_sign = 0;
    for (size_t k = 0; k < w->words; k++) {
        uint64_t active = w->mask[k] & x->mask[k];
        uint64_t same_sign_active = active & ~(w->sign[k] ^ x->sign[k]);
        total_active    += __builtin_popcountll(active);
        total_same_sign += __builtin_popcountll(same_sign_active);
    }
    return 2 * total_same_sign - total_active;
}

/* Version AVX-512 (solo AVX512F+BW, sin VPOPCNTDQ): el popcount sale de dos busquedas pshufb por nibble y la suma de
 * bytes de vpsadbw. Cuenta entera => resultado identico. Para filas cortas (< AVX512_DOT_MIN_WORDS palabras) el
 * popcnt escalar gana, asi que se delega en el. */
#ifndef AVX512_DOT_MIN_WORDS
#define AVX512_DOT_MIN_WORDS 8
#endif
#define T512 __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,popcnt")))
T512 static inline __m512i pop8_bytes(__m512i v, __m512i lut, __m512i low) {
    const __m512i lo = _mm512_and_si512(v, low), hi = _mm512_and_si512(_mm512_srli_epi16(v, 4), low);
    return _mm512_add_epi8(_mm512_shuffle_epi8(lut, lo), _mm512_shuffle_epi8(lut, hi));
}
T512 static int32_t ternary_dot_avx512(const TernaryVec *w, const TernaryVec *x) {
    const size_t nw = w->words;
    if (nw < AVX512_DOT_MIN_WORDS) return ternary_dot_popcnt(w, x);
    const __m512i lut = _mm512_broadcast_i32x4(_mm_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4)),
                  low = _mm512_set1_epi8(0x0f), eight = _mm512_set1_epi8(8), zero = _mm512_setzero_si512();
    __m512i acc = _mm512_setzero_si512();
    size_t k = 0, iters = 0;
    for (; k < nw; k += 8, iters++) {
        const __mmask8 lm = (nw - k >= 8) ? (__mmask8)0xFF : (__mmask8)((1u << (nw - k)) - 1u);
        const __m512i wm = _mm512_maskz_loadu_epi64(lm, w->mask + k), ws = _mm512_maskz_loadu_epi64(lm, w->sign + k);
        const __m512i xm = _mm512_maskz_loadu_epi64(lm, x->mask + k), xs = _mm512_maskz_loadu_epi64(lm, x->sign + k);
        const __m512i active = _mm512_and_si512(wm, xm);
        const __m512i same = _mm512_and_si512(active, _mm512_xor_si512(_mm512_xor_si512(ws, xs), _mm512_set1_epi32(-1)));
        const __m512i pa = pop8_bytes(active, lut, low), ps = pop8_bytes(same, lut, low);
        const __m512i u = _mm512_add_epi8(_mm512_sub_epi8(_mm512_add_epi8(ps, ps), pa), eight);   /* 2*ps - pa + 8, en 0..24 */
        acc = _mm512_add_epi64(acc, _mm512_sad_epu8(u, zero));
    }
    return (int32_t)(_mm512_reduce_add_epi64(acc) - (long long)iters * 8 * 64);
}
#endif

/* Punto de despacho: arranca apuntando a la version escalar (siempre
 * correcta, en cualquier CPU) y en x86 con POPCNT se cambia a la
 * version popcount (o ternary_core_set_dot_impl()). Ninguna capa (dense/conv2d) necesita saber que
 * esto existe -- siguen llamando ternary_dot() como siempre. */
static TernaryDotFn g_dot_impl = ternary_dot_scalar;

#if defined(__x86_64__) && defined(__GNUC__)
__attribute__((constructor))
static void ternary_core_pick_x86_dot(void) {
    __builtin_cpu_init();
    if (__builtin_cpu_supports("popcnt")) g_dot_impl = ternary_dot_popcnt;
    {
        const char *off = getenv("TERNARYAI_NO_AVX512");
        if (__builtin_cpu_supports("popcnt") && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
            __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512vl") && !(off && off[0] && off[0] != '0'))
            g_dot_impl = ternary_dot_avx512;
    }
}
#endif

void ternary_core_set_dot_impl(TernaryDotFn fn) {
    if (fn) g_dot_impl = fn;
}

int32_t ternary_dot(const TernaryVec *w, const TernaryVec *x) {
    return g_dot_impl(w, x);
}

int32_t ternary_dot_reference(const TernaryVec *w, const TernaryVec *x) {
    int32_t acc = 0;
    for (size_t i = 0; i < w->n; i++) {
        acc += ternary_vec_get(w, i) * ternary_vec_get(x, i);
    }
    return acc;
}

static _Thread_local char *g_diag_buf;
static _Thread_local size_t g_diag_cap;

void ternary_diag_capture(char *buf, size_t cap) {
    g_diag_buf = (buf && cap) ? buf : NULL;
    g_diag_cap = g_diag_buf ? cap : 0;
    if (g_diag_buf) g_diag_buf[0] = '\0';
}

void ternary_diag(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    if (g_diag_buf) {
        if (g_diag_buf[0] == '\0') {   /* solo el primer mensaje: es la causa */
            vsnprintf(g_diag_buf, g_diag_cap, fmt, ap);
            size_t n = strlen(g_diag_buf);
            while (n > 0 && (g_diag_buf[n - 1] == '\n' || g_diag_buf[n - 1] == ' ')) g_diag_buf[--n] = '\0';
        }
    } else {
        vfprintf(stderr, fmt, ap);
    }
    va_end(ap);
}
