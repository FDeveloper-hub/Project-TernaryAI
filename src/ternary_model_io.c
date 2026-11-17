#include "ternary_model_io.h"
#include <pthread.h>
#include <string.h>

/* Tablas: byte (0..242) -> 5 bits de mascara y 5 bits de signo (bit j = trit j); y (mascara,signo) -> byte. */
static uint8_t g_mask5[243], g_sign5[243], g_code5[32][32];
static pthread_once_t g_once = PTHREAD_ONCE_INIT;

static void build_tables(void) {
    for (unsigned code = 0; code < 243; code++) {
        unsigned c = code, m = 0, s = 0;
        for (unsigned j = 0; j < 5; j++) {
            unsigned digit = c % 3u; c /= 3u;           /* 0,1,2 = -1,0,+1 */
            if (digit != 1u) m |= 1u << j;
            if (digit == 0u) s |= 1u << j;
        }
        g_mask5[code] = (uint8_t)m; g_sign5[code] = (uint8_t)s;
    }
    for (unsigned m = 0; m < 32; m++)
        for (unsigned s = 0; s < 32; s++) {
            unsigned code = 0, mul = 1;
            for (unsigned j = 0; j < 5; j++) {
                unsigned digit = !((m >> j) & 1u) ? 1u : (((s >> j) & 1u) ? 0u : 2u);
                code += digit * mul; mul *= 3u;
            }
            g_code5[m][s] = (uint8_t)code;
        }
}

/* 5 bits a partir de la posicion 'pos' de un arreglo de palabras de 64 bits (bit 0 de words[0] = posicion 0). */
static inline unsigned get5(const uint64_t *w, size_t nwords, size_t pos) {
    size_t i = pos >> 6, sh = pos & 63;
    uint64_t v = w[i] >> sh;
    if (sh > 59 && i + 1 < nwords) v |= w[i + 1] << (64 - sh);
    return (unsigned)(v & 31u);
}

size_t ternary_pack_trits(const TernaryVec *v, size_t n, uint8_t *out_bytes) {
    pthread_once(&g_once, build_tables);
    size_t nbytes = ternary_packed_trits_bytes(n);
    for (size_t b = 0; b < nbytes; b++) {
        size_t pos = b * 5;
        unsigned m = get5(v->mask, v->words, pos), s = get5(v->sign, v->words, pos);
        if (pos + 5 > n) m &= (1u << (n - pos)) - 1u;       /* relleno = cero ternario */
        out_bytes[b] = g_code5[m][s & m];
    }
    return nbytes;
}

int ternary_unpack_trits(const uint8_t *in_bytes, size_t n, TernaryVec *out) {
    pthread_once(&g_once, build_tables);
    size_t nbytes = ternary_packed_trits_bytes(n);
    memset(out->mask, 0, out->words * sizeof(uint64_t));
    memset(out->sign, 0, out->words * sizeof(uint64_t));
    uint64_t cm = 0, cs = 0;
    size_t pos = 0, wi = 0;
    for (size_t b = 0; b < nbytes; b++) {
        unsigned code = in_bytes[b];
        if (code >= 243u) return 0;                          /* 3^5: byte fuera de rango, archivo corrupto */
        uint64_t m = g_mask5[code], s = g_sign5[code];
        if (b + 1 == nbytes && n % 5) { uint64_t keep = ((uint64_t)1 << (n % 5)) - 1u; m &= keep; s &= keep; }
        cm |= m << pos; cs |= s << pos;
        pos += 5;
        if (pos >= 64) {
            out->mask[wi] = cm; out->sign[wi] = cs; wi++;
            pos -= 64;
            cm = pos ? m >> (5 - pos) : 0;
            cs = pos ? s >> (5 - pos) : 0;
        }
    }
    if (pos > 0 && wi < out->words) { out->mask[wi] = cm; out->sign[wi] = cs; }
    return 1;
}
