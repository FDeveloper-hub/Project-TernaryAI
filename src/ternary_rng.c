#include "ternary_rng.h"
#include <math.h>

#define RNG_DEG 31
#define RNG_SEP 3
#define RNG_MAX 2147483647

void ternary_rng_seed(TernaryRng *r, unsigned int seed) {
    if (seed == 0) seed = 1;
    r->state[0] = (int32_t)seed;
    int32_t word = (int32_t)seed;
    for (int i = 1; i < RNG_DEG; i++) {
        /* state[i] = (16807 * state[i-1]) % 2147483647 sin desbordar 31 bits */
        long hi = word / 127773;
        long lo = word % 127773;
        long w = 16807 * lo - 2836 * hi;
        if (w < 0) w += RNG_MAX;
        word = (int32_t)w;
        r->state[i] = word;
    }
    r->fptr = RNG_SEP;
    r->rptr = 0;
    for (int i = 0; i < RNG_DEG * 10; i++) (void)ternary_rng_next(r);
}

int32_t ternary_rng_next(TernaryRng *r) {
    uint32_t val = (uint32_t)r->state[r->fptr] + (uint32_t)r->state[r->rptr];
    r->state[r->fptr] = (int32_t)val;
    int32_t result = (int32_t)(val >> 1);
    if (++r->fptr >= RNG_DEG) {
        r->fptr = 0;
        ++r->rptr;
    } else if (++r->rptr >= RNG_DEG) {
        r->rptr = 0;
    }
    return result;
}

int ternary_rng_sample_faninit(TernaryRng *r, size_t fan_in) {
    double p_nonzero = 1.0 / sqrt((double)fan_in);
    if (p_nonzero < 0.15) p_nonzero = 0.15;
    if (p_nonzero > 0.9) p_nonzero = 0.9;
    double p_each = p_nonzero / 2.0;
    double u = (double)ternary_rng_next(r) / ((double)RNG_MAX + 1.0);
    if (u < p_each) return -1;
    if (u < p_each + (1.0 - p_nonzero)) return 0;
    return 1;
}
