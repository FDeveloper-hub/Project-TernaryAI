#include "ternary_mem.h"
#include "ternary_loss.h"
#include <math.h>
#include <string.h>
#include "ternary_fexp.h"
#include "ternary_avx512.h"
#include "ternary_neon_policy.h"

double ternary_xent_row(const float *l, size_t vocab, size_t t, size_t n, float *g, int *correct_out) {
    if (ternary_avx512_enabled()) {
        double loss; ternary_avx512_xent_row(l, vocab, t, n, g, correct_out, &loss);
        return loss;
    }
    if (TN_ON(HEAD)) {
        double loss; ternary_neon_xent_row(l, vocab, t, n, g, correct_out, &loss);
        return loss;
    }
    size_t arg = 0;
    float m = l[0];
    for (size_t v = 1; v < vocab; v++) if (l[v] > m) { m = l[v]; arg = v; }
    *correct_out = (arg == t);
    double dl[16] = {0};                       /* suma en 16 carriles (orden canonico, igual que AVX-512) */
    for (size_t v = 0; v < vocab; v++) dl[v & 15] += (double)ternary_expf_nonpos(l[v] - m);
    const double sum = ternary_canon_reduce_d16(dl);
    const double loss = log(sum) - (double)(l[t] - m);
    if (g) {
        const double inv = 1.0 / (sum * (double)n);
        for (size_t v = 0; v < vocab; v++) g[v] = (float)((double)ternary_expf_nonpos(l[v] - m) * inv);
        g[t] -= (float)(1.0 / (double)n);
    }
    return loss;
}

double ternary_xent(const float *logits, size_t rows, size_t vocab, const int32_t *targets,
                    float *grad_out, size_t *n_valid_out, size_t *n_correct_out) {
    if (n_valid_out) *n_valid_out = 0;
    if (n_correct_out) *n_correct_out = 0;
    if (grad_out) memset(grad_out, 0, rows * vocab * sizeof(float));
    size_t n = 0;
    for (size_t p = 0; p < rows; p++) {
        if (targets[p] >= 0 && (size_t)targets[p] >= vocab) return (double)NAN;
        if (targets[p] >= 0) n++;
    }
    if (n == 0) return 0.0;
    double total = 0.0;
    size_t correct = 0;
    for (size_t p = 0; p < rows; p++) {
        if (targets[p] < 0) continue;
        int ok = 0;
        total += ternary_xent_row(logits + p * vocab, vocab, (size_t)targets[p], n, grad_out ? grad_out + p * vocab : NULL, &ok);
        correct += (size_t)ok;
    }
    if (n_valid_out) *n_valid_out = n;
    if (n_correct_out) *n_correct_out = correct;
    return total / (double)n;
}
