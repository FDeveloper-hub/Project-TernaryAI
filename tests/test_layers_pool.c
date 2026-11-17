/* Con y sin pool de hilos de calculo, las capas lm_head, swiglu y gqa dan
 * EXACTAMENTE lo mismo: salidas de forward en cada paso y, al seguir
 * entrenando, los mismos pesos (se ve en las salidas posteriores). */
#include "ternary_stack.h"
#include "ternary_layer.h"
#include "ternary_rng.h"
#include "ternary_threadpool.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg) do { if (!(cond)) { fprintf(stderr, "FALLO: %s (%s, linea %d)\n", msg, name, __LINE__); return 1; } } while (0)

static TernaryStack *build(const char *name, const size_t *shape, size_t nd, const double *p, size_t np, unsigned seed) {
    TernaryRng rng; ternary_rng_seed(&rng, seed);
    TernaryLayer l; char err[256];
    if (!ternary_layer_type_build(ternary_layer_type_by_name(name), shape, nd, p, np, &rng, &l, err, sizeof(err))) {
        fprintf(stderr, "build %s: %s\n", name, err); return NULL;
    }
    TernaryStack *s = ternary_stack_create();
    if (!ternary_stack_add(s, l, TERNARY_ACTIVATION_UINT8)) return NULL;
    return s;
}

static int run(const char *name, const size_t *shape, const double *p, size_t np, TernaryThreadPool *pool) {
    TernaryStack *a = build(name, shape, 2, p, np, 5), *b = build(name, shape, 2, p, np, 5);
    if (!a || !b) return 1;
    size_t n = shape[0] * shape[1], on = ternary_stack_output_len(a);
    TernaryVec *x = ternary_vec_alloc(n); size_t xd[2] = { shape[0], shape[1] };
    TernaryTensor *xt = ternary_tensor_view_ternary(xd, 2, x);
    TernaryCapabilities cn; memset(&cn, 0, sizeof(cn)); cn.num_cores = 1; cn.dot_impl_name = "scalar";
    float *oa = malloc(on * 4), *ob = malloc(on * 4), *go = malloc(on * 4);
    srand(11);
    for (int step = 0; step < 6; step++) {
        ternary_stack_zero_gradients(a); ternary_stack_zero_gradients(b);
        for (int s = 0; s < 4; s++) {
            for (size_t i = 0; i < n; i++) ternary_vec_set(x, i, rand() % 3 - 1);
            for (size_t i = 0; i < on; i++) go[i] = ((rand() % 2001) - 1000) / 1000.0f;
            ternary_set_compute_pool(NULL);
            ternary_stack_forward(a, xt, oa, &cn);
            ternary_stack_backward(a, xt, go);
            ternary_set_compute_pool(pool);
            ternary_stack_forward(b, xt, ob, &cn);
            ternary_stack_backward(b, xt, go);
            CHECK(memcmp(oa, ob, on * 4) == 0, "salida distinta con pool");
        }
        ternary_set_compute_pool(NULL); ternary_stack_apply_gradients(a, 0.001f);
        ternary_set_compute_pool(pool); ternary_stack_apply_gradients(b, 0.001f);
    }
    ternary_set_compute_pool(NULL);
    printf("[pool] %s: 6 pasos x 4 muestras, salidas identicas bit a bit con y sin pool de 4 hilos\n", name);
    ternary_tensor_free(xt); ternary_vec_free(x); ternary_stack_free(a); ternary_stack_free(b);
    free(oa); free(ob); free(go);
    return 0;
}

int main(void) {
    TernaryThreadPool *pool = ternary_threadpool_create(4);
    size_t sh[2] = { 24, 128 };
    double head[1] = { 300 }, swi[3] = { 192, 0, 0.2 }, gqa[5] = { 4, 2, 1, 10.0, 2 };
    if (run("lm_head", sh, head, 1, pool)) return 1;
    if (run("swiglu", sh, swi, 3, pool)) return 1;
    if (run("gqa", sh, gqa, 5, pool)) return 1;
    ternary_threadpool_destroy(pool);
    printf("test_layers_pool: OK\n");
    return 0;
}
