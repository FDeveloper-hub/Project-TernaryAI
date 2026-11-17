/* Fija el pool de hilos de calculo de los benchmarks: variable BENCH_THREADS (1 = sin pool). */
#include "ternary_threadpool.h"
#include <stdlib.h>
static void bench_pool_init(void) {
    const char *e = getenv("BENCH_THREADS"); int n = e ? atoi(e) : 1;
    if (n > 1) ternary_set_compute_pool(ternary_threadpool_create(n)); else ternary_set_compute_pool(NULL);
    fprintf(stderr, "[bench] hilos=%d\n", n > 1 ? n : 1);
}
