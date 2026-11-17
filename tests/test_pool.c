/* Pool de hilos: cada indice se ejecuta exactamente una vez, con cualquier numero de hilos (incluso mas que nucleos),
 * llamadas seguidas, dormir y despertar, llamadas anidadas y desde varios hilos, y el pool global. */
#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE
#include "ternary_threadpool.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FALLO (linea %d): ", __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); exit(1); } } while (0)

static void marcar(size_t i, void *ctx) { atomic_fetch_add(&((_Atomic int *)ctx)[i], 1); }
static void rango(size_t b, size_t e, void *ctx) { for (size_t i = b; i < e; i++) atomic_fetch_add(&((_Atomic int *)ctx)[i], 1); }

static void una_vez(TernaryThreadPool *p, size_t b, size_t e, int use_ranges, size_t grain) {
    _Atomic int *v = (_Atomic int *)calloc(e + 1, sizeof(_Atomic int));
    if (use_ranges) ternary_parallel_ranges(p, b, e, grain, rango, v); else ternary_parallel_for(p, b, e, marcar, v);
    for (size_t i = 0; i < e; i++) CHECK(atomic_load(&v[i]) == (i >= b ? 1 : 0), "indice %zu ejecutado %d veces (rango %zu..%zu)", i, atomic_load(&v[i]), b, e);
    free(v);
}

/* Partes fijas: cada parte se ejecuta una vez, y (con espera suficiente) en el hilo que le toca. Con mas hilos que partes, o mas partes que nucleos, tambien. */
static void parte(size_t b, size_t e, void *ctx) { for (size_t i = b; i < e; i++) atomic_fetch_add(&((_Atomic int *)ctx)[i], 1); }
static void estaticas(TernaryThreadPool *p, size_t nparts) {
    _Atomic int *v = (_Atomic int *)calloc(nparts + 1, sizeof(_Atomic int));
    ternary_parallel_static(p, nparts, parte, v);
    for (size_t i = 0; i < nparts; i++) CHECK(atomic_load(&v[i]) == 1, "parte fija %zu ejecutada %d veces (de %zu)", i, atomic_load(&v[i]), nparts);
    CHECK(atomic_load(&v[nparts]) == 0, "parte fija inexistente %zu ejecutada", nparts);
    free(v);
}

typedef struct { TernaryThreadPool *p; _Atomic int *cnt; size_t n; } Anid;
static void externa(size_t i, void *c_) { Anid *c = (Anid *)c_; (void)i; ternary_parallel_for(c->p, 0, c->n, marcar, c->cnt); }

static void *desde_hilo(void *a) {
    TernaryThreadPool *p = (TernaryThreadPool *)a;
    for (int r = 0; r < 200; r++) una_vez(p, 0, 37 + (size_t)r, r & 1, 0);
    return NULL;
}

int main(void) {
    const int hilos[] = {1, 2, 3, 4, 7, 16, 64};
    for (size_t h = 0; h < sizeof(hilos) / sizeof(hilos[0]); h++) {
        TernaryThreadPool *p = ternary_threadpool_create(hilos[h]);
        CHECK(p && ternary_threadpool_num_threads(p) == hilos[h], "pool de %d hilos", hilos[h]);
        CHECK(ternary_threadpool_verify(p) == 0, "verify");
        const size_t tam[] = {1, 2, 3, 5, 16, 100, 1000, 4097, 100000};
        for (size_t t = 0; t < sizeof(tam) / sizeof(tam[0]); t++) {
            una_vez(p, 0, tam[t], 0, 0); una_vez(p, 0, tam[t], 1, 0); una_vez(p, 0, tam[t], 1, 1); una_vez(p, 0, tam[t], 1, 7);
            if (tam[t] > 3) una_vez(p, 3, tam[t], 1, 0);
        }
        for (int r = 0; r < 3000; r++) una_vez(p, 0, 1 + (size_t)(r % 60), r & 1, 0);          /* muchas llamadas seguidas */
        ternary_parallel_for(p, 5, 5, marcar, NULL); ternary_parallel_for(p, 7, 3, marcar, NULL);   /* vacias */
        usleep(5000);                                                                        /* los trabajadores duermen... */
        una_vez(p, 0, 500, 0, 0);                                                           /* ...y despiertan */
        ternary_threadpool_set_spin_us(p, 0); usleep(2000); una_vez(p, 0, 321, 1, 0); una_vez(p, 0, 321, 0, 0);   /* sin giro: duermen siempre */
        ternary_threadpool_set_spin_us(p, 200);
        { _Atomic int *cnt = (_Atomic int *)calloc(50, sizeof(_Atomic int)); Anid a = { p, cnt, 50 };
          ternary_parallel_for(p, 0, 20, externa, &a);                                       /* anidada: corre en serie, sin bloqueo */
          for (int i = 0; i < 50; i++) CHECK(atomic_load(&cnt[i]) == 20, "anidada %d: %d", i, atomic_load(&cnt[i]));
          free(cnt); }
        { pthread_t th[3]; for (int i = 0; i < 3; i++) pthread_create(&th[i], NULL, desde_hilo, p);   /* varios llamadores */
          for (int i = 0; i < 3; i++) pthread_join(th[i], NULL); }
        for (size_t np = 1; np <= (size_t)hilos[h] && np <= 30; np++) for (int r = 0; r < 200; r++) estaticas(p, np);   /* partes <= hilos */
        estaticas(p, (size_t)hilos[h] + 5); estaticas(p, 40);                                                        /* partes > hilos: en serie */
        usleep(3000); estaticas(p, hilos[h] < 30 ? (size_t)hilos[h] : 30);                                           /* tras dormir */
        CHECK(ternary_threadpool_verify(p) == 0, "verify final");
        ternary_threadpool_destroy(p);
        printf("pool de %2d hilos: OK\n", hilos[h]);
    }
    CHECK(ternary_threadpool_num_threads(NULL) == 1, "NULL = 1 hilo");
    una_vez(NULL, 0, 100, 0, 0);
    /* pool global */
    CHECK(ternary_get_num_threads() >= 1, "global inicial");
    ternary_set_num_threads(5); CHECK(ternary_get_num_threads() == 5 && ternary_global_pool() && ternary_get_compute_pool() == ternary_global_pool(), "set 5");
    una_vez(ternary_get_compute_pool(), 0, 1234, 1, 0);
    ternary_set_num_threads(1); CHECK(ternary_get_num_threads() == 1 && ternary_global_pool() == NULL, "set 1");
    ternary_set_num_threads(200); una_vez(ternary_get_compute_pool(), 0, 1234, 0, 0); ternary_set_num_threads(1);
    printf("test_pool: OK\n");
    return 0;
}
