#define _POSIX_C_SOURCE 200809L
#include "ternary_threadpool.h"
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define CPU_RELAX() _mm_pause()
#elif defined(__arm__) || defined(__aarch64__)
#define CPU_RELAX() __asm__ __volatile__("yield" ::: "memory")
#else
#define CPU_RELAX() sched_yield()
#endif

#define POOL_CANARY 0x7E27A11Fu
#define SEALED 0xFFFFFFFFu            /* indice de un trabajo ya terminado: ningun trabajador puede reclamar mas */

/* Protocolo (sin trabajadores que esperar uno por uno):
 *  claim = (seq << 32) | indice_siguiente.  El llamador publica un trabajo escribiendo sus campos y despues claim=(seq,0).
 *  Un trabajador lee claim, despues los campos, y reclama un trozo con compare-and-swap sobre ESE valor de claim: si el trabajo
 *  ya termino (claim sellado o de otra seq) el CAS falla y no ejecuta nada. 'remaining' cuenta los indices sin terminar:
 *  el llamador sigue cuando llega a 0, sella claim y solo entonces puede escribir el trabajo siguiente. Un trabajador lento o
 *  dormido nunca bloquea al llamador (con mas hilos que nucleos, solo avanza el que esta en un nucleo). */
struct TernaryThreadPool {
    uint32_t canary_head;
    int num_threads;                 /* hilos logicos, incluido el llamador (>= 1) */
    int num_workers;                 /* trabajadores arrancados (num_threads - 1 si todo fue bien) */
    pthread_t *threads, *threads_shadow;
    _Atomic uint64_t claim;
    _Atomic uint64_t remaining;
    _Atomic(TernaryParallelRangeFn) fn;
    _Atomic(void *) ctx;
    _Atomic size_t begin, total, grain;
    _Atomic int static_mode;         /* 1: el trabajo actual son partes fijas (parte i = hilo i), claim = (seq<<32)|mascara de partes tomadas */
    _Atomic int sleepers;
    _Atomic int shutdown;
    _Atomic int spin_us;
    uint32_t seq;                    /* solo lo toca el llamador (bajo call_mutex) */
    pthread_mutex_t call_mutex;      /* un llamador a la vez; otro que llegue corre en serie */
    pthread_mutex_t sleep_mutex;
    pthread_cond_t sleep_cond;
    uint32_t canary_tail;
};

static _Thread_local int tls_busy = 0;       /* dentro de una tarea del pool: las llamadas anidadas corren en serie */

typedef struct { TernaryThreadPool *pool; int tid; } WorkerArg;

static inline uint64_t now_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* Reclama y ejecuta trozos del trabajo 'seq' hasta que no queden. */
static void run_chunks(TernaryThreadPool *p, uint32_t seq) {
    for (;;) {
        uint64_t c = atomic_load_explicit(&p->claim, memory_order_acquire);
        if ((uint32_t)(c >> 32) != seq) return;
        const uint32_t idx = (uint32_t)c;
        if (idx == SEALED) return;
        TernaryParallelRangeFn fn = atomic_load_explicit(&p->fn, memory_order_relaxed);
        void *ctx = atomic_load_explicit(&p->ctx, memory_order_relaxed);
        const size_t total = atomic_load_explicit(&p->total, memory_order_relaxed);
        const size_t grain = atomic_load_explicit(&p->grain, memory_order_relaxed);
        const size_t begin = atomic_load_explicit(&p->begin, memory_order_relaxed);
        if (idx >= total) return;
        const size_t take = (total - idx < grain) ? (total - idx) : grain;
        if (!atomic_compare_exchange_weak_explicit(&p->claim, &c, c + take, memory_order_acq_rel, memory_order_acquire)) continue;
        tls_busy = 1;
        fn(begin + idx, begin + idx + take, ctx);
        tls_busy = 0;
        atomic_fetch_sub_explicit(&p->remaining, take, memory_order_release);
    }
}

/* Parte fija 'part' de un trabajo estatico: la toma quien la reclame primero (bit en la mascara). */
static int run_static_part(TernaryThreadPool *p, uint32_t seq, int part) {
    for (;;) {
        uint64_t c = atomic_load_explicit(&p->claim, memory_order_acquire);
        if ((uint32_t)(c >> 32) != seq) return 0;
        const uint32_t mask = (uint32_t)c;
        if (mask == SEALED || (mask >> part) & 1u) return 0;
        TernaryParallelRangeFn fn = atomic_load_explicit(&p->fn, memory_order_relaxed);
        void *ctx = atomic_load_explicit(&p->ctx, memory_order_relaxed);
        const size_t begin = atomic_load_explicit(&p->begin, memory_order_relaxed);
        if ((size_t)part >= atomic_load_explicit(&p->total, memory_order_relaxed)) return 0;      /* hilo sin parte en este trabajo */
        if (!atomic_compare_exchange_weak_explicit(&p->claim, &c, c | (1ull << part), memory_order_acq_rel, memory_order_acquire)) continue;
        tls_busy = 1;
        fn(begin + (size_t)part, begin + (size_t)part + 1, ctx);
        tls_busy = 0;
        atomic_fetch_sub_explicit(&p->remaining, 1, memory_order_release);
        return 1;
    }
}

static void *worker_main(void *arg_) {
    WorkerArg *arg = (WorkerArg *)arg_;
    TernaryThreadPool *p = arg->pool;
    const int tid = arg->tid;
    free(arg);
    uint32_t seen = 0;
    for (;;) {
        /* esperar un trabajo nuevo: girar, luego ceder el nucleo, luego dormir */
        const uint64_t spin_ns = (uint64_t)atomic_load_explicit(&p->spin_us, memory_order_relaxed) * 1000ull;
        uint64_t t0 = spin_ns ? now_ns() : 0;
        unsigned it = 0;
        int phase = spin_ns ? 0 : 2;
        for (;;) {
            const uint64_t c = atomic_load_explicit(&p->claim, memory_order_acquire);
            if ((uint32_t)(c >> 32) != seen) { seen = (uint32_t)(c >> 32); break; }
            if (atomic_load_explicit(&p->shutdown, memory_order_acquire)) return NULL;
            if (phase == 0) {
                CPU_RELAX();
                if ((++it & 63u) == 0 && now_ns() - t0 > spin_ns) phase = 1;
            } else if (phase == 1) {
                sched_yield();
                if ((++it & 15u) == 0 && now_ns() - t0 > spin_ns * 2) phase = 2;
            } else {
                pthread_mutex_lock(&p->sleep_mutex);
                atomic_fetch_add_explicit(&p->sleepers, 1, memory_order_seq_cst);
                while ((uint32_t)(atomic_load_explicit(&p->claim, memory_order_seq_cst) >> 32) == seen &&
                       !atomic_load_explicit(&p->shutdown, memory_order_seq_cst))
                    pthread_cond_wait(&p->sleep_cond, &p->sleep_mutex);
                atomic_fetch_sub_explicit(&p->sleepers, 1, memory_order_seq_cst);
                pthread_mutex_unlock(&p->sleep_mutex);
                phase = spin_ns ? 0 : 2; it = 0; if (spin_ns) t0 = now_ns();
            }
        }
        if (atomic_load_explicit(&p->static_mode, memory_order_acquire)) run_static_part(p, seen, tid); else run_chunks(p, seen);
    }
}

TernaryThreadPool *ternary_threadpool_create(int num_threads) {
    if (num_threads < 1) num_threads = 1;
    TernaryThreadPool *p = (TernaryThreadPool *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->canary_head = p->canary_tail = POOL_CANARY;
    p->num_threads = num_threads;
    atomic_init(&p->claim, (uint64_t)0 | SEALED);       /* seq 0 sellada: nada que reclamar */
    atomic_init(&p->remaining, 0);
    atomic_init(&p->static_mode, 0); atomic_init(&p->sleepers, 0); atomic_init(&p->shutdown, 0); atomic_init(&p->spin_us, 200);
    atomic_init(&p->fn, NULL); atomic_init(&p->ctx, NULL);
    atomic_init(&p->begin, 0); atomic_init(&p->total, 0); atomic_init(&p->grain, 1);
    pthread_mutex_init(&p->call_mutex, NULL);
    pthread_mutex_init(&p->sleep_mutex, NULL);
    pthread_cond_init(&p->sleep_cond, NULL);
    {   /* con hilos <= nucleos, los trabajadores giran hasta 5 ms antes de dormir (la espera entre regiones cuesta ~1.7 us); con mas hilos que\n         * nucleos, girar le quita el nucleo a quien si trabaja: plazo corto */
        const long nc = sysconf(_SC_NPROCESSORS_ONLN);
        if (nc >= num_threads) atomic_store(&p->spin_us, 5000);
    }
    const int want = num_threads - 1;
    if (want > 0) {
        p->threads = (pthread_t *)calloc((size_t)want, sizeof(pthread_t));
        p->threads_shadow = (pthread_t *)calloc((size_t)want, sizeof(pthread_t));
        if (!p->threads || !p->threads_shadow) { free(p->threads); free(p->threads_shadow); free(p); return NULL; }
    }
    int started = 0;
    for (int i = 0; i < want; i++) {
        WorkerArg *a = (WorkerArg *)malloc(sizeof(*a));
        if (!a) break;
        a->pool = p; a->tid = started + 1;
        if (pthread_create(&p->threads[started], NULL, worker_main, a) != 0) { free(a); break; }
        p->threads_shadow[started] = p->threads[started];
        started++;
    }
    if (started != want) fprintf(stderr, "ternary_threadpool_create: solo arrancaron %d de %d trabajadores pedidos.\n", started, want);
    p->num_workers = started;
    p->num_threads = started + 1;
    return p;
}

void ternary_threadpool_destroy(TernaryThreadPool *p) {
    if (!p) return;
    pthread_mutex_lock(&p->sleep_mutex);
    atomic_store_explicit(&p->shutdown, 1, memory_order_seq_cst);
    pthread_cond_broadcast(&p->sleep_cond);
    pthread_mutex_unlock(&p->sleep_mutex);
    for (int i = 0; i < p->num_workers; i++) pthread_join(p->threads[i], NULL);
    pthread_mutex_destroy(&p->call_mutex); pthread_mutex_destroy(&p->sleep_mutex); pthread_cond_destroy(&p->sleep_cond);
    free(p->threads); free(p->threads_shadow); free(p);
}

int ternary_threadpool_verify(const TernaryThreadPool *p) {
    if (!p) return 1;
    int problems = 0;
    if (p->canary_head != POOL_CANARY) { fprintf(stderr, "[pool-verify] canario inicial pisado (0x%x)\n", p->canary_head); problems++; }
    if (p->canary_tail != POOL_CANARY) { fprintf(stderr, "[pool-verify] canario final pisado (0x%x)\n", p->canary_tail); problems++; }
    if (p->num_workers < 0 || p->num_workers + 1 != p->num_threads) {
        fprintf(stderr, "[pool-verify] num_workers invalido (%d, num_threads=%d)\n", p->num_workers, p->num_threads);
        return problems + 1;
    }
    for (int i = 0; i < p->num_workers; i++)
        if (p->threads[i] != p->threads_shadow[i]) { fprintf(stderr, "[pool-verify] threads[%d] alterado\n", i); problems++; }
    return problems;
}

void ternary_threadpool_set_spin_us(TernaryThreadPool *p, int us) { if (p) atomic_store(&p->spin_us, us < 0 ? 0 : us); }
int ternary_threadpool_num_threads(const TernaryThreadPool *p) { return p ? p->num_threads : 1; }

void ternary_parallel_ranges(TernaryThreadPool *p, size_t begin, size_t end, size_t grain, TernaryParallelRangeFn fn, void *ctx) {
    if (end <= begin) return;
    const size_t total = end - begin;
    if (!p || p->num_threads <= 1 || total < 2 || total >= (size_t)SEALED || tls_busy || pthread_mutex_trylock(&p->call_mutex) != 0) {
        fn(begin, end, ctx);
        return;
    }
    if (grain == 0) {                        /* ~4 trozos por hilo: se balancea sin pagar un CAS por indice */
        grain = total / ((size_t)p->num_threads * 4);
        if (grain < 1) grain = 1;
    }
    const uint32_t seq = ++p->seq ? p->seq : ++p->seq;     /* nunca 0 */
    atomic_store_explicit(&p->fn, fn, memory_order_relaxed);
    atomic_store_explicit(&p->ctx, ctx, memory_order_relaxed);
    atomic_store_explicit(&p->begin, begin, memory_order_relaxed);
    atomic_store_explicit(&p->total, total, memory_order_relaxed);
    atomic_store_explicit(&p->grain, grain, memory_order_relaxed);
    atomic_store_explicit(&p->remaining, total, memory_order_relaxed);
    atomic_store_explicit(&p->static_mode, 0, memory_order_relaxed);
    atomic_store_explicit(&p->claim, (uint64_t)seq << 32, memory_order_seq_cst);        /* publica */
    if (atomic_load_explicit(&p->sleepers, memory_order_seq_cst) > 0) {
        pthread_mutex_lock(&p->sleep_mutex);
        pthread_cond_broadcast(&p->sleep_cond);
        pthread_mutex_unlock(&p->sleep_mutex);
    }
    run_chunks(p, seq);                                  /* el llamador tambien trabaja */
    while (atomic_load_explicit(&p->remaining, memory_order_acquire) != 0) CPU_RELAX();
    atomic_store_explicit(&p->claim, ((uint64_t)seq << 32) | SEALED, memory_order_seq_cst);  /* sella: nadie reclama mas */
    atomic_thread_fence(memory_order_seq_cst);           /* los campos del proximo trabajo se escriben despues del sello */
    pthread_mutex_unlock(&p->call_mutex);
}

/* Partes FIJAS: la parte i la corre el hilo i (el llamador es el hilo 0), asi cada hilo reutiliza siempre los mismos datos en su cache.
 * Si un trabajador tarda en tomar su parte (mas hilos que nucleos, nucleo ocupado), el llamador la toma pasado un breve plazo. */
void ternary_parallel_static(TernaryThreadPool *p, size_t nparts, TernaryParallelRangeFn fn, void *ctx) {
    if (nparts == 0) return;
    if (!p || p->num_threads <= 1 || nparts < 2 || nparts > 30 || nparts > (size_t)p->num_threads || tls_busy || pthread_mutex_trylock(&p->call_mutex) != 0) {
        fn(0, nparts, ctx);
        return;
    }
    const uint32_t seq = ++p->seq ? p->seq : ++p->seq;
    atomic_store_explicit(&p->fn, fn, memory_order_relaxed);
    atomic_store_explicit(&p->ctx, ctx, memory_order_relaxed);
    atomic_store_explicit(&p->begin, 0, memory_order_relaxed);
    atomic_store_explicit(&p->total, nparts, memory_order_relaxed);
    atomic_store_explicit(&p->grain, 1, memory_order_relaxed);
    atomic_store_explicit(&p->remaining, nparts, memory_order_relaxed);
    atomic_store_explicit(&p->static_mode, 1, memory_order_relaxed);
    atomic_store_explicit(&p->claim, (uint64_t)seq << 32, memory_order_seq_cst);
    if (atomic_load_explicit(&p->sleepers, memory_order_seq_cst) > 0) {
        pthread_mutex_lock(&p->sleep_mutex);
        pthread_cond_broadcast(&p->sleep_cond);
        pthread_mutex_unlock(&p->sleep_mutex);
    }
    run_static_part(p, seq, 0);
    for (size_t j = 1; j < nparts; j++) {                      /* partes de los trabajadores: esperar un plazo corto a que las tomen */
        const uint64_t t0 = now_ns();
        for (;;) {
            const uint64_t c = atomic_load_explicit(&p->claim, memory_order_acquire);
            if (((uint32_t)c >> j) & 1u) break;
            if (now_ns() - t0 > 100000ull) { run_static_part(p, seq, (int)j); break; }
            CPU_RELAX();
        }
    }
    while (atomic_load_explicit(&p->remaining, memory_order_acquire) != 0) CPU_RELAX();
    atomic_store_explicit(&p->claim, ((uint64_t)seq << 32) | SEALED, memory_order_seq_cst);
    atomic_thread_fence(memory_order_seq_cst);
    pthread_mutex_unlock(&p->call_mutex);
}

typedef struct { TernaryParallelTaskFn fn; void *ctx; } ForCtx;
static void for_adapter(size_t b, size_t e, void *c_) {
    ForCtx *c = (ForCtx *)c_;
    for (size_t i = b; i < e; i++) c->fn(i, c->ctx);
}

void ternary_parallel_for(TernaryThreadPool *p, size_t begin, size_t end, TernaryParallelTaskFn fn, void *ctx) {
    if (end <= begin) return;
    if (!p || p->num_threads <= 1) { for (size_t i = begin; i < end; i++) fn(i, ctx); return; }
    ForCtx c = { fn, ctx };
    ternary_parallel_ranges(p, begin, end, 0, for_adapter, &c);
}

/* ---------------------------------------------------------------- pool global */
static TernaryThreadPool *g_pool = NULL;
static int g_threads = 0;                       /* 0 = sin inicializar */
static TernaryThreadPool *g_compute_override = NULL;
static pthread_mutex_t g_pool_lock = PTHREAD_MUTEX_INITIALIZER;

static void global_init_locked(void) {
    if (g_threads != 0) return;
    int n = 1;
    const char *e = getenv("TERNARYAI_THREADS");
    if (e && *e) { const int v = atoi(e); if (v > 1) n = v; }
    g_threads = n;
    if (n > 1) g_pool = ternary_threadpool_create(n);
    if (g_pool) g_threads = ternary_threadpool_num_threads(g_pool); else g_threads = 1;
}

void ternary_set_num_threads(int n) {
    if (n < 1) n = 1;
    pthread_mutex_lock(&g_pool_lock);
    global_init_locked();
    if (n != g_threads) {
        ternary_threadpool_destroy(g_pool); g_pool = NULL;
        if (n > 1) g_pool = ternary_threadpool_create(n);
        g_threads = g_pool ? ternary_threadpool_num_threads(g_pool) : 1;
    }
    pthread_mutex_unlock(&g_pool_lock);
}
int ternary_get_num_threads(void) { pthread_mutex_lock(&g_pool_lock); global_init_locked(); const int n = g_threads; pthread_mutex_unlock(&g_pool_lock); return n; }
TernaryThreadPool *ternary_global_pool(void) { pthread_mutex_lock(&g_pool_lock); global_init_locked(); TernaryThreadPool *p = g_pool; pthread_mutex_unlock(&g_pool_lock); return p; }

void ternary_set_compute_pool(TernaryThreadPool *pool) { g_compute_override = pool; }
TernaryThreadPool *ternary_get_compute_pool(void) { return g_compute_override ? g_compute_override : ternary_global_pool(); }

/* ---- ayudas elemento a elemento ---- */
typedef struct { unsigned char *dst; const unsigned char *src; size_t bytes; } ParMem;
#define PAR_MEM_BLOCK 65536
static void par_mem_range(size_t b0, size_t b1, void *c_) {
    ParMem *c = (ParMem *)c_;
    const size_t lo = b0 * PAR_MEM_BLOCK; size_t hi = b1 * PAR_MEM_BLOCK; if (hi > c->bytes) hi = c->bytes;
    if (c->src) memcpy(c->dst + lo, c->src + lo, hi - lo); else memset(c->dst + lo, 0, hi - lo);
}
void ternary_par_zero(void *p, size_t bytes) {
    TernaryThreadPool *pool = ternary_get_compute_pool();
    if (!pool || bytes < TERNARY_PAR_MIN_BYTES || ternary_threadpool_num_threads(pool) <= 1) { memset(p, 0, bytes); return; }
    ParMem c = { (unsigned char *)p, NULL, bytes };
    ternary_parallel_ranges(pool, 0, (bytes + PAR_MEM_BLOCK - 1) / PAR_MEM_BLOCK, 1, par_mem_range, &c);
}
void ternary_par_copy(void *dst, const void *src, size_t bytes) {
    TernaryThreadPool *pool = ternary_get_compute_pool();
    if (!pool || bytes < TERNARY_PAR_MIN_BYTES || ternary_threadpool_num_threads(pool) <= 1) { memcpy(dst, src, bytes); return; }
    ParMem c = { (unsigned char *)dst, (const unsigned char *)src, bytes };
    ternary_parallel_ranges(pool, 0, (bytes + PAR_MEM_BLOCK - 1) / PAR_MEM_BLOCK, 1, par_mem_range, &c);
}
