#define _POSIX_C_SOURCE 200809L
#include "ternary_prof.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int g_tprof_on = 0;

typedef struct {
    const char *name; int idx; int parent;
    uint64_t calls, total_ns, child_ns;
} ProfNode;

#define PROF_MAX_NODES 4096
#define PROF_MAX_DEPTH 64
static ProfNode g_nodes[PROF_MAX_NODES];
static int g_n = 0;
static struct { int node; uint64_t t0; } g_stack[PROF_MAX_DEPTH];
static int g_depth = 0;
static pthread_t g_owner;
static int g_owner_set = 0;
static double g_overhead_ns = 0.0;

static inline uint64_t now_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static int find_or_add(int parent, const char *name, int idx) {
    for (int i = 0; i < g_n; i++)
        if (g_nodes[i].parent == parent && g_nodes[i].idx == idx &&
            (g_nodes[i].name == name || strcmp(g_nodes[i].name, name) == 0)) return i;
    if (g_n >= PROF_MAX_NODES) return -1;
    ProfNode *n = &g_nodes[g_n];
    memset(n, 0, sizeof *n); n->name = name; n->idx = idx; n->parent = parent;
    return g_n++;
}

void ternary_prof_enter(const char *name, int idx) {
    if (!g_owner_set || !pthread_equal(pthread_self(), g_owner)) return;
    if (g_depth >= PROF_MAX_DEPTH) { g_depth++; return; }       /* demasiado profundo: no se cuenta, pero leave() queda balanceado */
    int parent = g_depth ? g_stack[g_depth - 1].node : -1;
    int id = find_or_add(parent, name, idx);
    g_stack[g_depth].node = id;
    g_stack[g_depth].t0 = now_ns();
    g_depth++;
}

void ternary_prof_leave(void) {
    if (!g_owner_set || !pthread_equal(pthread_self(), g_owner)) return;
    if (g_depth <= 0) return;
    if (g_depth > PROF_MAX_DEPTH) { g_depth--; return; }
    uint64_t t1 = now_ns();
    g_depth--;
    int id = g_stack[g_depth].node;
    if (id < 0) return;
    uint64_t dt = t1 - g_stack[g_depth].t0;
    g_nodes[id].calls++; g_nodes[id].total_ns += dt;
    if (g_depth > 0 && g_stack[g_depth - 1].node >= 0) g_nodes[g_stack[g_depth - 1].node].child_ns += dt;
}

/* Costo de un par enter/leave vacio (incluye las dos lecturas del reloj): se mide una vez al activar. */
static void calibrate(void) {
    const int N = 200000;
    int saved_n = g_n, saved_depth = g_depth;
    uint64_t t0 = now_ns();
    for (int i = 0; i < N; i++) { ternary_prof_enter("__calib__", -1); ternary_prof_leave(); }
    uint64_t t1 = now_ns();
    g_overhead_ns = (double)(t1 - t0) / N;
    g_n = saved_n; g_depth = saved_depth;      /* se descarta el nodo de calibracion */
}

void ternary_prof_enable(int on) {
    if (on) {
        g_owner = pthread_self(); g_owner_set = 1; g_depth = 0;
        g_tprof_on = 1;
        if (g_overhead_ns == 0.0) calibrate();
    } else {
        g_tprof_on = 0; g_depth = 0;
    }
}

void ternary_prof_reset(void) {
    for (int i = 0; i < g_n; i++) { g_nodes[i].calls = 0; g_nodes[i].total_ns = 0; g_nodes[i].child_ns = 0; }
}

size_t ternary_prof_dump_json(char *buf, size_t cap) {
    size_t need = 0;
    char tmp[512];
    #define EMIT(...) do { int k = snprintf(tmp, sizeof tmp, __VA_ARGS__); \
        if (k > 0) { if (buf && need + (size_t)k < cap) memcpy(buf + need, tmp, (size_t)k); need += (size_t)k; } } while (0)
    EMIT("{\"overhead_ns_per_scope\":%.2f,\"nodes\":[", g_overhead_ns);
    for (int i = 0; i < g_n; i++) {
        const ProfNode *n = &g_nodes[i];
        uint64_t self = n->total_ns > n->child_ns ? n->total_ns - n->child_ns : 0;
        EMIT("%s{\"id\":%d,\"parent\":%d,\"name\":\"%s\",\"idx\":%d,\"calls\":%llu,\"total_ns\":%llu,\"self_ns\":%llu}",
             i ? "," : "", i, n->parent, n->name, n->idx, (unsigned long long)n->calls,
             (unsigned long long)n->total_ns, (unsigned long long)self);
    }
    EMIT("]}");
    #undef EMIT
    if (buf && cap) buf[need < cap ? need : cap - 1] = '\0';
    return need;
}
