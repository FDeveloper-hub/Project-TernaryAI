#ifndef TERNARY_MEM_H
#define TERNARY_MEM_H
/* Memoria alineada a linea de cache (64 B). Los buffers que varios hilos escriben por trozos (una cabeza, un bloque de filas)
 * deben empezar en frontera de linea: con el malloc normal (16 B) dos hilos comparten linea en los bordes de sus trozos y el
 * protocolo de coherencia los frena (falso compartir; medido: la atencion hacia atras iba 2.3x MAS LENTA con 2 hilos).
 * Se liberan con free(). */
#include <stdlib.h>
#include <string.h>
/* aligned_alloc de C11 no esta declarado en Android (bionic) por debajo de API 28: alli se usa posix_memalign (siempre disponible). */
#if defined(__ANDROID__)
int posix_memalign(void **, size_t, size_t);
static inline void *ternary_aligned_alloc(size_t align, size_t bytes) { void *p = NULL; return posix_memalign(&p, align, bytes) == 0 ? p : NULL; }
#else
static inline void *ternary_aligned_alloc(size_t align, size_t bytes) { return aligned_alloc(align, bytes); }
#endif
static inline void *ternary_malloc64(size_t bytes) {
    const size_t b = (bytes + 63) / 64 * 64;
    return ternary_aligned_alloc(64, b ? b : 64);
}
static inline void *ternary_calloc64(size_t n, size_t sz) {
    if (sz != 0 && n > (size_t)-1 / sz) return NULL;
    const size_t bytes = n * sz, b = (bytes + 63) / 64 * 64;
    void *p = ternary_aligned_alloc(64, b ? b : 64);
    if (p) memset(p, 0, b ? b : 64);
    return p;
}
#endif
