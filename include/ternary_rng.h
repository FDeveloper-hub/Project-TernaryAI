#ifndef TERNARY_RNG_H
#define TERNARY_RNG_H
#include <stddef.h>
#include <stdint.h>
/*
 * Generador pseudoaleatorio propio de cada modelo -- reemplaza srand()/
 * rand() (estado GLOBAL del proceso: dos modelos creandose a la vez
 * desde hilos distintos se pisarian, y cada libc trae su propio
 * algoritmo) por un estado local y determinista.
 *
 * Implementa EXACTAMENTE el algoritmo de rand() de glibc (registro
 * aditivo TYPE_3, grado 31, separacion 3): para una misma semilla da la
 * misma secuencia que srand(seed)+rand() de glibc, en cualquier
 * plataforma. Asi la inicializacion de la fachada reproduce bit a bit
 * la del programa de referencia original (que usaba rand()) sin depender de la
 * libc del sistema. tests/test_simple.c lo verifica contra rand().
 */
typedef struct {
    int32_t state[31];
    int fptr, rptr;
} TernaryRng;

void ternary_rng_seed(TernaryRng *r, unsigned int seed);
/* Igual que rand() de glibc: entero en [0, 2147483647]. */
int32_t ternary_rng_next(TernaryRng *r);
/* Muestrea {-1,0,1} con la distribucion dependiente del fan-in de
 * la referencia original: p_nonzero=clip(1/sqrt(fan_in),0.15,0.9),
 * p(-1)=p(+1)=p_nonzero/2 (mismo criterio que el programa de referencia original). */
int ternary_rng_sample_faninit(TernaryRng *r, size_t fan_in);
#endif /* TERNARY_RNG_H */
