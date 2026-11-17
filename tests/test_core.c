#include "ternary_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

static void test_ejemplo_a_mano(void) {
    /* w=[1,-1,0,1], x=[1,1,-1,-1] -> esperado -1 (verificado en la conversacion) */
    TernaryVec *w = ternary_vec_alloc(4);
    TernaryVec *x = ternary_vec_alloc(4);
    int wv[4] = {1, -1, 0, 1};
    int xv[4] = {1, 1, -1, -1};
    for (int i = 0; i < 4; i++) { ternary_vec_set(w, i, wv[i]); ternary_vec_set(x, i, xv[i]); }

    int32_t got = ternary_dot(w, x);
    int32_t ref = ternary_dot_reference(w, x);
    printf("[ejemplo a mano] popcount=%d referencia=%d esperado=-1\n", got, ref);
    assert(got == -1 && ref == -1);

    ternary_vec_free(w); ternary_vec_free(x);
}

static void test_get_set_roundtrip(void) {
    TernaryVec *v = ternary_vec_alloc(200); /* cruza varias palabras de 64 */
    int values[200];
    srand(1);
    for (int i = 0; i < 200; i++) {
        values[i] = (rand() % 3) - 1; /* -1, 0, 1 */
        ternary_vec_set(v, i, values[i]);
    }
    for (int i = 0; i < 200; i++) {
        assert(ternary_vec_get(v, i) == values[i]);
    }
    printf("[roundtrip set/get] 200 valores OK\n");
    ternary_vec_free(v);
}

static void test_bordes_n_no_multiplo_64(void) {
    /* n=1, n=63, n=64, n=65, n=127, n=128, n=129 */
    size_t tamanos[] = {1, 63, 64, 65, 127, 128, 129};
    for (size_t t = 0; t < sizeof(tamanos)/sizeof(tamanos[0]); t++) {
        size_t n = tamanos[t];
        TernaryVec *w = ternary_vec_alloc(n);
        TernaryVec *x = ternary_vec_alloc(n);
        srand((unsigned)(n * 7 + 3));
        for (size_t i = 0; i < n; i++) {
            ternary_vec_set(w, i, (rand() % 3) - 1);
            ternary_vec_set(x, i, (rand() % 3) - 1);
        }
        int32_t got = ternary_dot(w, x);
        int32_t ref = ternary_dot_reference(w, x);
        printf("[borde n=%zu] popcount=%d referencia=%d %s\n",
               n, got, ref, (got == ref) ? "OK" : "FALLO");
        assert(got == ref);

        /* Confirmar que los bits sobrantes de la ultima palabra quedaron en 0 */
        size_t bits_en_ultima_palabra_usados = n % 64;
        if (bits_en_ultima_palabra_usados != 0) {
            uint64_t mask_bits_extra = ~((((uint64_t)1) << bits_en_ultima_palabra_usados) - 1);
            assert((w->mask[w->words - 1] & mask_bits_extra) == 0);
            assert((x->mask[w->words - 1] & mask_bits_extra) == 0);
        }

        ternary_vec_free(w); ternary_vec_free(x);
    }
}

static void test_barrido_aleatorio_masivo(void) {
    /* 2000 pares de vectores de tamano aleatorio (1..500), valores aleatorios */
    srand(42);
    int fallos = 0;
    for (int trial = 0; trial < 2000; trial++) {
        size_t n = 1 + (size_t)(rand() % 500);
        TernaryVec *w = ternary_vec_alloc(n);
        TernaryVec *x = ternary_vec_alloc(n);
        for (size_t i = 0; i < n; i++) {
            ternary_vec_set(w, i, (rand() % 3) - 1);
            ternary_vec_set(x, i, (rand() % 3) - 1);
        }
        int32_t got = ternary_dot(w, x);
        int32_t ref = ternary_dot_reference(w, x);
        if (got != ref) {
            fallos++;
            printf("[FALLO trial=%d n=%zu] popcount=%d referencia=%d\n", trial, n, got, ref);
        }
        ternary_vec_free(w); ternary_vec_free(x);
    }
    printf("[barrido masivo] 2000 pruebas, %d fallos\n", fallos);
    assert(fallos == 0);
}

static void test_memoria_sin_desperdicio(void) {
    /* 4 valores ternarios: con codigo de 2 bits entrelazado ocuparian 1 byte.
     * Con 2 planos de 1 bit, cada plano necesita 1 palabra de 64 bits minima
     * (no hay tipo mas chico que uint64_t aqui), asi que a esta escala SI hay
     * "desperdicio" de granularidad -- es esperado y se explica: el ahorro
     * real aparece a partir de que el vector ocupa >= 1 palabra completa de
     * 64 valores. Confirmamos la cuenta exacta, no una intuicion. */
    TernaryVec *v64 = ternary_vec_alloc(64);
    printf("[memoria] 64 pesos ternarios -> %zu bytes (esperado 16: 8+8)\n",
           ternary_vec_bytes(v64));
    assert(ternary_vec_bytes(v64) == 16); /* 2 bits/peso * 64 pesos / 8 = 16 bytes */
    ternary_vec_free(v64);

    TernaryVec *v256 = ternary_vec_alloc(256);
    printf("[memoria] 256 pesos ternarios -> %zu bytes (esperado 64)\n",
           ternary_vec_bytes(v256));
    assert(ternary_vec_bytes(v256) == 64);
    ternary_vec_free(v256);
}

int main(void) {
    test_ejemplo_a_mano();
    test_get_set_roundtrip();
    test_bordes_n_no_multiplo_64();
    test_barrido_aleatorio_masivo();
    test_memoria_sin_desperdicio();
    printf("\nTODAS LAS PRUEBAS PASARON\n");
    return 0;
}
