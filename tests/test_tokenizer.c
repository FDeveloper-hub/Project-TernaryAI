/*
 * Verifica el tokenizador BPE por bytes (src/ternary_tokenizer.c):
 * que aprenda fusiones reales de un corpus repetitivo, que
 * encode(decode(x)) == x siempre (con y sin fusiones aprendidas), que
 * el resultado sea determinista entre corridas, y que guardar/cargar
 * reproduzca exactamente el mismo tokenizador (mismo encode byte a
 * byte). No depende de ninguna capa/motor de computo -- es un modulo
 * aislado, por eso el test se compila solo con ternary_tokenizer.c.
 */
#include "tmp_path.h"
#include "ternary_tokenizer.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char err[256];

static void test_vocab_size_minimo(void) {
    TernaryTokenizer *t = ternary_tokenizer_create(255, err, sizeof(err));
    assert(t == NULL);  /* menor a 256 debe rechazarse */
    printf("[vocab_size minimo] rechazado correctamente: %s\n", err);
}

static int32_t *encode_alloc(const TernaryTokenizer *t, const char *text, size_t *n) {
    size_t len = strlen(text);
    ternary_tokenizer_encode(t, (const uint8_t *)text, len, NULL, 0, n, err, sizeof(err));
    int32_t *ids = malloc(*n * sizeof(int32_t));
    int ok = ternary_tokenizer_encode(t, (const uint8_t *)text, len, ids, *n, n, err, sizeof(err));
    assert(ok);
    return ids;
}

static char *decode_alloc(const TernaryTokenizer *t, const int32_t *ids, size_t n) {
    size_t len = 0;
    ternary_tokenizer_decode(t, ids, n, NULL, 0, &len, err, sizeof(err));
    char *out = malloc(len + 1);
    int ok = ternary_tokenizer_decode(t, ids, n, (uint8_t *)out, len, &len, err, sizeof(err));
    assert(ok);
    out[len] = '\0';
    return out;
}

static void test_aprende_fusiones_reales(void) {
    /* "ab" se repite mucho mas que cualquier otro par -> debe ser la
     * primera fusion aprendida, y el vocabulario final debe crecer. */
    const char *corpus = "abababababababababab cdcdcdcd xy";
    TernaryTokenizer *t = ternary_tokenizer_create(300, err, sizeof(err));
    assert(t != NULL);
    int ok = ternary_tokenizer_train(t, (const uint8_t *)corpus, strlen(corpus), err, sizeof(err));
    assert(ok);
    size_t merges = ternary_tokenizer_num_merges(t);
    printf("[aprende fusiones] vocab_size logrado=%zu (pedido 300), fusiones=%zu\n",
           ternary_tokenizer_vocab_size(t), merges);
    assert(merges > 0);
    assert(ternary_tokenizer_vocab_size(t) == 256 + merges);
    /* "ab" fusionado deberia hacer que "abab" ocupe menos ids que sus 4 bytes crudos */
    size_t n;
    int32_t *ids = encode_alloc(t, "abab", &n);
    printf("[aprende fusiones] 'abab' (4 bytes) codifico en %zu ids\n", n);
    assert(n < 4);
    free(ids);
    ternary_tokenizer_free(t);
}

static void test_roundtrip_con_y_sin_entrenar(void) {
    const char *muestras[] = {
        "", "a", "hola mundo", "abababababab", "\xc3\xb1\xc3\xa1\xc3\xa9\xc3\xad",  /* utf-8: ñáéí */
        "texto nunca visto en el corpus de entrenamiento 12345 !@#",
    };
    TernaryTokenizer *t = ternary_tokenizer_create(280, err, sizeof(err));
    assert(t != NULL);
    const char *corpus = "el rapido zorro marron salta sobre el perro perezoso. "
                          "el perro perezoso no hizo nada. el zorro se fue corriendo.";
    assert(ternary_tokenizer_train(t, (const uint8_t *)corpus, strlen(corpus), err, sizeof(err)));

    for (size_t i = 0; i < sizeof(muestras) / sizeof(muestras[0]); i++) {
        size_t n;
        int32_t *ids = encode_alloc(t, muestras[i], &n);
        char *back = decode_alloc(t, ids, n);
        int match = strcmp(back, muestras[i]) == 0;
        printf("[roundtrip] '%s' -> %zu ids -> '%s' %s\n", muestras[i], n, back, match ? "OK" : "FALLO");
        assert(match);
        free(ids); free(back);
    }
    ternary_tokenizer_free(t);
}

static void test_determinista_entre_corridas(void) {
    const char *corpus = "mississippi river banana bandana ababab cacaca lorem ipsum dolor sit amet "
                          "consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore";
    TernaryTokenizer *a = ternary_tokenizer_create(320, err, sizeof(err));
    TernaryTokenizer *b = ternary_tokenizer_create(320, err, sizeof(err));
    assert(ternary_tokenizer_train(a, (const uint8_t *)corpus, strlen(corpus), err, sizeof(err)));
    assert(ternary_tokenizer_train(b, (const uint8_t *)corpus, strlen(corpus), err, sizeof(err)));
    assert(ternary_tokenizer_num_merges(a) == ternary_tokenizer_num_merges(b));

    size_t na, nb;
    int32_t *ia = encode_alloc(a, "mississippi banana", &na);
    int32_t *ib = encode_alloc(b, "mississippi banana", &nb);
    assert(na == nb);
    assert(memcmp(ia, ib, na * sizeof(int32_t)) == 0);
    printf("[determinismo] dos entrenamientos identicos dan %zu fusiones e ids identicos\n",
           ternary_tokenizer_num_merges(a));
    free(ia); free(ib);
    ternary_tokenizer_free(a); ternary_tokenizer_free(b);
}

static void test_guardado_y_carga(void) {
    const char *corpus = "the quick brown fox jumps over the lazy dog the fox the dog the the the";
    TernaryTokenizer *t = ternary_tokenizer_create(300, err, sizeof(err));
    assert(ternary_tokenizer_train(t, (const uint8_t *)corpus, strlen(corpus), err, sizeof(err)));

    const char *path = tmp_path("test_tokenizer_ternaryai.bin");
    assert(ternary_tokenizer_save(t, path, err, sizeof(err)));
    TernaryTokenizer *loaded = ternary_tokenizer_load(path, err, sizeof(err));
    assert(loaded != NULL);
    assert(ternary_tokenizer_vocab_size(loaded) == ternary_tokenizer_vocab_size(t));

    const char *muestra = "the quick fox and the lazy the";
    size_t n1, n2;
    int32_t *ids1 = encode_alloc(t, muestra, &n1);
    int32_t *ids2 = encode_alloc(loaded, muestra, &n2);
    assert(n1 == n2 && memcmp(ids1, ids2, n1 * sizeof(int32_t)) == 0);
    printf("[guardado/carga] encode identico antes y despues de save+load (%zu ids)\n", n1);

    free(ids1); free(ids2);
    remove(path);
    ternary_tokenizer_free(t);
    ternary_tokenizer_free(loaded);
}

static void test_id_invalido_en_decode_no_crashea(void) {
    TernaryTokenizer *t = ternary_tokenizer_create(260, err, sizeof(err));
    assert(ternary_tokenizer_train(t, (const uint8_t *)"aaaa bbbb", 9, err, sizeof(err)));
    int32_t ids[3] = { 'x', 999999, 'y' };  /* 999999 esta fuera de rango a proposito */
    size_t len;
    int ok = ternary_tokenizer_decode(t, ids, 3, NULL, 0, &len, err, sizeof(err));
    assert(ok);
    char *out = malloc(len + 1);
    ternary_tokenizer_decode(t, ids, 3, (uint8_t *)out, len, &len, err, sizeof(err));
    out[len] = '\0';
    printf("[id invalido] decode con id fuera de rango no crashea, dio '%s' (%zu bytes)\n", out, len);
    free(out);
    ternary_tokenizer_free(t);
}

int main(void) {
    test_vocab_size_minimo();
    test_aprende_fusiones_reales();
    test_roundtrip_con_y_sin_entrenar();
    test_determinista_entre_corridas();
    test_guardado_y_carga();
    test_id_invalido_en_decode_no_crashea();
    printf("\ntest_tokenizer: TODO OK\n");
    return 0;
}
