/*
 * Verifica los TOKENS ESPECIALES del tokenizador BPE (src/ternary_tokenizer.c):
 * cadenas reservadas con ids fijos (256, 257, ...) que el BPE nunca parte
 * ni fusiona con el texto de al lado.
 *
 *  - ids fijos y cuenta del vocabulario (256 + especiales + fusiones);
 *  - errores: vacio, repetido, agregar despues de entrenar, no caben en vocab_size;
 *  - entrenamiento: una marca es una frontera que ninguna fusion cruza;
 *  - encode: la marca aislada produce exactamente un id, y el resto del
 *    texto se codifica igual que si la marca no estuviera (la frontera
 *    es exacta, que es lo que permite enmascarar "solo la respuesta");
 *  - el especial mas largo gana; parse_special=0 lo trata como texto;
 *  - decode(encode(x)) == x en miles de mezclas aleatorias de bytes y marcas;
 *    skip_special las omite;
 *  - guardado/carga (TKZ2), sin especiales se sigue escribiendo TKZ1, y
 *    archivos corruptos se rechazan con mensaje.
 * Modulo aislado: se compila solo con ternary_tokenizer.c.
 */
#include "tmp_path.h"
#include "ternary_tokenizer.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char err[256];

#define U8(s) ((const uint8_t *)(s))

static int32_t *enc(const TernaryTokenizer *t, const char *s, size_t len, int parse, size_t *n) {
    assert(ternary_tokenizer_encode_ex(t, U8(s), len, parse, NULL, 0, n, err, sizeof err));
    int32_t *ids = malloc((*n + 1) * sizeof(int32_t));
    assert(ternary_tokenizer_encode_ex(t, U8(s), len, parse, ids, *n, n, err, sizeof err));
    return ids;
}

static char *dec(const TernaryTokenizer *t, const int32_t *ids, size_t n, int skip, size_t *len) {
    assert(ternary_tokenizer_decode_ex(t, ids, n, skip, NULL, 0, len, err, sizeof err));
    char *out = malloc(*len + 1);
    assert(ternary_tokenizer_decode_ex(t, ids, n, skip, (uint8_t *)out, *len, len, err, sizeof err));
    out[*len] = 0;
    return out;
}

static TernaryTokenizer *make(size_t vocab, const char *const *sp, size_t nsp, const char *corpus) {
    TernaryTokenizer *t = ternary_tokenizer_create(vocab, err, sizeof err);
    assert(t);
    for (size_t i = 0; i < nsp; i++)
        assert(ternary_tokenizer_add_special(t, U8(sp[i]), strlen(sp[i]), err, sizeof err));
    assert(ternary_tokenizer_train(t, U8(corpus), strlen(corpus), err, sizeof err));
    return t;
}

static const char *CORPUS =
    "<|user|>\nHola, como estas hoy?\n<|assistant|>\nMuy bien, gracias por preguntar.\n<|end|>\n"
    "<|user|>\nQue puedo hacer esta tarde?\n<|assistant|>\nPuedes salir a caminar o descansar.\n<|end|>\n"
    "<|user|>\nHola otra vez, como estas?\n<|assistant|>\nHola, aqui estoy para ayudarte.\n<|end|>\n";

static void test_ids_y_errores(void) {
    const char *sp[] = {"<|user|>", "<|assistant|>", "<|end|>"};
    TernaryTokenizer *t = ternary_tokenizer_create(300, err, sizeof err);
    assert(t);
    for (int i = 0; i < 3; i++) assert(ternary_tokenizer_add_special(t, U8(sp[i]), strlen(sp[i]), err, sizeof err));
    assert(ternary_tokenizer_num_special(t) == 3);
    assert(ternary_tokenizer_special_id(t, U8("<|user|>"), 8) == 256);
    assert(ternary_tokenizer_special_id(t, U8("<|assistant|>"), 13) == 257);
    assert(ternary_tokenizer_special_id(t, U8("<|end|>"), 7) == 258);
    assert(ternary_tokenizer_special_id(t, U8("<|nada|>"), 8) == -1);
    assert(ternary_tokenizer_special_id(t, U8("<|user|"), 7) == -1);  /* solo coincide el texto completo */

    assert(!ternary_tokenizer_add_special(t, U8(""), 0, err, sizeof err));
    printf("[errores] vacio rechazado: %s\n", err);
    assert(!ternary_tokenizer_add_special(t, U8("<|end|>"), 7, err, sizeof err));
    printf("[errores] repetido rechazado: %s\n", err);

    assert(ternary_tokenizer_train(t, U8(CORPUS), strlen(CORPUS), err, sizeof err));
    assert(ternary_tokenizer_vocab_size(t) == 256 + 3 + ternary_tokenizer_num_merges(t));
    assert(ternary_tokenizer_vocab_size(t) <= 300);
    assert(!ternary_tokenizer_add_special(t, U8("<|tarde|>"), 9, err, sizeof err));
    printf("[errores] agregar despues de entrenar rechazado: %s\n", err);

    uint8_t buf[64]; size_t L;
    assert(ternary_tokenizer_special_text(t, 1, buf, sizeof buf, &L, err, sizeof err));
    assert(L == 13 && memcmp(buf, "<|assistant|>", 13) == 0);
    assert(!ternary_tokenizer_special_text(t, 3, buf, sizeof buf, &L, err, sizeof err));
    ternary_tokenizer_free(t);

    /* no caben: vocab_size 257 solo admite 1 especial */
    t = ternary_tokenizer_create(257, err, sizeof err);
    assert(ternary_tokenizer_add_special(t, U8("<|a|>"), 5, err, sizeof err));
    assert(!ternary_tokenizer_add_special(t, U8("<|b|>"), 5, err, sizeof err));
    printf("[errores] no cabe otro especial en vocab_size: %s\n", err);
    assert(ternary_tokenizer_train(t, U8(CORPUS), strlen(CORPUS), err, sizeof err));
    assert(ternary_tokenizer_num_merges(t) == 0 && ternary_tokenizer_vocab_size(t) == 257);
    ternary_tokenizer_free(t);
    printf("[ids] fijos 256,257,258; vocab = 256 + especiales + fusiones\n");
}

static void test_frontera_en_entrenamiento(void) {
    /* "x<|s|>y<|s|>" repetido: sin la marca como frontera, pares como ('x','<') se repiten y se
     * fusionan; con ella como frontera no queda ningun par de texto contiguo. */
    char corpus[1024] = {0};
    for (int i = 0; i < 50; i++) strcat(corpus, "x<|s|>y<|s|>");
    const char *sp[] = {"<|s|>"};
    TernaryTokenizer *t = make(400, sp, 1, corpus);
    assert(ternary_tokenizer_num_merges(t) == 0);
    size_t n;
    int32_t *ids = enc(t, "xy", 2, 1, &n);
    assert(n == 2 && ids[0] == 'x' && ids[1] == 'y');
    free(ids);
    ternary_tokenizer_free(t);

    /* el mismo corpus SIN especial si aprende fusiones: confirma que la prueba tiene dientes */
    TernaryTokenizer *u = ternary_tokenizer_create(400, err, sizeof err);
    assert(ternary_tokenizer_train(u, U8(corpus), strlen(corpus), err, sizeof err));
    assert(ternary_tokenizer_num_merges(u) > 0);
    ternary_tokenizer_free(u);
    printf("[entrenamiento] la marca es una frontera: 0 fusiones con ella, >0 sin ella\n");
}

static void test_frontera_en_encode(void) {
    const char *sp[] = {"<|user|>", "<|assistant|>", "<|end|>"};
    TernaryTokenizer *t = make(400, sp, 3, CORPUS);
    const char *partes[] = {"\nHola, como estas hoy?\n", "\nMuy bien, gracias.\n", "\n", "x", ""};
    for (int a = 0; a < 5; a++) for (int b = 0; b < 5; b++) {
        char texto[512];
        int w = snprintf(texto, sizeof texto, "%s<|assistant|>%s", partes[a], partes[b]);
        size_t n, na, nb;
        int32_t *ids = enc(t, texto, (size_t)w, 1, &n);
        int32_t *ia = enc(t, partes[a], strlen(partes[a]), 1, &na);
        int32_t *ib = enc(t, partes[b], strlen(partes[b]), 1, &nb);
        assert(n == na + 1 + nb);
        assert(memcmp(ids, ia, na * sizeof(int32_t)) == 0);
        assert(ids[na] == 257);
        assert(memcmp(ids + na + 1, ib, nb * sizeof(int32_t)) == 0);
        free(ids); free(ia); free(ib);
    }
    printf("[encode] marca = un id exacto; texto a cada lado se codifica igual que suelto (25 combinaciones)\n");

    /* una conversacion entera: 1 aparicion de cada marca por turno, en orden */
    size_t n;
    int32_t *ids = enc(t, CORPUS, strlen(CORPUS), 1, &n);
    int cu = 0, ca = 0, ce = 0;
    for (size_t i = 0; i < n; i++) { cu += ids[i] == 256; ca += ids[i] == 257; ce += ids[i] == 258; }
    assert(cu == 3 && ca == 3 && ce == 3);
    free(ids);

    /* parse_special=0: la marca es texto corriente (ningun id >=256 por ser especial; solo bytes/fusiones) */
    ids = enc(t, "<|user|>", 8, 0, &n);
    for (size_t i = 0; i < n; i++) assert(ids[i] != 256 && ids[i] != 257 && ids[i] != 258);
    size_t L; char *back = dec(t, ids, n, 0, &L);
    assert(L == 8 && memcmp(back, "<|user|>", 8) == 0);
    free(back); free(ids);
    printf("[encode] parse_special=0 trata la marca como texto y sigue siendo reversible\n");
    ternary_tokenizer_free(t);
}

static void test_mas_largo_gana(void) {
    const char *sp[] = {"<|a|>", "<|a|>b", "<|a"};
    TernaryTokenizer *t = make(300, sp, 3, "relleno relleno");
    size_t n;
    int32_t *ids = enc(t, "x<|a|>by<|a|>z", 14, 1, &n);
    /* x [<|a|>b=257] y [<|a|>=256] z */
    assert(n == 5 && ids[0] == 'x' && ids[1] == 257 && ids[2] == 'y' && ids[3] == 256 && ids[4] == 'z');
    free(ids);
    ids = enc(t, "<|ab", 4, 1, &n);  /* "<|a" gana a nada; 'b' queda suelto */
    assert(n == 2 && ids[0] == 258 && ids[1] == 'b');
    free(ids);
    ternary_tokenizer_free(t);
    printf("[encode] gana el especial mas largo\n");
}

static void test_roundtrip_aleatorio(void) {
    const char *sp[] = {"<|user|>", "<|assistant|>", "<|end|>", "\xc2\xbf?"};  /* incluye uno con bytes no ASCII */
    TernaryTokenizer *t = make(420, sp, 4, CORPUS);
    const char *trozos[] = {"<|user|>", "<|assistant|>", "<|end|>", "<|", "|>", "<", "a", "Hola", " como ",
                            "\n", "\xc2\xbf", "?", "\xc3\xa9", "estas", "\x01\xff", "<|user", "end|>"};
    const int nt = (int)(sizeof trozos / sizeof trozos[0]);
    srand(7);
    for (int it = 0; it < 3000; it++) {
        char texto[1024]; size_t w = 0;
        int k = rand() % 20;
        for (int j = 0; j < k; j++) {
            const char *p = trozos[rand() % nt];
            size_t pl = strlen(p);
            memcpy(texto + w, p, pl); w += pl;
        }
        for (int parse = 0; parse <= 1; parse++) {
            size_t n, L;
            int32_t *ids = enc(t, texto, w, parse, &n);
            char *back = dec(t, ids, n, 0, &L);
            assert(L == w && memcmp(back, texto, w) == 0);
            free(back);
            if (!parse) for (size_t i = 0; i < n; i++) assert(!(ids[i] >= 256 && ids[i] < 260));
            free(ids);
        }
    }
    printf("[roundtrip] 3000 mezclas aleatorias de bytes y marcas (con y sin parse_special): decode(encode(x)) == x\n");

    size_t n, L;
    int32_t *ids = enc(t, "<|user|>\nHola\n<|assistant|>\nBien\n<|end|>\n", strlen("<|user|>\nHola\n<|assistant|>\nBien\n<|end|>\n"), 1, &n);
    char *plain = dec(t, ids, n, 1, &L);
    assert(L == strlen("\nHola\n\nBien\n\n") && memcmp(plain, "\nHola\n\nBien\n\n", L) == 0);
    free(plain);
    int32_t raro[] = {256, 99999, -5, 'h', 257};
    char *r = dec(t, raro, 5, 0, &L);
    assert(L == strlen("<|user|>h<|assistant|>") && memcmp(r, "<|user|>h<|assistant|>", L) == 0);  /* ids invalidos se ignoran */
    free(r); free(ids);
    printf("[decode] skip_special omite las marcas; ids invalidos se ignoran\n");
    ternary_tokenizer_free(t);
}

static void test_guardado_y_carga(void) {
    const char *sp[] = {"<|user|>", "<|assistant|>", "<|end|>"};
    TernaryTokenizer *t = make(420, sp, 3, CORPUS);
    const char *path = tmp_path("test_tokenizer_special.bin");
    assert(ternary_tokenizer_save(t, path, err, sizeof err));
    TernaryTokenizer *u = ternary_tokenizer_load(path, err, sizeof err);
    assert(u);
    assert(ternary_tokenizer_vocab_size(u) == ternary_tokenizer_vocab_size(t));
    assert(ternary_tokenizer_num_special(u) == 3);
    size_t n1, n2;
    int32_t *a = enc(t, CORPUS, strlen(CORPUS), 1, &n1);
    int32_t *b = enc(u, CORPUS, strlen(CORPUS), 1, &n2);
    assert(n1 == n2 && memcmp(a, b, n1 * sizeof(int32_t)) == 0);
    free(a); free(b);
    const char *path2 = tmp_path("test_tokenizer_special2.bin");
    assert(ternary_tokenizer_save(u, path2, err, sizeof err));
    FILE *f1 = fopen(path, "rb"), *f2 = fopen(path2, "rb");
    int same = 1, c1, c2;
    do { c1 = fgetc(f1); c2 = fgetc(f2); if (c1 != c2) same = 0; } while (c1 != EOF && c2 != EOF);
    fclose(f1); fclose(f2);
    assert(same);
    assert(!ternary_tokenizer_add_special(u, U8("<|x|>"), 5, err, sizeof err));  /* cargado = entrenado */
    ternary_tokenizer_free(u);
    printf("[guardado/carga] tokenizer.json reproduce el mismo tokenizador (encode y archivo identicos)\n");

    /* el archivo es JSON legible: formato, especiales con su id y fusiones como pares */
    {
        FILE *f = fopen(path, "rb");
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        char *txt = malloc((size_t)sz + 1);
        assert(fread(txt, 1, (size_t)sz, f) == (size_t)sz);
        fclose(f); txt[sz] = '\0';
        assert(strstr(txt, "\"format\": \"ternary-bpe\"") && strstr(txt, "\"content\": \"<|user|>\"") &&
               strstr(txt, "\"id\": 256") && strstr(txt, "\"merges\": ["));
        const char *bad = tmp_path("test_tokenizer_special_bad.json");
        /* corrupciones: cada una debe rechazarse con mensaje, nunca cargar a medias */
        struct { const char *from, *to, *what; } cases[] = {
            { "\"ternary-bpe\"", "\"otra-cosa\"", "formato desconocido" },
            { "\"format_version\": 1", "\"format_version\": 2", "version futura" },
            { "\"id\": 257", "\"id\": 300", "id de especial fuera de orden" },
        };
        for (size_t c = 0; c < sizeof cases / sizeof cases[0]; c++) {
            const char *at = strstr(txt, cases[c].from);
            assert(at);
            size_t pre = (size_t)(at - txt), fl = strlen(cases[c].from), tl = strlen(cases[c].to);
            char *mod = malloc((size_t)sz + tl + 1);
            memcpy(mod, txt, pre); memcpy(mod + pre, cases[c].to, tl); memcpy(mod + pre + tl, at + fl, (size_t)sz - pre - fl + 1);
            FILE *fb = fopen(bad, "wb"); fwrite(mod, 1, strlen(mod), fb); fclose(fb);
            assert(ternary_tokenizer_load(bad, err, sizeof err) == NULL && err[0]);
            printf("[corrupto] %s: %s\n", cases[c].what, err);
            free(mod);
        }
        {   /* fusion con id futuro: el primer miembro de la primera fusion pasa a 100000 */
            const char *key = "\"merges\": [\n    [";
            const char *at = strstr(txt, key);
            assert(at);
            const char *num = at + strlen(key), *comma = strchr(num, ',');
            assert(comma);
            size_t pre = (size_t)(num - txt);
            char *mod = malloc((size_t)sz + 16);
            memcpy(mod, txt, pre); memcpy(mod + pre, "100000", 6); strcpy(mod + pre + 6, comma);
            FILE *fm = fopen(bad, "wb"); fwrite(mod, 1, strlen(mod), fm); fclose(fm);
            assert(ternary_tokenizer_load(bad, err, sizeof err) == NULL && strstr(err, "ids invalidos"));
            printf("[corrupto] fusion con id futuro: %s\n", err);
            free(mod);
        }
        FILE *fb = fopen(bad, "wb"); fwrite(txt, 1, (size_t)sz / 2, fb); fclose(fb);   /* truncado */
        assert(ternary_tokenizer_load(bad, err, sizeof err) == NULL);
        printf("[corrupto] truncado: %s\n", err);
        remove(bad);
        free(txt);
    }
    ternary_tokenizer_free(t);

    /* sin especiales: "special_tokens" queda vacio */
    TernaryTokenizer *p = make(300, NULL, 0, CORPUS);
    assert(ternary_tokenizer_save(p, path, err, sizeof err));
    TernaryTokenizer *q = ternary_tokenizer_load(path, err, sizeof err);
    assert(q && ternary_tokenizer_num_special(q) == 0 && ternary_tokenizer_vocab_size(q) == ternary_tokenizer_vocab_size(p));
    ternary_tokenizer_free(p); ternary_tokenizer_free(q);
    remove(path); remove(path2);
    printf("[guardado/carga] sin especiales se guarda y se carga igual\n");

    /* un especial que no es UTF-8 valido viaja como content_hex */
    {
        const uint8_t raw[3] = { 0xff, 0xfe, 'x' };
        TernaryTokenizer *r = ternary_tokenizer_create(300, err, sizeof err);
        assert(r && ternary_tokenizer_add_special(r, raw, 3, err, sizeof err));
        assert(ternary_tokenizer_train(r, (const uint8_t *)CORPUS, strlen(CORPUS), err, sizeof err));
        assert(ternary_tokenizer_save(r, path, err, sizeof err));
        TernaryTokenizer *s = ternary_tokenizer_load(path, err, sizeof err);
        assert(s && ternary_tokenizer_special_id(s, raw, 3) == 256);
        ternary_tokenizer_free(r); ternary_tokenizer_free(s);
        remove(path);
        printf("[guardado/carga] especial no UTF-8 (content_hex) ida y vuelta\n");
    }
}

int main(void) {
    test_ids_y_errores();
    test_frontera_en_entrenamiento();
    test_frontera_en_encode();
    test_mas_largo_gana();
    test_roundtrip_aleatorio();
    test_guardado_y_carga();
    printf("\ntest_tokenizer_special: TODO OK\n");
    return 0;
}
