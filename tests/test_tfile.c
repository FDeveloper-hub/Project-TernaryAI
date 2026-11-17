/*
 * Formato de modelo: contenedor model.ternary (estilo safetensors) + config.json + training_state.ternary.
 *
 *  A) Contenedor: ida y vuelta de todos los dtypes, alineacion a 64 bytes, orden de la cabecera, y una bateria
 *     de archivos corruptos o malintencionados (cabecera mentirosa, offsets solapados o fuera del bloque,
 *     dtype/forma incoherentes, version futura, truncado en CADA longitud posible) que deben rechazarse con mensaje.
 *  B) JSON: flotantes que vuelven exactos, DOM, profundidad maxima, sangrado.
 *  C) Pila: pesos conocidos -> model.ternary decodificado A MANO (sin la libreria) coincide; el export no lleva
 *     training_state.ternary y borra uno viejo; el checkpoint si; config.json alterado / tensor faltante / forma
 *     equivocada / estado de otra red se rechazan con mensaje; instantanea en memoria identica.
 */
#include "ternary_stack.h"
#include "ternary_tfile.h"
#include "ternary_json.h"
#include "ternary_model_io.h"
#include "tmp_path.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { fprintf(stderr, "FALLO (linea %d): %s\n", __LINE__, msg); fails++; } } while (0)

/* ---- A) contenedor ------------------------------------------------------------------------------------- */
static void test_container_roundtrip(void) {
    TernaryTWriter *w = ternary_tw_create();
    float f[5] = { 1.5f, -2.25f, 0.0f, 1e-30f, 3.0e30f };
    int32_t i32[3] = { -7, 0, 2147483647 };
    int8_t i8[7] = { -128, -1, 0, 1, 127, 5, -5 };
    uint8_t u8[1] = { 200 };
    TernaryVec *v = ternary_vec_alloc(23);
    for (size_t k = 0; k < 23; k++) ternary_vec_set(v, k, (int)(k % 3) - 1);
    size_t s5[1] = { 5 }, s3[1] = { 3 }, s7[2] = { 7, 1 }, s1[1] = { 1 }, s23[2] = { 23, 1 }, s0[2] = { 0, 4 };
    CHECK(ternary_tw_meta(w, "autor", "prueba") && !ternary_tw_meta(w, "format", "x") && !ternary_tw_meta(w, "version", "9"), "meta: format/version son reservados");
    CHECK(ternary_tw_add(w, "a.f32", TTD_F32, s5, 1, f), "add f32");
    CHECK(ternary_tw_add(w, "b.i32", TTD_I32, s3, 1, i32), "add i32");
    CHECK(ternary_tw_add(w, "c.i8", TTD_I8, s7, 2, i8), "add i8");
    CHECK(ternary_tw_add(w, "d.u8", TTD_U8, s1, 1, u8), "add u8");
    CHECK(ternary_tw_add_trits(w, "e.trit", s23, 2, v), "add trits");
    CHECK(ternary_tw_add(w, "f.vacio", TTD_F32, s0, 2, f), "add tensor vacio");
    CHECK(!ternary_tw_add(w, "a.f32", TTD_F32, s5, 1, f), "nombre repetido rechazado");
    CHECK(!ternary_tw_add(w, "x", TTD_TRIT5, s5, 1, f), "TRIT5 solo entra por add_trits");
    CHECK(!ternary_tw_add_trits(w, "y", s7, 2, v), "forma que no coincide con el vector");
    const char *path = tmp_path("test_tfile_a.ternary");
    CHECK(ternary_tw_write_file(w, path), "escribir");
    char err[256] = "";
    TernaryTReader *r = ternary_tr_open(path, err, sizeof err);
    CHECK(r != NULL, err);
    if (r) {
        CHECK(ternary_tr_count(r) == 6, "6 tensores");
        const TernaryTInfo *t = ternary_tr_find(r, "a.f32");
        CHECK(t && t->dtype == TTD_F32 && t->nelem == 5 && t->nbytes == 20 && ternary_tinfo_f32(t, 3) == 1e-30f && ternary_tinfo_f32(t, 4) == 3.0e30f, "f32");
        t = ternary_tr_find(r, "b.i32");
        int32_t g[3]; if (t) memcpy(g, t->data, 12);
        CHECK(t && g[0] == -7 && g[2] == 2147483647, "i32");
        t = ternary_tr_find(r, "c.i8");
        CHECK(t && t->ndims == 2 && t->shape[0] == 7 && ((const int8_t *)t->data)[0] == -128 && ((const int8_t *)t->data)[4] == 127, "i8");
        t = ternary_tr_find(r, "d.u8");
        CHECK(t && t->data[0] == 200, "u8");
        t = ternary_tr_find(r, "e.trit");
        CHECK(t && t->dtype == TTD_TRIT5 && t->nbytes == 5, "trit5: 23 pesos -> 5 bytes");
        TernaryVec *u = ternary_vec_alloc(23);
        CHECK(t && ternary_tinfo_read_trits(t, u), "leer trits");
        int same = 1; for (size_t k = 0; k < 23; k++) same &= ternary_vec_get(u, k) == (int)(k % 3) - 1;
        CHECK(same, "trits identicos");
        ternary_vec_free(u);
        t = ternary_tr_find(r, "f.vacio");
        CHECK(t && t->nelem == 0 && t->nbytes == 0, "tensor vacio");
        CHECK(ternary_tr_find(r, "no.existe") == NULL, "find de un nombre inexistente");
        CHECK(ternary_tr_meta(r, "autor") && !strcmp(ternary_tr_meta(r, "autor"), "prueba") && !strcmp(ternary_tr_meta(r, "format"), "ternary") &&
              !strcmp(ternary_tr_meta(r, "version"), "1"), "metadata");
        /* alineacion: cada tensor empieza en multiplo de 64 dentro del archivo (y, con mmap, en memoria) */
        for (size_t k = 0; k < ternary_tr_count(r); k++) {
            const TernaryTInfo *q = ternary_tr_at(r, k);
            CHECK(((uintptr_t)q->data % TERNARY_TFILE_ALIGN) == 0, "tensor alineado a 64 bytes");
        }
        CHECK(ternary_tr_expect(r, "a.f32", TTD_F32, s5, 1, err, sizeof err) != NULL, "expect correcto");
        CHECK(ternary_tr_expect(r, "a.f32", TTD_F32, s3, 1, err, sizeof err) == NULL && strstr(err, "a.f32") && strstr(err, "[5]"), "expect: forma equivocada con mensaje");
        CHECK(ternary_tr_expect(r, "a.f32", TTD_I32, s5, 1, err, sizeof err) == NULL, "expect: dtype equivocado");
        CHECK(ternary_tr_expect(r, "nada", TTD_I32, s5, 1, err, sizeof err) == NULL && strstr(err, "nada"), "expect: tensor faltante");
        ternary_tr_close(r);
    }
    /* el archivo: 8 bytes de largo + JSON + relleno, inicio de datos multiplo de 64, orden de insercion, JSON valido */
    long len = 0; char *raw = read_whole_file(path, &len);
    CHECK(raw != NULL, "releer");
    if (raw) {
        uint64_t hn = 0; for (int k = 0; k < 8; k++) hn |= (uint64_t)(uint8_t)raw[k] << (8 * k);
        CHECK((8 + hn) % 64 == 0 && raw[8] == '{' && (raw[8 + hn - 1] == ' ' || raw[8 + hn - 1] == '}'), "cabecera alineada a 64 y rellena con espacios");
        char *a = strstr(raw + 8, "\"a.f32\""), *b = strstr(raw + 8, "\"b.i32\""), *e = strstr(raw + 8, "\"e.trit\"");
        CHECK(a && b && e && a < b && b < e, "la cabecera conserva el orden de insercion");
        char jerr[100]; TernaryJNode *root = ternary_jdom_parse(raw + 8, hn, jerr, sizeof jerr);
        CHECK(root && ternary_jdom_get(ternary_jdom_get(root, "a.f32"), "data_offsets"), "la cabecera es JSON valido");
        ternary_jdom_free(root);
        free(raw);
    }
    ternary_tw_free(w); ternary_vec_free(v);
    remove(path);
}

/* arma un archivo con una cabecera JSON dada y 'blob' bytes de datos */
static uint8_t *make_file(const char *hdr, size_t blob, size_t *out_len, int hn_override) {
    size_t hl = strlen(hdr), total = 8 + hl, pad = ((total + 63) & ~(size_t)63) - total;
    uint64_t n = (uint64_t)(hl + pad);
    if (hn_override >= 0) n = (uint64_t)hn_override;
    uint8_t *buf = (uint8_t *)calloc(8 + hl + pad + blob + 1, 1);
    for (int k = 0; k < 8; k++) buf[k] = (uint8_t)(n >> (8 * k));
    memcpy(buf + 8, hdr, hl); memset(buf + 8 + hl, ' ', pad);
    for (size_t k = 0; k < blob; k++) buf[8 + hl + pad + k] = (uint8_t)(k * 31 + 7);
    *out_len = 8 + hl + pad + blob;
    return buf;
}

static void expect_reject(const char *hdr, size_t blob, int hn_override, const char *needle, const char *what) {
    size_t len; uint8_t *buf = make_file(hdr, blob, &len, hn_override);
    char err[256] = "";
    TernaryTReader *r = ternary_tr_from_memory(buf, len, err, sizeof err);
    int ok = (r == NULL) && err[0] && (!needle || strstr(err, needle));
    if (!ok) fprintf(stderr, "  (caso '%s': mensaje='%s')\n", what, err);
    CHECK(ok, what);
    if (r) ternary_tr_close(r);
    free(buf);
}

static void test_container_corrupt(void) {
    const char *M = "\"__metadata__\":{\"format\":\"ternary\",\"version\":\"1\"}";
    char h[600];
    /* un archivo valido minimo se acepta */
    snprintf(h, sizeof h, "{%s,\"w\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,16]}}", M);
    { size_t len; uint8_t *b = make_file(h, 16, &len, -1); char err[200]; TernaryTReader *r = ternary_tr_from_memory(b, len, err, sizeof err);
      CHECK(r != NULL, err); ternary_tr_close(r); free(b); }
    snprintf(h, sizeof h, "{%s,\"w\":{\"dtype\":\"F64\",\"shape\":[4],\"data_offsets\":[0,32]}}", M);
    expect_reject(h, 32, -1, "dtype desconocido", "dtype desconocido");
    snprintf(h, sizeof h, "{%s,\"w\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,8]}}", M);
    expect_reject(h, 16, -1, "ocupan", "bytes que no cuadran con dtype x forma");
    snprintf(h, sizeof h, "{%s,\"w\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,16]}}", M);
    expect_reject(h, 8, -1, "truncado", "tensor mas alla del bloque");
    snprintf(h, sizeof h, "{%s,\"w\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[8,16]}}", M);
    expect_reject(h, 64, -1, "alinear", "inicio sin alinear a 64");
    snprintf(h, sizeof h, "{%s,\"a\":{\"dtype\":\"F32\",\"shape\":[32],\"data_offsets\":[0,128]},\"b\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[64,80]}}", M);
    expect_reject(h, 192, -1, "solapan", "tensores solapados");
    snprintf(h, sizeof h, "{%s,\"w\":{\"dtype\":\"F32\",\"shape\":[-4],\"data_offsets\":[0,16]}}", M);
    expect_reject(h, 16, -1, "shape", "forma negativa");
    snprintf(h, sizeof h, "{%s,\"w\":{\"dtype\":\"F32\",\"shape\":[1,1,1,1,1,1,1,1,1],\"data_offsets\":[0,4]}}", M);
    expect_reject(h, 16, -1, "shape", "demasiadas dimensiones");
    snprintf(h, sizeof h, "{%s,\"w\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[16,0]}}", M);
    expect_reject(h, 16, -1, "data_offsets", "offsets invertidos");
    snprintf(h, sizeof h, "{%s,\"w\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,16,32]}}", M);
    expect_reject(h, 16, -1, "data_offsets", "offsets con 3 valores");
    snprintf(h, sizeof h, "{%s,\"w\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,16]},\"w\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[64,80]}}", M);
    expect_reject(h, 128, -1, "repetido", "nombre repetido en la cabecera");
    snprintf(h, sizeof h, "{\"__metadata__\":{\"format\":\"ternary\",\"version\":\"2\"},\"w\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,16]}}");
    expect_reject(h, 16, -1, "version", "version de formato futura");
    snprintf(h, sizeof h, "{\"__metadata__\":{\"format\":\"safetensors\",\"version\":\"1\"}}");
    expect_reject(h, 0, -1, "format", "otro formato");
    snprintf(h, sizeof h, "{\"w\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,16]}}");
    expect_reject(h, 16, -1, "format", "sin __metadata__");
    snprintf(h, sizeof h, "{\"__metadata__\":{\"format\":1,\"version\":\"1\"}}");
    expect_reject(h, 0, -1, "string", "metadata que no es string");
    expect_reject("[1,2,3]", 0, -1, "objeto", "cabecera que no es un objeto");
    expect_reject("{\"a\":", 0, -1, "JSON", "JSON incompleto");
    snprintf(h, sizeof h, "{%s} basura", M);
    expect_reject(h, 0, -1, "sobrante", "texto sobrante tras la cabecera");
    snprintf(h, sizeof h, "{%s}", M);
    expect_reject(h, 0, 100000, "no cabe", "largo de cabecera mayor que el archivo");
    expect_reject(h, 0, 33, "alineado", "inicio de datos sin alinear");
    { size_t len; uint8_t *b = make_file(h, 0, &len, -1); char err[200]; b[7] = 0x7f;   /* largo absurdo */
      CHECK(ternary_tr_from_memory(b, len, err, sizeof err) == NULL, "largo absurdo"); free(b); }
    { char err[100]; uint8_t tiny[5] = {1,2,3,4,5}; CHECK(ternary_tr_from_memory(tiny, 5, err, sizeof err) == NULL && strstr(err, "corto"), "archivo de 5 bytes"); }
    { char err[100]; CHECK(ternary_tr_open(tmp_path("test_tfile_no_existe.ternary"), err, sizeof err) == NULL && err[0], "archivo inexistente"); }
    /* truncado en CADA longitud posible de un archivo real */
    {
        TernaryTWriter *w = ternary_tw_create();
        float f[40]; for (int k = 0; k < 40; k++) f[k] = (float)k;
        TernaryVec *v = ternary_vec_alloc(200);
        for (size_t k = 0; k < 200; k++) ternary_vec_set(v, k, (int)(k % 3) - 1);
        size_t s40[1] = { 40 }, s200[1] = { 200 };
        ternary_tw_add(w, "f", TTD_F32, s40, 1, f); ternary_tw_add_trits(w, "t", s200, 1, v);
        uint8_t *buf; size_t len;
        CHECK(ternary_tw_serialize(w, &buf, &len), "serializar en memoria");
        int rejected = 0;
        for (size_t cut = 0; cut < len; cut++) {
            char err[200];
            TernaryTReader *r = ternary_tr_from_memory(buf, cut, err, sizeof err);
            if (!r) rejected++; else ternary_tr_close(r);
        }
        CHECK(rejected == (int)len, "todo truncado (de 0 a len-1 bytes) se rechaza");
        char err[200]; TernaryTReader *r = ternary_tr_from_memory(buf, len, err, sizeof err);
        CHECK(r != NULL, "el archivo completo si carga");
        /* un byte de trits fuera de 0..242 se detecta al decodificar */
        const TernaryTInfo *t = ternary_tr_find(r, "t");
        uint8_t *mut = (uint8_t *)t->data - (uint8_t *)buf + buf;
        TernaryVec *u = ternary_vec_alloc(200);
        CHECK(ternary_tinfo_read_trits(t, u), "decodifica el original");
        uint8_t keep = mut[3]; mut[3] = 243;
        CHECK(!ternary_tinfo_read_trits(t, u), "byte 243 en un tensor TRIT5 se rechaza");
        mut[3] = keep;
        ternary_vec_free(u); ternary_tr_close(r); free(buf); ternary_tw_free(w); ternary_vec_free(v);
    }
}

/* ---- B) JSON ------------------------------------------------------------------------------------------- */
static void test_json(void) {
    const float vals[] = { 0.1f, 1e-5f, 0.3f, 1.0f / 3.0f, 3.4028235e38f, -3.4028235e38f, 1e-45f, 1.17549435e-38f, 123456.789f, -0.0f, 10.0f, 0.5f };
    int exact = 1;
    for (size_t k = 0; k < sizeof vals / sizeof vals[0]; k++) {
        TernaryJBuf b; ternary_jbuf_init(&b); ternary_jbuf_float(&b, vals[k]);
        float back = strtof(ternary_jbuf_cstr(&b), NULL);
        if (memcmp(&back, &vals[k], sizeof back) != 0 && !(back == 0.0f && vals[k] == 0.0f)) { exact = 0; fprintf(stderr, "  %s no vuelve exacto\n", ternary_jbuf_cstr(&b)); }
        ternary_jbuf_free(&b);
    }
    CHECK(exact, "un float escrito en JSON vuelve exacto");
    { TernaryJBuf b; ternary_jbuf_init(&b); ternary_jbuf_float(&b, 1e-5f); CHECK(b.len == 5 && !memcmp(b.buf, "1e-05", 5), "forma mas corta (1e-05, no 9.99999975e-06)"); ternary_jbuf_free(&b); }
    { TernaryJBuf b; ternary_jbuf_init(&b); ternary_jbuf_float(&b, NAN); CHECK(b.failed, "NaN no se puede escribir en JSON"); ternary_jbuf_free(&b); }
    /* DOM y escapes */
    char err[120];
    const char *src = "{\"a\":[1,2.5,-3e2,true,null],\"s\":\"h\\u00f3la \\\"x\\\" \\ud83d\\ude00\",\"o\":{\"k\":123456789012}}";
    TernaryJNode *r = ternary_jdom_parse(src, strlen(src), err, sizeof err);
    CHECK(r != NULL, err);
    if (r) {
        const TernaryJNode *a = ternary_jdom_get(r, "a");
        CHECK(a && a->n == 5 && a->kids[0].is_int && a->kids[0].i == 1 && !a->kids[1].is_int && a->kids[1].num == 2.5 &&
              a->kids[2].num == -300.0 && a->kids[3].b == 1 && a->kids[4].type == TJ_NULL, "arreglo mixto");
        const char *s = ternary_jdom_get_str(r, "s");
        CHECK(s && !strcmp(s, "h\xc3\xb3la \"x\" \xf0\x9f\x98\x80"), "escapes \\u y pares subrogados a UTF-8");
        int64_t big; CHECK(ternary_jdom_get_int(ternary_jdom_get(r, "o"), "k", &big) && big == 123456789012LL, "entero grande exacto");
        CHECK(!ternary_jdom_get_int(r, "a", &big), "getter tipado rechaza otro tipo");
        ternary_jdom_free(r);
    }
    char deep[400]; memset(deep, '[', 150); deep[150] = '\0';
    CHECK(ternary_jdom_parse(deep, 150, err, sizeof err) == NULL, "JSON demasiado anidado se rechaza (sin desbordar la pila)");
    CHECK(ternary_jdom_parse("{\"a\":1,}", 8, err, sizeof err) == NULL, "coma sobrante");
    CHECK(ternary_jdom_parse("[1 2]", 5, err, sizeof err) == NULL, "falta coma");
    CHECK(ternary_jdom_parse("\"abc", 4, err, sizeof err) == NULL, "string sin cerrar");
    CHECK(ternary_jdom_parse("", 0, err, sizeof err) == NULL, "vacio");
    /* sangrado: legible y equivalente */
    const char *c = "{\"a\":{\"b\":[1,2,3],\"c\":[{\"d\":1},{\"e\":\"x, y\"}]},\"f\":[]}";
    size_t pl; char *p = ternary_json_pretty(c, strlen(c), &pl);
    CHECK(p && strstr(p, "\"b\": [1, 2, 3]") && strstr(p, "\"f\": []") && strstr(p, "\n    \"c\": ["), "sangrado: arreglos escalares en una linea, objetos expandidos");
    TernaryJNode *r1 = ternary_jdom_parse(c, strlen(c), err, sizeof err), *r2 = p ? ternary_jdom_parse(p, pl, err, sizeof err) : NULL;
    CHECK(r1 && r2 && ternary_jdom_get(ternary_jdom_get(r2, "a"), "c")->n == 2 && !strcmp(ternary_jdom_get_str(&ternary_jdom_get(ternary_jdom_get(r2, "a"), "c")->kids[1], "e"), "x, y"), "el JSON con sangrado vuelve a leerse igual");
    ternary_jdom_free(r1); ternary_jdom_free(r2); free(p);
}

/* ---- C) pila --------------------------------------------------------------------------------------------- */
#define WI(o, i) ((int)(((o) * 7 + (i) * 3 + (o) * (i)) % 3) - 1)
static TernaryStack *known_net(void) {
    TernaryDense *l1 = ternary_dense_alloc(8, 6, 1), *l2 = ternary_dense_alloc(6, 3, 0);
    for (size_t o = 0; o < 6; o++) {
        for (size_t i = 0; i < 8; i++) ternary_dense_set_weight(l1, o, i, WI(o, i));
        ternary_vec_set(l1->bias, o, (int)(o % 3) - 1);
    }
    for (size_t o = 0; o < 3; o++) for (size_t i = 0; i < 6; i++) ternary_dense_set_weight(l2, o, i, WI(o + 2, i + 1));
    l1->scale = 0.125f; l2->scale = 0.3f;
    for (size_t k = 0; k < 48; k++) l1->counter_W[k] = (int8_t)((int)k % 11 - 5);
    for (size_t k = 0; k < 18; k++) l2->counter_W[k] = (int8_t)(7 - (int)k % 13);
    for (size_t k = 0; k < 6; k++) l1->counter_b[k] = (int8_t)(k - 2);
    TernaryStack *s = ternary_stack_create();
    ternary_stack_add(s, ternary_layer_from_dense(l1), TERNARY_ACTIVATION_UINT8);
    ternary_stack_add(s, ternary_layer_from_dense(l2), TERNARY_ACTIVATION_UINT8);
    return s;
}

static void write_file_text(const char *path, const char *txt) { FILE *f = fopen(path, "wb"); if (!f) { CHECK(0, "no se pudo crear el archivo de prueba"); return; } fwrite(txt, 1, strlen(txt), f); fclose(f); }

static void replace_in_file(const char *path, const char *from, const char *to) {
    long n = 0; char *t = read_whole_file(path, &n);
    if (!t) { CHECK(0, "no se pudo leer para alterar"); return; }
    t[n] = '\0';
    char *at = strstr(t, from);
    if (!at) { fprintf(stderr, "  no hay '%s' en %s\n", from, path); CHECK(0, "texto a alterar no encontrado"); free(t); return; }
    FILE *f = fopen(path, "wb");
    fwrite(t, 1, (size_t)(at - t), f); fwrite(to, 1, strlen(to), f); fwrite(at + strlen(from), 1, (size_t)n - (size_t)(at - t) - strlen(from), f);
    fclose(f); free(t);
}

static int decode_by_hand(const uint8_t *bytes, size_t n, int *out) {   /* base 3, 5 por byte, SIN usar la libreria */
    for (size_t b = 0; b < (n + 4) / 5; b++) {
        unsigned c = bytes[b];
        if (c > 242) return 0;
        for (int j = 0; j < 5; j++) { unsigned d = c % 3; c /= 3; if (b * 5 + (size_t)j < n) out[b * 5 + (size_t)j] = (int)d - 1; else if (d != 1) return 0; }
    }
    return 1;
}

static void test_stack_files(void) {
    const char *ck = tmp_path("test_tfile_ck"), *ex = tmp_path("test_tfile_ex"), *bad = tmp_path("test_tfile_bad");
    model_dir_remove(ck); model_dir_remove(ex); model_dir_remove(bad);
    TernaryStack *s = known_net();
    CHECK(ternary_stack_save(s, ck, 1) && ternary_stack_save(s, ex, 0), "guardar checkpoint y export");
    CHECK(access(model_file_path(ck, "training_state.ternary"), F_OK) == 0 && access(model_file_path(ex, "training_state.ternary"), F_OK) != 0, "solo el checkpoint tiene training_state.ternary");
    CHECK(access(model_file_path(ex, "config.json"), F_OK) == 0 && access(model_file_path(ex, "model.ternary"), F_OK) == 0, "el export tiene config.json y model.ternary");
    CHECK(access(model_file_path(ex, "model.ternary.tmp"), F_OK) != 0 && access(model_file_path(ex, "config.json.tmp"), F_OK) != 0, "no quedan archivos .tmp");

    /* model.ternary decodificado a mano coincide con los pesos conocidos */
    char err[256] = "";
    TernaryTReader *r = ternary_tr_open(model_file_path(ex, "model.ternary"), err, sizeof err);
    CHECK(r != NULL, err);
    if (r) {
        const TernaryTInfo *w0 = ternary_tr_find(r, "layers.0.weight"), *w1 = ternary_tr_find(r, "layers.1.weight"),
                           *b0 = ternary_tr_find(r, "layers.0.bias"), *s0 = ternary_tr_find(r, "layers.0.scale");
        CHECK(w0 && w0->dtype == TTD_TRIT5 && w0->ndims == 2 && w0->shape[0] == 6 && w0->shape[1] == 8 && w0->nbytes == 10, "layers.0.weight: TRIT5 [6,8], 10 bytes");
        CHECK(w1 && w1->shape[0] == 3 && w1->shape[1] == 6 && ternary_tr_find(r, "layers.1.bias") == NULL, "layers.1.weight [3,6] y sin bias");
        CHECK(s0 && ternary_tinfo_f32(s0, 0) == 0.125f && ternary_tinfo_f32(ternary_tr_find(r, "layers.1.scale"), 0) == 0.3f, "escalas exactas");
        int d0[48], d1[18], db[6];
        CHECK(w0 && decode_by_hand(w0->data, 48, d0) && w1 && decode_by_hand(w1->data, 18, d1) && b0 && decode_by_hand(b0->data, 6, db), "decodificar a mano");
        int ok = 1;
        for (size_t o = 0; o < 6; o++) { for (size_t i = 0; i < 8; i++) ok &= d0[o * 8 + i] == WI(o, i); ok &= db[o] == (int)(o % 3) - 1; }
        for (size_t o = 0; o < 3; o++) for (size_t i = 0; i < 6; i++) ok &= d1[o * 6 + i] == WI(o + 2, i + 1);
        CHECK(ok, "los pesos del archivo son los pesos de la red (fila-mayor, [salida][entrada])");
        CHECK(ternary_tr_find(r, "layers.0.weight_counters") == NULL, "el export no lleva contadores");
        ternary_tr_close(r);
    }
    /* el checkpoint guarda los contadores en training_state.ternary */
    r = ternary_tr_open(model_file_path(ck, "training_state.ternary"), err, sizeof err);
    CHECK(r != NULL, err);
    if (r) {
        const TernaryTInfo *v0 = ternary_tr_find(r, "layers.0.weight_counters"), *vb = ternary_tr_find(r, "layers.0.bias_counters");
        CHECK(v0 && v0->dtype == TTD_I8 && v0->shape[0] == 6 && v0->shape[1] == 8 && ((const int8_t *)v0->data)[7] == (int8_t)(7 % 11 - 5), "contadores I8 [6,8]");
        CHECK(vb && ((const int8_t *)vb->data)[0] == -2, "contadores del bias");
        CHECK(ternary_tr_find(r, "layers.0.weight") == NULL, "training_state no repite los pesos");
        ternary_tr_close(r);
    }
    /* config.json legible */
    long n = 0; char *cfg = read_whole_file(model_file_path(ex, "config.json"), &n);
    if (cfg) {
        cfg[n] = '\0';
        CHECK(strstr(cfg, "\"format\": \"ternary\"") && strstr(cfg, "\"type\": \"dense\"") && strstr(cfg, "\"input_dim\": 8") &&
              strstr(cfg, "\"checkpoint\": false") && strstr(cfg, "\"name\": \"layers.1\"") && strstr(cfg, "\"activation\": \"uint8\""), "config.json legible");
        free(cfg);
    }
    /* cargar y volver a guardar: carpetas identicas byte a byte (checkpoint y export) */
    TernaryStack *c = ternary_stack_load(ck), *e = ternary_stack_load(ex);
    CHECK(c && e, "cargar checkpoint y export");
    if (c && e) {
        const char *ck2 = tmp_path("test_tfile_ck2"), *ex2 = tmp_path("test_tfile_ex2");
        model_dir_remove(ck2); model_dir_remove(ex2);
        CHECK(ternary_stack_save(c, ck2, 1) && model_dirs_equal(ck, ck2), "checkpoint: guardar -> cargar -> guardar identico");
        CHECK(ternary_stack_save(e, ex2, 0) && model_dirs_equal(ex, ex2), "export: guardar -> cargar -> guardar identico");
        CHECK(ternary_stack_save(c, ex2, 0) && model_dirs_equal(ex, ex2), "un checkpoint cargado exporta el mismo export");
        model_dir_remove(ck2); model_dir_remove(ex2);
    }
    /* un export sobre la carpeta de un checkpoint borra el estado de entrenamiento viejo */
    CHECK(ternary_stack_save(s, ck, 0), "exportar sobre la carpeta del checkpoint");
    CHECK(access(model_file_path(ck, "training_state.ternary"), F_OK) != 0, "el training_state viejo se borro");
    CHECK(ternary_stack_save(s, ck, 1), "volver a guardar el checkpoint");

    /* instantanea en memoria: restaurar devuelve exactamente el estado guardado */
    {
        TernaryStackSnapshot snap = { NULL, 0 };
        CHECK(ternary_stack_snapshot(s, &snap), "instantanea");
        TernaryStack *t = known_net();
        ((TernaryDense *)ternary_stack_layer(t, 0).impl)->counter_W[3] = 99;
        ternary_dense_set_weight((TernaryDense *)ternary_stack_layer(t, 1).impl, 0, 0, WI(2, 1) == 1 ? -1 : 1);   /* cambia un peso */
        CHECK(ternary_stack_restore(t, &snap), "restaurar");
        const char *a = tmp_path("test_tfile_sa"), *b = tmp_path("test_tfile_sb");
        model_dir_remove(a); model_dir_remove(b);
        CHECK(ternary_stack_save(s, a, 1) && ternary_stack_save(t, b, 1) && model_dirs_equal(a, b), "tras restaurar, el estado es identico al de la instantanea");
        TernaryStackSnapshot bad_snap = { (char *)"corto", 5 };
        CHECK(!ternary_stack_restore(t, &bad_snap), "instantanea corrupta rechazada");
        model_dir_remove(a); model_dir_remove(b);
        ternary_stack_snapshot_free(&snap); ternary_stack_free(t);
    }

    /* archivos alterados: cada uno debe rechazarse con un mensaje que diga que paso */
    struct { const char *file, *from, *to, *needle, *what; } cases[] = {
        { "config.json", "\"format_version\": 2", "\"format_version\": 7", "format_version", "version futura de config.json" },
        { "config.json", "\"type\": \"dense\"", "\"type\": \"conv9d\"", "conv9d", "tipo de capa desconocido" },
        { "config.json", "\"activation\": \"uint8\"", "\"activation\": \"sign\"", "sign", "activacion eliminada" },
        { "config.json", "\"input_dim\": 8", "\"input_dim\": 9", "layers.0", "dimension que no coincide con el tensor" },
        { "config.json", "\"output_dim\": 6,", "\"output_dim\": 5,", "layers.", "dimensiones que no encajan" },
        { "config.json", "\"has_bias\": true", "\"has_bias\": false", "", "sin bias el tensor sobra pero la red cambia: carga distinta" },
    };
    for (size_t k = 0; k < sizeof cases / sizeof cases[0]; k++) {
        model_dir_remove(bad); model_dir_copy(ex, bad);
        if (!strcmp(cases[k].what, "sin bias el tensor sobra pero la red cambia: carga distinta")) continue;   /* no es un error: un tensor de mas se ignora */
        replace_in_file(model_file_path(bad, cases[k].file), cases[k].from, cases[k].to);
        char why[300] = "";
        TernaryStack *t = ternary_stack_load_ex(bad, why, sizeof why);
        int ok = t == NULL && why[0] && strstr(why, cases[k].needle);
        if (!ok) fprintf(stderr, "  caso '%s': mensaje='%s'\n", cases[k].what, why);
        CHECK(ok, cases[k].what);
        if (t) ternary_stack_free(t);
    }
    { /* sin config.json, sin model.ternary, carpeta inexistente, un archivo en vez de carpeta */
        char why[300];
        model_dir_remove(bad); model_dir_copy(ex, bad); remove(model_file_path(bad, "config.json"));
        CHECK(ternary_stack_load_ex(bad, why, sizeof why) == NULL && strstr(why, "config.json"), "falta config.json");
        model_dir_remove(bad); model_dir_copy(ex, bad); remove(model_file_path(bad, "model.ternary"));
        CHECK(ternary_stack_load_ex(bad, why, sizeof why) == NULL && strstr(why, "model.ternary"), "falta model.ternary");
        CHECK(ternary_stack_load_ex(tmp_path("test_tfile_no_existe"), why, sizeof why) == NULL && strstr(why, "carpeta"), "carpeta inexistente");
        model_dir_remove(bad);
        write_file_text(bad, "esto es un archivo");
        CHECK(ternary_stack_load_ex(bad, why, sizeof why) == NULL && strstr(why, "carpeta"), "un archivo en vez de una carpeta");
        remove(bad);
        model_dir_remove(bad); model_dir_copy(ex, bad); write_file_text(model_file_path(bad, "config.json"), "{ esto no es json");
        CHECK(ternary_stack_load_ex(bad, why, sizeof why) == NULL && strstr(why, "config.json"), "config.json que no es JSON");
        model_dir_remove(bad); model_dir_copy(ex, bad); write_file_text(model_file_path(bad, "config.json"), "{\"format\":\"ternary\",\"format_version\":2,\"layers\":[]}");
        CHECK(ternary_stack_load_ex(bad, why, sizeof why) == NULL && strstr(why, "layers"), "layers vacio");
    }
    { /* estado de entrenamiento de OTRA red, o con la forma cambiada */
        TernaryDense *d = ternary_dense_alloc(8, 6, 1), *d2 = ternary_dense_alloc(6, 4, 0);   /* l2 con 4 salidas, no 3 */
        TernaryStack *o = ternary_stack_create();
        ternary_stack_add(o, ternary_layer_from_dense(d), TERNARY_ACTIVATION_UINT8);
        ternary_stack_add(o, ternary_layer_from_dense(d2), TERNARY_ACTIVATION_UINT8);
        const char *oc = tmp_path("test_tfile_other");
        model_dir_remove(oc);
        CHECK(ternary_stack_save(o, oc, 1), "guardar la otra red");
        model_dir_remove(bad); model_dir_copy(ck, bad);
        long sn = 0; char *st = read_whole_file(model_file_path(oc, "training_state.ternary"), &sn);
        FILE *f = fopen(model_file_path(bad, "training_state.ternary"), "wb"); fwrite(st, 1, (size_t)sn, f); fclose(f); free(st);
        char why[300];
        CHECK(ternary_stack_load_ex(bad, why, sizeof why) == NULL && strstr(why, "layers.1"), "training_state de otra red");
        model_dir_remove(oc); ternary_stack_free(o);
    }
    { /* un byte 243 en model.ternary (en la zona de pesos) se rechaza al cargar */
        model_dir_remove(bad); model_dir_copy(ex, bad);
        char why[300]; TernaryTReader *rr = ternary_tr_open(model_file_path(bad, "model.ternary"), why, sizeof why);
        const TernaryTInfo *w0 = ternary_tr_find(rr, "layers.0.weight");
        long off = (long)(w0->data - (const uint8_t *)ternary_tr_find(rr, "layers.0.scale")->data);   /* solo para ubicar: abajo se busca por contenido */
        (void)off;
        size_t pos = 0; long len = 0; char *raw = read_whole_file(model_file_path(bad, "model.ternary"), &len);
        uint64_t hn = 0; for (int q = 0; q < 8; q++) hn |= (uint64_t)(uint8_t)raw[q] << (8 * q);
        pos = 8 + hn;   /* el primer tensor empieza al inicio del bloque de datos */
        CHECK(memcmp(raw + pos, w0->data, 10) == 0, "el primer tensor es layers.0.weight");
        raw[pos + 2] = (char)243;
        FILE *f = fopen(model_file_path(bad, "model.ternary"), "wb"); fwrite(raw, 1, (size_t)len, f); fclose(f); free(raw);
        ternary_tr_close(rr);
        CHECK(ternary_stack_load_ex(bad, why, sizeof why) == NULL && strstr(why, "layers.0.weight"), "byte 243 en los pesos");
    }
    model_dir_remove(bad); model_dir_remove(ck); model_dir_remove(ex);
    ternary_stack_free(s); ternary_stack_free(c); ternary_stack_free(e);
}

int main(void) {
    test_container_roundtrip();
    test_container_corrupt();
    test_json();
    test_stack_files();
    if (fails) { fprintf(stderr, "test_tfile: %d FALLO(S)\n", fails); return 1; }
    printf("test_tfile: OK (contenedor model.ternary, JSON, config.json, training_state, corrupciones rechazadas con mensaje)\n");
    return 0;
}
