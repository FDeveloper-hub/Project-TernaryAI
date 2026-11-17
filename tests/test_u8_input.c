/* Entrada uint8 por la fachada: ternaryai_model_train_f32 / predict_f32.
 *  - aprende una tarea lineal con entradas float,
 *  - es determinista (dos modelos iguales -> mismos bits),
 *  - predict_f32 == salida del stack con la muestra cuantizada a mano,
 *  - rechazos: primera capa que no acepta uint8, floats no finitos, y las activaciones eliminadas dan error claro. */
#include "ternaryai.h"
#include "ternary_stack.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fallas = 0;
#define CHECK(c, ...) do { if (!(c)) { fallas++; printf("FALLO (linea %d): ", __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)
static char err[256];
static const double HYPER[TERNARYAI_NUM_HYPER] = { 0.02 };

#define N 64
#define F 12

static TernaryAIModel *mk(unsigned seed) {
    size_t in[1] = { F }, sh[4]; int nd;
    double d1[2] = { 16, 1 }, d2[2] = { 1, 1 };
    TernaryAIModel *m = ternaryai_model_create(in, 1, seed, err, sizeof(err));
    if (!m || !ternaryai_model_add_layer(m, "dense", d1, 2, "uint8", sh, &nd, err, sizeof(err)) ||
        !ternaryai_model_add_layer(m, "dense", d2, 2, "uint8", sh, &nd, err, sizeof(err))) { printf("no arma: %s\n", err); exit(2); }
    return m;
}

int main(void) {
    static float x[N * F], y[N], a[N], b[N];
    unsigned st = 12345;
    for (int s = 0; s < N; s++) {
        double sum = 0;
        for (int i = 0; i < F; i++) {
            st = st * 1664525u + 1013904223u;
            x[s * F + i] = ((int)((st >> 8) % 2001) - 1000) / 500.0f;
            if (i < 4) sum += x[s * F + i];
        }
        y[s] = sum > 0 ? 1.0f : -1.0f;
    }
    double l1[300], l2[300];
    TernaryAIModel *m1 = mk(7), *m2 = mk(7);
    CHECK(ternaryai_model_train_f32(m1, x, y, N, 0, 300, HYPER, l1, err, sizeof(err)), "train m1: %s", err);
    CHECK(ternaryai_model_train_f32(m2, x, y, N, 0, 300, HYPER, l2, err, sizeof(err)), "train m2: %s", err);
    CHECK(memcmp(l1, l2, sizeof(l1)) == 0, "no determinista");
    CHECK(l1[299] < l1[0], "la perdida no baja: %.4f -> %.4f", l1[0], l1[299]);
    CHECK(ternaryai_model_predict_f32(m1, x, N, a, err, sizeof(err)), "predict a: %s", err);
    CHECK(ternaryai_model_predict_f32(m2, x, N, b, err, sizeof(err)), "predict b: %s", err);
    CHECK(memcmp(a, b, sizeof(a)) == 0, "predict no determinista");
    int ok = 0; for (int s = 0; s < N; s++) ok += (a[s] > 0) == (y[s] > 0);
    printf("  entrada uint8: perdida %.4f -> %.4f, acierto %d/%d\n", l1[0], l1[299], ok, N);
    CHECK(ok > N * 6 / 10, "acierto bajo");

    /* floats no finitos */
    float bad[F]; memset(bad, 0, sizeof(bad)); bad[3] = NAN;
    float o1[1];
    CHECK(!ternaryai_model_predict_f32(m1, bad, 1, o1, err, sizeof(err)) && strstr(err, "finite"), "NaN deberia rechazarse");

    /* primera capa que no acepta uint8 (Conv2D) */
    {
        size_t in[3] = { 4, 4, 1 }, sh[4]; int nd; double c[5] = { 2, 2, 1, 1, 1 }, d[2] = { 1, 1 };
        (void)c;
        TernaryAIModel *m = ternaryai_model_create(in, 3, 1, err, sizeof(err));
        double conv[5] = { 2, 1, 2, 2, 1 };
        if (m && ternaryai_model_add_layer(m, "conv2d", conv, 5, "uint8", sh, &nd, err, sizeof(err)) &&
            ternaryai_model_add_layer(m, "dense", d, 2, "uint8", sh, &nd, err, sizeof(err))) {
            float xi[16] = { 0 }, oo[1];
            CHECK(!ternaryai_model_predict_f32(m, xi, 1, oo, err, sizeof(err)) && strstr(err, "uint8"), "conv2d primero: %s", err);
        }
        ternaryai_model_free(m);
    }
    /* activaciones eliminadas */
    {
        size_t in[1] = { F }, sh[4]; int nd; double d1[2] = { 4, 1 };
        TernaryAIModel *m = ternaryai_model_create(in, 1, 1, err, sizeof(err));
        CHECK(!ternaryai_model_add_layer(m, "dense", d1, 2, "sign", sh, &nd, err, sizeof(err)) && strstr(err, "removed"), "sign: %s", err);
        CHECK(!ternaryai_model_add_layer(m, "dense", d1, 2, "band", sh, &nd, err, sizeof(err)) && strstr(err, "removed"), "band: %s", err);
        CHECK(!ternaryai_model_add_layer(m, "dense", d1, 2, "relu", sh, &nd, err, sizeof(err)) && strstr(err, "unknown"), "relu: %s", err);
        ternaryai_model_free(m);
    }
    ternaryai_model_free(m1); ternaryai_model_free(m2);
    if (fallas) { printf("%d fallas\n", fallas); return 1; }
    printf("OK\n");
    return 0;
}
