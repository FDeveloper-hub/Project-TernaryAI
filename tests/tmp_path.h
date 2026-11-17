#ifndef TERNARY_TESTS_TMP_PATH_H
#define TERNARY_TESTS_TMP_PATH_H
/* Carpeta temporal portable para los tests: $TMPDIR si existe (hay plataformas sin /tmp), si no /tmp.
 * tmp_path() devuelve un buffer propio rotativo (32 ranuras), valido mientras no se hagan 32 llamadas mas. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
static const char *tmp_dir(void) {
    const char *d = getenv("TMPDIR");
    return (d && d[0]) ? d : "/tmp";
}
static inline const char *tmp_path(const char *name) {
    static char buf[32][1024];
    static int k = 0;
    char *b = buf[k++ & 31];
    snprintf(b, sizeof buf[0], "%s/%s", tmp_dir(), name);
    return b;
}

/* ---- carpetas de modelo (config.json + model.ternary + training_state.ternary opcional) ---- */
static const char *const MODEL_DIR_FILES[3] = { "config.json", "model.ternary", "training_state.ternary" };

static inline char *model_file_path(const char *dir, const char *name) {
    static char buf[16][1280];
    static int k = 0;
    char *b = buf[k++ & 15];
    snprintf(b, sizeof buf[0], "%s/%s", dir, name);
    return b;
}
/* Suma de los bytes de los archivos del modelo; -1 si falta config.json o model.ternary. */
static inline long model_dir_bytes(const char *dir) {
    long total = 0;
    for (int i = 0; i < 3; i++) {
        struct stat st;
        if (stat(model_file_path(dir, MODEL_DIR_FILES[i]), &st) != 0) { if (i < 2) return -1; continue; }
        total += (long)st.st_size;
    }
    return total;
}
static inline char *read_whole_file(const char *path, long *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); *len = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)*len + 1);
    if (!buf || fread(buf, 1, (size_t)*len, f) != (size_t)*len) { if (buf) free(buf); fclose(f); return NULL; }
    fclose(f);
    return buf;
}
/* Dos carpetas de modelo identicas byte a byte (mismos archivos presentes y mismo contenido). */
static inline int model_dirs_equal(const char *a, const char *b) {
    for (int i = 0; i < 3; i++) {
        long la = 0, lb = 0;
        char *x = read_whole_file(model_file_path(a, MODEL_DIR_FILES[i]), &la), *y = read_whole_file(model_file_path(b, MODEL_DIR_FILES[i]), &lb);
        int eq = (!x && !y) || (x && y && la == lb && memcmp(x, y, (size_t)la) == 0);
        free(x); free(y);
        if (!eq) return 0;
    }
    return model_dir_bytes(a) > 0;
}
/* Borra los archivos del modelo y la carpeta (si 'path' es un archivo suelto, lo borra). */
static inline void model_dir_remove(const char *dir) {
    for (int i = 0; i < 3; i++) {
        char tmp[1300];
        remove(model_file_path(dir, MODEL_DIR_FILES[i]));
        snprintf(tmp, sizeof tmp, "%s.tmp", model_file_path(dir, MODEL_DIR_FILES[i]));
        remove(tmp);
    }
    if (rmdir(dir) != 0) remove(dir);
}
/* Copia una carpeta de modelo (los archivos que existan) a 'dst' (se crea). */
static inline void model_dir_copy(const char *src, const char *dst) {
    mkdir(dst, 0777);
    for (int i = 0; i < 3; i++) {
        long n = 0;
        char *buf = read_whole_file(model_file_path(src, MODEL_DIR_FILES[i]), &n);
        if (!buf) continue;
        FILE *f = fopen(model_file_path(dst, MODEL_DIR_FILES[i]), "wb");
        if (f) { fwrite(buf, 1, (size_t)n, f); fclose(f); }
        free(buf);
    }
}
#endif
