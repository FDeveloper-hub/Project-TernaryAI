#ifndef TERNARY_JSON_H
#define TERNARY_JSON_H
#include <stddef.h>
/*
 * Parser JSON minimalista, de proposito especifico: no arma un DOM ni
 * devuelve valores genericos, solo dos operaciones sobre un cursor de
 * texto -- extraer un string (con sus escapes decodificados a UTF-8) y
 * saltar cualquier valor completo (objeto/arreglo/string/numero/
 * true/false/null) sin extraerlo. Quien lo usa (ternary_dataset.c)
 * conoce el esquema exacto que espera y camina objetos/arreglos "a
 * mano" combinando estas dos operaciones -- no hace falta un parser
 * JSON de proposito general para un esquema fijo conocido.
 */

typedef struct {
    const char *s;
    size_t len;
    size_t pos;
} TernaryJsonParser;

void ternary_json_init(TernaryJsonParser *p, const char *s, size_t len);

/* Avanza el cursor sobre espacios/tabs/saltos de linea. */
void ternary_json_skip_ws(TernaryJsonParser *p);

/* Mira el proximo caracter sin consumirlo. Devuelve 0 si ya no hay mas
 * (fin del buffer) -- en ese caso *out queda sin definir. */
int ternary_json_peek(const TernaryJsonParser *p, char *out);

/* Si el proximo caracter es exactamente 'c', lo consume y devuelve 1;
 * si no, no mueve el cursor y devuelve 0 (sin error -- lo pone el
 * llamador, que sabe si ahi era obligatorio o no). */
int ternary_json_match(TernaryJsonParser *p, char c);

/* Igual que match, pero es un error si no aparece (para sintaxis
 * obligatoria como el ':' de "clave":valor). */
int ternary_json_expect(TernaryJsonParser *p, char c, char *err, size_t err_cap);

/* Parsea un string JSON (con comillas) empezando en el cursor,
 * decodifica sus escapes (\", \\, \/, \b, \f, \n, \r, \t, \uXXXX
 * incluidos pares subrogados) a UTF-8 en un buffer propio (malloc,
 * el llamador hace free). *out queda SIN el byte nulo final; usar
 * *out_len para su longitud real (puede contener bytes 0 si el JSON
 * los tenia escapados como \u0000, aunque no es lo esperable en texto
 * de entrenamiento real). */
int ternary_json_parse_string(TernaryJsonParser *p, char **out, size_t *out_len,
                               char *err, size_t err_cap);

/* Salta un valor JSON completo de cualquier tipo (para ignorar campos
 * del esquema que no interesan) sin extraer nada. */
int ternary_json_skip_value(TernaryJsonParser *p, char *err, size_t err_cap);

/* ======================================================================
 * DOM de solo lectura + constructor de texto (config.json, cabecera de
 * model.ternary, tokenizer.json). Mismo parser de strings de arriba; los
 * numeros enteros se guardan exactos (int64) y todo numero tambien como
 * double. Profundidad maxima 64 (un archivo hostil no desborda la pila).
 * ====================================================================== */
#include <stdint.h>

typedef enum { TJ_NULL = 0, TJ_BOOL, TJ_NUM, TJ_STR, TJ_ARR, TJ_OBJ } TernaryJType;

typedef struct TernaryJNode TernaryJNode;
struct TernaryJNode {
    TernaryJType type;
    int          b;                 /* TJ_BOOL */
    double       num;               /* TJ_NUM */
    int64_t      i;                 /* TJ_NUM, valido si is_int */
    int          is_int;            /* el literal no tenia '.', 'e' ni 'E' y cabe en int64 */
    char        *str;               /* TJ_STR (con byte nulo final; str_len = largo real) */
    size_t       str_len;
    size_t       n;                 /* hijos de TJ_ARR / TJ_OBJ */
    char       **keys;              /* TJ_OBJ: n claves */
    TernaryJNode *kids;             /* TJ_ARR / TJ_OBJ: n hijos contiguos */
};

/* Parsea todo el texto (basura despues del valor es un error). NULL si falla (mensaje en err). */
TernaryJNode *ternary_jdom_parse(const char *s, size_t len, char *err, size_t err_cap);
void ternary_jdom_free(TernaryJNode *root);

/* Valor de la clave en un objeto (la ultima si esta repetida), o NULL. */
const TernaryJNode *ternary_jdom_get(const TernaryJNode *obj, const char *key);
/* Getters tipados: devuelven 1 y escriben *out si la clave existe con ese tipo; si no, 0 y *out no cambia. */
int ternary_jdom_get_int(const TernaryJNode *obj, const char *key, int64_t *out);
int ternary_jdom_get_float(const TernaryJNode *obj, const char *key, float *out);
int ternary_jdom_get_bool(const TernaryJNode *obj, const char *key, int *out);
const char *ternary_jdom_get_str(const TernaryJNode *obj, const char *key);   /* NULL si falta o no es string */
const TernaryJNode *ternary_jdom_at(const TernaryJNode *arr, size_t i);        /* NULL fuera de rango */

/* Constructor de texto JSON. Las funciones insertan la coma solas cuando el ultimo caracter no es '{', '[', ',' ni ':'. */
typedef struct { char *buf; size_t len, cap; int failed; } TernaryJBuf;
void ternary_jbuf_init(TernaryJBuf *b);
void ternary_jbuf_free(TernaryJBuf *b);
const char *ternary_jbuf_cstr(TernaryJBuf *b);                        /* el texto como string C (agrega el NUL sin contarlo) */
void ternary_jbuf_raw(TernaryJBuf *b, const char *s);                 /* texto tal cual (sin coma automatica) */
void ternary_jbuf_open(TernaryJBuf *b, char c);                       /* coma si hace falta + '{' o '[' */
void ternary_jbuf_close(TernaryJBuf *b, char c);                      /* '}' o ']' */
void ternary_jbuf_key(TernaryJBuf *b, const char *key);               /* coma + "key": */
void ternary_jbuf_str(TernaryJBuf *b, const char *s);                 /* coma + string escapado (UTF-8 tal cual) */
void ternary_jbuf_int(TernaryJBuf *b, int64_t v);                     /* coma + entero */
void ternary_jbuf_float(TernaryJBuf *b, float v);                     /* coma + la forma decimal mas corta que vuelve exacta al mismo float */
void ternary_jbuf_bool(TernaryJBuf *b, int v);
void ternary_jbuf_kint(TernaryJBuf *b, const char *key, int64_t v);   /* "key":v */
void ternary_jbuf_kfloat(TernaryJBuf *b, const char *key, float v);
void ternary_jbuf_kbool(TernaryJBuf *b, const char *key, int v);
void ternary_jbuf_kstr(TernaryJBuf *b, const char *key, const char *v);
/* Reescribe JSON compacto con sangrado de 2 espacios: los objetos se expanden; los arreglos que solo
 * contienen escalares quedan en una linea. Devuelve un buffer malloc (el llamador hace free) o NULL. */
char *ternary_json_pretty(const char *s, size_t len, size_t *out_len);

#endif /* TERNARY_JSON_H */
