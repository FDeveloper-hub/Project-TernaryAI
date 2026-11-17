#ifndef TERNARY_TFILE_H
#define TERNARY_TFILE_H

#include "ternary_core.h"
#include <stddef.h>
#include <stdint.h>

/*
 * Contenedor de tensores "model.ternary" (mismo estilo que model.safetensors).
 *
 *   bytes 0..7        N: largo de la cabecera JSON (uint64 little-endian)
 *   bytes 8..8+N-1    cabecera JSON compacta, rellena con espacios al final para que 8+N sea multiplo de 64
 *   bytes 8+N..       bloque binario de tensores
 *
 * La cabecera es un objeto JSON:
 *   { "__metadata__": {"format":"ternary","version":"1", ...strings...},
 *     "<nombre del tensor>": {"dtype":"TRIT5","shape":[512,128],"data_offsets":[ini,fin]}, ... }
 * data_offsets son bytes relativos al inicio del bloque binario; fin-ini es EXACTAMENTE el tamano del tensor
 * (el relleno de alineacion queda fuera). Los tensores van en el orden en que se agregaron (orden de ejecucion de la
 * red) y cada uno empieza en un multiplo de 64 bytes (linea de cache / vector AVX-512): con mmap, un motor de
 * inferencia puede leer los pesos empaquetados directamente de la pagina del archivo, sin copiarlos.
 *
 * dtypes: F32, I32 (little-endian), I8, U8 y TRIT5 (pesos ternarios, 5 por byte en base 3: byte = t0 + 3*t1 +
 * 9*t2 + 27*t3 + 81*t4, ti = valor+1; el ultimo byte se completa con ceros; ver ternary_model_io.h).
 * Todo lo que no es un tensor (arquitectura, hiperparametros, tokenizador) va en JSON aparte: config.json y
 * tokenizer.json, igual que en HuggingFace.
 */

#define TERNARY_TFILE_ALIGN 64
#define TERNARY_TFILE_FORMAT "ternary"
#define TERNARY_TFILE_VERSION 1
/* config.json: version 2 = contadores R2 (tensores "_counters"), sin vote_limit; el checkpoint de entrenamiento lleva "r2". */
#define TERNARY_CONFIG_VERSION 2
#define TERNARY_TFILE_MAX_DIMS 8

typedef enum { TTD_F32 = 0, TTD_I32, TTD_I8, TTD_U8, TTD_TRIT5 } TernaryTDType;

const char *ternary_tdtype_name(TernaryTDType t);
/* Bytes de n elementos de ese dtype (TRIT5: (n+4)/5). */
size_t ternary_tdtype_bytes(TernaryTDType t, size_t n);

/* ---- escritura ---------------------------------------------------------------------------------------- */
typedef struct TernaryTWriter TernaryTWriter;

TernaryTWriter *ternary_tw_create(void);
void ternary_tw_free(TernaryTWriter *w);
/* Pares clave/valor de texto para __metadata__ (format y version se escriben solos). 1 en exito. */
int ternary_tw_meta(TernaryTWriter *w, const char *key, const char *value);
/* Agrega un tensor. 'data' NO se copia: debe seguir valido hasta serializar/escribir. Nombre unico. 1 en exito. */
int ternary_tw_add(TernaryTWriter *w, const char *name, TernaryTDType dt, const size_t *shape, size_t nd, const void *data);
/* Tensor TRIT5 desde un TernaryVec (v->n == producto de shape): empaqueta y guarda su propio buffer. */
int ternary_tw_add_trits(TernaryTWriter *w, const char *name, const size_t *shape, size_t nd, const TernaryVec *v);
/* Atajos: vector de n floats / int8 (forma {n}). */
int ternary_tw_add_f32(TernaryTWriter *w, const char *name, const float *data, size_t n);
int ternary_tw_add_i8(TernaryTWriter *w, const char *name, const int8_t *data, size_t n);
/* Escribe el archivo completo en 'path' (primero path.tmp y luego rename: nunca queda un archivo a medias). 1 en exito. */
int ternary_tw_write_file(const TernaryTWriter *w, const char *path);
/* Serializa en memoria (malloc; el llamador hace free). 1 en exito. */
int ternary_tw_serialize(const TernaryTWriter *w, uint8_t **out, size_t *out_len);

/* ---- lectura ------------------------------------------------------------------------------------------ */
typedef struct TernaryTReader TernaryTReader;

typedef struct {
    const char   *name;
    TernaryTDType dtype;
    size_t        ndims;
    size_t        shape[TERNARY_TFILE_MAX_DIMS];
    size_t        nelem;
    size_t        nbytes;
    const uint8_t *data;       /* apunta dentro del archivo (mmap) o del buffer: valido hasta cerrar el lector */
} TernaryTInfo;

/* Abre y valida TODA la cabecera: JSON bien formado, dtypes conocidos, tamanos coherentes con dtype y forma,
 * offsets alineados, dentro del archivo y sin solaparse. NULL + mensaje en err si algo no cuadra. */
TernaryTReader *ternary_tr_open(const char *path, char *err, size_t err_cap);
/* Igual sobre un buffer en memoria (no se copia: debe seguir valido hasta cerrar el lector). */
TernaryTReader *ternary_tr_from_memory(const void *data, size_t len, char *err, size_t err_cap);
void ternary_tr_close(TernaryTReader *r);

size_t ternary_tr_count(const TernaryTReader *r);
const TernaryTInfo *ternary_tr_at(const TernaryTReader *r, size_t i);       /* orden alfabetico */
const TernaryTInfo *ternary_tr_find(const TernaryTReader *r, const char *name);
const char *ternary_tr_meta(const TernaryTReader *r, const char *key);      /* NULL si no esta */
/* Busca 'name' y exige dtype y forma exactos; si no coincide devuelve NULL con el motivo en err. */
const TernaryTInfo *ternary_tr_expect(const TernaryTReader *r, const char *name, TernaryTDType dt,
                                      const size_t *shape, size_t nd, char *err, size_t err_cap);

/* Lee un tensor TRIT5 completo en 'out' (out->n == nelem). 0 si hay un byte fuera de 0..242 (archivo corrupto). */
int ternary_tinfo_read_trits(const TernaryTInfo *t, TernaryVec *out);
/* i-esimo valor de un tensor F32 / I8 (seguro aunque 'data' no este alineado). */
float ternary_tinfo_f32(const TernaryTInfo *t, size_t i);

#endif /* TERNARY_TFILE_H */
