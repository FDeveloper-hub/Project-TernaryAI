#ifndef TERNARY_TOKENIZER_H
#define TERNARY_TOKENIZER_H
#include <stddef.h>
#include <stdint.h>
/*
 * Tokenizador BPE a nivel de bytes (mismo principio que GPT-2/GPT-4):
 * el vocabulario base son los 256 valores de byte posibles (siempre
 * presentes, id == valor del byte) mas "fusiones" (merges) aprendidas
 * de un corpus, cada una con su propio id nuevo (256, 257, ...) hasta
 * llegar al tamano de vocabulario pedido. Por eso alcanza con pedir un
 * tamano para construirlo: no hace falta vocabulario cerrado ni
 * normalizacion, y cualquier texto UTF-8 se puede codificar siempre
 * (en el peor caso, byte a byte, sin OOV).
 *
 * v1: las fusiones se aprenden sobre el corpus COMPLETO como una sola
 * secuencia de bytes, sin pre-tokenizar por palabra/regex antes (a
 * diferencia de GPT-2) -- mas simple de construir y razonar, a costa
 * de que una fusion podria llegar a cruzar un limite de palabra. El
 * formato de archivo (lista de pares) no cambia si mas adelante se
 * agrega un paso de pre-tokenizacion.
 *
 * Entrenamiento: O(vocab_size * largo_del_corpus) -- cada fusion
 * recorre el corpus completo para contar pares y volver a fusionar.
 * Simple y correcto; si el corpus de entrenamiento real resulta
 * demasiado grande para el dispositivo, se puede optimizar despues
 * (conteo incremental) sin cambiar esta API.
 */

typedef struct TernaryTokenizer TernaryTokenizer;

/* vocab_size pedido; el logrado puede ser menor si el corpus se queda
 * sin pares repetidos antes de llegar ahi (ver ternary_tokenizer_vocab_size).
 * Debe ser >= 256 (los 256 bytes crudos siempre estan). */
TernaryTokenizer *ternary_tokenizer_create(size_t vocab_size, char *err, size_t err_cap);
void ternary_tokenizer_free(TernaryTokenizer *t);

/* Aprende las fusiones a partir de 'corpus' (corpus_len bytes). Se
 * llama una sola vez sobre un tokenizador recien creado (entrenar dos
 * veces es un error: no tiene sentido re-entrenar sobre fusiones ya
 * aprendidas). Detenerse antes de vocab_size por falta de pares
 * repetidos NO es un error. */
int ternary_tokenizer_train(TernaryTokenizer *t, const uint8_t *corpus, size_t corpus_len,
                             char *err, size_t err_cap);

size_t ternary_tokenizer_vocab_size(const TernaryTokenizer *t);  /* 256 + especiales + fusiones logradas */
size_t ternary_tokenizer_num_merges(const TernaryTokenizer *t);

/* Mismo patron que snprintf para medir/escribir: llamar primero con
 * out_ids=NULL/out_cap=0 para saber cuantos ids hacen falta en
 * *out_len, despues con un buffer de ese tamano. Devuelve 1 en exito
 * (incluso si out_cap < *out_len: eso no es un error, es "medir"). */
int ternary_tokenizer_encode(const TernaryTokenizer *t, const uint8_t *text, size_t text_len,
                              int32_t *out_ids, size_t out_cap, size_t *out_len,
                              char *err, size_t err_cap);

/* Mismo patron de medicion que encode, pero de ids a bytes. Ids fuera
 * de rango [0, vocab_size) se ignoran (no escriben nada). */
int ternary_tokenizer_decode(const TernaryTokenizer *t, const int32_t *ids, size_t num_ids,
                              uint8_t *out_bytes, size_t out_cap, size_t *out_len,
                              char *err, size_t err_cap);

/* ---- Tokens especiales ----------------------------------------------
 * Cadenas de bytes (p. ej. "<|user|>") reservadas con ids FIJOS: el
 * primero agregado es el id 256, el siguiente 257, etc.; las fusiones
 * aprendidas empiezan despues. El BPE nunca las parte ni las fusiona
 * con el texto de al lado, asi la frontera entre una marca y su
 * contenido es exacta (util para enmascarar "solo la respuesta").
 *
 * - Se agregan ANTES de entrenar (despues cambiarian los ids de las
 *   fusiones). vocab_size cuenta los 256 bytes + los especiales + las
 *   fusiones, asi que debe ser >= 256 + num_special.
 * - En el entrenamiento cada aparicion de un especial en el corpus es una
 *   frontera: ninguna fusion la cruza y no cuenta como texto.
 * - encode reconoce los especiales en el texto (el mas largo si varios
 *   coinciden); encode_ex con parse_special=0 los trata como texto
 *   corriente (sirve para texto de usuario no confiable).
 * - decode los escribe de vuelta como sus bytes; decode_ex con
 *   skip_special=1 los omite.
 * Sin especiales todo se comporta (y se guarda) exactamente como antes. */
int ternary_tokenizer_add_special(TernaryTokenizer *t, const uint8_t *text, size_t len, char *err, size_t err_cap);
size_t ternary_tokenizer_num_special(const TernaryTokenizer *t);
/* Texto del especial 'index' (mismo patron de medir/escribir). */
int ternary_tokenizer_special_text(const TernaryTokenizer *t, size_t index, uint8_t *out, size_t out_cap,
                                    size_t *out_len, char *err, size_t err_cap);
/* Id del especial con exactamente ese texto, o -1 si no es un especial. */
int32_t ternary_tokenizer_special_id(const TernaryTokenizer *t, const uint8_t *text, size_t len);
int ternary_tokenizer_encode_ex(const TernaryTokenizer *t, const uint8_t *text, size_t text_len, int parse_special,
                                 int32_t *out_ids, size_t out_cap, size_t *out_len,
                                 char *err, size_t err_cap);
int ternary_tokenizer_decode_ex(const TernaryTokenizer *t, const int32_t *ids, size_t num_ids, int skip_special,
                                 uint8_t *out_bytes, size_t out_cap, size_t *out_len,
                                 char *err, size_t err_cap);

/* tokenizer.json: texto JSON legible (vocab_size, tokens especiales, fusiones). Ver ternary_tokenizer.c. */
int ternary_tokenizer_save(const TernaryTokenizer *t, const char *path, char *err, size_t err_cap);
TernaryTokenizer *ternary_tokenizer_load(const char *path, char *err, size_t err_cap);
TernaryTokenizer *ternary_tokenizer_load_mem(const char *text, size_t len, const char *label, char *err, size_t err_cap);   /* desde memoria (no toma posesion de text) */

#endif /* TERNARY_TOKENIZER_H */
