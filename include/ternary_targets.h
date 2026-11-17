#ifndef TERNARY_TARGETS_H
#define TERNARY_TARGETS_H
#include <stddef.h>
#include <stdint.h>
/*
 * Objetivos de entrenamiento con mascara de perdida a partir de ids ya tokenizados (ver ternary_dataset.h para el formato de
 * texto y ternary_tokenizer.h para los tokens especiales).
 *
 * Un ejemplo conversacional es "<|rol|>\ntexto\n<|rol|>\ntexto\n...<|end|>\n". Cada marcador de rol abre un TRAMO que dura
 * hasta el proximo marcador o <|end|>. Un tramo es "entrenable" (el modelo debe generarlo: p. ej. assistant) o "enmascarado"
 * (es contexto: user, system, tool...). La salida es el objetivo siguiente-token de la API de entrenamiento:
 *
 *      y[i] = ids[i+1]  si el token ids[i+1] se entrena,   y[i] = -1 (ignorado)  si no.    y[n-1] = -1.
 *
 * Que se entrena: los tokens del contenido de un tramo entrenable MAS el marcador/<|end|> que lo cierra (asi el modelo
 * aprende a terminar su turno). El marcador que ABRE el tramo no se entrena (es parte del prompt: en la generacion lo pone quien
 * llama). Los especiales que no son marcadores (<|think|>, <|/think|>...) son tokens normales del tramo en que aparecen.
 * Texto sin ningun marcador (registros 'text', corpus plano) se entrena entero si unmarked_train = 1.
 *
 * Los ids especiales ocupan el rango contiguo [special_lo, special_hi) (256, 257, ... por construccion del tokenizador);
 * kind[id - special_lo] dice que es cada uno.
 */
enum {
    TERNARY_TARGETS_OTHER = 0,   /* especial que no abre ni cierra tramos (<|think|>...): token normal */
    TERNARY_TARGETS_TRAIN = 1,   /* marcador de rol cuyo tramo se entrena */
    TERNARY_TARGETS_MASK  = 2,   /* marcador de rol cuyo tramo es contexto (enmascarado) */
    TERNARY_TARGETS_END   = 3,   /* <|end|>: cierra el tramo y vuelve al estado inicial (sin marcador) */
};

typedef struct {
    int32_t special_lo, special_hi;   /* ids especiales: [lo, hi); vacio si lo >= hi */
    const int8_t *kind;               /* [hi - lo], valores TERNARY_TARGETS_* */
    int unmarked_train;               /* 1: el texto fuera de todo tramo se entrena */
} TernaryTargetsCfg;

/* Un ejemplo de n ids -> y[n]. n = 0 no escribe nada. Devuelve 1 (siempre valido si cfg lo es). */
int ternary_targets_build(const int32_t *ids, size_t n, const TernaryTargetsCfg *cfg, int32_t *y);

/* n_ex ejemplos concatenados: el ejemplo e ocupa [offs[e], offs[e+1]) en ids y en y (offs tiene n_ex+1 elementos, no decrece).
 * Cada ejemplo se procesa por separado (el estado vuelve al inicial); se reparte en el pool de hilos de la libreria
 * (el resultado es el mismo con cualquier numero de hilos: cada ejemplo escribe solo su tramo de y). */
int ternary_targets_build_batch(const int32_t *ids, const size_t *offs, size_t n_ex, const TernaryTargetsCfg *cfg, int32_t *y);

/* Primer indice i >= from, i < n, con ids[i] en [lo, hi); n si no hay. Nucleo vectorial (AVX-512 -> NEON -> escalar); los tres
 * dan siempre el mismo resultado. find_range_scalar es la referencia. */
size_t ternary_targets_find_range(const int32_t *ids, size_t from, size_t n, int32_t lo, int32_t hi);
size_t ternary_targets_find_range_scalar(const int32_t *ids, size_t from, size_t n, int32_t lo, int32_t hi);
#endif
