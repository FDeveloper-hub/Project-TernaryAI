#ifndef TERNARY_LOSS_H
#define TERNARY_LOSS_H
#include <stddef.h>
#include <stdint.h>
/*
 * Perdida de entropia cruzada con softmax sobre logits, para predecir el siguiente token.
 *
 *   L = (1/n) * sum_{filas validas} ( log(sum_v exp(l[v])) - l[target] )       n = filas validas
 *   dL/dl[v] = (softmax(l)[v] - [v == target]) / n
 *
 * logits: rows*vocab floats (la salida de la ultima capa, p.ej. ternary_head). targets: rows ids; un id
 * NEGATIVO marca una fila IGNORADA (relleno / posiciones sin objetivo): no suma a la perdida, su gradiente
 * es 0 y no cuenta en n. Un id >= vocab es un error del llamador: devuelve NAN y deja grad_out en cero
 * (nunca se ignora en silencio). Sin filas validas: perdida 0 y gradiente 0.
 *
 * Estable numericamente: se resta el maximo de la fila antes de exponenciar (logits de +-1e4 no dan inf/NaN)
 * y la suma va en double. Es el UNICO lugar con exp/log en el camino del modelo de lenguaje, y vive fuera de
 * las capas (se llama una vez por muestra, sobre los logits finales).
 *
 * grad_out (opcional, NULL si solo se quiere la perdida): rows*vocab, se SOBREESCRIBE. n_valid_out y
 * n_correct_out (opcionales): filas validas y cuantas tienen argmax == target (empate: el id mas bajo).
 * Devuelve la perdida media.
 */
double ternary_xent(const float *logits, size_t rows, size_t vocab, const int32_t *targets,
                    float *grad_out, size_t *n_valid_out, size_t *n_correct_out);

/* Una fila: perdida (sin dividir por n) y, si g != NULL, su gradiente (ya dividido por n, n = filas validas de la
 * muestra). *correct_out = 1 si argmax == t. Es el cuerpo de ternary_xent, expuesto para la ruta por trozos. */
double ternary_xent_row(const float *logits_row, size_t vocab, size_t t, size_t n, float *g, int *correct_out);

#endif /* TERNARY_LOSS_H */
