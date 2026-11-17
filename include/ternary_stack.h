#ifndef TERNARY_STACK_H
#define TERNARY_STACK_H

#include "ternary_layer.h"
#include "ternary_activation.h"

/*
 * Contenedor que encadena capas (Dense, Conv2D, o lo que se agregue
 * despues -- ver ternary_layer.h) con una activacion entre cada una
 * (ver ternary_activation.h), y sabe guardar/cargar la red entera a
 * disco en dos modos (ver ternary_stack_save).
 *
 * Nace de tres decisiones explicitas: (1) agregar un tipo de capa
 * nuevo debe ser facil -- por eso el contenedor solo conoce la
 * interfaz generica de ternary_layer.h, nunca Dense/Conv2D en
 * concreto; (2) el guardado es SELECTIVO -- checkpoint completo (se
 * puede retomar el entrenamiento) o export final empaquetado al minimo
 * de bits (solo inferencia); (3) la activacion entre capas la decide
 * el contenedor, no el codigo de entrenamiento a mano, para poder
 * agregar mas activaciones despues sin tocar cada test de entrenamiento
 * uno por uno.
 */
typedef struct TernaryStack TernaryStack;

TernaryStack *ternary_stack_create(void);
/* Libera TODAS las capas agregadas (via vt->destroy) y el contenedor
 * mismo -- nunca liberar una capa a mano despues de agregarla. */
void ternary_stack_free(TernaryStack *s);

/* Agrega una capa YA CREADA (ver ternary_layer_from_dense/_from_conv2d)
 * con la activacion que se le aplica a SU salida antes de pasarla a la
 * siguiente capa (o de dejarla como salida final de la red).
 *
 * Devuelve 0 (y NO agrega nada, el llamador sigue siendo dueno de
 * 'layer') si:
 *   - layer.vt->propagates_grad_x(layer.impl)==0 y esta NO es la
 *     primera capa que se agrega (ver comentario grande en
 *     ternary_layer.h: hoy es el caso de Conv2D) -- entrenar asi
 *     dejaria las capas anteriores sin gradiente nunca, en silencio.
 *   - el input_len() de esta capa no coincide con el output_len() de
 *     la capa anterior ya agregada (arquitectura invalida).
 * 1 en exito. */
int ternary_stack_add(TernaryStack *s, TernaryLayer layer, TernaryActivation act);

/* Forma logica de la entrada de la red (p.ej. {28,28,1} para una imagen,
 * {2} para XOR) -- metadato que la red no necesita para calcular (solo
 * usa input_len()) pero que ternary_stack_save() guarda para que un
 * modelo cargado recuerde con que forma se lo armo. ndims en 1..4; una
 * forma cuyo producto no coincida con la de la PRIMERA capa se rechaza
 * (devuelve 0). Si nunca se fija, ternary_stack_input_shape() devuelve
 * la forma plana {input_len}. */
int ternary_stack_set_input_shape(TernaryStack *s, const size_t *dims, size_t ndims);
/* Copia la forma en 'dims' (capacidad TERNARY_MAX_SHAPE_DIMS) y devuelve
 * ndims (0 si la red no tiene capas). */
size_t ternary_stack_input_shape(const TernaryStack *s, size_t *dims);

/* Acceso de solo lectura a la capa i / su activacion (para describir la
 * arquitectura sin conocer los tipos concretos). Capa con impl==NULL o
 * activacion NULL si i esta fuera de rango. */
TernaryLayer ternary_stack_layer(const TernaryStack *s, size_t i);
const TernaryActivation *ternary_stack_activation(const TernaryStack *s, size_t i);

size_t ternary_stack_num_layers(const TernaryStack *s);
size_t ternary_stack_input_len(const TernaryStack *s);  /* de la PRIMERA capa */
size_t ternary_stack_output_len(const TernaryStack *s); /* de la ULTIMA capa */

/* Forward de la red completa. out_final: ya reservado por el llamador,
 * longitud ternary_stack_output_len(s) -- es la salida ESCALADA de la
 * ultima capa (antes de su activacion, igual convencion que
 * ternary_dense_forward/ternary_conv2d_forward de toda la vida: la
 * decision ternaria final, si hace falta, la toma el llamador aplicando
 * sign() el mismo, o via ternary_stack_last_activated() de abajo).
 * Cachea internamente cada salida intermedia para que
 * ternary_stack_backward() no tenga que repetir este forward. */
void ternary_stack_forward(TernaryStack *s, const TernaryTensor *x, float *out_final,
                            const TernaryCapabilities *caps);

/* Version ya cuantizada (por la activacion de la ULTIMA capa) de la
 * ultima salida calculada por ternary_stack_forward() -- valida hasta
 * el proximo forward. Util si se le puso una activacion a la ultima
 * capa a proposito (p.ej. para una decision de clasificacion ternaria). */
const TernaryVec *ternary_stack_last_activated(const TernaryStack *s);

/* Debe llamarse UNA vez al empezar cada epoca/batch, antes de la
 * primera ternary_stack_backward() de ese batch -- pone en cero los
 * acumuladores de gradiente de TODAS las capas. */
void ternary_stack_zero_gradients(TernaryStack *s);

/* Backward de UNA muestra, para TODAS las capas -- requiere haber
 * llamado ternary_stack_forward() con el MISMO 'x' inmediatamente antes
 * (usa las salidas intermedias que quedaron cacheadas). ACUMULA en los
 * buffers de gradiente internos (no aplica nada todavia) -- llamar
 * ternary_stack_apply_gradients() al final del batch/epoca.
 * grad_final: dL/d(salida escalada de la ULTIMA capa), longitud
 * ternary_stack_output_len(s) -- el llamador calcula la perdida y este
 * gradiente el mismo, igual que siempre en este proyecto. */
void ternary_stack_backward(TernaryStack *s, const TernaryTensor *x, const float *grad_final);

/* Entrenamiento de UNA muestra con entropia cruzada SIN materializar los logits rows*vocab: solo si la ultima
 * capa es la cabeza 'lm_head' (ternary_stack_xent_fusable; TERNARYAI_NO_FUSED_XENT=1 lo apaga). Equivale
 * bit a bit a forward + ternary_xent + (gradiente *= inv_n) + backward. Devuelve la perdida media de la muestra,
 * o NAN si un target >= vocab (nada cambia). */
int ternary_stack_xent_fusable(const TernaryStack *s);
double ternary_stack_train_xent(TernaryStack *s, const TernaryTensor *x, const TernaryCapabilities *caps,
                                const int32_t *targets, float inv_n, size_t *n_valid, size_t *n_correct);

/* ENTRENAMIENTO POR BLOQUES de muestras (solo si todas las capas lo admiten:
 * hoy, pilas de Dense). Hace forward + perdida + backward de las n muestras
 * procesandolas de S en S: cada fila de W se expande una vez por bloque y
 * cada fila de gradiente se toca una vez por bloque en vez de una por
 * muestra. Mismos bits que el camino muestra a muestra (solo cambia el
 * orden de los bucles). Acumula en los gradientes internos: llamar antes
 * ternary_stack_zero_gradients() y despues ternary_stack_apply_gradients().
 * loss_fn(sample, out, grad, ctx) se llama EN ORDEN de muestra con la salida
 * de la ultima capa y debe escribir dL/d(salida) en grad.
 * Devuelve 1 si lo hizo; 0 (sin tocar nada) si la pila no es bloqueable,
 * S==0 o no hubo memoria -- el llamador cae al camino de siempre. */
typedef void (*TernaryLossGradFn)(size_t sample, const float *out, float *grad, void *ctx);
int ternary_stack_blockable(const TernaryStack *s);
int ternary_stack_train_epoch_blocked(TernaryStack *s, TernaryVec *const *xs, size_t n, size_t S,
                                      const TernaryCapabilities *caps,
                                      TernaryLossGradFn loss_fn, void *ctx);

/* Inferencia por bloques: igual que llamar ternary_stack_forward n veces (enteros,
 * mismos bits), pero S muestras por vez para que cada fila de W se lea una vez por
 * bloque. out: n * output_len floats. Devuelve 0 (sin tocar nada) si alguna capa no
 * admite forward por bloques o no hay memoria. */
int ternary_stack_predict_blocked(TernaryStack *s, TernaryVec *const *xs, size_t n, size_t S,
                                  const TernaryCapabilities *caps, float *out);

/* Aplica a TODAS las capas los gradientes acumulados desde el ultimo
 * ternary_stack_zero_gradients() -- mismo scale_lr para toda
 * la red (igual que el resto del proyecto: no hay LR por capa). */
void ternary_stack_apply_gradients(TernaryStack *s, float scale_lr);

/* Regla R2 del stack (ver ternary_dense.h; tau en (0,1], beta en [0,1)). ternary_stack_apply_gradients la aplica a todas las capas. */
void ternary_stack_set_update_rule(TernaryStack *s, float tau, float beta);
/* Resto de parametros de R2 del stack: inc > 0, zero_target < 1 (negativo = sin objetivo de ceros), reset en [0,1]. El checkpoint de
 * entrenamiento (full_checkpoint=1) guarda los cinco en config.json ("r2") y la carga los restaura; el export no los lleva. */
void ternary_stack_set_r2_params(TernaryStack *s, float inc, float zero_target, float reset);
/* Memoria (beta2) de la normalizacion por elemento de la cabeza lm_head, en [0,1): 0 la apaga (queda la R2 por fila). Defecto 0.99. */
void ternary_stack_set_head_beta2(TernaryStack *s, float head_beta2);
TernaryUpdateRule ternary_stack_get_update_rule(const TernaryStack *s);

/*
 * Guardado/carga de la red completa a un solo archivo.
 *
 * full_checkpoint=1 ("checkpoint"): pesos+votos+bias+escala de CADA
 * capa tal cual estan en memoria (2 bits/peso) -- pensado para pausar
 * un entrenamiento largo y retomarlo despues EXACTO (ver
 * tests/test_model_io.c: entrenar N+M epocas de un tiron da bit a bit
 * lo mismo que entrenar N, guardar, cargar, entrenar M mas).
 *
 * full_checkpoint=0 ("export"): SOLO pesos+bias de cada capa,
 * empaquetados al MINIMO de bits (ver ternary_model_io.h,
 * ternary_pack_trits: ~1.58 bits/peso en vez de 2), SIN votos -- listo
 * para repartir/desplegar, NO se puede seguir entrenando despues de
 * cargarlo (los votos vuelven a arrancar de 0, como una capa recien
 * creada).
 *
 * Devuelve 1/un TernaryStack no-NULL en exito. ternary_stack_load()
 * devuelve NULL si el archivo no existe, esta corrupto, o menciona un
 * tipo de capa o de activacion que esta version no reconoce (nunca
 * carga a medias en silencio). */
int ternary_stack_save(const TernaryStack *s, const char *dir, int full_checkpoint);
TernaryStack *ternary_stack_load(const char *dir);
/* Igual que ternary_stack_load pero devuelve el motivo del fallo en err. */
TernaryStack *ternary_stack_load_ex(const char *dir, char *err, size_t err_cap);

/* Instantanea EN MEMORIA del estado completo (pesos, votos, bias, escalas) y restauracion en el mismo stack (mismas
 * formas). Sirve para volver al mejor punto de un entrenamiento. snap debe empezar en {NULL, 0}; devuelven 1 en exito. */
typedef struct { char *buf; size_t size; } TernaryStackSnapshot;
int ternary_stack_snapshot(const TernaryStack *s, TernaryStackSnapshot *snap);
int ternary_stack_restore(TernaryStack *s, const TernaryStackSnapshot *snap);
void ternary_stack_snapshot_free(TernaryStackSnapshot *snap);

/* Checkpoint por bloque: las capas con estado grande (Z guardados, vectores ternarios por posicion, buffers de las
 * residuales...) lo guardan en UN espacio de trabajo compartido por todo el stack, y el backward recalcula el forward de
 * cada una justo antes de su backward (la ultima no hace falta). Los gradientes de entrada tambien se comparten (dos
 * buffers que alternan). Bits identicos con y sin checkpoint; cuesta ~un forward extra por bloque a cambio de pasar de
 * ~12 KB a ~1 KB por token y por bloque. No se deshace ni admite agregar capas despues. Tambien se activa con la variable
 * de entorno TERNARYAI_CHECKPOINT=1 (se lee en el primer forward). 0 ok, -1 error. */
int ternary_stack_set_checkpoint(TernaryStack *s, int on);

#endif /* TERNARY_STACK_H */
