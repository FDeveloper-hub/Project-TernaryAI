#ifndef TERNARYAI_H
#define TERNARYAI_H
/*
 * API plana de libternaryai -- la unica superficie que ve la fachada Python
 * (python/ternaryai/_native.py, via ctypes). Todo es punteros opacos,
 * enteros, doubles y buffers: sin structs por valor ni callbacks, para
 * que cualquier FFI la pueda usar directo.
 *
 * Convenciones:
 *  - Las funciones que pueden fallar devuelven 1 (ok) / 0 (error) o un
 *    puntero NULL, y dejan un mensaje legible en 'err' (buffer del
 *    llamador, 'err_cap' bytes; puede ser NULL).
 *  - Las formas son arreglos de size_t de hasta TERNARYAI_MAX_DIMS dims
 *    (HWC para imagenes).
 *  - Los parametros de una capa viajan como un arreglo plano de double
 *    en el orden del registro (un parametro "pair" ocupa 2 posiciones;
 *    bool = 0/1).
 *  - Un TernaryAIModel NO se comparte entre hilos. Distintos modelos si
 *    pueden usarse desde hilos distintos: el computo se serializa
 *    internamente (comparten el pool de hilos de CPU).
 *  - Solo usa la familia CPU (escalar + pool); no toca Vulkan ni
 *    OpenCL.
 *
 * Agregar un tipo de capa NO cambia esta API: se registra en
 * ternary_layer.c (ver ternary_layer.h) y aparece solo en el registro
 * de abajo.
 */
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__)
#  define TERNARYAI_API __attribute__((visibility("default")))
#else
#  define TERNARYAI_API
#endif

/* Sube cuando cambia algo de esta API; la fachada Python lo comprueba. */
#define TERNARYAI_ABI_VERSION 30   /* v30: motor: conversacion configurable (ternaryai_infer_chat_encode_fmt / chat_reply_ex / think_split) y presupuesto de razonamiento en TernaryInferGenParams (4 campos al final); v29: ternaryai_dataset_load_ex (opciones de razonamiento: conservar o quitar); se agrega, nada de lo anterior cambia;  v28: ternaryai_dataset_targets (objetivos con mascara de perdida a partir de ids; se agrega, nada de lo anterior cambia);  v27: ternaryai_set_threads / get_threads (reparto en hilos de toda la libreria; se agregan, nada de lo anterior cambia);  v26: motor de inferencia (include/ternary_infer.h, funciones ternaryai_infer_*; se agregan, nada de lo anterior cambia);  v25: normalizacion por elemento del gradiente de la cabeza lm_head (estado de 4 bits logaritmicos + escala por fila; ternaryai_model_set_head_beta2 / get_head_beta2, 'head_beta2' en la regla R2 y en el checkpoint); v24: se elimina V ternario (parametro v_ternary de GQA; V es siempre de 8 bits en la ruta uint8; un parametro menos por capa GQA, config sin 'v_ternary'); v23: se eliminan la regla de votos, la seleccion de regla (set_update_rule recibe tau y beta), el calendario k/umbral y vote_limit; hyper = solo scale_lr; v22: regla R2 (kind 2) reemplaza a la regla de momento (kind 1) y se elimina ternaryai_model_set_layer_tau; v21: ternaryai_model_set_layer_tau (tau por capa); v20: ternaryai_model_attention_probs (diagnostico de atencion); v19: ternaryai_prof_enable/reset/dump_json (perfilador por ambitos, apagado por defecto); v18: formato de modelo tipo safetensors (carpeta con model.ternary + config.json [+ training_state.ternary]) y tokenizer.json; save/load reciben una CARPETA; v17: ternaryai_model_ffn_zero_stats (sparsity de la activacion uint8 del feed-forward); v16: ternaryai_model_set_update_rule (regla 1: momento entero); v15: entrada uint8 (ternaryai_model_train_f32 / predict_f32), sign y band eliminadas (solo 'uint8'); v14: ternaryai_model_set_batching (mini-lotes en train / train_ids); v13: ternaryai_layer_param_has_max/max (vote_limit <= 127: los votos son int8); v12: ternaryai_model_hold_best / restore_held_best (restore_best ya no depende de los trozos de verbose); v11: tokens especiales del tokenizador (add_special, encode_ex, decode_ex, ...); v10: ternaryai_model_set_checkpoint; v9: ternaryai_model_set_restore_best; v8: GQA y SwiGLU suman el parametro out_scale (un valor mas por capa); v7: bloques residuales (begin/end/cancel_residual, layer_info_at, layer_children); v6: ternaryai_model_train_ids / predict_ids (ids + entropia cruzada); v5: ternaryai_dataset_load */
#define TERNARYAI_MAX_DIMS 4
#define TERNARYAI_MAX_SLOTS 16
#define TERNARYAI_NAME_CAP 64
#define TERNARYAI_MAX_RESIDUAL_DEPTH 8   /* bloques residuales abiertos a la vez (anidados) */
/* Orden de los hiperparametros de ternaryai_model_train(): solo la tasa de la escala. */
#define TERNARYAI_NUM_HYPER 1
enum { TERNARYAI_H_SCALE_LR = 0 };

enum { TERNARYAI_PARAM_INT = 0, TERNARYAI_PARAM_FLOAT = 1, TERNARYAI_PARAM_BOOL = 2, TERNARYAI_PARAM_PAIR = 3 };

typedef struct TernaryAIModel TernaryAIModel;

TERNARYAI_API int ternaryai_abi_version(void);

/* --- Registro de capas y activaciones (solo lectura) --- */
TERNARYAI_API int         ternaryai_layer_type_count(void);
TERNARYAI_API const char *ternaryai_layer_type_name(int type);    /* "dense" */
TERNARYAI_API const char *ternaryai_layer_type_class(int type);   /* "Dense" */
TERNARYAI_API const char *ternaryai_layer_type_doc(int type);
TERNARYAI_API int         ternaryai_layer_param_count(int type);
TERNARYAI_API const char *ternaryai_layer_param_name(int type, int param);
TERNARYAI_API int         ternaryai_layer_param_kind(int type, int param);      /* TERNARYAI_PARAM_* */
TERNARYAI_API int         ternaryai_layer_param_required(int type, int param);
TERNARYAI_API double      ternaryai_layer_param_default(int type, int param);
TERNARYAI_API int         ternaryai_layer_param_has_min(int type, int param);
TERNARYAI_API double      ternaryai_layer_param_min(int type, int param);
TERNARYAI_API int         ternaryai_layer_param_has_max(int type, int param);
TERNARYAI_API double      ternaryai_layer_param_max(int type, int param);
TERNARYAI_API const char *ternaryai_layer_param_doc(int type, int param);
TERNARYAI_API int         ternaryai_activation_count(void);
TERNARYAI_API const char *ternaryai_activation_name(int i);

/* --- Modelo --- */
/* Modelo vacio para entradas de forma in_shape; 'seed' fija los pesos iniciales. */
TERNARYAI_API TernaryAIModel *ternaryai_model_create(const size_t *in_shape, int in_ndims, unsigned int seed,
                                            char *err, size_t err_cap);
/* Agrega una capa por nombre de tipo; devuelve su forma de salida. Si falla
 * (forma, posicion, parametros) el modelo queda EXACTAMENTE como estaba. */
TERNARYAI_API int ternaryai_model_add_layer(TernaryAIModel *m, const char *type_name, const double *params,
                                      int num_slots, const char *activation, size_t *out_shape,
                                      int *out_ndims, char *err, size_t err_cap);
/* Bloques residuales x + f(x). begin_residual abre un bloque (activation: la que se aplica a su salida);
 * las capas que se agregan despues con ternaryai_model_add_layer van a su cadena f, y end_residual lo cierra
 * y lo agrega al modelo (o al bloque padre: se pueden anidar, hasta TERNARYAI_MAX_RESIDUAL_DEPTH).
 * Las reglas de la cadena las valida add_layer/end_residual con un mensaje (la primera sub-capa debe tomar
 * entrada float, la ultima debe devolver el largo de x, ninguna toma ids ni deja de propagar gradiente).
 * Si add_layer falla el bloque sigue abierto y sin rastro de la capa rechazada; si end_residual falla el bloque
 * se DESCARTA entero y el modelo queda como antes de abrirlo. cancel_residual descarta el bloque abierto mas
 * interno a pedido. Con un bloque abierto, train/predict/save fallan (el modelo no esta completo).
 * end_residual devuelve la forma de salida (= la de x). */
TERNARYAI_API int ternaryai_model_begin_residual(TernaryAIModel *m, const char *activation, char *err, size_t err_cap);
TERNARYAI_API int ternaryai_model_end_residual(TernaryAIModel *m, size_t *out_shape, int *out_ndims,
                                         char *err, size_t err_cap);
TERNARYAI_API int ternaryai_model_cancel_residual(TernaryAIModel *m, char *err, size_t err_cap);
TERNARYAI_API int ternaryai_model_open_residuals(const TernaryAIModel *m);
/* Como layer_info pero con una ruta: path[0] = capa del modelo, path[1] = sub-capa de esa residual, ...
 * Una residual se describe con type_name "residual" y 0 parametros. layer_children: cuantas sub-capas tiene la
 * capa de la ruta (-1 si no es una residual o la ruta no existe). */
TERNARYAI_API int ternaryai_model_layer_info_at(const TernaryAIModel *m, const int *path, int depth, char *type_name,
                                          double *params, int *num_slots, char *activation, size_t *out_shape,
                                          int *out_ndims);
TERNARYAI_API int ternaryai_model_layer_children(const TernaryAIModel *m, const int *path, int depth);
TERNARYAI_API int ternaryai_model_num_layers(const TernaryAIModel *m);
TERNARYAI_API int ternaryai_model_input_shape(const TernaryAIModel *m, size_t *dims, int *ndims);
/* Arquitectura de la capa i: tipo, parametros reales, activacion, forma de salida.
 * type_name/activation: buffers de TERNARYAI_NAME_CAP bytes; params: TERNARYAI_MAX_SLOTS doubles;
 * out_shape: TERNARYAI_MAX_DIMS size_t. */
TERNARYAI_API int ternaryai_model_layer_info(const TernaryAIModel *m, int i, char *type_name, double *params,
                                       int *num_slots, char *activation, size_t *out_shape,
                                       int *out_ndims);

/* Entrena 'count' epocas de lote completo, numeradas desde first_epoch.
 * x: int8 (n, input_len) con valores -1/0/1; y: float (n, output_len).
 * hyper: TERNARYAI_NUM_HYPER doubles. loss_out: 'count' doubles (MSE de cada
 * epoca, medido en sus forwards, antes de aplicar esa epoca). */
TERNARYAI_API int ternaryai_model_train(TernaryAIModel *m, const int8_t *x, const float *y, size_t n,
                                  size_t first_epoch, size_t count, const double *hyper,
                                  double *loss_out, char *err, size_t err_cap);
/* Salida escalada de la ultima capa: out = float (n, output_len). */
/* Entrada uint8 (opcion alternativa a la ternaria int8): x son floats (n * features); cada muestra se cuantiza a uint8
 * (escala y zero-point por fila, el cero exacto) y la primera capa la lee como uint8. Rechaza modelos cuya primera
 * capa no acepte uint8. Mismos hiperparametros y salida que ternaryai_model_train / predict. */
TERNARYAI_API int ternaryai_model_train_f32(TernaryAIModel *m, const float *x, const float *y, size_t n,
                                            size_t first_epoch, size_t count, const double *hyper, double *loss_out,
                                            char *err, size_t err_cap);
TERNARYAI_API int ternaryai_model_predict_f32(TernaryAIModel *m, const float *x, size_t n, float *out,
                                              char *err, size_t err_cap);

TERNARYAI_API int ternaryai_model_predict(TernaryAIModel *m, const int8_t *x, size_t n, float *out,
                                    char *err, size_t err_cap);

/* 1 si la primera capa toma ids de token (usar *_ids), 0 si toma ternarios (o no hay capas). */
TERNARYAI_API int ternaryai_model_takes_ids(const TernaryAIModel *m);

/* Modelos de lenguaje: la primera capa toma ids (Embedding) y la salida es un vector de logits
 * por posicion, (seq, vocab) con seq = largo de la entrada. Mismo calendario de epocas e
 * hiperparametros que ternaryai_model_train.
 * ids: int32 (n, seq); targets: int32 (n, seq) con el id esperado por posicion (el siguiente
 * token, normalmente); un valor NEGATIVO ignora esa posicion (relleno): no suma a la perdida
 * ni genera gradiente. Un target >= vocab es un error. Un id de entrada negativo es relleno
 * (vector cero); uno >= vocab del Embedding es un error.
 * loss_out: 'count' doubles, la entropia cruzada de cada epoca (media sobre las muestras de la
 * media por muestra de sus posiciones validas), medida en sus forwards, antes del update.
 * acc_out (puede ser NULL): 'count' doubles, fraccion de posiciones validas acertadas (argmax). */
/* on=1 (defecto): train_ids restaura al final el estado de menor perdida de entrenamiento; on=0: conserva el ultimo. */
TERNARYAI_API void ternaryai_model_set_restore_best(TernaryAIModel *m, int on);
/* Mini-lotes para ternaryai_model_train / train_ids. batch_size = 0 (o >= n) es el lote completo de siempre: una sola
 * actualizacion por epoca, sin permutar, con exactamente los mismos bits que antes. Con batch_size < n cada epoca
 * recorre las muestras en pasos de batch_size (el ultimo puede ser mas chico): cada paso promedia el gradiente de SU
 * lote y aplica sus votos y su escala. El calendario (umbral, k) sigue siendo por epoca. shuffle != 0 permuta las
 * muestras cada epoca con una permutacion determinista que depende solo de (seed, epoca global), no de como se
 * parta el fit en trozos. El ajuste vive en el modelo hasta que se cambie. */
TERNARYAI_API void ternaryai_model_set_batching(TernaryAIModel *m, size_t batch_size, int shuffle, uint64_t seed);
/* Regla R2 del modelo (contador int8 con histeresis y objetivo de ceros, gradiente normalizado por fila, ver ternary_dense.h):
 * tau en (0,1] y beta en [0,1). Devuelve 1, o 0 con mensaje en err si algun valor es invalido (el modelo no cambia).
 * Vive en el modelo hasta que se cambie. */
TERNARYAI_API int ternaryai_model_set_update_rule(TernaryAIModel *m, double tau, double beta, char *err, size_t err_cap);
/* Resto de parametros de R2: inc (> 0, def. 0.03), zero_target (< 1; negativo = sin objetivo de ceros; def. 0.2) y reset (en [0,1], def. 0.5).
 * Viven en el modelo y viajan en el checkpoint de entrenamiento (save con full_checkpoint=1), no en el export. */
TERNARYAI_API int ternaryai_model_set_r2_params(TernaryAIModel *m, double inc, double zero_target, double reset, char *err, size_t err_cap);
/* Parametros vigentes: out5 = { tau, beta, inc, zero_target, reset }. Devuelve 1, o 0 si algun puntero es NULL. */
TERNARYAI_API int ternaryai_model_get_update_rule(const TernaryAIModel *m, double *out5);
/* Memoria (beta2) de la normalizacion por elemento del gradiente de la cabeza lm_head, en [0, 1): 0 la apaga y deja la R2 por fila. Defecto 0.99. */
TERNARYAI_API int ternaryai_model_set_head_beta2(TernaryAIModel *m, double head_beta2, char *err, size_t err_cap);
TERNARYAI_API int ternaryai_model_get_head_beta2(const TernaryAIModel *m, double *out);
/* Retener el mejor estado entre llamadas a train_ids (los trozos de un fit con verbose): hold_best(m,1) lo activa y
 * borra lo retenido; hold_best(m,0) lo descarta. Mientras esta activo, train_ids no restaura al final de cada llamada;
 * restore_held_best(m, ultima_perdida) vuelve al mejor estado (si la ultima perdida es peor) y lo desactiva
 * (devuelve 1 si restauro). Asi "verbose" no cambia el resultado del entrenamiento. */
TERNARYAI_API void ternaryai_model_hold_best(TernaryAIModel *m, int on);
TERNARYAI_API int ternaryai_model_restore_held_best(TernaryAIModel *m, double last_loss);
/* on=1: checkpoint por bloque (ver ternary_stack_set_checkpoint): las capas con estado grande comparten un espacio de
 * trabajo y el backward recalcula su forward; los resultados son identicos bit a bit y la memoria por token baja ~10x en
 * modelos con varios bloques. No se deshace. Llamar con el modelo ya armado (capas agregadas). 0 ok, -1 error. */
TERNARYAI_API int ternaryai_model_set_checkpoint(TernaryAIModel *m, int on);

/* Sparsity del feed-forward (capas swiglu, de cualquier gate_mode, tambien dentro de residuales, en orden de recorrido).
 * Para cada una devuelve cuantos valores de la activacion uint8 que entra a W_down cayeron exactamente en el zero-point
 * de su fila (zeros[i]) sobre cuantos se vieron (total[i]) desde el ultimo reset. Escribe a lo sumo 'max' capas y
 * devuelve cuantas capas swiglu tiene el modelo. reset != 0 pone los contadores a cero. Solo cuenta: no cambia resultados. */
TERNARYAI_API int ternaryai_model_ffn_zero_stats(const TernaryAIModel *m, uint64_t *zeros, uint64_t *total, int max, int reset);

/* Perfilador por ambitos (ver include/ternary_prof.h). Apagado: cuesta una comparacion por ambito y no cambia ningun
 * resultado. enable(1) hace que el hilo que llama sea el unico que se mide (llamar desde el hilo que entrena).
 * dump_json escribe el arbol (nodos con llamadas, tiempo total y propio en ns) y devuelve el largo necesario sin el NUL. */
/* Diagnostico de atencion: con gqa_index < 0 devuelve cuantas capas GQA tiene el modelo (en orden de recorrido). Con
 * gqa_index >= 0 escribe en out (cap >= seq*seq floats) las probabilidades de atencion de esa capa y cabeza segun el ULTIMO
 * forward (predict): out[i*seq + j], j <= i. 0 en exito, -1 si no se puede (ventana, indice fuera de rango, cap chico). */
TERNARYAI_API int ternaryai_model_attention_probs(const TernaryAIModel *m, int gqa_index, int head, float *out, size_t cap);
/* Hilos de calculo de toda la libreria (entrenamiento e inferencia): lo fija el usuario a mano. n <= 1 = un solo hilo (valor inicial: TERNARYAI_THREADS o 1).
 * El resultado es identico con cualquier numero de hilos. No cambiarlo mientras se entrena. */
TERNARYAI_API void ternaryai_set_threads(int n);
TERNARYAI_API int ternaryai_get_threads(void);
TERNARYAI_API void ternaryai_prof_enable(int on);
TERNARYAI_API void ternaryai_prof_reset(void);
TERNARYAI_API size_t ternaryai_prof_dump_json(char *buf, size_t cap);

TERNARYAI_API int ternaryai_model_train_ids(TernaryAIModel *m, const int32_t *ids, const int32_t *targets,
                                      size_t n, size_t first_epoch, size_t count, const double *hyper,
                                      double *loss_out, double *acc_out, char *err, size_t err_cap);
/* Logits sin softmax: out = float (n, seq, vocab). */
TERNARYAI_API int ternaryai_model_predict_ids(TernaryAIModel *m, const int32_t *ids, size_t n, float *out,
                                        char *err, size_t err_cap);

/* Un modelo guardado es una CARPETA (se crea; la carpeta padre debe existir):
 *   model.ternary          pesos (contenedor de tensores estilo safetensors, ver ternary_tfile.h)
 *   config.json            arquitectura e hiperparametros
 *   training_state.ternary solo con full_checkpoint=1: votos/momento, para retomar el entrenamiento
 * full_checkpoint=1: retomable; 0: export minimo para inferencia (borra un training_state.ternary viejo de esa carpeta).
 * load acepta ambas (si encuentra training_state.ternary lo carga). El tokenizador (tokenizer.json) se guarda aparte. */
TERNARYAI_API int ternaryai_model_save(const TernaryAIModel *m, const char *path, int full_checkpoint,
                                 char *err, size_t err_cap);
TERNARYAI_API TernaryAIModel *ternaryai_model_load(const char *path, char *err, size_t err_cap);
TERNARYAI_API void ternaryai_model_free(TernaryAIModel *m);   /* acepta NULL */

/* --- Tokenizador BPE por bytes (ver ternary_tokenizer.h) --- */
typedef struct TernaryAITokenizer TernaryAITokenizer;

TERNARYAI_API TernaryAITokenizer *ternaryai_tokenizer_create(size_t vocab_size, char *err, size_t err_cap);
TERNARYAI_API int ternaryai_tokenizer_train(TernaryAITokenizer *t, const uint8_t *corpus, size_t corpus_len,
                                      char *err, size_t err_cap);
TERNARYAI_API size_t ternaryai_tokenizer_vocab_size(const TernaryAITokenizer *t);
TERNARYAI_API size_t ternaryai_tokenizer_num_merges(const TernaryAITokenizer *t);
TERNARYAI_API int ternaryai_tokenizer_encode(const TernaryAITokenizer *t, const uint8_t *text, size_t text_len,
                                       int32_t *out_ids, size_t out_cap, size_t *out_len,
                                       char *err, size_t err_cap);
TERNARYAI_API int ternaryai_tokenizer_decode(const TernaryAITokenizer *t, const int32_t *ids, size_t num_ids,
                                       uint8_t *out_bytes, size_t out_cap, size_t *out_len,
                                       char *err, size_t err_cap);
/* Tokens especiales (ver ternary_tokenizer.h): se agregan antes de entrenar; ids fijos 256, 257, ... */
TERNARYAI_API int ternaryai_tokenizer_add_special(TernaryAITokenizer *t, const uint8_t *text, size_t len,
                                            char *err, size_t err_cap);
TERNARYAI_API size_t ternaryai_tokenizer_num_special(const TernaryAITokenizer *t);
TERNARYAI_API int ternaryai_tokenizer_special_text(const TernaryAITokenizer *t, size_t index, uint8_t *out,
                                             size_t out_cap, size_t *out_len, char *err, size_t err_cap);
TERNARYAI_API int32_t ternaryai_tokenizer_special_id(const TernaryAITokenizer *t, const uint8_t *text, size_t len);
TERNARYAI_API int ternaryai_tokenizer_encode_ex(const TernaryAITokenizer *t, const uint8_t *text, size_t text_len,
                                          int parse_special, int32_t *out_ids, size_t out_cap, size_t *out_len,
                                          char *err, size_t err_cap);
TERNARYAI_API int ternaryai_tokenizer_decode_ex(const TernaryAITokenizer *t, const int32_t *ids, size_t num_ids,
                                          int skip_special, uint8_t *out_bytes, size_t out_cap, size_t *out_len,
                                          char *err, size_t err_cap);
TERNARYAI_API int ternaryai_tokenizer_save(const TernaryAITokenizer *t, const char *path, char *err, size_t err_cap);
TERNARYAI_API TernaryAITokenizer *ternaryai_tokenizer_load(const char *path, char *err, size_t err_cap);
TERNARYAI_API void ternaryai_tokenizer_free(TernaryAITokenizer *t);  /* acepta NULL */

/* --- Carga de datasets de entrenamiento: texto, JSON, JSONL (ver ternary_dataset.h) --- */
enum {
    TERNARYAI_DATASET_TEXT_LINES = 0,
    TERNARYAI_DATASET_TEXT_WHOLE = 1,
    TERNARYAI_DATASET_JSON       = 2,
    TERNARYAI_DATASET_JSONL      = 3,
};

TERNARYAI_API int ternaryai_dataset_load(const char *path, int format, int strict,
                                   uint8_t *out_buf, size_t out_cap, size_t *out_len, size_t *out_count,
                                   size_t *out_skipped, char *err, size_t err_cap);

/* Igual, con flags (ver ternary_dataset_load_ex): TERNARYAI_DATASET_REASONING_DROP quita el razonamiento; 0 lo conserva normalizado
 * a <|think|>...<|/think|>. */
enum { TERNARYAI_DATASET_REASONING_KEEP = 0, TERNARYAI_DATASET_REASONING_DROP = 1 };
TERNARYAI_API int ternaryai_dataset_load_ex(const char *path, int format, int strict, int flags,
                                   uint8_t *out_buf, size_t out_cap, size_t *out_len, size_t *out_count,
                                   size_t *out_skipped, char *err, size_t err_cap);

/* --- Objetivos con mascara de perdida (ver ternary_targets.h) ---
 * ids: n_examples ejemplos concatenados; el ejemplo e ocupa [offsets[e], offsets[e+1]) (offsets: n_examples+1 valores, no decrecientes).
 * Escribe y (misma disposicion que ids): y[i] = ids[i+1] si ese token se entrena, -1 si no; el ultimo de cada ejemplo es -1.
 * kind[id - special_lo] (special_hi - special_lo valores): 0 especial normal, 1 rol entrenable, 2 rol enmascarado, 3 <|end|>.
 * Devuelve 1, o 0 con el mensaje en err (argumentos invalidos). */
TERNARYAI_API int ternaryai_dataset_targets(const int32_t *ids, const size_t *offsets, size_t n_examples,
                                            int32_t special_lo, int32_t special_hi, const int8_t *kind, int unmarked_train,
                                            int32_t *y, char *err, size_t err_cap);

#ifdef __cplusplus
}
#endif
#endif /* TERNARYAI_H */
