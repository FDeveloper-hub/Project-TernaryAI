#ifndef TERNARY_LAYER_H
#define TERNARY_LAYER_H

#include "ternary_core.h"
#include "ternary_dispatch.h"
#include "ternary_dense.h"
#include "ternary_conv2d.h"
#include "ternary_embedding.h"
#include "ternary_gqa.h"
#include "ternary_rmsnorm.h"
#include "ternary_swiglu.h"
#include "ternary_head.h"
#include "ternary_rng.h"
#include "ternary_tensor.h"
#include "ternary_activation.h"
#include "ternary_tfile.h"
#include "ternary_json.h"
#include <stdio.h>
#include <stdint.h>

/*
 * Interfaz generica de capa -- existe para que TernaryStack (ver
 * ternary_stack.h) pueda encadenar CUALQUIER tipo de capa sin conocer
 * su tipo concreto, y para que agregar un tipo de capa nuevo en el
 * futuro no requiera tocar el contenedor: alcanza con escribir estas
 * ~13 funciones de adaptacion (ver ternary_layer.c, ternary_layer_from_dense
 * y ternary_layer_from_conv2d son el ejemplo a copiar) y sumar el tipo
 * a ternary_layer_import_by_type() (cargar un modelo).
 *
 * Todas las capas de este proyecto comparten la misma forma: una matriz
 * de pesos "filas x ancho_de_fila" (output_dim*input_dim en Dense,
 * out_channels*kernel_size en Conv2D), un bias ternario opcional de
 * "filas" elementos, y un escalar de escala por capa -- por eso el
 * gradiente se acumula en un buffer GENERICO que vive en TernaryStack
 * (ver TernaryStackEntry), no en un tipo opaco por-capa: alcanza para
 * Dense y Conv2D tal cual estan hoy, y probablemente para lo que se
 * agregue despues si sigue el mismo patron.
 */
/* Contexto de exportacion/importacion de una capa al formato model.ternary (ver ternary_tfile.h).
 * model: pesos (siempre). state: estado de entrenamiento (votos), NULL en un export de inferencia. */
typedef struct { TernaryTWriter *model; TernaryTWriter *state; } TernaryExportCtx;
typedef struct {
    const TernaryTReader *model;
    const TernaryTReader *state;     /* NULL: sin estado de entrenamiento (votos en cero) */
    char *err; size_t err_cap;       /* primer mensaje de error (puede ser NULL) */
} TernaryImportCtx;

typedef struct {
    /* Se escribe tal cual en config.json (ternary_stack_save) y
     * se compara tal cual al cargar (ternary_stack_load) -- nunca
     * cambiar el nombre de un tipo ya usado en archivos existentes. */
    const char *type_name;

    size_t (*input_len)(const void *self);
    size_t (*output_len)(const void *self);
    /* Cantidad de matrices de pesos entrenables de esta capa (>= 1).
     * Dense, Conv2D y Embedding tienen 1; una capa como GQA tiene varias
     * (Wq, Wk, Wv, Wo). Cada matriz m tiene sus propias filas/ancho, su
     * bias opcional, su acumulador de gradiente y su PROPIA escala (una
     * escala por matriz, estilo BitNet). El mecanismo de votos/umbral/
     * escala no cambia: TernaryStack lo aplica una vez por matriz, en el
     * mismo orden (filas, bias, escala) que antes tenia con una sola. */
    size_t (*num_matrices)(const void *self);
    size_t (*grad_rows)(const void *self, size_t m);       /* output_dim / out_channels */
    size_t (*grad_row_width)(const void *self, size_t m);  /* input_dim / kernel_size */
    int    (*has_bias)(const void *self, size_t m);
    /* 0 si esta capa NUNCA calcula gradiente hacia su entrada (hoy:
     * Conv2D, que asume ser siempre la primera capa de la red -- ver su
     * propio comentario en ternary_conv2d.h). ternary_stack_add()
     * rechaza agregar una capa asi en cualquier posicion que no sea la
     * 0, en vez de entrenar en silencio con las capas anteriores
     * congeladas (sin gradiente llegandoles nunca). */
    int    (*propagates_grad_x)(const void *self);

    /* Tipo de dato que esta capa espera recibir como 'x' en forward/
     * backward (ver ternary_tensor.h). TERNARY para Dense/Conv2D (como
     * siempre); ID para una capa que hace busqueda en vez de producto
     * punto (Embeddings); FLOAT para una capa que quiere el crudo de la
     * anterior sin cuantizar. TernaryStack usa esto para decidir, al
     * conectar la salida de una capa con la entrada de la siguiente, si
     * hace falta cuantizar (llama a la activacion de la capa, como
     * siempre) o pasar el float crudo directo -- ver ternary_stack.c. Si
     * esta capa es la PRIMERA de la red, este es tambien el tipo que
     * debe tener el tensor 'x' que el llamador le pasa a
     * ternary_stack_forward/backward. Solo la capa 0 puede declarar ID:
     * ternary_stack_add() rechaza agregarla en cualquier otra posicion
     * (nadie produce ids como salida, ver ternary_tensor.h). */
    TernaryDType (*input_dtype)(const void *self);

    void (*forward)(const void *self, const TernaryTensor *x, float *out, int32_t *out_raw_opt,
                     const TernaryCapabilities *caps);

    /* grad_x_out: buffer de longitud input_len(), YA PUESTO A CERO por
     * el llamador antes de cada llamada (grad_x_out se ACUMULA, no se
     * pisa -- misma convencion que ternary_dense_backward). Ignorado
     * (puede llegar NULL) si propagates_grad_x()==0. grad_w_out: arreglo de
     * num_matrices() buffers; grad_w_out[m] tiene longitud
     * grad_rows(m)*grad_row_width(m), se SOBREESCRIBE (no se acumula) -- el llamador lo sube a su propio acumulador de epoca
     * despues de esta llamada. grad_scale_out: arreglo de num_matrices()
     * floats; grad_bias_out[m]: buffer de grad_rows(m) floats, o NULL si
     * !has_bias(m). Ambos SI se acumulan (el llamador los deja intactos entre muestras de la
     * misma epoca, en cero solo al principio). */
    void (*backward)(const void *self, const TernaryTensor *x, const float *grad_out,
                      float *const *grad_w_out, float *grad_x_out,
                      float *grad_scale_out, float *const *grad_bias_out,
                      const int32_t *z_raw_opt);

    void (*apply_row_gradients)(void *self, size_t m, size_t row, const float *grad_row);
    void (*apply_bias_gradients)(void *self, size_t m, const float *grad_bias);
    void (*apply_scale_gradient)(void *self, size_t m, float grad_scale, float scale_lr);

    void (*destroy)(void *self);

    /* OPCIONAL (NULL = no soportado). Igual que backward, pero grad_w_out[m]
     * se ACUMULA (+=) en vez de sobreescribirse, de modo que el stack lo
     * apunta directo al acumulador de la epoca y se ahorra el buffer
     * temporal y la pasada acc += scratch. Debe dar bits identicos a
     * backward + suma. */
    void (*backward_accumulate)(const void *self, const TernaryTensor *x, const float *grad_out,
                                 float *const *grad_w_out, float *grad_x_out,
                                 float *grad_scale_out, float *const *grad_bias_out,
                                 const int32_t *z_raw_opt);

    /* OPCIONAL (NULL = la capa no admite entrenamiento por bloques). Solo
     * para capas sin estado interno entre forward y backward (el backward
     * depende solo de la entrada, grad_out y z_raw): Dense. Ver
     * ternary_dense_backward_block. grad_w_out: arreglo de num_matrices()
     * buffers (se ACUMULA); grad_x[s] se acumula (NULL = no hace falta). */
    void (*backward_block)(const void *self, size_t S, const TernaryTensor *const *xs,
                            const float *const *grad_out, const int32_t *const *z_raw,
                            float *const *grad_w_out, float *const *grad_x,
                            float *grad_scale_out, float *const *grad_bias_out);

    /* OPCIONAL (NULL = el stack llama apply_row_gradients fila por fila).
     * Aplica los gradientes de TODAS las filas de la matriz m; debe dar los
     * mismos bits que el bucle fila por fila. */
    void (*apply_matrix_gradients)(void *self, size_t m, const float *grad_w);

    /* OPCIONAL (NULL = el stack llama forward muestra a muestra). Forward de S
     * muestras; mismos resultados que S llamadas a forward. */
    void (*forward_block)(const void *self, size_t S, const TernaryTensor *const *xs,
                           float *const *outs, int32_t *const *raws, const TernaryCapabilities *caps);

    /* OPCIONALES (NULL = la capa no guarda estado grande entre forward y backward). Checkpoint por bloque
     * (ternary_stack_set_checkpoint): el estado de la capa que crece con la secuencia (Z guardados, vectores
     * ternarios por posicion, buffers de una residual...) puede vivir en un ESPACIO DE TRABAJO que comparten todas
     * las capas del stack, porque el stack recalcula el forward de la capa justo antes de su backward.
     *   state_bytes: bytes que necesita ese estado (0 = nada que compartir).
     *   bind_state:  a partir de ahora el estado vive en 'base' (alineado a 64 bytes, state_bytes() bytes); la capa
     *                libera el que tenia propio. No se deshace. El forward debe reescribir TODO lo que el backward lee. */
    size_t (*state_bytes)(const void *self);
    void (*bind_state)(void *self, void *base);

    /* OPCIONAL (NULL = no). Devuelve el largo de fila (row_len) del tensor UINT8 (ternary_u8.h) que la capa acepta como 'x'
     * (0 = no lo acepta), ademas del dtype que declara input_dtype(): lo decide el contenedor al conectar una capa cuya activacion de salida es uint8
     * (ternary_stack_add rechaza la conexion si esta no la acepta). Una capa que lo acepta debe mirar x->dtype en
     * forward/backward. */
    size_t (*accepts_u8)(const void *self);
    /* OPCIONAL (NULL = la salida entera es UNA fila). Largo de cada fila (token) de la salida: la activacion uint8
     * cuantiza cada fila con su propia escala y zero-point. output_len() debe ser multiplo de este valor. */
    size_t (*output_row_len)(const void *self);

    /* Persistencia en el formato model.ternary (ver ternary_tfile.h): lo que es un tensor (pesos, escalas, contadores R2) va
     * a los escritores de 'cx' con nombres "<prefix>.<hoja>", y el resto de la capa (dimensiones, hiperparametros) se
     * escribe como miembros del objeto JSON "config" que el llamador ya abrio en 'cfg'. Con cx->state != NULL se
     * guarda tambien el estado de entrenamiento (checkpoint).
     *   export_layer: 1 en exito.
     *   import_layer: reconstruye la capa desde 'cfg' (el objeto "config") y los tensores 'prefix.*'; NULL si falta
     *                 algo o no cuadra (el motivo queda en cx->err). */
    int   (*export_layer)(const void *self, const TernaryExportCtx *cx, const char *prefix, TernaryJBuf *cfg);
    void *(*import_layer)(const TernaryImportCtx *cx, const char *prefix, const TernaryJNode *cfg);
} TernaryLayerVTable;

typedef struct {
    void *impl;                     /* TernaryDense*, TernaryConv2D*, ... */
    const TernaryLayerVTable *vt;
} TernaryLayer;

/* Envuelven una capa YA CREADA (alloc'd por el llamador) -- el
 * TernaryStack que la reciba se vuelve dueno: la libera via
 * vt->destroy() en ternary_stack_free/ternary_stack_remove, nunca hace
 * falta (ni corresponde) llamar ternary_dense_free/ternary_conv2d_free
 * a mano despues de envolverla. */
TernaryLayer ternary_layer_from_dense(TernaryDense *layer);
TernaryLayer ternary_layer_from_conv2d(TernaryConv2D *layer);
TernaryLayer ternary_layer_from_embedding(TernaryEmbedding *layer);
TernaryLayer ternary_layer_from_gqa(TernaryGQA *layer);
TernaryLayer ternary_layer_from_rmsnorm(TernaryRMSNorm *layer);
TernaryLayer ternary_layer_from_swiglu(TernarySwiGLU *layer);
TernaryLayer ternary_layer_from_head(TernaryHead *layer);

/* ===================================================================
 * Registro de tipos de capa
 *
 * UNA tabla (ternary_layer.c, TERNARY_LAYER_TYPES) describe cada tipo de
 * capa: su nombre, sus parametros de construccion (nombre, tipo, valor
 * por defecto, minimo), como construirla con pesos iniciales, como
 * describirla de vuelta y su vtable. De esa misma tabla salen:
 *   - ternary_layer_import_by_type() (cargar un modelo), y
 *   - la fachada Python (ternary_simple.h), que arma tn.Dense,
 *     tn.Conv2D, ... a partir de ella sin una linea de Python por capa.
 * Agregar un tipo de capa nuevo = escribir su vtable (arriba) + su
 * build/describe + UNA entrada en la tabla.
 *
 * Los valores de los parametros viajan como un arreglo plano de double
 * en el orden de la tabla; un parametro PAIR ocupa 2 posiciones.
 * =================================================================== */
typedef enum {
    TERNARY_PARAM_INT = 0,
    TERNARY_PARAM_FLOAT = 1,
    TERNARY_PARAM_BOOL = 2,
    TERNARY_PARAM_PAIR = 3   /* dos enteros (p.ej. alto x ancho de un kernel) */
} TernaryParamKind;

#define TERNARY_MAX_SHAPE_DIMS 4     /* dims maximas de una forma (entrada/salida de capa) */
#define TERNARY_MAX_LAYER_SLOTS 16   /* posiciones maximas de parametros de una capa */

typedef struct {
    const char *name;
    TernaryParamKind kind;
    int required;           /* 1: sin valor por defecto */
    double default_value;   /* ignorado si required (en PAIR vale para ambos) */
    int has_min;
    double min_value;
    const char *doc;
    int has_max;            /* opcional: al final para que los inicializadores sin maximo sigan valiendo */
    double max_value;
} TernaryParamSpec;

typedef struct {
    const char *type_name;   /* igual a vt->type_name: es lo que se escribe en el archivo */
    const char *class_name;  /* nombre de la clase publica ("Dense", "Conv2D") */
    const char *doc;
    const TernaryParamSpec *params;
    size_t num_params;
    const TernaryLayerVTable *vt;
    /* Crea la capa para una entrada de forma in_shape y la inicializa
     * con pesos de 'rng'. params: ya validados contra 'params' de
     * arriba. Devuelve 1 y llena *out, o 0 y un mensaje en err. */
    int (*build)(const size_t *in_shape, size_t in_ndims, const double *params,
                 TernaryRng *rng, TernaryLayer *out, char *err, size_t err_cap);
    /* Inversa de build: parametros reales de una capa ya creada (mismo
     * orden de posiciones) y su forma de salida (hasta
     * TERNARY_MAX_SHAPE_DIMS dims; devuelve ndims). */
    size_t (*describe)(const void *impl, double *params_out, size_t *out_shape);
} TernaryLayerType;

size_t ternary_layer_type_count(void);
const TernaryLayerType *ternary_layer_type_at(size_t i);          /* NULL si fuera de rango */
const TernaryLayerType *ternary_layer_type_by_name(const char *type_name); /* NULL si no existe */
/* Cantidad de posiciones que ocupan los parametros de 'type' en el arreglo plano. */
size_t ternary_layer_type_num_slots(const TernaryLayerType *type);
/* Valida 'params' (num_slots valores) contra la tabla del tipo (enteros
 * enteros, minimos, booleanos 0/1) y construye la capa. Devuelve 1 en
 * exito; 0 con mensaje en err si los parametros o la forma no sirven. */
int ternary_layer_type_build(const TernaryLayerType *type, const size_t *in_shape, size_t in_ndims,
                             const double *params, size_t num_slots, TernaryRng *rng,
                             TernaryLayer *out, char *err, size_t err_cap);

/* Escribe {"type":..,"activation":..,"config":{..}} (y los tensores 'prefix.*') de UNA capa. 1 en exito. */
int ternary_layer_export_entry(TernaryLayer l, const char *act_name, const TernaryExportCtx *cx, const char *prefix,
                               TernaryJBuf *out);
/* Inversa: lee una entrada {"type","activation","config"} y reconstruye la capa y su activacion. 1 en exito;
 * 0 con el motivo en cx->err. */
int ternary_layer_import_entry(const TernaryImportCtx *cx, const char *prefix, const TernaryJNode *entry,
                               TernaryLayer *layer_out, const TernaryActivation **act_out);
/* Reconoce 'type_name' (tabla publica + contenedores) y reconstruye la capa con vt->import_layer. */
TernaryLayer ternary_layer_import_by_type(const char *type_name, const TernaryImportCtx *cx, const char *prefix,
                                          const TernaryJNode *cfg, int *ok);

#endif /* TERNARY_LAYER_H */
