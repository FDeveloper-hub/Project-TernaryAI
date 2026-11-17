#ifndef TERNARY_TENSOR_H
#define TERNARY_TENSOR_H
#include "ternary_core.h"
#include "ternary_u8.h"
#include <stddef.h>
#include <stdint.h>
/*
 * Tensor generico que atraviesa la interfaz de capa (ver ternary_layer.h)
 * en lugar del TernaryVec fijo de antes. Nace de un problema concreto:
 * Embeddings no hace un producto punto (como Dense/Conv2D), hace una
 * BUSQUEDA -- su entrada natural es un id entero, no un vector ternario
 * empaquetado. Forzarla a la forma vieja (one-hot de ancho=vocab_size)
 * funcionaria matematicamente pero haria O(vocab_size) trabajo por una
 * operacion que es O(1) (copiar una fila).
 *
 * Alcance DELIBERADAMENTE acotado (decision de diseno): forma con
 * N dimensiones (N decidido en tiempo de ejecucion, ver 'dims' abajo) y
 * 3 tipos de dato -- FLOAT, ID, TERNARIO. Esto cubre Embeddings, GQA y
 * SwiGLU (todo lo que hay en el horizonte real); NO es un motor de
 * tensores de proposito general (sin grafos con ramas, sin tipos
 * definidos por el usuario, sin difusion/broadcasting).
 *
 * Dimensiones REALMENTE dinamicas (decision explicita de diseno, con el
 * costo conocido): 'dims' se reserva con malloc del tamano exacto que
 * hace falta, sin ningun limite fijo -- a diferencia de TernaryVec (que
 * sigue siendo un arreglo de bits de ancho fijo, sin cambios) esto SI
 * paga una asignacion de memoria por tensor creado. Por eso el
 * contenedor (ternary_stack.c) crea estos tensores UNA vez por capa al
 * armar la red, nunca dentro del forward/backward de cada muestra --
 * la forma de una capa no cambia entre muestras, asi que no hay
 * necesidad real de recrear el tensor en el camino caliente, solo de
 * actualizar que buffer de datos apunta.
 */

typedef enum {
    TERNARY_DTYPE_FLOAT   = 0,  /* float continuo -- salida cruda de cualquier capa antes de cuantizar */
    TERNARY_DTYPE_ID      = 1,  /* enteros (ids de token/indice) -- solo como entrada, nunca tiene gradiente */
    TERNARY_DTYPE_TERNARY = 2,  /* empaquetado {-1,0,1} de ternary_core.h (TernaryVec) */
    TERNARY_DTYPE_UINT8   = 3,  /* bytes con escala y zero-point por fila (ternary_u8.h) -- solo como VISTA de una activacion */
} TernaryDType;

typedef struct {
    size_t *dims;       /* malloc'd, 'ndims' elementos; NULL si ndims==0 */
    size_t ndims;
    size_t numel;        /* producto de dims, cacheado; 0 si ndims==0 */
    TernaryDType dtype;

    /* Buffer de datos: en un tensor "dueno" de su memoria (ver
     * ternary_tensor_alloc) exactamente UNO de estos tres es no-NULL,
     * segun dtype, y le pertenece (ternary_tensor_free lo libera). En un
     * tensor "vista" (ver ternary_tensor_view_*) el puntero es prestado
     * -- apunta al buffer real de otra estructura (p.ej. el act_out de
     * una capa) y ternary_tensor_free() de una vista NUNCA libera datos,
     * solo 'dims'. El campo 'owns_data' distingue los dos casos. */
    float *data_f;
    int32_t *data_i;
    TernaryVec *data_t;
    TernaryU8 *data_u8;  /* dtype UINT8 (siempre prestado) */
    int owns_data;
} TernaryTensor;

/* Crea un tensor DUEÑO de su buffer de datos (reservado en 0 para
 * FLOAT/ID; TERNARY se reserva vacio via ternary_vec_alloc, ver
 * ternary_core.h). NULL si no hay memoria. */
TernaryTensor *ternary_tensor_alloc(const size_t *dims, size_t ndims, TernaryDType dtype);

/* Crea una VISTA sobre un buffer ya existente (no reserva ni copia
 * datos, no libera 'data' en ternary_tensor_free) -- para que el
 * contenedor le pase a una capa el buffer de la capa anterior sin
 * copiarlo. Exactamente uno de data_f/data_i/data_t debe ser el que
 * corresponda a 'dtype'; los otros dos se ignoran. */
TernaryTensor *ternary_tensor_view_float(const size_t *dims, size_t ndims, float *data);
TernaryTensor *ternary_tensor_view_id(const size_t *dims, size_t ndims, int32_t *data);
TernaryTensor *ternary_tensor_view_ternary(const size_t *dims, size_t ndims, TernaryVec *data);
TernaryTensor *ternary_tensor_view_u8(const size_t *dims, size_t ndims, TernaryU8 *data);

/* Reapunta una vista YA CREADA a otro buffer del mismo dtype, sin tocar
 * 'dims'/'ndims'/'numel' -- para el camino caliente: el contenedor crea
 * la vista una vez por capa y en cada forward solo actualiza a que
 * buffer apunta (nunca vuelve a hacer malloc de 'dims'). */
void ternary_tensor_rebind_float(TernaryTensor *t, float *data);
void ternary_tensor_rebind_id(TernaryTensor *t, int32_t *data);
void ternary_tensor_rebind_ternary(TernaryTensor *t, TernaryVec *data);
void ternary_tensor_rebind_u8(TernaryTensor *t, TernaryU8 *data);

void ternary_tensor_free(TernaryTensor *t);

#endif /* TERNARY_TENSOR_H */
