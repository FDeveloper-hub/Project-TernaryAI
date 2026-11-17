#ifndef TERNARY_ACTIVATION_H
#define TERNARY_ACTIVATION_H

#include "ternary_core.h"
#include "ternary_u8.h"
#include <stddef.h>

/*
 * Activacion entre capas -- TernaryStack la decide y la aplica sola (requisito de diseno: todo se
 * determina en el contenedor). Hoy hay UNA sola, uint8: cuantizacion afin por token (fila) a bytes 0..255 con
 * escala y zero-point; el 0 se representa exacto y los negativos viven por debajo del zero-point. Conserva la
 * magnitud con 255 niveles. Ver ternary_u8.h. (sign y band, que aplastaban cada valor a {-1,0,+1}, se
 * eliminaron: medido, tirar la magnitud era lo que mas costaba.)
 *
 * El contenedor la aplica SOLO cuando la capa siguiente lee la salida como TERNARY (es decir, como activacion
 * cuantizada): la ultima capa y las que alimentan a una capa FLOAT (p.ej. RMSNorm) no cuantizan nada.
 * Agregar otra activacion = escribir una funcion 'apply_u8' (o un 'apply' nuevo si produjera otro formato) y
 * sumarla a ternary_activation.c.
 *
 * 'name' se escribe tal cual en el archivo de modelo.
 */
typedef struct {
    const char *name;
    /* Cuantiza 'out' (salida cruda en float de una capa, dst->rows * dst->row_len valores) fila por fila en 'dst',
     * que el contenedor reserva con la forma de la salida de la capa. */
    void (*apply_u8)(const float *out, TernaryU8 *dst);
} TernaryActivation;

extern const TernaryActivation TERNARY_ACTIVATION_UINT8;

/* Busca una activacion por nombre (para ternary_stack_load). NULL si no existe: nunca se asume una por defecto en
 * silencio. En particular "sign" y "band" ya no existen. */
const TernaryActivation *ternary_activation_by_name(const char *name);

/* Lista de las activaciones nativas (para que la fachada Python las descubra sola, igual que el registro de capas). */
size_t ternary_activation_count(void);
const TernaryActivation *ternary_activation_at(size_t i); /* NULL si i fuera de rango */

#endif /* TERNARY_ACTIVATION_H */
