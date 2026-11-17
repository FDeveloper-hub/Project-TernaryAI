#ifndef TERNARY_RESIDUAL_H
#define TERNARY_RESIDUAL_H
#include "ternary_layer.h"
#include "ternary_activation.h"
#include <stddef.h>
/*
 * Conexion residual como capa envoltorio: contiene una cadena de sub-capas f y calcula
 *
 *     out = x + f(x)
 *
 * por dentro, de modo que el TernaryStack la ve como UNA sola capa (sin tocar el contenedor). Es la forma
 * de cada sub-bloque pre-norm de un transformer:  x + GQA(banda(RMSNorm(x)))  y  x + SwiGLU(banda(RMSNorm(x))).
 *
 * El flujo residual es FLOAT: la capa recibe la salida escalada cruda de la capa anterior (dtype FLOAT) y
 * entrega x + f(x) tambien en float; el stack le aplica a ese resultado la activacion de la capa residual
 * (la que se pase al agregarla) antes de la capa siguiente.
 *
 * Reglas de la cadena interna (se validan al agregar, nunca en silencio):
 *  - la PRIMERA sub-capa debe tomar entrada FLOAT (p.ej. RMSNorm): recibe x tal cual;
 *  - cada sub-capa siguiente recibe la salida de la anterior, cuantizada por la activacion de esa
 *    sub-capa si pide TERNARY, o cruda si pide FLOAT -- igual que dentro de un TernaryStack;
 *  - TODAS deben propagar gradiente hacia su entrada (x recibe gradiente de la rama f ademas del atajo);
 *  - ninguna puede pedir ids como entrada;
 *  - la ULTIMA debe entregar exactamente el largo de x (se comprueba en ternary_layer_from_residual).
 *    Su activacion no se usa (se suma su salida cruda); se guarda igual por simetria con el stack.
 *
 * Backward: la rama f recibe dL/dout tal cual, y x recibe dL/dout (atajo) + dL/dx de la rama. Los
 * gradientes de TODAS las matrices internas se entregan al stack como matrices de esta capa, en orden
 * (las de la sub-capa 0, luego las de la 1...), asi que el optimizador del stack las actualiza igual
 * que a cualquier otra capa: no hay un camino de entrenamiento aparte.
 *
 * Guardado: cada sub-capa se guarda con su nombre de tipo y su activacion, igual que en un stack; la
 * carga las reconstruye por nombre (checkpoint retomable exacto, export empaquetado). Una capa residual
 * no se construye con parametros numericos del registro (contiene capas): se arma en C con
 * ternary_residual_create/_add, y se recupera por nombre ("residual") al cargar un modelo guardado.
 * Se permite anidar una residual dentro de otra.
 */
#define TERNARY_RESIDUAL_MAX_LAYERS 64

typedef struct TernaryResidual TernaryResidual;

TernaryResidual *ternary_residual_create(void);
/* Libera el envoltorio y TODAS las sub-capas agregadas (via vt->destroy). */
void ternary_residual_free(TernaryResidual *r);

/* Agrega una sub-capa YA CREADA con la activacion que se le aplica a SU salida antes de pasarla a la
 * siguiente. 1 en exito (la residual pasa a ser duena de 'layer'); 0 (con mensaje en stderr) si rompe
 * una regla de la lista de arriba -- en ese caso NO se agrega y el llamador sigue siendo dueno. */
int ternary_residual_add(TernaryResidual *r, TernaryLayer layer, TernaryActivation act);

size_t ternary_residual_num_layers(const TernaryResidual *r);
TernaryLayer ternary_residual_layer(const TernaryResidual *r, size_t i);
const TernaryActivation *ternary_residual_activation(const TernaryResidual *r, size_t i);

/* Capa para ternary_stack_add. Devuelve {NULL,NULL} (el llamador sigue siendo dueno de 'r') si la cadena
 * esta vacia o su ultima sub-capa no entrega el largo de la entrada. Tras ternary_stack_add exitoso la
 * dueña es el stack (no liberar 'r' a mano). */
TernaryLayer ternary_layer_from_residual(TernaryResidual *r);
const TernaryLayerVTable *ternary_residual_vtable(void);

#endif /* TERNARY_RESIDUAL_H */
