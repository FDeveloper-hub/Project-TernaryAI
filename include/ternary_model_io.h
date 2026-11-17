#ifndef TERNARY_MODEL_IO_H
#define TERNARY_MODEL_IO_H

#include "ternary_core.h"
#include <stdint.h>
#include <stddef.h>

/*
 * Empaquetado ternario al MINIMO de bits, para el export final de un
 * modelo (full_checkpoint=0 en TernaryLayerVTable::save/load) -- requisito
 * de diseno: no se empaca a 2 bits, se juntan todos los
 * pesos en 1.58 bits, al final se usa el minimo de bits necesarios.
 *
 * Un peso ternario tiene 3 estados (-1,0,1): log2(3) = 1.58496...
 * bits/peso es el limite de entropia. El empaquetado de 2 bits/peso que
 * usa el resto del proyecto (mask+sign, un bit cada uno) es el que
 * conviene para ENTRENAR y para los kernels CPU/GPU (permite operar con
 * AND/XOR/popcount directo, sin decodificar), pero desperdicia 0.415
 * bits/peso -- 21% de mas -- para GUARDAR el modelo ya entrenado.
 *
 * La tecnica (estandar para empaquetar valores base-3): agrupar 5 pesos
 * por byte. 3^5 = 243 <= 255, entra justo en un byte sin desperdiciar
 * un bit entero: byte = t0 + 3*t1 + 9*t2 + 27*t3 + 81*t4, con cada
 * ti en {0,1,2} (mapeado de {-1,0,1} sumando 1). Esto da 8 bits/5 pesos
 * = 1.6 bits/peso -- a 0.015 bits/peso del limite teorico, y MUY por
 * debajo de los 2 bits/peso de siempre (20% menos de tamano de archivo
 * en los pesos, que son la enorme mayoria de los bytes de un modelo
 * grande).
 *
 * El ultimo grupo, si 'n' no es multiplo de 5, se completa con ceros
 * (peso ternario 0) que ternary_unpack_trits() descarta solo porque
 * conoce el 'n' real (nunca se guarda relleno de mas, ni hace falta:
 * el 'n' exacto ya esta en el header de la capa, escrito aparte).
 */

/* Bytes que ocupan 'n' pesos ternarios empaquetados de a 5 por byte. */
static inline size_t ternary_packed_trits_bytes(size_t n) {
    return (n + 4) / 5;
}

/* Empaqueta los primeros 'n' valores de 'v' (leidos con ternary_vec_get,
 * cada uno debe ser -1, 0 o 1) en 'out_bytes', que debe tener espacio
 * para AL MENOS ternary_packed_trits_bytes(n) bytes. Devuelve la
 * cantidad de bytes escritos. */
size_t ternary_pack_trits(const TernaryVec *v, size_t n, uint8_t *out_bytes);

/* Contraparte exacta: desempaqueta 'n' valores desde 'in_bytes' (que
 * debe tener al menos ternary_packed_trits_bytes(n) bytes validos, ya
 * leidos del archivo) hacia 'out', que debe estar reservado con
 * capacidad >= n. Cualquier byte de entrada fuera de rango 0..242
 * (corrupcion de archivo) se trata como error: devuelve 0 y deja 'out'
 * sin terminar de llenar; 1 en exito. */
int ternary_unpack_trits(const uint8_t *in_bytes, size_t n, TernaryVec *out);

#endif /* TERNARY_MODEL_IO_H */
