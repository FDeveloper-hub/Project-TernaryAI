#ifndef TERNARY_IMAGE_H
#define TERNARY_IMAGE_H

#include "ternary_core.h"
#include <stdint.h>

/*
 * Convierte una imagen de pixeles uint8 (0..255), en layout HWC
 * (fila-mayor: indice = (h*width + w)*channels + c), a un TernaryVec de
 * longitud height*width*channels.
 *
 * Regla de cuantizacion (zona muerta simetrica alrededor de un punto
 * central, tipicamente 128 para 8 bits sin signo):
 *   valor_centrado = (int)pixel - center
 *   > +dead_zone   -> ternario 1
 *   < -dead_zone   -> ternario -1
 *   en el medio    -> ternario 0
 *
 * dead_zone=0 da una cuantizacion binaria pura (sign), sin ceros -- util
 * si se prefiere no perder informacion en la zona intermedia. dead_zone
 * mayor produce imagenes mas dispersas (mas ceros), lo cual ademas
 * acelera el forward (menos bits activos en el AND del popcount).
 */
TernaryVec *ternary_encode_image_u8(const uint8_t *pixels,
                                     size_t height, size_t width, size_t channels,
                                     int center, int dead_zone);

/* Indice plano HWC, expuesto para que capas como Conv2D puedan direccionar
 * la imagen sin duplicar esta cuenta en cada sitio. */
static inline size_t ternary_image_index(size_t h, size_t w, size_t c,
                                          size_t width, size_t channels) {
    return (h * width + w) * channels + c;
}

#endif /* TERNARY_IMAGE_H */
