#include "ternary_mem.h"
#include "ternary_image.h"
#include <stdlib.h>

TernaryVec *ternary_encode_image_u8(const uint8_t *pixels,
                                     size_t height, size_t width, size_t channels,
                                     int center, int dead_zone) {
    size_t n = height * width * channels;
    TernaryVec *v = ternary_vec_alloc(n);
    if (!v) return NULL;

    for (size_t i = 0; i < n; i++) {
        int centrado = (int)pixels[i] - center;
        int valor;
        if (centrado > dead_zone)       valor = 1;
        else if (centrado < -dead_zone) valor = -1;
        else                            valor = 0;
        ternary_vec_set(v, i, valor);
    }
    return v;
}
