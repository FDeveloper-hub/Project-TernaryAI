#include "ternary_mem.h"
#include "ternary_activation.h"
#include <string.h>

static void uint8_apply_u8(const float *out, TernaryU8 *dst) { ternary_u8_quantize(dst, out); }
const TernaryActivation TERNARY_ACTIVATION_UINT8 = { "uint8", uint8_apply_u8 };

/* Tabla unica de activaciones nativas: agregar una activacion nueva es sumar una linea aca (ternary_activation_by_name
 * y la lista que ve la fachada Python salen de esta misma tabla). */
static const TernaryActivation *const ACTIVATIONS[] = { &TERNARY_ACTIVATION_UINT8 };

size_t ternary_activation_count(void) { return sizeof(ACTIVATIONS) / sizeof(ACTIVATIONS[0]); }

const TernaryActivation *ternary_activation_at(size_t i) {
    return (i < ternary_activation_count()) ? ACTIVATIONS[i] : NULL;
}

const TernaryActivation *ternary_activation_by_name(const char *name) {
    for (size_t i = 0; i < ternary_activation_count(); i++)
        if (strcmp(name, ACTIVATIONS[i]->name) == 0) return ACTIVATIONS[i];
    return NULL;
}
