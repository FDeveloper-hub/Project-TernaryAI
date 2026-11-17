/* Nucleos NEON de 128 bits: planos ternarios (mask/sign): gradw, gradx, pack_thr, sum_abs, bwd1_row, unpack_row, expand_trits, pack_sign_i32.
 * Interfaz y reglas: include/ternary_neon.h. Los esqueletos para compilaciones sin NEON estan en ternary_neon_stubs.c. */
#include "ternary_neon.h"
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON) && defined(__GNUC__)
#include <arm_neon.h>

/* (sin implementar) */

#endif
