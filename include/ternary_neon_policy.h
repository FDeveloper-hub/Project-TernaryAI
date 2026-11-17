#ifndef TERNARY_NEON_POLICY_H
#define TERNARY_NEON_POLICY_H
/* Politica de uso de los nucleos NEON: orden de despacho AVX-512 -> NEON -> escalar. Cada grupo tiene un interruptor
 * (1 = usar NEON cuando la CPU lo tiene, 0 = camino escalar). Vectorizar solo vale si es MAS RAPIDO que el escalar:
 * los valores se fijan con `make tests/bench_neon` ejecutado en el dispositivo (ver README, seccion NEON). Se puede
 * cambiar sin editar el codigo con -DTERNARY_NEON_<GRUPO>=0 en CFLAGS. Con igualdad de bits, apagar un grupo no
 * cambia ningun resultado, solo la velocidad. */
#include "ternary_neon.h"
#include "ternary_avx512.h"

#ifndef TERNARY_NEON_ELEM
#define TERNARY_NEON_ELEM 1       /* add, acc_add2, scale, shift_round, sign, scale_i32, scatter, smear */
#endif
#ifndef TERNARY_NEON_QUANT
#define TERNARY_NEON_QUANT 1      /* cuantizador uint8 por fila */
#endif
#ifndef TERNARY_NEON_R2
#define TERNARY_NEON_R2 1         /* regla R2 */
#endif
#ifndef TERNARY_NEON_ELEMNORM
#define TERNARY_NEON_ELEMNORM 1   /* normalizacion por elemento */
#endif
#ifndef TERNARY_NEON_ATTN
#define TERNARY_NEON_ATTN 1       /* atencion flash, softmax entero y RoPE */
#endif
#ifndef TERNARY_NEON_RMS
#define TERNARY_NEON_RMS 1        /* RMSNorm */
#endif
#ifndef TERNARY_NEON_SWIGLU
#define TERNARY_NEON_SWIGLU 1     /* SwiGLU / ReLU2 */
#endif
#ifndef TERNARY_NEON_HEAD
#define TERNARY_NEON_HEAD 1       /* entropia cruzada y dot_gz de la cabeza */
#endif

#ifndef TERNARY_NEON_SCAN
#define TERNARY_NEON_SCAN 0       /* busqueda de marcadores especiales (ternary_targets): APAGADO por defecto: en el dispositivo no compensa con marcadores muy juntos, y solo se usa al preparar un dataset (despreciable). Se activa con -DTERNARY_NEON_SCAN=1 */
#endif

#define TN_ON(G) (TERNARY_NEON_##G && ternary_neon_enabled())
/* Algun nucleo vectorial (AVX-512 o NEON) para el grupo G. */
#define TN_VEC(G) (ternary_avx512_enabled() || TN_ON(G))
#endif
