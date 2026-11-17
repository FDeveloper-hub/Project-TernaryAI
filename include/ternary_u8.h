#ifndef TERNARY_U8_H
#define TERNARY_U8_H

#include <stddef.h>
#include <stdint.h>

/*
 * Activaciones uint8 con escala y punto cero (zero-point) POR FILA (por token).
 *
 * Una fila de 'row_len' valores float x se guarda como bytes q en 0..255 mas una escala float 'scale' y un
 * zero-point entero 'zp' en 0..255:
 *
 *     x_i  ~=  scale * (q_i - zp)
 *
 * El rango de la fila siempre incluye el 0 (lo = min(min_x, 0), hi = max(max_x, 0)), de modo que el valor 0 se
 * representa EXACTO (q = zp): los ceros (relleno, SiLU apagada...) no meten sesgo. Los valores negativos viven
 * por debajo del zero-point; no hay ninguna perdida de signo.
 *
 * Producto con una fila de pesos ternarios (mask/sign de ternary_core.h) -- sin multiplicaciones:
 *
 *     sum_i w_i * (q_i - zp)  =  sum_{w_i=+1} q_i  -  sum_{w_i=-1} q_i  -  zp * (n_pos - n_neg)
 *
 * es un ENTERO exacto (n_pos/n_neg son popcounts de la propia fila de pesos). El backward usa el estimador
 * directo: el gradiente atraviesa la cuantizacion como si fuera la identidad.
 */
typedef struct {
    uint8_t *q;        /* rows * row_len */
    float *scale;      /* rows */
    int32_t *zp;       /* rows, cada uno en 0..255 */
    size_t rows, row_len;
} TernaryU8;

/* NULL si rows o row_len son 0 o no hay memoria. Todo arranca en 0 (escala 1, zp 0). */
TernaryU8 *ternary_u8_alloc(size_t rows, size_t row_len);
void ternary_u8_free(TernaryU8 *t);

/* Cuantiza x (rows*row_len floats) fila por fila. Una fila de ceros queda con escala 1 y zp 0. */
void ternary_u8_quantize(TernaryU8 *dst, const float *x);

/* Valores representados: out[r*row_len+i] = scale[r] * (q - zp[r]). */
void ternary_u8_dequant(const TernaryU8 *t, float *out);

/* Producto entero exacto de una fila de pesos ternarios (n bits en 'wm'/'ws', bits sobrantes de la ultima palabra
 * en 0) con una fila q de n bytes y su zero-point: sum_i w_i * (q_i - zp). Escalar. */
int32_t ternary_dot_u8_scalar(const uint64_t *wm, const uint64_t *ws, size_t n, const uint8_t *q, int32_t zp);

/* Referencia lenta elemento a elemento (para las pruebas): w[i] en {-1,0,1}. */
int32_t ternary_dot_u8_reference(const int8_t *w, size_t n, const uint8_t *q, int32_t zp);

#endif /* TERNARY_U8_H */
