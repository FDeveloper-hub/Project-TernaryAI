#ifndef TERNARY_SWIGLU_H
#define TERNARY_SWIGLU_H
#include "ternary_core.h"
#include "ternary_dense.h"
#include "ternary_u8.h"
#include <stddef.h>
#include <stdint.h>
/*
 * SwiGLU ternario (la red feed-forward de un bloque transformer). Tres proyecciones con votos propios
 * (reutilizan TernaryDense tal cual, como GQA): W_gate (hidden x dim), W_up (hidden x dim) y
 * W_down (dim x hidden), todas sin bias. Por cada posicion (fila):
 *
 *     h_j = compuerta(z_gate_j) * up_j         j = 0..hidden-1
 *     t   = cuantizar_con_banda(h)             {-1,0,+1}
 *     out = escala_down * (W_down . t)
 *
 * El tramo intermedio es una PIEZA INTERCAMBIABLE (gate_mode), para medir una contra otra con datos:
 *
 *  - TERNARY_SWIGLU_GATE_SILU (opcion 1): compuerta = silu(escala_gate * z_gate), con z_gate ENTERO crudo
 *    en [-dim, dim]: silu sale de una TABLA de 2*dim+1 valores (misma idea que la tabla de exp del softmax,
 *    sin expf en el camino caliente). up = escala_up * z_up. El producto h es float, y solo vive dentro de
 *    la capa: se ternariza antes de W_down.
 *  - TERNARY_SWIGLU_GATE_SIGN (opcion 2): compuerta = sign(z_gate) en {-1,0,+1}; h = sign(z_gate) * z_up,
 *    todo entero. Se pierde la modulacion suave de silu (cuanto cuesta eso es justo lo que se mide).
 *  - TERNARY_SWIGLU_GATE_RELU2 (opcion 3, estilo BitNet, SIN compuerta): h_j = relu(escala_up * z_up_j)^2.
 *    Solo DOS matrices (W_up y W_down; wg == NULL, sin su memoria ni su calculo): un tercio menos de pesos y de
 *    multiplicaciones en el feed-forward a igual 'hidden'. Con z entero crudo igual que las otras. La escala de up
 *    entra al cuadrado en h, pero como h se cuantiza por fila (banda relativa / uint8 con escala por fila) es un factor
 *    comun: no cambia t y solo se mueve junto con la de down, asi que tampoco se aprende. NO es el default (el
 *    default sigue siendo silu, GATE_SILU = 0). Ojo con los indices de matriz: con esta opcion num_matrices = 2 y
 *    m = 0 es W_up, m = 1 es W_down (grad_w_out[0..1], grad_scale_out[0..1]).
 *
 * Cuantizacion con banda de ceros (por fila): t_j = sign(h_j) si |h_j| > band * mean_j|h_j|, si no 0.
 * El umbral es RELATIVO a la fila, asi que un factor positivo comun a todo h no cambia t: por eso la
 * escala de up nunca influye en la salida (ni, en la opcion 2, la de gate) -- esas escalas no reciben
 * gradiente, igual que Wq/Wk/Wv en GQA. Solo cuentan escala_down (siempre) y escala_gate (solo opcion 1,
 * porque deforma silu).
 *
 * Gradiente: exacto en W_down, en el producto y en silu (tabla de silu'); la cuantizacion con banda NO se
 * deriva (estimador directo, igual que cada sign() del proyecto). En la opcion 2 el sign() de la compuerta
 * tambien pasa de largo (dh/dz_gate := up, estimador directo).
 *
 * Entrada: UN TernaryVec de rows*dim (todas las posiciones concatenadas, igual que GQA/RMSNorm). Salida:
 * rows*dim floats. La conexion residual NO esta aqui: es del bloque.
 */
#define TERNARY_SWIGLU_GATE_SILU 0
#define TERNARY_SWIGLU_GATE_SIGN 1
#define TERNARY_SWIGLU_GATE_RELU2 2   /* sin compuerta: wg == NULL, 2 matrices */
#define TERNARY_SWIGLU_MAX_DIM 65536

typedef struct {
    size_t rows, dim, hidden;
    int gate_mode;
    float band;                     /* >= 0; 0 = sin banda (solo los h exactamente 0 dan 0) */

    TernaryDense *wg;               /* dim -> hidden (NULL con GATE_RELU2) */
    TernaryDense *wu;               /* dim -> hidden */
    TernaryDense *wd;               /* hidden -> dim */

    /* Tablas de silu y de su derivada sobre el entero crudo z de la compuerta (indice z + dim), para la
     * escala de gate vigente. Se reconstruyen al cambiar esa escala (ver ternary_swiglu_refresh_tables). */
    float *silu_tab, *dsilu_tab;    /* 2*dim + 1 */

    /* --- scratch reservado UNA vez --- */
    size_t chunk;                   /* posiciones por bloque de Dense (min(rows, 48)) */
    TernaryVec **xc;                /* chunk vectores de dim: las posiciones del bloque actual */
    /* Estado que crece con rows (se reparte de UNA region contigua, ver ternary_swiglu_bind_state): */
    TernaryVec **tpos;              /* rows vectores de hidden: t de cada posicion, para el backward */
    int32_t *zg, *zu, *zd;          /* rows*hidden, rows*hidden, rows*dim (Z entero crudo) */
    void *tp_raw;                   /* tpos + TernaryVec de cada posicion (punteros: fuera del espacio compartido) */
    unsigned char *state_raw;       /* region propia (NULL si el estado vive en un espacio de trabajo ajeno) */
    float *tmp;                     /* max(hidden, dim): salida escalada que el forward de Dense exige y aqui no se usa */
    float *h;                       /* hidden */
    float *f_a, *f_u;               /* hidden cada uno: a = escala_gate*sx*z_gate y u = escala_up*sx*z_up (ruta uint8) */
    uint64_t zq_zeros, zq_total;    /* contador de sparsity (ruta uint8): valores de hq == zero-point / valores vistos; no afecta a ningun calculo */
    TernaryU8 hq;                   /* h cuantizado a uint8 por posicion (rows x hidden), entrada de W_down en la ruta uint8; vive en el estado */
    float *grad_t, *grad_a, *grad_u;   /* chunk*hidden cada uno */
    float *grad_xc;                 /* chunk*dim */
    /* vistas (punteros por posicion del bloque actual) */
    const TernaryVec **xcv, **tcv;  /* chunk */
    const float **gv, **gav, **guv; /* chunk */
    const int32_t **zgv, **zuv, **zdv; /* chunk */
    float **gtv, **gxv, **ov;       /* chunk */
    int32_t **rv;                   /* chunk */
    /* Sin buffers de gradiente del tamano de las matrices: el backward acumula directo en grad_w_out. */
} TernarySwiGLU;

/* NULL si alguna dimension es 0 o > TERNARY_SWIGLU_MAX_DIM, gate_mode invalido, band < 0 o sin memoria.
 * Los pesos arrancan en 0: el llamador los sortea (ver swiglu_build en ternary_layer.c). */
TernarySwiGLU *ternary_swiglu_alloc(size_t rows, size_t dim, size_t hidden, int gate_mode, float band);
void ternary_swiglu_free(TernarySwiGLU *s);
/* Bytes del estado que crece con rows (Z de gate/up/down y los t por posicion) y su reubicacion en una region ajena
 * (checkpoint por bloque, ver TernaryLayerVTable.state_bytes). */
size_t ternary_swiglu_state_bytes(const TernarySwiGLU *s);
void ternary_swiglu_bind_state(TernarySwiGLU *s, void *base);

/* Reconstruye las tablas de silu para la escala de gate actual. Hay que llamarla tras cambiar
 * s->wg->scale a mano (apply_scale_gradient y load ya lo hacen). */
void ternary_swiglu_refresh_tables(TernarySwiGLU *s);   /* no hace nada con GATE_RELU2 (no hay compuerta) */

/* Cantidad de matrices de pesos de la capa: 3 (gate, up, down) o 2 con GATE_RELU2 (up, down). */
size_t ternary_swiglu_num_matrices(const TernarySwiGLU *s);
/* Matriz m en el orden de num_matrices (ver arriba). NULL si m esta fuera de rango. */
TernaryDense *ternary_swiglu_matrix(const TernarySwiGLU *s, size_t m);

/* x: rows*dim ternarios. out: rows*dim floats. out_raw_opt (opcional): Z entero crudo de W_down. */
void ternary_swiglu_forward(const TernarySwiGLU *s, const TernaryVec *x, float *out, int32_t *out_raw_opt);

/* ---- Ruta uint8 (ver ternary_u8.h) ----------------------------------------------------------------------------
 * x: rows filas de dim bytes, cada una con su escala sx y zero-point. Por posicion:
 *     a = escala_gate * sx * Zg,  u = escala_up * sx * Zu       (Zg, Zu enteros exactos)
 *     h = silu(a) * u   (opcion 1)   |   h = sign(Zg) * u   (opcion 2)
 *     hq = uint8(h)  (una escala y un zero-point por posicion, cero exacto)
 *     out = escala_down * sx_h * (W_down . (hq - zp_h)).
 * silu se evalua con la exponencial rapida compartida (ternary_fexp.h), la misma en escalar y en AVX-512.
 * El backward usa el estimador directo en la cuantizacion de h; la escala de up no se aprende (solo se mueve con
 * la de down: es redundante con ella). grad_w_out[0..2] se SOBREESCRIBEN; grad_x_out y grad_scale_out se ACUMULAN. */
void ternary_swiglu_forward_u8(const TernarySwiGLU *s, const TernaryU8 *x, float *out, int32_t *out_raw_opt);
void ternary_swiglu_backward_u8(TernarySwiGLU *s, const TernaryU8 *x, const float *grad_out,
                                float *const *grad_w_out, float *grad_x_out, float *grad_scale_out);

/* Sparsity de la activacion cuantizada h (entrada de W_down): acumula, en cada forward uint8, cuantos valores de hq caen
 * exactamente en el zero-point de su fila (h cuantizado a cero). Con ReLU² son los h <= 0; con silu, los h que redondean
 * a cero. Solo cuenta; los resultados no cambian. */
void ternary_swiglu_zero_stats(const TernarySwiGLU *s, uint64_t *zeros, uint64_t *total, int reset);

/* Producto intermedio h (antes de cuantizar) de la fila 'row' -- hidden floats. Expuesto para verificar
 * silu y su derivada por diferencias finitas (la cuantizacion posterior no es derivable). */
void ternary_swiglu_product(const TernarySwiGLU *s, const TernaryVec *x, size_t row, float *h_out);

/* grad_out: rows*dim. grad_w_out[0..2] = W_gate, W_up, W_down (mismo orden que num_matrices), cada uno
 * SOBREESCRITO con el total de la muestra (con GATE_RELU2 solo [0]=W_up y [1]=W_down). grad_x_out: rows*dim, se ACUMULA. grad_scale_out[0..2] se
 * ACUMULAN (la de up queda siempre en 0, la de gate solo en la opcion 1). Requiere el forward inmediato. */
void ternary_swiglu_backward(TernarySwiGLU *s, const TernaryVec *x, const float *grad_out,
                              float *const *grad_w_out, float *grad_x_out, float *grad_scale_out);

#endif /* TERNARY_SWIGLU_H */
