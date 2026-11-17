#ifndef TERNARY_DENSE_H
#define TERNARY_DENSE_H

#include "ternary_core.h"
#include "ternary_u8.h"
#include "ternary_threadpool.h"
#include "ternary_dispatch.h"
#include <stdint.h>

/*
 * Capa densa.
 *
 * REDISEÑO (antes: output_dim TernaryVec independientes, uno por fila --
 * cada uno pagaba un minimo de 1 palabra de 64 bits POR PLANO aunque la
 * fila tuviera 2 pesos. Con capas angostas (tipico en la primera capa,
 * cerca de la entrada) eso dominaba el tamano real y violaba "sin
 * espacios vacios"). AHORA: una sola matriz ternaria (W) de longitud
 * output_dim*input_dim, fila mayor -- el peso (o,i) vive en el indice
 * o*input_dim+i. El relleno de la ultima palabra parcial pasa UNA vez
 * por capa entera, no una vez por fila.
 *
 * El costo de este cambio: extraer la fila 'o' para el producto punto ya
 * no es "leer las palabras tal cual" (pueden no empezar en un limite de
 * palabra) -- hay que desplazar bits para alinear la ventana. Ese trabajo
 * es proporcional al LARGO DE LA FILA (input_dim/64 palabras), no al
 * tamano de la capa entera, y sigue siendo por-palabra (bulk), no
 * por-bit. Ver ternary_dense.c: extract_row().
 *
 * Los contadores de R2 siguen sin empaquetar (son estado del optimizador, con rango
 * [-127,127], no bits ternarios) pero en UN arreglo plano output_dim*input_dim en
 * vez de un arreglo de punteros -- tambien elimina output_dim mallocs separados.
 */
typedef struct {
    size_t input_dim;
    size_t output_dim;

    TernaryVec *W;        /* matriz completa, longitud output_dim*input_dim */
    TernaryVec *bias;     /* longitud output_dim, NULL si no hay bias */
    int8_t *counter_W;      /* plano, longitud output_dim*input_dim */
    int8_t *counter_b;      /* longitud output_dim, NULL si no hay bias */

    /*
     * Escala estilo BitNet: UN float por capa (no por peso). Sin ella la
     * precision del modelo cae drasticamente (~35% medido en la version
     * Python) porque el rango entero crudo del dot product no tiene
     * relacion con el rango del target. Z_salida = scale * (dot + bias).
     * Update directo (sin inercia/votos): ya se verifico empiricamente
     * en Python que darle inercia a la escala baja la precision (46% vs
     * 91%) -- un solo numero se beneficia de reaccionar cada paso.
     */
    float scale;

    /* Buffer de trabajo reservado UNA vez (no en el hot path): ventana
     * de fila extraida antes del popcount. */
    TernaryVec *row_scratch;
} TernaryDense;

TernaryDense *ternary_dense_alloc(size_t input_dim, size_t output_dim,
                                   int use_bias);
void ternary_dense_free(TernaryDense *layer);

/* Acceso a un peso individual (o,i) -- envoltorio fino sobre
 * ternary_vec_get/set con el indice plano o*input_dim+i. */
int  ternary_dense_get_weight(const TernaryDense *layer, size_t o, size_t i);
void ternary_dense_set_weight(TernaryDense *layer, size_t o, size_t i, int value);

/* Forward: out[o] = scale * (ternary_dot(fila_o, x) + bias[o]).
 * out debe apuntar a un arreglo ya reservado de output_dim floats.
 * out_raw_opt (opcional, puede ser NULL): si se pasa, recibe el Z entero
 * crudo (antes de escalar) -- lo necesita el backward para el gradiente
 * de la escala. */
void ternary_dense_forward(const TernaryDense *layer, const TernaryVec *x,
                            float *out, int32_t *out_raw_opt);

void ternary_dense_forward_parallel(const TernaryDense *layer, const TernaryVec *x,
                                     float *out, int32_t *out_raw_opt, TernaryThreadPool *pool);

/*
 * Punto de entrada recomendado: forward escalar de la capa; usa
 * ternary_dense_forward_parallel() con caps->cpu_pool (creado una vez en
 * ternary_init()) -- forward_parallel trae su propio umbral interno para
 * capas chicas, asi que nunca es peor que llamar al serial a mano. */
void ternary_dense_forward_auto(const TernaryDense *layer, const TernaryVec *x,
                                 float *out, int32_t *out_raw_opt,
                                 const TernaryCapabilities *caps);

/*
 * Backward. grad_out[o] = dL/d(salida escalada) de esta capa (lo que
 * llega desde la capa siguiente o de la funcion de perdida).
 * Calcula:
 *   - grad_w_out[o*input_dim+i] = dL/dW[o][i]  (para apply_row_gradients)
 *   - grad_x_out[i]             += dL/dX[i]     (se ACUMULA, no se pisa)
 *   - *grad_scale_out           += dL/dscale     (se ACUMULA)
 *   - grad_bias_out[o]          += dL/dbias[o]   (se ACUMULA; NULL si no interesa)
 * El gradiente de PESOS y BIAS usa grad_out DIRECTO (sin multiplicar por
 * la escala). Solo grad_x_out (lo que se propaga a la capa anterior) se
 * multiplica por la escala. El bias se acumula igual que G_W pero sin
 * multiplicar por X (su "entrada" es siempre 1 implicita).
 */
void ternary_dense_backward(const TernaryDense *layer, const TernaryVec *x,
                             const float *grad_out, float *grad_w_out,
                             float *grad_x_out, float *grad_scale_out,
                             float *grad_bias_out,
                             const int32_t *z_raw_opt);

/* Igual que ternary_dense_backward, pero grad_w_out se ACUMULA (+=) en vez de pisarse.
 * REGLA PARA CAPAS NUEVAS: una capa que aplica un Dense a varias posiciones (o varias veces) por muestra
 * NO debe reservar un buffer propio del tamano de la matriz para sumar los resultados parciales: el
 * stack ya le entrega grad_w_out[m] (que ella pone en cero al empezar el backward) y esta funcion
 * acumula directo ahi. Un buffer de gradiente propio cuesta 4 bytes por peso, mas que el peso mismo
 * (0.25 bytes) y su voto (1 byte) juntos. */
/* Forward de un BLOQUE de S muestras: cada fila de W se lee una vez y se usa
 * para las S muestras (en vez de leer toda W S veces). Enteros: mismos bits
 * que S llamadas a ternary_dense_forward_auto. outs[s]: output_dim floats;
 * raws[s]: output_dim int32 (Z crudo). Si el motor elegido es GPU, o la fila
 * no cabe en el buffer de pila, cae a S llamadas de forward_auto. */
void ternary_dense_forward_block(const TernaryDense *layer, size_t S,
                                 const TernaryVec *const *xs, float *const *outs,
                                 int32_t *const *raws, const TernaryCapabilities *caps);

/* Backward de un BLOQUE de S muestras a la vez (entrenamiento por bloques).
 * Mismos bits que llamar S veces ternary_dense_backward_accumulate en
 * orden de muestra: cada elemento de grad_w se acumula en orden
 * ascendente de muestra, grad_x[s] en orden ascendente de fila, y
 * grad_scale/grad_bias en orden (muestra, fila) -- solo cambia el orden
 * de los BUCLES, no el de las sumas de ningun elemento. Ventajas: cada
 * fila de W se expande UNA vez por bloque (no una por muestra) y cada
 * fila de grad_w se lee/escribe UNA vez por bloque.
 *  xs[s]      entrada de la muestra s
 *  grad_out[s] dL/d(salida escalada), output_dim floats
 *  z_raw[s]   Z crudo del forward de la muestra s (obligatorio)
 *  grad_x[s]  se ACUMULA (+=); el llamador lo pone en cero antes. El
 *             arreglo entero puede ser NULL si no hace falta grad_x. */
void ternary_dense_backward_block(const TernaryDense *layer, size_t S,
                                  const TernaryVec *const *xs, const float *const *grad_out,
                                  const int32_t *const *z_raw,
                                  float *grad_w_out, float *const *grad_x,
                                  float *grad_scale_out, float *grad_bias_out);

void ternary_dense_backward_accumulate(const TernaryDense *layer, const TernaryVec *x,
                                        const float *grad_out, float *grad_w_out,
                                        float *grad_x_out, float *grad_scale_out,
                                        float *grad_bias_out,
                                        const int32_t *z_raw_opt);


/* ---- entrada uint8 (ver ternary_u8.h) -----------------------------------------------------------------------
 * x: UNA fila (rows == 1, row_len == input_dim) uint8 con escala sx y zero-point. Calculo (enteros, sin
 * multiplicaciones):  z_raw[o] = sum_i w[o][i]*(q_i - zp) + bias[o]   (int32, bias en unidades de sx)
 *                     out[o]   = scale * sx * z_raw[o].
 * out_raw_opt recibe z_raw. Escalar y en serie (la ruta de produccion con kernels AVX-512 viene aparte). */
void ternary_dense_forward_u8(const TernaryDense *layer, const TernaryU8 *x, float *out, int32_t *out_raw_opt);
/* Igual, sobre UNA fila q (input_dim bytes) con su escala sx y zero-point zp (para capas que aplican el Dense a
 * varias posiciones, como la cabeza). */
void ternary_dense_forward_u8_row(const TernaryDense *layer, const uint8_t *q, float sx, int32_t zp,
                                  float *out, int32_t *out_raw_opt);
void ternary_dense_backward_u8_row(const TernaryDense *layer, const uint8_t *q, float sx, int32_t zp,
                                   const float *grad_out, float *grad_w_out, float *grad_x_out,
                                   float *grad_scale_out, float *grad_bias_out, const int32_t *z_raw_opt);

/* Backward con entrada uint8 (estimador directo a traves de la cuantizacion). Mismas convenciones que
 * ternary_dense_backward_accumulate: grad_w_out[o*in+i] += g_o * xdq_i  con xdq_i = sx*(q_i - zp) (sin la escala de
 * la capa); grad_x_out[i] += scale * sum_o g_o*w[o][i] (sin sx: el estimador directo trata xdq como x);
 * *grad_scale_out += sum_o g_o * sx * z_raw[o]; grad_bias_out[o] += g_o * sx. Todo se ACUMULA. z_raw_opt = Z del
 * forward (NULL = se recalcula). */
/* Piezas del backward uint8 para repartir en hilos (ver ternary_dense_backward_u8_row). Seguras para hilos. */
void ternary_dense_u8_to_f32(const uint8_t *q, size_t n, float sx, int32_t zp, float *out);
void ternary_dense_u8_gradx(const TernaryDense *layer, const float *grad_out, float *grad_x_out);
/* Forward entero por lotes (AVX-512 + VNNI): mismos bits que ternary_dense_forward_u8_row por posicion. fw se crea una vez por llamada
 * del llamador (NULL sin VNNI: cae al camino por posicion) y es de solo lectura (compartible entre hilos). q: P filas de input_dim
 * bytes con paso qs; sx[p] y zp[p] por posicion; out y raw (cada uno puede ser NULL, no los dos) con pasos os y rs. */
typedef struct TernaryFwdW TernaryFwdW;
TernaryFwdW *ternary_dense_fwdw_create(const TernaryDense *layer);
void ternary_dense_fwdw_free(TernaryFwdW *fw);
void ternary_dense_forward_u8_batch(const TernaryDense *layer, const TernaryFwdW *fw, size_t P, const uint8_t *q, size_t qs,
                                    const float *sx, const int32_t *zp, float *out, size_t os, int32_t *raw, size_t rs);

/* grad_x de P posiciones a la vez (AVX-512): mismos bits que llamar P veces a ternary_dense_u8_gradx. Los planos se crean una
 * vez por backward (NULL si no hay AVX-512: entonces cae al camino por posicion) y se comparten entre hilos. */
typedef struct TernaryPlanes TernaryPlanes;
TernaryPlanes *ternary_dense_planes_create(const TernaryDense *layer);
void ternary_dense_planes_free(TernaryPlanes *pl);
void ternary_dense_u8_gradx_batch(const TernaryDense *layer, const TernaryPlanes *pl, size_t P, const float *grad_out,
                                  size_t go_stride, float *grad_x_out, size_t gx_stride);
void ternary_dense_u8_gradw(const TernaryDense *layer, size_t o0, size_t o1, size_t P,
                            const float *gout, size_t gout_stride, const float *xf, size_t xf_stride,
                            const float *sx, const int32_t *z, size_t z_stride, float *gw,
                            float *scale_terms, size_t terms_stride);
void ternary_dense_backward_u8(const TernaryDense *layer, const TernaryU8 *x, const float *grad_out,
                               float *grad_w_out, float *grad_x_out, float *grad_scale_out,
                               float *grad_bias_out, const int32_t *z_raw_opt);

/*
 * REGLA DE ACTUALIZACION DE PESOS: R2. Cada peso tiene un contador int8 (counter_W) que la regla mueve con el gradiente.
 * Por fila (unidad de salida) se normaliza el gradiente y se recorta,
 *        g_hat = clip(g / RMS_fila(g), -3, 3)
 * y el contador integra el sentido de descenso (int8, escala 127 = 1.0), casi sin decaimiento:
 *        c <- clamp(rint(beta*c - inc*127*g_hat), -127, 127)            (inc def. 0.03; beta ~0.999: con enteros no decae)
 *        si |c| >= T:  w <- clamp(w + sign(c), -1, 1);  c <- -sign(c) * reset * T          (histeresis: volver atras exige evidencia opuesta)
 * con T distinto segun la transicion para sostener un objetivo de ceros por fila (zero_target, def. 0.2; <0 lo desactiva):
 * T0 (0 -> +-1) = round(tau*fin*127), T1 (+-1 -> 0) = round(tau*fout*127), T (saturado) = round(tau*127), con
 *        d = 2*(ceros_fila/n - zero_target);  fin = clamp(1-d, 0.4, 1.6);  fout = clamp(1+d, 0.6, 2.5).
 * Usa la MAGNITUD del gradiente (normalizada): no hay umbral absoluto, ni piso, ni k. 1 byte por peso. Parametros: tau en (0,1]
 * (def. 0.7), beta en [0,1) (def. 0.999), inc (def. 0.03), zero_target (def. 0.2) y reset (def. 0.5): los cinco primeros forman TernaryUpdateRule; el sexto, head_beta2 (def. 0.99, 0 = apagada), es la memoria de la normalizacion por elemento de la cabeza (ver TernaryElemNorm).
 * Vale para Dense (pesos y bias), GQA, SwiGLU, RMSNorm, LMHead, Embedding y Conv2D.
 *
 * La regla vigente vale por HILO: ternary_stack_apply_gradients la fija (desde la del stack) mientras dura la aplicacion,
 * asi que dos modelos con reglas distintas no se pisan. Quien llame a las funciones apply_* a mano puede fijarla antes con
 * ternary_update_rule_set (sin hacerlo rige tau 0.7, beta 0.999); con un TernaryStack basta la regla del stack.
 * Un checkpoint de entrenamiento guarda los parametros de R2 (tau, beta, inc, zero_target, reset), ver ternary_stack.h.
 */
#define TERNARY_R2_TAU 0.7f
#define TERNARY_R2_BETA 0.999f
#define TERNARY_R2_INC 0.03f
#define TERNARY_R2_ZERO_TARGET 0.2f
#define TERNARY_R2_RESET 0.5f
#define TERNARY_R2_HEAD_BETA2 0.99f   /* memoria de la normalizacion por elemento de la cabeza (0 = apagada) */
/* Parametros completos de R2: tau en (0,1], beta en [0,1), inc > 0 (paso del contador por unidad de gradiente normalizado),
 * zero_target en [0,1) (fraccion de ceros por fila; < 0 lo desactiva), reset en [0,1] (fraccion del umbral al reiniciar). */
typedef struct { float tau, beta, inc, zero_target, reset, head_beta2; } TernaryUpdateRule;
#define TERNARY_R2_RULE_DEFAULT { TERNARY_R2_TAU, TERNARY_R2_BETA, TERNARY_R2_INC, TERNARY_R2_ZERO_TARGET, TERNARY_R2_RESET, TERNARY_R2_HEAD_BETA2 }
void ternary_r2_set_params(float inc, float zero_target, float reset);   /* defaults 0.03, 0.2, 0.5 */
void ternary_r2_get_params(float *inc, float *zero_target, float *reset);
/* Piezas comunes del escalar y del AVX-512 (garantizan los mismos bits): ceros de una fila, umbrales (ti, tin, tout) y reinicio (ri) a partir de ellos. */
size_t ternary_r2_count_zeros(const TernaryVec *w, size_t base, size_t n);
void ternary_r2_thresholds(size_t nz, size_t n, const TernaryUpdateRule *r, int *ti, int *tin, int *tout, int *ri, float *inc);
void ternary_update_rule_set(const TernaryUpdateRule *r);   /* para el hilo que llama; NULL = los valores por defecto de R2 */
TernaryUpdateRule ternary_update_rule_get(void);
/* Regla R2 sobre UNA fila de n pesos que empiezan en el indice 'base' de w, con su contador en cnt[base..base+n) (cnt = plano completo, mismo indice que los pesos).
 * Es la funcion unica de la regla (Dense y Embedding la usan; AVX-512 si esta activo, mismos bits); la suma de cuadrados es la canonica de ternary_fexp.h. */
void ternary_r2_update_row(TernaryVec *w, int8_t *cnt, size_t base, const float *g, size_t n, const TernaryUpdateRule *r);

/* Variante de la funcion anterior para un gradiente YA normalizado (pre != 0): se omite la division por el RMS de la fila (g_hat = clip(g,-3,3)). */
void ternary_r2_update_row_ex(TernaryVec *w, int8_t *cnt, size_t base, const float *g, size_t n, const TernaryUpdateRule *r, int pre);

/* ---- Normalizacion por elemento del gradiente (cabeza de salida) ----------------------------------------------------
 * R2 normaliza el gradiente de cada fila por su RMS: en la cabeza eso amplifica hasta el paso maximo las filas de tokens raros
 * (cuyo gradiente es pequeno y casi siempre del mismo signo), y sus pesos derivan todos en la misma direccion. Aqui cada PESO se
 * normaliza por el RMS de su propio historial (media movil de g^2, como RMSprop con correccion de sesgo):
 *     v <- beta2*v + (1-beta2)*g^2 ;  g <- g / (sqrt(v/(1-beta2^t)) + eps) ;  eps = 1e-3*RMS(g de toda la matriz) + 1e-20
 * y R2 recibe ese g ya normalizado (sigue recortando a +-3). El estado v se guarda en 4 BITS por peso: un codigo logaritmico r en 0..15
 * (v = rmax_fila * 2^-r, redondeo al mas cercano en el dominio logaritmico) mas UNA escala float (rmax = maximo de la fila) por fila:
 * 0,5 byte por peso en vez de 4 (float32). Resultados con el micromodelo en la nota del README ("Normalizacion por elemento").
 * Escalar y AVX-512 dan los mismos bits. */
#define TERNARY_ELEMNORM_LEVELS 16
typedef struct {
    size_t rows, cols;
    uint8_t *code;      /* rows * ((cols+1)/2) bytes: dos codigos por byte (el de indice par en el nibble bajo) */
    float *rmax;        /* rows: escala por fila (maximo del v de la fila antes de cuantizar); 0 = fila sin historia */
    uint32_t steps;     /* actualizaciones aplicadas (t de la correccion de sesgo) */
    float *tmp;         /* cols: scratch (no es estado) */
} TernaryElemNorm;
TernaryElemNorm *ternary_elemnorm_alloc(size_t rows, size_t cols);
void ternary_elemnorm_free(TernaryElemNorm *n);
size_t ternary_elemnorm_state_bytes(const TernaryElemNorm *n);   /* code + rmax (sin el scratch) */
/* g: rows*cols, se SOBREESCRIBE con el gradiente normalizado; beta2 en (0,1). Actualiza el estado. */
void ternary_elemnorm_apply(TernaryElemNorm *n, float *g, float beta2);
void ternary_elemnorm_apply_scalar(TernaryElemNorm *n, float *g, float beta2);   /* referencia escalar (los tests la comparan con AVX-512) */
/* Todas las filas con un gradiente ya normalizado (pre): ver ternary_r2_update_row_ex. */
void ternary_dense_apply_matrix_gradients_pre(TernaryDense *layer, const float *grad_w);

/* Aplica gradientes (R2) a una fila completa del plano de pesos. */
void ternary_dense_apply_row_gradients(TernaryDense *layer, size_t row, const float *grad_row);

/* Todas las filas de una vez: el mismo resultado bit a bit que fila por fila, repartido entre los hilos del pool. */
void ternary_dense_apply_matrix_gradients(TernaryDense *layer, const float *grad_w);

/* Igual, para el bias completo (un solo vector de longitud output_dim, con su propio contador). */
void ternary_dense_apply_bias_gradients(TernaryDense *layer, const float *grad_bias);

/* Update directo de la escala (sin inercia -- ver comentario en el struct). */
void ternary_dense_apply_scale_gradient(TernaryDense *layer, float grad_scale, float scale_lr);

#endif /* TERNARY_DENSE_H */
