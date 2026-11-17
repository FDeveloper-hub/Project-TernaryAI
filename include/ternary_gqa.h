#ifndef TERNARY_GQA_H
#define TERNARY_GQA_H
#include "ternary_core.h"
#include "ternary_dense.h"
#include "ternary_softmax.h"
#include <stddef.h>
#include <stdint.h>
/*
 * Grouped Query Attention. Cuatro proyecciones ternarias con votos
 * propios (Wq, Wk, Wv, Wo -- reutilizan TernaryDense tal cual, sin
 * reescribir su mecanismo de entrenamiento), mas la mecanica propia de
 * la atencion (particion en cabezas, puntajes, softmax causal, suma
 * ponderada) que no tiene pesos.
 *
 * Decisiones de diseno:
 *  - Q y K se cuantizan a ternario por SIGNO antes del producto punto
 *    (misma activacion que usa todo el proyecto) -- asi Q.K cae en
 *    [-head_dim, head_dim], el rango entero que ya espera
 *    ternary_softmax.h.
 *  - V NO es ternario (se elimino la opcion v_ternary: V ternario hacia dano en las
 *    mediciones): V = escala_v * Z cuantizado SIEMPRE a 8 bits
 *    (int8 simetrico con escala por posicion, STE), en todas las rutas.
 *  - La salida de la atencion (promedio ponderado, siempre continua)
 *    tambien se cuantiza por signo antes de Wo, por la misma razon que
 *    Q/K: Wo es un TernaryDense normal, necesita entrada ternaria.
 *  - La codificacion posicional queda para despues: esta capa funciona
 *    sin nocion de orden mas alla de la mascara causal.
 *
 * Gradiente: igual que el resto del proyecto, la cuantizacion por signo
 * NO se deriva (estimador directo/straight-through) -- el gradiente
 * pasa de largo por cada sign(), exactamente como ya pasa en cada
 * frontera entre capas de TernaryStack.
 *
 * Entrada: UN TernaryVec de seq_len*embed_dim (todas las posiciones
 * concatenadas -- misma convencion "plana" que Conv2D). Salida: mismo
 * tamano.
 */
typedef struct {
    size_t embed_dim, seq_len;
    size_t n_heads, n_kv_heads, head_dim, group_size;
    TernaryDense *wq;  /* embed_dim -> n_heads*head_dim, sin bias */
    TernaryDense *wk;  /* embed_dim -> n_kv_heads*head_dim, sin bias */
    TernaryDense *wv;  /* embed_dim -> n_kv_heads*head_dim, sin bias */
    TernaryDense *wo;  /* n_heads*head_dim -> embed_dim, sin bias */
    TernarySoftmax *softmax;

    /* RoPE entero (opcional, ver ternary_gqa_enable_rope): tablas en punto
     * fijo (escala TERNARY_ROPE_SCALE), calculadas UNA vez. */
    int rope;
    float rope_base;                    /* base de frecuencias (para describe/guardar) */
    size_t rope_pairs;                  /* head_dim / 2 */
    int16_t *rope_cos, *rope_sin;       /* seq_len * rope_pairs */

    /* Puntaje entero (ver ternary_gqa_enable_rope_int): q.k se calcula con los
     * Q/K ROTADOS sin binarizar (escala TERNARY_ROPE_SCALE). */
    int qk_int;
    int qk_shift;                       /* score = (q_rot . k_rot) >> (16 + qk_shift) */
    int32_t *q_rot, *k_rot;             /* seq_len*qw y seq_len*kvw, guardados en el forward */
    int32_t qk_lim;                     /* recorte del puntaje entero: +-qk_lim (por defecto 16*head_dim) */

    /* --- scratch reutilizable, reservado UNA vez (nunca en el forward/backward de una muestra) --- */
    TernaryVec *x_pos;                  /* embed_dim */
    int32_t *q_z, *k_z, *v_z;           /* seq_len*n_heads*head_dim / seq_len*n_kv_heads*head_dim (Z crudo) */
    TernaryVec *q_bits, *k_bits;        /* cuantizados por signo */
    float *v_float;                     /* seq_len*n_kv_heads*head_dim (siempre cuantizado a 8 bits) */
    TernaryVec *qh, *kh;                /* head_dim: vista de una cabeza para ternary_dot */
    int32_t *scores;                    /* seq_len */
    float *probs;                       /* seq_len */
    float *attn_cat;                    /* seq_len*n_heads*head_dim, entrada de Wo antes de cuantizar */
    TernaryVec *attn_bits;
    int32_t *wo_z;                      /* seq_len*embed_dim */

    float *grad_q_raw, *grad_k_raw, *grad_v_raw;  /* mismo ancho que q_z/k_z/v_z */
    float *grad_attn_cat;                          /* seq_len*n_heads*head_dim */
    float *ds_scratch;                             /* seq_len */

    /* Atencion "flash" ternaria (solo con puntaje entero, ver ternary_gqa.c). Como FlashAttention-2 guarda el
     * logsumexp por fila para no repetir el softmax en el backward, aqui se guardan el maximo ENTERO de la fila y
     * 1/suma: p_j = exp(-(m - s_j)*t) * inv se reconstruye sin ningun pase extra. Reservado una vez. */
    float *exp_tab;                                /* exponenciales compactas, exp_k0+1 entradas (ver ternary_gqa.c) */
    uint32_t exp_k0;
    /* Disposicion AVX-512 de la atencion flash (solo si la CPU lo permite): K y V traspuestas con paso tpad (multiplo
     * de 16, relleno en 0) y trabajo por grupo de cabezas. NULL = camino escalar. */
    size_t tpad;
    int32_t *kT;                                   /* n_kv_heads*head_dim filas de tpad (k_rot traspuesta) */
    float *vT;                                     /* idem (V como float traspuesta) */
    int32_t *sc_blk; float *e_blk, *p_blk, *ds_blk, *qf_blk;   /* group_size*tpad (qf: group_size*head_dim) */
    const int32_t **qptr; int32_t *m_blk; float *inv_blk;       /* group_size */
    int32_t *row_m;                                /* seq_len*n_heads */
    float *row_inv;                                /* seq_len*n_heads */

    /* Bloques de posiciones para Dense (forward de Wo y backward de Wo/Wq/Wk/Wv por backward_block). */
    size_t chunk;                                  /* min(seq_len, 48) */
    TernaryVec **xc;                               /* chunk vectores de embed_dim */
    TernaryVec **rc;                               /* chunk vectores de n_heads*head_dim */
    float *gxc;                                    /* chunk*embed_dim: grad_x temporal por bloque */
    void **views;                                  /* 11*chunk punteros por muestra del bloque (ver ternary_gqa.c) */

    /* --- Ventana deslizante (ver ternary_gqa_alloc_window) y camino "streaming" ---------------------------------
     * window = 0: atencion causal completa (todo lo de arriba). window = W >= 1: la fila i atiende a las claves
     * j en (i-W, i] mas, si sinks = S > 0, a las S primeras posiciones (attention sinks). El tiempo es O(T*(W+S)) y la
     * memoria de la capa es O(W) mas 2 bits por elemento de la salida de atencion: NINGUN buffer de tamano T*ancho
     * en floats/enteros (Z de Q/K/V, Q/K rotadas, V, gradientes, salida y Z de Wo se recalculan en el backward). */
    size_t window, sinks;
    int full_bound;                     /* atencion completa: buffers de tamano seq_len en el espacio compartido (ver ternary_gqa_bind_state) */
    int attn_bound;                     /* attn_bits vive en un espacio de trabajo ajeno (ver ternary_gqa_bind_state) */
    size_t w_n;                         /* window + sinks: claves por fila como maximo */
    size_t w_C0;                        /* posiciones del primer bloque, fijas en las ranuras [0, C0) (solo si sinks > 0) */
    size_t w_Lb, w_base, w_ld;          /* buffer deslizante: Lb ranuras para las posiciones [w_base, w_base+Lb) tras C0; paso de las traspuestas */
    int32_t *w_kr, *w_qz, *w_kz, *w_vz; /* (C0+Lb) ranuras de ancho kvw / qw / kvw / kvw */
    int32_t *w_kT; float *w_vT;         /* K rotada y V traspuestas (kvw filas de paso w_ld), solo con AVX-512 */
    float *w_vf, *w_gq, *w_gk, *w_gv;
    int32_t *w_qrot;                    /* chunk*qw: Q rotada del bloque actual */
    float *w_attn;                      /* chunk*qw: salida de atencion (forward) / gradiente de la salida (backward) del bloque */
    int32_t *w_zo;                      /* chunk*embed_dim: Z de Wo del bloque (recalculado en el backward) */
    float *w_tmpf;                      /* max(qw, kvw, embed_dim) */
    int32_t *w_tmpi;                    /* max(qw, kvw) */
    float *w_o;                         /* head_dim: salida recalculada de una cabeza */
    double *rope_theta;                 /* pares: frecuencias (solo en modo ventana, sin tablas de seq_len filas) */
    int16_t *rope_c, *rope_s;           /* pares: cos/sin de la posicion actual (modo ventana) */
    float *dprobs;                      /* seq_len, solo camino legado (sin puntaje entero) */

    /* Ruta uint8 (ver ternary_gqa_forward_u8): scratch propio, reservado la primera vez que se usa */
    int32_t *u8_tmp;                    /* max(qw, kvw): Z de Q/K reescalado antes de rotar */
    TernaryU8 *attn_u8;                 /* seq_len x n_heads*head_dim: salida de atencion cuantizada, entrada de Wo */

    /* Sin buffers de gradiente del tamano de las matrices: el backward acumula directo en grad_w_out. */

    /* "Smeared keys" (ver ternary_gqa_set_smear): K'_j = (K_j + K_{j-1}) >> 1 antes de rotar. smear_buf: 2*kvw enteros
     * (clave anterior sin mezclar y clave mezclada de trabajo), reservado al activar la opcion. */
    int smear;
    int32_t *smear_buf;
} TernaryGQA;

TernaryGQA *ternary_gqa_alloc(size_t embed_dim, size_t seq_len, size_t n_heads, size_t n_kv_heads);
/* Igual que ternary_gqa_alloc pero con ventana deslizante de ancho 'window' (>= 1) y 'sinks' posiciones iniciales
 * siempre visibles (0 <= sinks <= min(seq_len, 48)). Exige RoPE entero: llamar luego a ternary_gqa_enable_rope_int.
 * window >= seq_len con sinks = 0 equivale a la atencion completa (mismos bits que ternary_gqa_alloc). */
TernaryGQA *ternary_gqa_alloc_window(size_t embed_dim, size_t seq_len, size_t n_heads, size_t n_kv_heads,
                                      size_t window, size_t sinks);
void ternary_gqa_free(TernaryGQA *g);
/* Solo en modo ventana: el unico estado que crece con la secuencia son los bits de la salida de atencion (2 bits por
 * elemento); pueden vivir en un espacio de trabajo compartido (checkpoint por bloque). 0 si no hay nada que compartir. */
size_t ternary_gqa_state_bytes(const TernaryGQA *g);
void ternary_gqa_bind_state(TernaryGQA *g, void *base);

#define TERNARY_ROPE_SCALE 256   /* cos/sin en punto fijo: valor real = tabla / 256 */

/* Activa RoPE sobre Q y K (por pares de dimensiones (2i,2i+1) de cada
 * cabeza; requiere head_dim par). Frecuencia del par i: base^(-2i/head_dim).
 * La rotacion se aplica al Z ENTERO crudo (antes del signo) con enteros
 * puros: como despues solo importa el signo, no hay redondeo ni float en
 * el forward. El unico float es la construccion de la tabla (una vez) y
 * el giro inverso del gradiente en el backward. Devuelve 0 si ok, -1 si
 * head_dim es impar o no hay memoria. */
int ternary_gqa_enable_rope(TernaryGQA *g, float base);

/* RoPE + puntaje ENTERO sin binarizar. La propiedad relativa de RoPE
 * (q_p . k_r depende solo de p-r) vale para los valores rotados SIN
 * cuantizar; binarizar el signo de Q y K por separado la rompe (medido:
 * 48% de la varianza del puntaje explicada por la distancia relativa, vs
 * 100% con el producto entero). Aqui el puntaje es
 *     (q_rot . k_rot) >> (16 + score_shift)     (int64 acumulado, sin float)
 * con q_rot/k_rot = Z entero rotado con la tabla en punto fijo. score_shift
 * ajusta la magnitud para el softmax (tipico 2-4). La tabla del softmax
 * se amplia a +-16*head_dim. Backward: dscore/dq_real = k_rot/(256*2^shift)
 * en float (controlado), luego el giro inverso ya existente. Los pesos
 * siguen siendo ternarios; V, Wo y sus escalas no cambian. 0 ok, -1 error. */
int ternary_gqa_enable_rope_int(TernaryGQA *g, float base, int score_shift);

/* "Smeared keys" (Olsson et al. 2022, "In-context Learning and Induction Heads"): la clave de cada posicion se mezcla con
 * la de la anterior, K'_j = (K_j + K_{j-1}) >> 1 (entero, desplazamiento aritmetico; K'_0 = K_0), sobre el Z entero de K ANTES
 * de rotar con RoPE. Asi una sola capa puede comparar "el token actual" con "el token que precede a cada clave", que es lo que
 * necesita una cabeza de induccion ([A][B] ... [A] -> [B]) sin esperar a que dos capas se compongan bien con pesos ternarios.
 * Medido en este proyecto (tarea clave-valor sintetica, 2 bloques, dim 128): ~60% de exactitud con smear contra 0-10% sin el.
 * Sin parametros nuevos que aprender: el backward reparte el gradiente de K'_j entre K_j y K_{j-1} (0.5 y 0.5; K_0 recibe todo el
 * de K'_0). Solo con puntaje entero (RoPE modo 1, tras ternary_gqa_enable_rope_int) y atencion completa (window == 0).
 * 0 = ok, -1 = no permitido o sin memoria. on = 0 lo desactiva. Debe fijarse antes de entrenar o inferir: cambia las salidas. */
int ternary_gqa_set_smear(TernaryGQA *g, int on);

/* Diagnostico: probabilidades de atencion de la cabeza 'head' tal como las uso el ULTIMO forward (puntaje entero -> tabla
 * exponencial -> normalizado), escritas en out[i*seq_len + j] (j <= i; 0 para j > i), seq_len*seq_len valores. Solo lee el
 * estado guardado del forward (q_rot/k_rot): no cambia ningun resultado. 0 en exito, -1 si no aplica (ventana, puntaje no
 * entero, cabeza fuera de rango o sin forward previo). */
int ternary_gqa_attention_probs(const TernaryGQA *g, size_t head, float *out);

/* Cambia el recorte del puntaje entero (por defecto 16*head_dim) y reconstruye
 * la tabla del softmax. Solo tras ternary_gqa_enable_rope_int. Experimental:
 * NO se guarda en el checkpoint. 0 ok, -1 error. */
int ternary_gqa_set_score_clip(TernaryGQA *g, int32_t lim);

void ternary_gqa_forward(const TernaryGQA *g, const TernaryVec *x, float *out, int32_t *out_raw_opt);

/* ---- Ruta uint8 (ver ternary_u8.h) ---------------------------------------------------------------------------
 * x: seq_len filas de embed_dim bytes, cada una con su escala sx y zero-point (la salida de un RMSNorm cuantizado).
 *   - Wq/Wk/Wv: Z entero exacto = sum_i w_i*(q_i - zp)  (el valor "normalizado por fila": la escala sx de cada token
 *     no entra en Q/K, igual que en la ruta ternaria entra solo el signo). Q/K = Z >> TERNARY_GQA_U8_SHIFT (con
 *     redondeo), que devuelve Z al rango entero que ya espera el puntaje (RoPE entero, softmax, atencion flash: todo
 *     igual que antes). V: escala_v * sx * Z, cuantizado a int8 simetrico por posicion.
 *   - La salida de atencion (por posicion) se cuantiza a uint8 y Wo la consume con su producto entero.
 * Requiere puntaje entero (ternary_gqa_enable_rope_int). Funciona con atencion completa y con ventana deslizante (en ventana la
 * salida de atencion cuantizada ocupa seq_len*n_heads*head_dim bytes: 1 byte por elemento en vez de 2 bits).
 * Backward con el estimador directo en cada cuantizacion; grad_w se SOBREESCRIBE, grad_x y grad_scale se ACUMULAN. */
#ifndef TERNARY_GQA_U8_SHIFT
#define TERNARY_GQA_U8_SHIFT 8
#endif
void ternary_gqa_forward_u8(const TernaryGQA *g, const TernaryU8 *x, float *out, int32_t *out_raw_opt);
void ternary_gqa_backward_u8(TernaryGQA *g, const TernaryU8 *x, const float *grad_out,
                             float *const *grad_w_out, float *grad_x_out, float *grad_scale_out);

/* grad_out: seq_len*embed_dim. grad_w_out[0..3] = Wq,Wk,Wv,Wo (mismo
 * orden que num_matrices), cada uno SOBREESCRITO con el total de la
 * muestra. grad_x_out: seq_len*embed_dim, se ACUMULA (convencion de
 * siempre). grad_scale_out[0..3]: se ACUMULAN. */
void ternary_gqa_backward(TernaryGQA *g, const TernaryVec *x, const float *grad_out,
                           float *const *grad_w_out, float *grad_x_out, float *grad_scale_out,
                           const int32_t *out_raw_opt);

#endif /* TERNARY_GQA_H */
