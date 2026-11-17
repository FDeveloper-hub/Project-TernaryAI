#ifndef TERNARY_EMBEDDING_H
#define TERNARY_EMBEDDING_H
#include "ternary_core.h"
#include <stddef.h>
#include <stdint.h>
/*
 * Capa de embeddings: tabla ternaria vocab_size x embed_dim indexada por
 * id de token. Es una BUSQUEDA (copiar una fila), no un producto punto:
 * O(seq_len*embed_dim) por muestra, independiente del tamano del
 * vocabulario. Mismo mecanismo de entrenamiento que Dense/Conv2D (votos
 * int8 + umbral dinamico + escala unica por capa estilo BitNet), aplicado
 * por fila -- reutiliza ternary_r2_update_row() de ternary_dense.h,
 * no hay un optimizador nuevo.
 *
 * Entrada: seq_len ids enteros por muestra (dtype ID, ver
 * ternary_tensor.h). Salida: seq_len*embed_dim floats, posicion p en
 * [p*embed_dim, (p+1)*embed_dim). Sin bias.
 */
typedef struct {
    size_t vocab_size;
    size_t embed_dim;
    size_t seq_len;
    TernaryVec *W;      /* vocab_size*embed_dim, fila v en [v*embed_dim, (v+1)*embed_dim) */
    int8_t *counter_W;    /* mismo layout que W */
    float scale;        /* escala unica por capa, como Dense/Conv2D (inicia en 1.0) */
} TernaryEmbedding;

TernaryEmbedding *ternary_embedding_alloc(size_t vocab_size, size_t embed_dim, size_t seq_len);
void ternary_embedding_free(TernaryEmbedding *layer);

int  ternary_embedding_get_weight(const TernaryEmbedding *layer, size_t v, size_t d);
void ternary_embedding_set_weight(TernaryEmbedding *layer, size_t v, size_t d, int value);

/* out[p*embed_dim+d] = scale * W[ids[p]][d]. out_raw_opt (puede ser NULL),
 * misma longitud: el valor sin escalar W[ids[p]][d] (-1/0/1) -- es el
 * "Z crudo" de esta capa, igual que el de Dense, y lo usa backward para
 * el gradiente de la escala. Un id fuera de [0, vocab_size) se trata
 * como fila de ceros (no se valida por rendimiento: ids validos son
 * responsabilidad del tokenizador/llamador). */
void ternary_embedding_forward(const TernaryEmbedding *layer, const int32_t *ids,
                                float *out, int32_t *out_raw_opt);

/* grad_out: longitud seq_len*embed_dim. grad_w_out: longitud
 * vocab_size*embed_dim, se SOBREESCRIBE (todo en 0 salvo las filas que
 * aparecen en ids; si un id se repite, sus gradientes se SUMAN en su
 * fila). *grad_scale_out se ACUMULA. */
void ternary_embedding_backward(const TernaryEmbedding *layer, const int32_t *ids,
                                 const float *grad_out, float *grad_w_out, float *grad_scale_out,
                                 const int32_t *z_raw_opt);

/* Misma semantica que ternary_dense_apply_row_gradients (una fila =
 * embed_dim gradientes). Una fila con gradiente todo en cero no cambia. */
void ternary_embedding_apply_row_gradients(TernaryEmbedding *layer, size_t row, const float *grad_row);
void ternary_embedding_apply_scale_gradient(TernaryEmbedding *layer, float grad_scale,
                                             float scale_lr);

#endif /* TERNARY_EMBEDDING_H */
