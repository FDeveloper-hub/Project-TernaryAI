#ifndef TERNARY_CORE_H
#define TERNARY_CORE_H

#include <stddef.h>
#include <stdint.h>

/*
 * Representacion UNICA de un vector de pesos/activaciones ternarias
 * {-1, 0, 1}: dos planos de bits empaquetados en palabras de 64 bits,
 * SIN int8 intermedio en ningun momento de su ciclo de vida.
 *
 *   mask[i] = 1  <=>  el peso i es distinto de cero
 *   sign[i] = 1  <=>  el peso i es negativo (solo tiene sentido si mask=1)
 *
 * Costo: exactamente 2 bits por peso (1 en mask + 1 en sign), igual que un
 * codigo entrelazado de 2 bits -- la separacion en dos planos no agrega
 * memoria, solo permite operar sobre TODA una palabra de 64 pesos a la vez
 * con AND/OR/XOR/popcount en vez de desempaquetar.
 *
 * El unico "espacio no usado" posible es el resto de la ULTIMA palabra
 * cuando n no es multiplo de 64 -- inevitable en cualquier empaquetado de
 * ancho fijo (le pasa igual a un empaquetado de 4 valores por byte si n no
 * es multiplo de 4). Esos bits sobrantes se mantienen siempre en 0 para
 * que nunca contaminen un popcount.
 */
typedef struct {
    uint64_t *mask;   /* palabras de "activo/inactivo" */
    uint64_t *sign;   /* palabras de "signo" (valido donde mask=1) */
    size_t n;         /* cantidad real de valores ternarios */
    size_t words;     /* = ceil(n/64), tamano de mask[] y sign[] */
} TernaryVec;

/* Crea un vector de n valores ternarios, todos inicializados en 0.
 * Retorna NULL si falla la asignacion. */
TernaryVec *ternary_vec_alloc(size_t n);

/* Reparto secuencial (a 64 bytes) de una region contigua entre varios buffers. Con base == NULL no reparte nada:
 * solo cuenta (off = bytes necesarios), para calcular el tamano de un espacio de trabajo con el mismo codigo. */
typedef struct { unsigned char *base; size_t off; } TernaryCarve;
static inline void *ternary_carve(TernaryCarve *c, size_t bytes) {
    const size_t o = (c->off + 63u) & ~(size_t)63u;
    c->off = o + bytes;
    return c->base ? (void *)(c->base + o) : NULL;
}
void ternary_vec_free(TernaryVec *v);

/* value debe ser -1, 0 o 1. Sin chequeo en release por velocidad; el
 * llamador es responsable (uso esperado: solo en construccion/tests, no
 * en el hot path de entrenamiento). */
void ternary_vec_set(TernaryVec *v, size_t i, int value);
int  ternary_vec_get(const TernaryVec *v, size_t i);

/* Cuantos bytes REALES ocupa el vector (para medir RAM, no un estimado). */
size_t ternary_vec_bytes(const TernaryVec *v);

/*
 * Producto punto entre dos vectores ternarios de igual longitud, via
 * popcount -- sin una sola multiplicacion.
 *
 *   activo    = mask_w & mask_x
 *   mismo_signo = ~(sign_w ^ sign_x)              (XNOR)
 *   resultado = 2*popcount(activo & mismo_signo) - popcount(activo)
 *
 * Es el nucleo de TODO forward de una capa: cada fila de pesos de la
 * matriz se reduce a esto contra el vector de entrada.
 */
int32_t ternary_dot(const TernaryVec *w, const TernaryVec *x);

/* Version escalar de referencia (siempre correcta, cualquier CPU) -- es
 * la que usa ternary_dot() por defecto, y la que usan las pruebas para
 * comparar contra cualquier version acelerada (popcount, etc). */
int32_t ternary_dot_scalar(const TernaryVec *w, const TernaryVec *x);

/* Version de referencia, lenta, SIN popcount ni bits -- multiplica
 * elemento a elemento con enteros normales. Existe solo para verificar
 * ternary_dot() por comparacion directa en las pruebas; nunca se usa en
 * el motor real. */
int32_t ternary_dot_reference(const TernaryVec *w, const TernaryVec *x);


/* Mensajes de diagnostico de las validaciones de arquitectura (ternary_stack_add, ternary_residual_add...).
 * Por defecto van a stderr, como siempre; mientras haya un buffer capturador activo (por hilo) el primer
 * mensaje se copia ahi, sin salto de linea final, y NO se imprime: asi la API plana puede devolver la razon
 * exacta de un rechazo en vez de un mensaje generico. ternary_diag_capture(NULL, 0) desactiva la captura. */
void ternary_diag_capture(char *buf, size_t cap);
void ternary_diag(const char *fmt, ...);

#endif /* TERNARY_CORE_H */
