#ifndef TERNARY_THREADPOOL_H
#define TERNARY_THREADPOOL_H

#include <stddef.h>

/*
 * Pool de hilos persistente y de baja latencia (fork-join).
 *
 *  - num_threads cuenta TODOS los hilos que trabajan, incluido el que llama: se crean num_threads-1 trabajadores y el hilo que
 *    llama tambien reparte trabajo (nunca se queda esperando sin hacer nada). Sin tope: se puede pedir cualquier cantidad.
 *  - Los trabajadores esperan unos microsegundos girando (asi una operacion detras de otra no paga el despertar del sistema)
 *    y despues duermen; se despiertan solos cuando llega trabajo.
 *  - El trabajo se reparte en trozos (grain indices) que se reclaman con un contador atomico: el reparto se balancea solo.
 *    Cada indice se ejecuta exactamente una vez. El reparto NO afecta el resultado mientras cada indice escriba datos
 *    propios (particion por filas/posiciones/cabezas): por eso los resultados son identicos con 1 o con N hilos.
 *  - Una llamada dentro de otra (anidada) o desde un segundo hilo mientras el pool esta ocupado corre en serie en el hilo que llama.
 *
 *   ternary_parallel_for(pool, 0, 256, mi_funcion_por_fila, &contexto);
 */
typedef struct TernaryThreadPool TernaryThreadPool;

/* num_threads <= 1: pool serial (sin hilos de mas). */
TernaryThreadPool *ternary_threadpool_create(int num_threads);
void ternary_threadpool_destroy(TernaryThreadPool *pool);

typedef void (*TernaryParallelTaskFn)(size_t idx, void *ctx);
typedef void (*TernaryParallelRangeFn)(size_t begin, size_t end, void *ctx);

/* fn(idx) una vez por cada idx en [begin, end). NULL o pool de 1 hilo: serial. */
void ternary_parallel_for(TernaryThreadPool *pool, size_t begin, size_t end, TernaryParallelTaskFn fn, void *ctx);
/* fn(b, e) sobre trozos [b, e) de a lo sumo 'grain' indices (grain 0 = automatico). Menos llamadas que parallel_for. */
/* nparts partes fijas (parte i en el hilo i; fn(i, i+1, ctx)); nparts <= hilos. Para reutilizar siempre los mismos datos en cache. */
void ternary_parallel_static(TernaryThreadPool *pool, size_t nparts, TernaryParallelRangeFn fn, void *ctx);
void ternary_parallel_ranges(TernaryThreadPool *pool, size_t begin, size_t end, size_t grain, TernaryParallelRangeFn fn, void *ctx);

int ternary_threadpool_num_threads(const TernaryThreadPool *pool);   /* pool NULL = 1 */
/* Microsegundos que un trabajador gira antes de dormir (por defecto 200). 0 = duerme de inmediato. */
void ternary_threadpool_set_spin_us(TernaryThreadPool *pool, int us);

/* Autochequeo de integridad (canarios + copia sombra de los pthread_t).
 * 0 = intacto; >0 = cantidad de problemas (se imprimen a stderr). */
int ternary_threadpool_verify(const TernaryThreadPool *pool);

/* ---- Pool global de la libreria: lo fija el usuario a mano (Python: tn.set_threads(n)). ----
 * n <= 1: sin reparto (un solo hilo). Valor inicial: variable de entorno TERNARYAI_THREADS, o 1. No llamar mientras se entrena. */
void ternary_set_num_threads(int n);
int ternary_get_num_threads(void);
TernaryThreadPool *ternary_global_pool(void);               /* NULL = un solo hilo */

/* Pool de calculo que usan las capas: el global, salvo que un test/benchmark ponga el suyo (NULL = volver al global). */
void ternary_set_compute_pool(TernaryThreadPool *pool);
TernaryThreadPool *ternary_get_compute_pool(void);

/* Ayudas de reparto para operaciones elemento a elemento (el resultado no depende del reparto). Por debajo de un umbral
 * (TERNARY_PAR_MIN_BYTES) se ejecutan en el hilo que llama, para no pagar la sincronizacion en trabajos chicos. */
#define TERNARY_PAR_MIN_BYTES 262144
void ternary_par_zero(void *p, size_t bytes);                       /* memset(p, 0, bytes) */
void ternary_par_copy(void *dst, const void *src, size_t bytes);    /* memcpy (sin solape) */

#endif /* TERNARY_THREADPOOL_H */
