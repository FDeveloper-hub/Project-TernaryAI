#ifndef TERNARY_DISPATCH_H
#define TERNARY_DISPATCH_H

#include "ternary_core.h"
#include "ternary_threadpool.h"

typedef int32_t (*TernaryDotFn)(const TernaryVec *, const TernaryVec *);

/* Permite reemplazar la implementacion que usa ternary_dot() en TODO el
 * motor (hoy solo la usa ternary_core.c para elegir la version con POPCNT
 * en x86), sin que ternary_dense.c / ternary_conv2d.c lo sepan. */
void ternary_core_set_dot_impl(TernaryDotFn fn);

/*
 * Todo el calculo corre en el camino escalar de CPU (popcount sobre las
 * palabras empaquetadas mask/sign), con el pool de hilos. NEON, Vulkan y
 * OpenCL ya no calculan nada: Vulkan y OpenCL solo se DETECTAN y se
 * reportan (has_vulkan / has_opencl) para diagnostico.
 */
typedef enum {
    TERNARY_ENGINE_CPU_SCALAR = 0
} TernaryEngine;

/* Capacidades detectadas en tiempo de ejecucion. */
typedef struct {
    int num_cores;
    int has_vulkan;    /* loader Vulkan cargable Y con una GPU real con cola compute (solo reporte) */
    int has_opencl;    /* libOpenCL cargable Y con al menos 1 device (solo reporte) */
    const char *dot_impl_name; /* "scalar", para diagnostico/log */

    /* Pool de hilos FIJO para el forward de CPU (creado UNA vez aca,
     * nunca en el hot path -- ver ternary_threadpool.h). NULL si
     * ternary_threadpool_create() fallo (memoria) o si num_cores<=1. */
    TernaryThreadPool *cpu_pool;
} TernaryCapabilities;

/* Detecta nucleos, crea el pool de hilos y detecta (solo reporta) Vulkan
 * y OpenCL. Idempotente. */
TernaryCapabilities ternary_init(void);

/* Igual que ternary_init() pero SIN tocar Vulkan ni OpenCL (has_vulkan/
 * has_opencl quedan en 0). Es lo que usa la fachada Python (libternaryai)
 * por defecto. Se libera igual, con ternary_shutdown(). */
TernaryCapabilities ternary_init_cpu(void);

/* Libera el pool de hilos. Llamar una sola vez al terminar. */
void ternary_shutdown(TernaryCapabilities *caps);

/* Motor para una capa de 'layer_words' palabras de 64 bits: siempre el
 * escalar (se conserva la firma para quien ya la llama). */
TernaryEngine ternary_choose_engine(const TernaryCapabilities *caps, size_t layer_words);

#endif /* TERNARY_DISPATCH_H */
