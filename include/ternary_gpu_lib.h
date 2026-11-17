#ifndef TERNARY_GPU_LIB_H
#define TERNARY_GPU_LIB_H

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * Carga de librerias de driver de GPU (Vulkan/OpenCL), compartida por
 * ternary_dispatch.c y ternary_vulkan.c (solo
 * funciones static inline: no hay .c nuevo que agregar al Makefile).
 *
 * Con TERNARY_VK_TRACE=1 se imprime cada candidato probado y el motivo
 * exacto (dlerror) por el que fallo -- la unica forma de saber si una
 * libreria "no existe" o si el linker de Android la bloquea por
 * namespace. Si NINGUN candidato abre, se imprime igual una linea con el
 * ultimo error (una sola vez por biblioteca buscada).
 */

/* Orden Vulkan: PRIMERO el loader de la plataforma Android (es el unico
 * que llega al driver real de la GPU del dispositivo, p.ej. Mali), DESPUES
 * los nombres desnudos: en Termux "libvulkan.so"/".so.1" resuelve al
 * loader de Termux, cuyo unico ICD instalado suele ser Mesa lavapipe
 * (Vulkan emulado en la CPU), que NO es la GPU. */
#define TERNARY_VULKAN_LIB_CANDIDATES { \
    "/system/lib64/libvulkan.so", "/system/lib/libvulkan.so", \
    "/vendor/lib64/libvulkan.so", "/vendor/lib/libvulkan.so", \
    "libvulkan.so", "libvulkan.so.1" }

/* Orden OpenCL (mismo criterio que Vulkan): rutas ABSOLUTAS del vendor
 * primero, nombres desnudos AL FINAL. En Termux "libOpenCL.so" a secas
 * resuelve (via LD_LIBRARY_PATH) al ICD loader de Termux (paquete
 * ocl-icd), que abre sin error pero no tiene ningun ICD registrado y
 * responde "0 plataformas" -- justo el sintoma "abre pero no detecta". */
#define TERNARY_OPENCL_LIB_CANDIDATES { \
    "/vendor/lib/libOpenCL.so", "/system/vendor/lib/libOpenCL.so", "/system/lib/libOpenCL.so", \
    "/vendor/lib64/libOpenCL.so", "/system/vendor/lib64/libOpenCL.so", "/system/lib64/libOpenCL.so", \
    "/vendor/lib/egl/libGLES_mali.so", "/system/vendor/lib/egl/libGLES_mali.so", \
    "/system/lib/egl/libGLES_mali.so", "/vendor/lib/libGLES_mali.so", \
    "/vendor/lib64/egl/libGLES_mali.so", "/system/vendor/lib64/egl/libGLES_mali.so", \
    "/system/lib64/egl/libGLES_mali.so", "/vendor/lib64/libGLES_mali.so", \
    "libOpenCL.so", "libOpenCL.so.1", "libGLES_mali.so", "libmali.so" }

static inline int ternary_gpu_trace_on(void) {
    const char *e = getenv("TERNARY_VK_TRACE");
    return (e && e[0] && e[0] != '0');
}

/* dlopen() de UNA ruta; con TERNARY_VK_TRACE=1 imprime el dlerror() si falla. */
static inline void *ternary_gpu_dlopen_one(const char *label, const char *path) {
    dlerror();
    void *lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (ternary_gpu_trace_on()) {
        if (lib) fprintf(stderr, "[gpu-lib] %s: dlopen('%s') OK\n", label, path);
        else { const char *e = dlerror(); fprintf(stderr, "[gpu-lib] %s: dlopen('%s') fallo: %s\n", label, path, e ? e : "(sin detalle)"); }
        fflush(stderr);
    }
    return lib;
}

/* dlopen() de la primera ruta que abra; *which_out (opcional) recibe cual. */
static inline void *ternary_gpu_dlopen_first(const char *label, const char *const *cands, size_t n,
                                              const char **which_out) {
    const char *last_err = NULL;
    int trace = ternary_gpu_trace_on();
    for (size_t i = 0; i < n; i++) {
        dlerror();
        void *lib = dlopen(cands[i], RTLD_NOW | RTLD_LOCAL);
        if (lib) {
            if (trace) { fprintf(stderr, "[gpu-lib] %s: dlopen('%s') OK\n", label, cands[i]); fflush(stderr); }
            if (which_out) *which_out = cands[i];
            return lib;
        }
        const char *err = dlerror();
        last_err = err ? err : "(sin detalle)";
        if (trace) { fprintf(stderr, "[gpu-lib] %s: dlopen('%s') fallo: %s\n", label, cands[i], last_err); fflush(stderr); }
    }
    fprintf(stderr, "ternary: %s -- ningun candidato de libreria abrio (ultimo error: %s). "
                    "Con TERNARY_VK_TRACE=1 se ve el motivo de cada ruta.\n",
            label, last_err ? last_err : "(sin detalle)");
    return NULL;
}

#endif /* TERNARY_GPU_LIB_H */
