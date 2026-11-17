#ifndef TERNARY_VULKAN_H
#define TERNARY_VULKAN_H

/*
 * Deteccion de Vulkan (SOLO deteccion: ya no hay pipeline de computo, shaders
 * ni buffers; todo el calculo corre en el camino escalar de CPU).
 *
 * ternary_vulkan_probe() prueba los loaders de TERNARY_VULKAN_LIB_CANDIDATES
 * (ver ternary_gpu_lib.h; TERNARY_VULKAN_LIB=<ruta> fuerza uno) y devuelve 1
 * si alguno da una GPU REAL con cola de computo. Un dispositivo de tipo CPU
 * (Mesa lavapipe de Termux: Vulkan emulado por software) se descarta salvo
 * TERNARY_VK_ALLOW_CPU=1. Imprime en stderr la GPU y la version de API.
 * Nunca hace dlclose del driver (ver comentario en el .c).
 */
int ternary_vulkan_probe(void);

#endif /* TERNARY_VULKAN_H */
