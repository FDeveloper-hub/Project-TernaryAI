#define VK_NO_PROTOTYPES
#include "ternary_vulkan.h"
#include <vulkan/vulkan_core.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ternary_gpu_lib.h"

/* NUNCA se hace dlclose() de un driver de GPU (libvulkan.so): el driver de
 * Mali crea hilos propios y registra destructores de TLS/atexit dentro de su
 * propia libreria; descargarla mientras esos hilos siguen vivos deja codigo
 * ejecutandose sobre direcciones ya desmapeadas (crash tipico Bus error /
 * SIGSEGV sin relacion visible). Android documenta que los drivers Vulkan no
 * deben descargarse. La libreria queda cargada hasta que termine el proceso. */

static int probe_from(const char *lib_path) {
    void *lib = ternary_gpu_dlopen_one("Vulkan (deteccion)", lib_path);
    if (!lib) return 0;

    PFN_vkCreateInstance create = (PFN_vkCreateInstance)dlsym(lib, "vkCreateInstance");
    PFN_vkDestroyInstance destroy = (PFN_vkDestroyInstance)dlsym(lib, "vkDestroyInstance");
    PFN_vkEnumeratePhysicalDevices enumerate =
        (PFN_vkEnumeratePhysicalDevices)dlsym(lib, "vkEnumeratePhysicalDevices");
    PFN_vkGetPhysicalDeviceQueueFamilyProperties qf_props_fn =
        (PFN_vkGetPhysicalDeviceQueueFamilyProperties)dlsym(lib, "vkGetPhysicalDeviceQueueFamilyProperties");
    PFN_vkGetPhysicalDeviceProperties props_fn =
        (PFN_vkGetPhysicalDeviceProperties)dlsym(lib, "vkGetPhysicalDeviceProperties"); /* opcional */
    if (!create || !destroy || !enumerate || !qf_props_fn) return 0;

    VkApplicationInfo app_info;
    memset(&app_info, 0, sizeof(app_info));
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "ternary_native_probe";
    app_info.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo inst_info;
    memset(&inst_info, 0, sizeof(inst_info));
    inst_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    inst_info.pApplicationInfo = &app_info;

    VkInstance instance = VK_NULL_HANDLE;
    if (create(&inst_info, NULL, &instance) != VK_SUCCESS || !instance) return 0;

    int found = 0;
    uint32_t device_count = 0;
    enumerate(instance, &device_count, NULL);
    VkPhysicalDevice *devices =
        device_count ? (VkPhysicalDevice *)malloc(device_count * sizeof(VkPhysicalDevice)) : NULL;
    if (devices && enumerate(instance, &device_count, devices) >= 0) {
        for (uint32_t d = 0; d < device_count && !found; d++) {
            uint32_t qf_count = 0;
            qf_props_fn(devices[d], &qf_count, NULL);
            if (qf_count == 0) continue;
            VkQueueFamilyProperties *qf = (VkQueueFamilyProperties *)malloc(qf_count * sizeof(*qf));
            if (!qf) continue;
            qf_props_fn(devices[d], &qf_count, qf);
            int has_compute = 0;
            for (uint32_t q = 0; q < qf_count; q++)
                if (qf[q].queueFlags & VK_QUEUE_COMPUTE_BIT) { has_compute = 1; break; }
            free(qf);
            if (!has_compute) continue;

            VkPhysicalDeviceProperties dp;
            memset(&dp, 0, sizeof(dp));
            if (props_fn) props_fn(devices[d], &dp);
            const char *allow = getenv("TERNARY_VK_ALLOW_CPU");
            int allow_cpu = (allow && allow[0] && allow[0] != '0');
            if (props_fn && dp.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU && !allow_cpu) {
                fprintf(stderr, "ternary: Vulkan '%s' via %s es un dispositivo de CPU (emulado por software, "
                                "no la GPU) -- se ignora.\n", dp.deviceName, lib_path);
                continue;
            }
            if (props_fn)
                fprintf(stderr, "ternary: Vulkan detectado: GPU='%s' api=%u.%u.%u via %s "
                                "(solo deteccion: el calculo corre en el camino escalar).\n",
                        dp.deviceName, VK_VERSION_MAJOR(dp.apiVersion), VK_VERSION_MINOR(dp.apiVersion),
                        VK_VERSION_PATCH(dp.apiVersion), lib_path);
            found = 1;
        }
    }
    free(devices);
    destroy(instance, NULL);
    return found;
}

int ternary_vulkan_probe(void) {
    const char *forced = getenv("TERNARY_VULKAN_LIB");
    if (forced && forced[0]) return probe_from(forced);
    static const char *const candidatos[] = TERNARY_VULKAN_LIB_CANDIDATES;
    for (size_t i = 0; i < sizeof(candidatos) / sizeof(candidatos[0]); i++)
        if (probe_from(candidatos[i])) return 1;
    fprintf(stderr, "ternary: ningun loader Vulkan dio una GPU real (TERNARY_VK_TRACE=1 muestra el motivo) "
                    "-- Vulkan no disponible.\n");
    return 0;
}
