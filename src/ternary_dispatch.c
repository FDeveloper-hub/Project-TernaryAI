#include "ternary_mem.h"
#include "ternary_dispatch.h"
#include "ternary_vulkan.h"
#include <unistd.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "ternary_gpu_lib.h"
#include <stdlib.h>

/* ---- Declaracion minima del ABI de OpenCL, para dlsym + llamar SIN linkear
 * contra libOpenCL (asi el binario arranca igual si no existe). Es la parte
 * ESTABLE del ABI (OpenCL 1.0) necesaria para una sola pregunta: "hay al
 * menos un dispositivo?". ---- */

/* -- OpenCL -- */
typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef struct _cl_platform_id *cl_platform_id;
typedef struct _cl_device_id *cl_device_id;
#define CL_SUCCESS 0
#define CL_DEVICE_TYPE_ALL 0xFFFFFFFFULL
typedef cl_int (*PFN_clGetPlatformIDs)(cl_uint, cl_platform_id *, cl_uint *);
typedef cl_int (*PFN_clGetDeviceIDs)(cl_platform_id, uint64_t, cl_uint, cl_device_id *, cl_uint *);

/* Cuenta el mayor indice de CPU listado en un archivo con el formato de
 * /sys/devices/system/cpu/{possible,present}: rangos y/o valores sueltos
 * separados por coma ("0-7", "0-3,4-7", "0,1,2,3"...). Devuelve -1 si no
 * pudo leer nada util. */
static int highest_cpu_index_in_file(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;

    char buf[256];
    int hi = -1;
    if (fgets(buf, sizeof(buf), f)) {
        char *tok = strtok(buf, ",\n");
        while (tok) {
            int lo, top;
            if (sscanf(tok, "%d-%d", &lo, &top) == 2) {
                if (top > hi) hi = top;
            } else if (sscanf(tok, "%d", &lo) == 1) {
                if (lo > hi) hi = lo;
            }
            tok = strtok(NULL, ",\n");
        }
    }
    fclose(f);
    return hi;
}

static int detect_num_cores(void) {
    /* _SC_NPROCESSORS_ONLN solo cuenta los nucleos ENCENDIDOS en este
     * instante. En Android, el gobernador de energia apaga nucleos por
     * cpu-hotplug cuando la carga es baja -- en el SoC objetivo (octa-core,
     * 8x Cortex-A53, sin big.LITTLE real: los 8 son identicos) es comun
     * que solo 4 esten online al momento exacto de esta llamada, aunque
     * el chip tenga 8 disponibles. _SC_NPROCESSORS_CONF cuenta los
     * nucleos CONFIGURADOS por el kernel (esten online o no), que es lo
     * que en realidad se quiere para dimensionar el thread pool: cuantos
     * nucleos existen, no cuantos estan prendidos ahora mismo. */
    long conf = sysconf(_SC_NPROCESSORS_CONF);
    long onln = sysconf(_SC_NPROCESSORS_ONLN);
    long n = (conf > onln) ? conf : onln;

    if (n <= 0) {
        /* Ultimo recurso si sysconf no responde nada util: leer
         * directamente el rango de CPUs que expone el kernel Linux.
         * "possible" es el limite duro del kernel (incluye nucleos que
         * nunca se prendieron); "present" es el hardware fisico presente.
         * Se prueban ambos por si alguno no existe en el kernel del
         * dispositivo. */
        int hi = highest_cpu_index_in_file("/sys/devices/system/cpu/possible");
        if (hi < 0) hi = highest_cpu_index_in_file("/sys/devices/system/cpu/present");
        if (hi >= 0) n = hi + 1;
    }

    return (n > 0) ? (int)n : 1;
}

/* Prueba real: no solo "existe el simbolo", sino "hay al menos un
 * dispositivo detras". dlsym exitoso confirma que el LOADER esta, pero
 * el loader puede estar instalado sin ningun ICD/driver real detras
 * (pasa en escritorios sin GPU, y es un caso comun en entornos de prueba).
 *
 * Nota sobre el Mali de gama baja: es un GPU de la familia
 * Midgard de ARM; el soporte de Vulkan en drivers Mali empezo con la
 * familia Bifrost (posterior). Es esperable que este GPU en particular
 * no tenga driver Vulkan aunque el codigo de deteccion este perfecto --
 * "no detecta Vulkan" puede ser el resultado CORRECTO, no un bug. */
/* NUNCA se hace dlclose() de un driver de GPU (libvulkan.so/libOpenCL.so/
 * libGLES_mali.so): el driver de Mali crea hilos propios y registra
 * destructores de TLS/atexit dentro de su propia libreria; descargarla
 * mientras esos hilos siguen vivos deja codigo ejecutandose sobre
 * direcciones ya desmapeadas (y que despues pueden reutilizarse para
 * otra cosa) -- crash tipico "Bus error"/SIGSEGV sin relacion visible
 * con la llamada que lo dispara. Android documenta que los drivers
 * Vulkan no deben descargarse. La libreria queda cargada hasta que
 * termine el proceso (costo: unos pocos MB, una sola vez). */
static void keep_gpu_lib_loaded(void *lib) { (void)lib; }

/* OpenCL en Android es mas fragil que Vulkan: a diferencia de
 * libvulkan.so, libOpenCL.so NUNCA formo parte de AOSP ni de la lista de
 * librerias publicas que el NDK garantiza poder dlopen-ear por nombre --
 * es siempre una libreria de vendor (Mali, Qualcomm, etc.), vive tipico
 * en /vendor/lib(64)/ o /system/vendor/lib(64)/, y el namespace del
 * linker puede bloquear su carga a una app normal (Termux/Pydroid 3 sin
 * root) aunque se le de la ruta exacta. Se prueban varias rutas
 * conocidas de Mali como mejor esfuerzo; si ninguna funciona,
 * confirmar desde una shell en el dispositivo si el archivo existe siquiera
 * (`find / -iname 'libOpenCL*' 2>/dev/null`) antes de asumir que es un
 * bug de deteccion -- puede ser que el fabricante no lo incluya. */
static int probe_opencl(void) {
    /* La sospecha de que "OpenCL se llama como la GPU" tiene fundamento
     * real, confirmado buscando precedentes con chips Mali: en
     * Mali, el blob que implementa OpenCL casi nunca se llama
     * literalmente "libOpenCL.so" -- es el mismo binario monolitico que
     * ya implementa EGL+GLES (libGLES_mali.so), y a veces aparece
     * ademas simplemente como libmali.so. Un usuario reporto
     * textualmente haber hecho andar un kernel OpenCL en una GPU Mali
     * copiando
     * /system/vendor/lib/libGLES_mali.so (junto a libOpenCL.so) --
     * fuente: foro de ARM Community, hilo "Mali G72 OpenCL library".
     * Tambien hay precedente de Android 6 poniendolo en
     * /system/lib/egl/libGLES_mali.so (sin "vendor" en la ruta). Se
     * agregan esas variantes ademas de las que ya se probaban. */
    static const char *const candidatos[] = TERNARY_OPENCL_LIB_CANDIDATES;
    size_t n_cand = sizeof(candidatos) / sizeof(candidatos[0]);
    int abrio_alguna = 0;

    /* Se prueba CADA candidato por separado y se sigue con el siguiente si
     * abre pero no sirve (antes se usaba la primera libreria que abriera
     * y, si esa no tenia plataformas, se daba OpenCL por inexistente sin
     * decir por que). Cada descarte se explica en stderr. */
    for (size_t c = 0; c < n_cand; c++) {
        void *lib = ternary_gpu_dlopen_one("OpenCL (deteccion)", candidatos[c]);
        if (!lib) continue;
        abrio_alguna = 1;

        PFN_clGetPlatformIDs get_platforms = (PFN_clGetPlatformIDs)dlsym(lib, "clGetPlatformIDs");
        if (!get_platforms) {
            fprintf(stderr, "ternary: OpenCL '%s' abre pero no exporta clGetPlatformIDs -- descartada.\n", candidatos[c]);
            keep_gpu_lib_loaded(lib);
            continue;
        }
        cl_uint num_platforms = 0;
        cl_int r = get_platforms(0, NULL, &num_platforms);
        if (r != CL_SUCCESS || num_platforms == 0) {
            fprintf(stderr, "ternary: OpenCL '%s' abre pero clGetPlatformIDs devolvio %d con %u plataformas "
                            "(-1001 = no hay ningun ICD/driver registrado) -- descartada.\n",
                    candidatos[c], (int)r, (unsigned)num_platforms);
            keep_gpu_lib_loaded(lib);
            continue;
        }

        /* Al menos una plataforma existe, pero eso no garantiza que tenga
         * dispositivos usables (puede ser solo CPU-only o estar vacia) --
         * confirmar tambien clGetDeviceIDs sobre las plataformas. */
        PFN_clGetDeviceIDs get_devices = (PFN_clGetDeviceIDs)dlsym(lib, "clGetDeviceIDs");
        int tiene_dispositivos = 0;
        if (get_devices) {
            cl_platform_id platforms[8];
            cl_uint got = 0;
            if (get_platforms(8, platforms, &got) == CL_SUCCESS) {
                for (cl_uint pl = 0; pl < got && !tiene_dispositivos; pl++) {
                    cl_uint num_devices = 0;
                    if (get_devices(platforms[pl], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices) == CL_SUCCESS
                        && num_devices > 0) {
                        tiene_dispositivos = 1;
                    }
                }
            }
        }
        keep_gpu_lib_loaded(lib);
        if (tiene_dispositivos) {
            fprintf(stderr, "ternary: OpenCL detectado en '%s' (%u plataforma(s)).\n", candidatos[c], (unsigned)num_platforms);
            return 1;
        }
        fprintf(stderr, "ternary: OpenCL '%s' tiene %u plataforma(s) pero ningun dispositivo -- descartada.\n",
                candidatos[c], (unsigned)num_platforms);
    }
    if (!abrio_alguna)
        fprintf(stderr, "ternary: OpenCL -- ninguna de las %zu rutas de libreria abrio "
                        "(TERNARY_VK_TRACE=1 muestra el dlerror() de cada una).\n", n_cand);
    return 0;
}

/* Parte CPU de la deteccion (nucleos, pool de hilos): la comparten
 * ternary_init() y ternary_init_cpu(). Deja Vulkan/OpenCL en cero. */
static void init_cpu_part(TernaryCapabilities *caps) {
    memset(caps, 0, sizeof(*caps));
    caps->num_cores = detect_num_cores();
    caps->dot_impl_name = "scalar";
    /* El reparto en hilos lo fija el usuario a mano (ternary_set_num_threads / tn.set_threads); las capas usan el pool global
     * (ternary_get_compute_pool). cpu_pool queda NULL salvo que un test ponga uno propio. */
    caps->cpu_pool = NULL;
}

TernaryCapabilities ternary_init_cpu(void) {
    TernaryCapabilities caps;
    init_cpu_part(&caps);
    return caps;
}

TernaryCapabilities ternary_init(void) {
    TernaryCapabilities caps;
    init_cpu_part(&caps);
    /* Solo deteccion: se reporta si hay GPU, pero nada se calcula en ella. */
    caps.has_vulkan = ternary_vulkan_probe();
    caps.has_opencl = probe_opencl();
    return caps;
}

void ternary_shutdown(TernaryCapabilities *caps) {
    if (!caps) return;
    caps->cpu_pool = NULL;                      /* el pool global no es de caps */
}

TernaryEngine ternary_choose_engine(const TernaryCapabilities *caps, size_t layer_words) {
    (void)caps; (void)layer_words;
    return TERNARY_ENGINE_CPU_SCALAR;
}
