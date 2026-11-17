#include <stdio.h>
#include <assert.h>
#include "ternary_dispatch.h"

/* Todo el calculo corre en el camino escalar de CPU: ternary_choose_engine()
 * siempre devuelve el escalar, sin importar el tamano de la capa ni si hay
 * GPU. Vulkan/OpenCL solo se detectan y se reportan. */
int main(void) {
    TernaryCapabilities fake = {0};
    fake.num_cores = 8; fake.has_vulkan = 1; fake.has_opencl = 1; fake.dot_impl_name = "scalar";
    assert(ternary_choose_engine(&fake, 1) == TERNARY_ENGINE_CPU_SCALAR);
    assert(ternary_choose_engine(&fake, 999999999u) == TERNARY_ENGINE_CPU_SCALAR);

    /* ternary_init() real: en un entorno sin GPU has_vulkan/has_opencl dan 0; en un dispositivo con GPU pueden dar 1.
     * Lo unico que se exige es que sea un booleano sano, que nunca crashee y que el motor siga siendo el escalar. */
    {
        TernaryCapabilities c = ternary_init();
        printf("ternary_init(): cores=%d vulkan(loader)=%d opencl(loader)=%d dot_impl=%s pool=%s\n",
               c.num_cores, c.has_vulkan, c.has_opencl, c.dot_impl_name, c.cpu_pool ? "si" : "no");
        assert(c.num_cores >= 1);
        assert(c.has_vulkan == 0 || c.has_vulkan == 1);
        assert(c.has_opencl == 0 || c.has_opencl == 1);
        assert(ternary_choose_engine(&c, 999999999u) == TERNARY_ENGINE_CPU_SCALAR);
        ternary_shutdown(&c);
    }
    {
        TernaryCapabilities c = ternary_init_cpu();
        assert(c.has_vulkan == 0 && c.has_opencl == 0);
        ternary_shutdown(&c);
    }
    printf("test_dispatch: OK\n");
    return 0;
}
