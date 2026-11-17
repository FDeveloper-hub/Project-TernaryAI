CC ?= gcc
CFLAGS = -std=c11 -O3 -ffp-contract=off -Wall -Wextra -Iinclude
# El backward usa fmaf (multiplicacion y suma fusionadas): en ARMv7 necesita VFPv4 para ser una instruccion (si no, es una llamada de software, correcta pero lenta).
ifneq (,$(filter armv7l armv8l,$(shell uname -m)))
CFLAGS += -mfpu=neon-vfpv4
endif
LDLIBS = -lm -ldl -lpthread

NEON_SRC = src/ternary_neon.c src/ternary_neon_stubs.c src/ternary_neon_planes.c src/ternary_neon_elem.c src/ternary_neon_attn.c src/ternary_neon_norm.c

SRC = src/ternary_core.c src/ternary_dense.c src/ternary_image.c src/ternary_conv2d.c \
      src/ternary_dispatch.c src/ternary_threadpool.c \
      src/ternary_vulkan.c src/ternary_avx512.c $(NEON_SRC) src/ternary_u8.c src/ternary_prof.c

# ternary_dispatch.c llama a ternary_vulkan_probe() (solo deteccion) --
# cualquier target que lo incluya necesita tambien este .c en el link.
GPU_SRC = src/ternary_vulkan.c src/ternary_avx512.c $(NEON_SRC) src/ternary_u8.c src/ternary_prof.c   # (u8: activaciones uint8, que Dense necesita en el link)  Vulkan: solo deteccion; avx512: nucleos AVX-512 con dispatch en runtime

.PHONY: test-python all test clean test-kernels bench test-infer engine-libs

# Esta distribucion (sdist) puede no incluir tests/: en ese caso 'make' solo construye la libreria.
HAS_TESTS := $(wildcard tests/test_core.c)
ifeq ($(HAS_TESTS),)
all: libternaryai.so
else
all: tests/test_targets tests/test_pool tests/test_core tests/test_dense tests/test_backward tests/test_image_conv tests/test_pool tests/test_threadpool tests/test_stack tests/test_threads_layers tests/test_model_io tests/test_tfile tests/test_embedding tests/test_multimatrix tests/test_softmax tests/test_gqa_rope tests/test_gqa_scale tests/test_gqa_smear tests/test_gqa_layer tests/test_gqa_learn tests/test_gqa_window tests/test_gqa_scaling tests/test_rmsnorm tests/test_rmsnorm_layer tests/test_swiglu tests/test_head tests/test_residual tests/test_tokenizer tests/test_tokenizer_special tests/test_dataset tests/test_dispatch tests/test_simple tests/test_ids tests/test_blocks tests/test_kernels tests/test_avx512 tests/test_avx512_u8 tests/test_neon_u8 tests/test_apply_parallel tests/test_layers_pool tests/test_u8 tests/test_u8_stack tests/test_swiglu_u8 tests/test_relu2 tests/test_r2 tests/test_elemnorm tests/test_gqa_learn_u8 tests/test_u8_block tests/test_gqa_u8_window tests/test_u8_input tests/test_infer tests/test_infer_kern tests/test_neon_attn tests/test_neon_norm tests/test_neon_elem libternaryai.so
endif

tests/test_core: $(SRC) tests/test_core.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

tests/test_dense: src/ternary_core.c src/ternary_dense.c src/ternary_dispatch.c src/ternary_threadpool.c $(GPU_SRC) tests/test_dense.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

tests/test_backward: src/ternary_core.c src/ternary_dense.c src/ternary_dispatch.c src/ternary_threadpool.c $(GPU_SRC) tests/test_backward.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

tests/test_pool: src/ternary_threadpool.c tests/test_pool.c
	$(CC) $(CFLAGS) src/ternary_threadpool.c tests/test_pool.c -o $@ -lpthread

tests/test_threadpool: src/ternary_core.c src/ternary_dense.c src/ternary_dispatch.c src/ternary_threadpool.c $(GPU_SRC) tests/test_threadpool.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

STACK_SRC = src/ternary_core.c src/ternary_dense.c src/ternary_conv2d.c src/ternary_dispatch.c src/ternary_threadpool.c src/ternary_tensor.c src/ternary_embedding.c src/ternary_layer.c src/ternary_activation.c src/ternary_model_io.c src/ternary_json.c src/ternary_tfile.c src/ternary_stack.c src/ternary_rng.c src/ternary_softmax.c src/ternary_gqa.c src/ternary_rmsnorm.c src/ternary_swiglu.c src/ternary_head.c src/ternary_loss.c src/ternary_residual.c $(GPU_SRC)

# Contenedor de capas (Dense+Conv2D via vtable generica) -- ver el
# comentario grande al principio de tests/test_stack.c.
tests/test_threads_layers: $(STACK_SRC) tests/test_threads_layers.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

tests/test_stack: $(STACK_SRC) tests/test_stack.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

# Guardado/carga de modelo (checkpoint completo y export empaquetado) --
# ver el comentario grande al principio de tests/test_model_io.c.
tests/test_embedding: $(STACK_SRC) tests/test_embedding.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

tests/test_multimatrix: $(STACK_SRC) tests/test_multimatrix.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

# Softmax con puntajes enteros: modulo aislado, solo necesita libm.
tests/test_softmax: src/ternary_softmax.c src/ternary_avx512.c $(NEON_SRC) tests/test_softmax.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

# Escala float de GQA (Wo) y escalas sin efecto (Wq/Wk/Wv).
tests/test_gqa_scale: $(STACK_SRC) tests/test_gqa_scale.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

# Smeared keys de GQA (nucleos AVX-512 bit a bit, reglas, causalidad, backward).
tests/test_gqa_smear: $(STACK_SRC) tests/test_gqa_smear.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

# GQA como capa del TernaryStack (registro, stack==manual, checkpoint, export).
# RMSNorm: matematica, propiedades y gradientes contra una referencia en double.
tests/test_rmsnorm: $(STACK_SRC) tests/test_rmsnorm.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

tests/test_rmsnorm_layer: $(STACK_SRC) tests/test_rmsnorm_layer.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

# SwiGLU: matematica contra referencia, gradientes, capa en el stack, las dos compuertas.
tests/test_swiglu: $(STACK_SRC) tests/test_swiglu.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

# Cabeza de salida (lm_head) y perdida de entropia cruzada: matematica, gradientes, capa en el stack, aprendizaje.
tests/test_apply_parallel: $(STACK_SRC) tests/test_apply_parallel.c
	$(CC) $(CFLAGS) $(STACK_SRC) tests/test_apply_parallel.c -o $@ -lm -ldl -lpthread

# Activaciones uint8 (ver include/ternary_u8.h): cuantizador, producto entero exacto y Dense con entrada uint8.
tests/test_swiglu_u8: $(STACK_SRC) tests/test_swiglu_u8.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

# Regla R2 (ver include/ternary_dense.h).
tests/test_r2: $(STACK_SRC) tests/test_r2.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

# Normalizacion por elemento del gradiente de la cabeza (ver include/ternary_dense.h).
tests/test_elemnorm: $(STACK_SRC) tests/test_elemnorm.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

# ReLU² sin compuerta (gate_mode=2 de swiglu).
tests/test_relu2: $(STACK_SRC) tests/test_relu2.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

tests/test_gqa_learn_u8: $(STACK_SRC) tests/test_gqa_learn_u8.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

tests/test_u8_block: $(STACK_SRC) tests/test_u8_block.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

tests/test_gqa_u8_window: $(STACK_SRC) tests/test_gqa_u8_window.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

tests/test_u8_stack: $(STACK_SRC) tests/test_u8_stack.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

tests/test_u8: src/ternary_core.c src/ternary_dense.c src/ternary_dispatch.c src/ternary_threadpool.c $(GPU_SRC) tests/test_u8.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

tests/test_layers_pool: $(STACK_SRC) tests/test_layers_pool.c
	$(CC) $(CFLAGS) $(STACK_SRC) tests/test_layers_pool.c -o $@ -lm -ldl -lpthread

tests/test_head: $(STACK_SRC) tests/test_head.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

# Conexion residual (capa envoltorio): reglas, forward/backward contra la cadena a mano, mini LM, persistencia.
tests/test_residual: $(STACK_SRC) tests/test_residual.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

tests/test_gqa_learn: $(STACK_SRC) tests/test_gqa_learn.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

tests/test_gqa_layer: $(STACK_SRC) tests/test_gqa_layer.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

# GQA con ventana deslizante: equivalencia con la atencion completa, localidad, linealidad del backward; y leyes de
# escala (memoria por token y pendiente del tiempo).
tests/test_gqa_window: $(STACK_SRC) tests/test_gqa_window.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

tests/test_gqa_scaling: $(STACK_SRC) tests/test_gqa_scaling.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

# GQA con RoPE entero (tabla en punto fijo, signo exacto).
tests/test_gqa_rope: $(STACK_SRC) tests/test_gqa_rope.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

tests/test_model_io: $(STACK_SRC) tests/test_model_io.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

# Formato de modelo (model.ternary + config.json + training_state.ternary): contenedor, JSON, corrupciones.
tests/test_tfile: $(STACK_SRC) tests/test_tfile.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

tests/test_image_conv: src/ternary_core.c src/ternary_image.c src/ternary_conv2d.c src/ternary_dense.c src/ternary_dispatch.c src/ternary_threadpool.c $(GPU_SRC) tests/test_image_conv.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

# Tokenizador BPE por bytes -- no depende de ninguna capa/motor de
# computo, solo de ternary_tokenizer.c -- ver el comentario grande al
# principio de tests/test_tokenizer.c.
tests/test_tokenizer: src/ternary_tokenizer.c src/ternary_json.c tests/test_tokenizer.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

# Tokens especiales del tokenizador (ids fijos, fronteras de entrenamiento y encode, TKZ2) --
# ver el comentario grande al principio de tests/test_tokenizer_special.c.
tests/test_tokenizer_special: src/ternary_tokenizer.c src/ternary_json.c tests/test_tokenizer_special.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

# Cargador de datasets (texto/json/jsonl) -- depende de ternary_json.c
# para el parseo JSON, pero no de ninguna capa/motor de computo.
tests/test_dataset: src/ternary_json.c src/ternary_dataset.c tests/test_dataset.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

# Objetivos con mascara de perdida (ternary_targets.h); AVX-512 / NEON se despachan en ejecucion.
tests/test_targets: src/ternary_targets.c src/ternary_threadpool.c src/ternary_avx512.c $(NEON_SRC) tests/test_targets.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

# Nucleos numericos de Dense (include/ternary_kernels.h): bit a bit contra el bucle escalar.
tests/test_kernels: include/ternary_kernels.h tests/test_kernels.c
	$(CC) $(CFLAGS) tests/test_kernels.c -o $@ -lm

# Nucleos AVX-512 contra el bucle escalar (se salta solo si la CPU no tiene AVX-512).
tests/test_avx512: src/ternary_avx512.c $(NEON_SRC) tests/test_avx512.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

tests/test_avx512_u8: src/ternary_avx512.c $(NEON_SRC) src/ternary_u8.c src/ternary_threadpool.c tests/test_avx512_u8.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

# Nucleos NEON de la ruta uint8 contra el escalar (se salta solo si no hay NEON en la compilacion o en la CPU).
tests/test_neon_u8: $(NEON_SRC) src/ternary_u8.c src/ternary_avx512.c src/ternary_threadpool.c tests/test_neon_u8.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

# Nucleos NEON (planes) contra el bucle escalar (se salta solo si no hay NEON).
tests/test_neon_planes: $(NEON_SRC) src/ternary_u8.c src/ternary_avx512.c src/ternary_threadpool.c tests/test_neon_planes.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

# Nucleos NEON (elem) contra el bucle escalar (se salta solo si no hay NEON).
tests/test_neon_elem: $(NEON_SRC) src/ternary_u8.c src/ternary_avx512.c src/ternary_threadpool.c tests/test_neon_elem.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

# Nucleos NEON (attn) contra el bucle escalar (se salta solo si no hay NEON).
tests/test_neon_attn: $(NEON_SRC) src/ternary_u8.c src/ternary_avx512.c src/ternary_threadpool.c tests/test_neon_attn.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

# Nucleos NEON (norm) contra el bucle escalar (se salta solo si no hay NEON).
tests/test_neon_norm: $(NEON_SRC) src/ternary_u8.c src/ternary_avx512.c src/ternary_threadpool.c tests/test_neon_norm.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

# Deteccion real de ternary_init() (CPU, hilos, Vulkan/OpenCL solo como reporte) y
# que el motor elegido sea siempre el escalar.
tests/test_dispatch: src/ternary_core.c src/ternary_dispatch.c src/ternary_threadpool.c $(GPU_SRC) tests/test_dispatch.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

ifneq ($(HAS_TESTS),)
test: all
	./tests/test_core
	./tests/test_kernels
	./tests/test_avx512
	./tests/test_avx512_u8
	./tests/test_neon_u8
	./tests/test_apply_parallel
	./tests/test_layers_pool
	./tests/test_u8
	./tests/test_u8_stack
	./tests/test_u8_input
	./tests/test_swiglu_u8
	./tests/test_relu2
	./tests/test_r2
	./tests/test_elemnorm
	./tests/test_gqa_learn_u8
	./tests/test_u8_block
	./tests/test_gqa_u8_window
	./tests/test_dense
	./tests/test_backward
	./tests/test_image_conv
	./tests/test_pool
	./tests/test_threadpool
	./tests/test_stack
	./tests/test_threads_layers
	TERNARYAI_NO_AVX512=1 ./tests/test_threads_layers
	./tests/test_model_io
	./tests/test_tfile
	./tests/test_embedding
	./tests/test_multimatrix
	./tests/test_softmax
	./tests/test_gqa_rope
	./tests/test_gqa_scale
	./tests/test_gqa_layer
	./tests/test_gqa_learn
	./tests/test_gqa_window
	./tests/test_gqa_scaling
	./tests/test_rmsnorm
	./tests/test_rmsnorm_layer
	./tests/test_swiglu
	./tests/test_head
	./tests/test_residual
	./tests/test_tokenizer
	./tests/test_tokenizer_special
	./tests/test_dataset
	./tests/test_targets
	./tests/test_simple
	./tests/test_ids
	./tests/test_blocks
	./tests/test_dispatch
	./tests/test_infer_kern
	./tests/test_infer

# Medicion de velocidad (en el dispositivo): make -j8 all bench && sh velocidad.sh
bench: tests/bench_neon tests/bench_infer_kern tests/bench_infer tests/bench_matvec_var
else
test bench:
	@echo "Esta distribucion no incluye las pruebas (tests/). Usa el proyecto completo para ejecutarlas." >&2; exit 1
endif

clean:
	rm -f tests/test_relu2 tests/test_r2 tests/test_elemnorm tests/test_avx512 tests/test_avx512_u8 tests/test_neon_u8 tests/test_apply_parallel tests/test_layers_pool tests/test_kernels tests/test_core tests/test_dense tests/test_backward tests/test_image_conv tests/test_threadpool tests/test_stack tests/test_threads_layers tests/test_model_io tests/test_tfile tests/test_embedding tests/test_multimatrix tests/test_softmax tests/test_gqa_rope tests/test_gqa_scale tests/test_gqa_layer tests/test_gqa_learn tests/test_gqa_window tests/test_gqa_scaling tests/test_rmsnorm tests/test_rmsnorm_layer tests/test_swiglu tests/test_head tests/test_residual tests/test_tokenizer tests/test_dataset tests/test_targets tests/test_dispatch tests/test_simple tests/test_ids tests/test_blocks tests/test_u8 tests/test_u8_stack tests/test_u8_input tests/test_infer tests/test_infer_kern tests/test_neon_attn tests/test_neon_norm tests/test_neon_elem tests/bench_neon tests/bench_infer_kern tests/bench_infer tests/bench_matvec_var libternaryai.so libternaryai_engine_scalar.so libternaryai_engine_avx512.so libternaryai_engine_neon.so

# --- libternaryai: libreria compartida que usa la fachada Python (ctypes) ---
# Solo se exportan los simbolos ternaryai_* (ver TERNARYAI_API en include/ternaryai.h).
LIB_SRC = src/ternary_core.c src/ternary_dense.c src/ternary_conv2d.c src/ternary_image.c \
          src/ternary_dispatch.c src/ternary_threadpool.c \
          src/ternary_tensor.c src/ternary_embedding.c src/ternary_layer.c src/ternary_activation.c src/ternary_model_io.c src/ternary_json.c src/ternary_tfile.c src/ternary_stack.c \
          src/ternary_rng.c src/ternary_softmax.c src/ternary_gqa.c src/ternary_rmsnorm.c src/ternary_swiglu.c src/ternary_head.c src/ternary_loss.c src/ternary_residual.c src/ternary_tokenizer.c src/ternary_dataset.c src/ternary_targets.c \
          src/ternary_infer_load.c src/ternary_infer_ternam.c src/ternary_infer_ops.c src/ternary_infer_k_scalar.c src/ternary_infer_k_avx512.c src/ternary_infer_k_neon.c src/ternary_infer_engine.c src/ternary_infer_gen.c src/ternary_infer_cache.c \
          src/ternaryai.c $(GPU_SRC)

libternaryai.so: $(LIB_SRC) include/ternaryai.h
	$(CC) $(CFLAGS) -fPIC -fvisibility=hidden -shared $(LIB_SRC) -o $@ $(LDLIBS)

# --- Libreria de SOLO motor (API plana ternaryai_infer_*), con UNA vectorizacion fija compilada (nunca las dos) ---
# Desde Python: tn.build_engine_lib(salida, target="scalar"|"avx512"|"auto", cc=...) hace lo mismo para la plataforma de destino.
ENGINE_SRC = src/ternary_threadpool.c src/ternary_infer_cache.c src/ternary_infer_engine.c src/ternary_infer_gen.c src/ternary_infer_load.c src/ternary_infer_ops.c src/ternary_infer_ternam.c src/ternary_json.c src/ternary_tokenizer.c
ENGINE_MAP = -Wl,--version-script=src/engine_export.map

libternaryai_engine_scalar.so: $(ENGINE_SRC) src/ternary_infer_k_scalar.c src/engine_export.map
	$(CC) $(CFLAGS) -DTERNARYAI_FIXED_SCALAR -fPIC -shared $(ENGINE_SRC) src/ternary_infer_k_scalar.c -o $@ $(ENGINE_MAP) -lm -lpthread

libternaryai_engine_avx512.so: $(ENGINE_SRC) src/ternary_infer_k_avx512.c src/engine_export.map
	$(CC) $(CFLAGS) -DTERNARYAI_FIXED_AVX512 -mavx512f -mavx512bw -mavx512vl -mavx512dq $(if $(VNNI),-mavx512vnni) -fPIC -shared $(ENGINE_SRC) src/ternary_infer_k_avx512.c -o $@ $(ENGINE_MAP) -lm -lpthread

# NEON (ARMv7/ARMv8): solo con compilador ARM; en ARM de 32 bits hace falta -mfpu=neon (o neon-vfpv4) si el compilador no lo trae por defecto.
libternaryai_engine_neon.so: $(ENGINE_SRC) src/ternary_infer_k_neon.c src/ternary_infer_k_scalar.c src/engine_export.map
	$(CC) $(CFLAGS) -DTERNARYAI_FIXED_NEON -fPIC -shared $(ENGINE_SRC) src/ternary_infer_k_neon.c src/ternary_infer_k_scalar.c -o $@ $(ENGINE_MAP) -lm -lpthread

ENGINE_LIBS = libternaryai_engine_scalar.so libternaryai_engine_avx512.so
ifneq (,$(filter arm% aarch64%,$(shell $(CC) -dumpmachine 2>/dev/null)))
ENGINE_LIBS += libternaryai_engine_neon.so
endif
engine-libs: $(ENGINE_LIBS)

# API plana de libternaryai contra el camino manual (ver el comentario grande
# al principio de tests/test_simple.c).
tests/test_simple: $(LIB_SRC) tests/test_simple.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

tests/test_u8_input: $(LIB_SRC) tests/test_u8_input.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

# ids + entropia cruzada por la API plana (ver el comentario al principio de tests/test_ids.c).
tests/test_ids: $(LIB_SRC) tests/test_ids.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

# bloques residuales por la API plana (ver el comentario al principio de tests/test_blocks.c).
tests/test_blocks: $(LIB_SRC) tests/test_blocks.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

ifneq ($(wildcard python/tests),)
# Tests de la interfaz Python (requiere python3 y numpy)
test-python: libternaryai.so
	cd python && python3 -m unittest discover -s tests -t .
else
test-python:
	@echo "Esta distribucion no incluye las pruebas (python/tests/). Usa el proyecto completo para ejecutarlas." >&2; exit 1
endif

# Motor de inferencia (include/ternary_infer.h): comparacion con la API de la libreria desde Python (python/tests/test_infer.py)
# y pruebas funcionales en C con un modelo exportado: make test-infer MODEL=carpeta
tests/test_infer_kern: $(LIB_SRC) tests/test_infer_kern.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

tests/test_infer: $(LIB_SRC) tests/test_infer.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

tests/bench_matvec_var: $(LIB_SRC) tests/bench_matvec_var.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

tests/bench_infer_kern: $(LIB_SRC) tests/bench_infer_kern.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

tests/bench_infer: $(LIB_SRC) tests/bench_infer.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

ifneq ($(HAS_TESTS),)
test-infer: tests/test_infer tests/bench_infer
	./tests/test_infer $(MODEL)
	./tests/bench_infer $(MODEL)
else
test-infer:
	@echo "Esta distribucion no incluye las pruebas (tests/). Usa el proyecto completo para ejecutarlas." >&2; exit 1
endif

# Benchmark NEON contra escalar (ejecutar en el dispositivo; bajo qemu solo comprueba que corre).
tests/bench_neon: $(NEON_SRC) src/ternary_u8.c src/ternary_avx512.c src/ternary_threadpool.c tests/bench_neon.c tests/bench_neon_elem.h
	$(CC) $(CFLAGS) $(filter %.c,$^) -o $@ $(LDLIBS)
