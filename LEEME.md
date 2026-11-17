# TernaryAI

*English version: [README.md](README.md)* · *Referencia completa de la sintaxis: [docs/MANUAL.md](docs/MANUAL.md)*

Redes neuronales cuyos pesos son estrictamente **−1, 0 o +1**, con el cálculo en **C11** y una interfaz en Python (`import ternaryai`).
Pensada para dispositivos pequeños: entrena y genera en un dispositivo Android de gama baja de 2018 (ARMv7 de 32 bits, ~1 GB de RAM) y aprovecha
AVX-512 en x86-64 y NEON en ARM cuando están disponibles.

- **Pesos ternarios sin copia float por peso.** La regla de actualización R2 guarda por cada peso un contador de 8 bits con histéresis (no un peso
  float "en la sombra" como en QAT o BitNet). Cada capa lleva una escala float final.
- **Entrenamiento y inferencia en el mismo paquete.** El motor de inferencia da logits idénticos bit a bit a `Model.predict`, y los núcleos
  escalar, AVX-512 y NEON dan los mismos bits.
- **Modelos de lenguaje de punta a punta.** Tokenizador BPE a nivel de bytes con tokens especiales, lector de datasets (formato propio y los de
  Hugging Face más usados), generación con razonamiento (`<|think|>…<|/think|>`), verificadores, pass@k y fine-tuning por rejection sampling.
- **Una sola dependencia de Python: NumPy**, y solo en la interfaz (arrays de entrada y salida). El cálculo no pasa por NumPy.

## Qué contiene

| Componente | Qué hace |
|---|---|
| Capas (`tn.Dense`, `Conv2D`, `Embedding`, `GQA`, `SwiGLU`, `RMSNorm`, `LMHead`, `Residual`) | Bloques para clasificadores y transformers (atención con consultas agrupadas, RoPE entera, ventana deslizante, activación SwiGLU o ReLU²) |
| `tn.Model` | `fit` (mini-lotes, máscara de pérdida, checkpoints), `predict`, `save` / `load`, `export` |
| `tn.Tokenizer` | BPE de bytes en C, con tokens especiales; `tn.CHAT_SPECIAL_TOKENS` trae los 13 de conversación y razonamiento |
| `tn.dataset` | `load`, `tokenize`, `targets`: formato propio `chat`, ChatML/OpenAI `messages`, ShareGPT `conversations`, `prompt`+`completion`, Alpaca y `text` |
| `tn.Engine` / `Session` | Inferencia con caché KV, muestreo (`temperature`, `top_k`, `top_p`, …), `reply`, `sample`, lotes de conversaciones, hilos |
| Exportación | `.ternam` (un solo archivo, carga con `mmap`), librería de solo motor (`build_engine_lib`) y librería fusionada con el modelo incrustado (`export_fused`) |
| `tn.reasoning` | Verificadores, `evaluate` (pass@k), `rejection_sample` y `rft` |

## Instalación

Hace falta Python ≥ 3.10 y NumPy ≥ 1.21.6. Esta es una versión alfa (1.0.0a1), así que `pip` necesita la opción `--pre`. Hay tres formas de instalar:

- **Desde PyPI:** `pip install --pre ternaryai`. En Linux x86-64 con glibc ≥ 2.34 instala un wheel ya compilado (incluye la librería nativa y las fuentes en C que necesita; no hace falta compilador). En otras plataformas, por ejemplo Termux en Android, pip lo compila desde el código fuente, y para eso hacen falta un compilador C11 y `make`.
- **Wheel desde GitHub:** descarga `ternaryai-1.0.0a1-py3-none-manylinux_2_34_x86_64.whl` desde la página de [Releases](https://github.com/FDeveloper-hub/Project-TernaryAI/releases) e instálalo con `pip install` seguido del nombre del archivo.
- **Desde el código fuente:** `pip install .` desde un clon de este repositorio. La instalación ejecuta el Makefile.

Para trabajar en la propia librería, se compila en el sitio y se usa el paquete desde el código fuente:

```bash
make libternaryai.so                 # compila la librería nativa (detecta AVX-512 / NEON en tiempo de ejecución)
export PYTHONPATH=$PWD/python        # o: pip install -e .
python3 -c "import ternaryai as tn; print(tn.__name__, 'ok')"
```

Probado en Linux x86-64 con Python 3.10, 3.11, 3.12 y 3.13 y NumPy 1.21.6, 1.24.4, 2.2.6, 2.4.6 y 2.5.5, y en ARMv7 (Termux en un dispositivo ARMv7 de 32 bits, y bajo qemu). No se ha probado en Windows ni en macOS.
`TERNARYAI_NO_AVX512=1` / `TERNARYAI_NO_NEON=1` fuerzan el núcleo escalar; `TERNARYAI_NO_VNNI=1` conserva AVX-512 pero no usa el forward entero con VNNI (el resultado es idéntico bit a bit).

## Ejemplo: de un dataset a un modelo que conversa

Un archivo `datos.jsonl` con una conversación por línea, en el formato propio (`user` / `model`):

```json
{"chat": [{"user": "Hola"}, {"model": "Hola, ¿en qué puedo ayudarte?"}]}
{"chat": [{"user": "¿Cuánto es 2+2?"}, {"model": "<think>2+2=4</think>Es 4."}]}
```

```python
import ternaryai as tn

examples, skipped = tn.dataset.load("datos.jsonl")               # una cadena por conversación, con los marcadores de rol
tok = tn.Tokenizer(400, special_tokens=list(tn.CHAT_SPECIAL_TOKENS))
tok.train("\n".join(examples).encode())
x, y = tn.dataset.tokenize(examples, tok, seq_len=64)            # ids + objetivos: solo se entrena lo que dice el asistente

model = tn.Model(64, [
    tn.Embedding(tok.vocab_size, 48),
    *tn.transformer_block(heads=4, hidden=96, kv_heads=2, gate_mode=2),
    tn.RMSNorm(activation="uint8"),
    tn.LMHead(tok.vocab_size),
])
model.fit(x, y, epochs=400)

engine = model.engine(tok)                                       # motor de inferencia (instantánea del modelo)
r = engine.session().reply("¿Cuánto es 2+2?", max_new_tokens=40)
print(r.text, "|", r.thinking)                                   # Es 4. | 2+2=4
print(engine.session().reply("Hola", max_new_tokens=40).text)   # Hola, ¿en qué puedo ayudarte?
```

(Con 8 conversaciones de cada tipo repetidas, como en esta prueba; el resultado exacto depende de la semilla y del tamaño del modelo.)

Para guardar el modelo: `model.save(carpeta)` (con los contadores, para seguir entrenando) o `model.export(carpeta)` (solo inferencia);
`tn.export_ternam(carpeta, "modelo.ternam")` lo deja en un archivo autocontenido que `tn.Engine("modelo.ternam")` abre con `mmap`.

## Conversación y razonamiento

`Session.reply(mensajes, ...)` arma el prompt igual que el cargador de datasets (`<|rol|>\ntexto\n`) y se detiene en `<|end|>` y `<|user|>`.
Devuelve un `Reply` con `.text`, `.thinking`, `.stop` e `.ids`.

- El bloque `<|think|>…<|/think|>` va a `.thinking` y se quita de `.text`. Con `show_thinking=True` queda dentro de `.text`, con sus etiquetas,
  para leer la cadena de pensamiento en su sitio.
- `think_budget=N` limita los tokens de razonamiento (al llegar se fuerza `<|/think|>`); `force_think=True` empieza ya dentro de `<|think|>`.
- `dataset.load(..., reasoning="keep" | "drop")` acepta `<think>…</think>` y los campos `reasoning_content`, `reasoning` y `thinking`.
- `Engine.chat_encode`, `Engine.split_thinking` y `Engine.special_id` sirven para construir y analizar secuencias a mano.

## Evaluación y fine-tuning por rejection sampling

```python
R = tn.reasoning
tasks = [R.Task("¿Cuánto es 2+2?", reference=4), ("¿Cuánto es 3+5?", 8)]
ok = R.all_of(R.numeric(), R.has_thinking())            # correcta Y con razonamiento bien cerrado
print(R.evaluate(engine, tasks, ok, n=8, k=(1, 4), temperature=0.8, max_new_tokens=64).summary())

R.rft(model, tok, tasks, ok, rounds=3, n=8, epochs=2)   # muestrea, conserva lo verificado y entrena con ello
```

- `Engine.sample(mensajes, n)` genera `n` respuestas a la misma conversación (el prompt se procesa una vez, las candidatas corren en un lote, la candidata
  `i` usa la semilla `seed + i`).
- Los verificadores (`exact_match`, `contains`, `regex_match`, `numeric`, `has_thinking`, `all_of`, o cualquier función `verify(reply, task)`)
  solo comparan texto: nunca ejecutan lo que genera el modelo.
- `evaluate` da pass@k (estimador sin sesgo), recompensa media, longitudes y motivos de parada. `rft` acepta `replay=(x, y)` para mezclar los datos de
  entrenamiento originales y limitar el olvido.

## Rendimiento

Medido contra un modelo BitNet b1.58 QAT escrito en PyTorch (solo CPU, PyTorch 2.14.1), con la misma arquitectura y el mismo número de pesos.
Condiciones: 1 hilo en 1 núcleo (Xeon de 2,1 GHz, AVX-512 VNNI), 30 épocas, lote 8, contexto 64, texto a nivel de byte (`pydoc_data.topics`), una sola semilla.
El modelo de PyTorch mantiene pesos sombra en float, los ternariza con absmean y un estimador straight-through, usa activaciones int8 por token, Adam (lr 0,01),
y embedding y cabeza de salida ternarios. Picos medidos en ese núcleo: fp32 215 GFLOP/s, int8 VNNI 677 GOP/s, DRAM 16 GB/s.

| | d=96, TernaryAI | d=96, PyTorch QAT | d=640, TernaryAI | d=640, PyTorch QAT |
|---|---|---|---|---|
| Pesos | 307 mil | 307 mil | 11,8 M | 11,8 M |
| Entrenamiento (tokens/s) | 18.798 | 12.289 | 803 | 819 |
| Pérdida de evaluación | 1,833 | 1,801 | 2,681 | 2,376 |
| Precisión de evaluación | 53,1 % | 56,1 % | 35,9 % | 40,0 % |
| Prefill (tokens/s) | 150.216 | 26.211 | 8.678 | 3.427 |
| Decodificación, 1 conversación (tokens/s) | 115.199 | 990 | 3.869 | 291 |
| Decodificación, lote de 32 (tokens/s) | 117.854 | 14.678 | 6.075 | 1.560 |
| RAM pico del proceso (anónima) | 34 MB | 196 MB | 193 MB | 616 MB |
| RSS pico del proceso | 65 MB | 303 MB | 227 MB | 725 MB |
| Modelo exportado | 0,08 MB | 0,31 MB (int8) | 2,4 MB | 11,8 MB (int8) |

Cómo leerlo:

- **La velocidad de entrenamiento es equivalente.** TernaryAI es 1,5 veces más rápido en d=96 y empata en d=640 (803 contra 819 tokens/s).
- **La inferencia y la memoria son lo que gana la librería.** Con un lote de 32 conversaciones es 8 veces (d=96) y 3,9 veces (d=640) más rápido; el prefill es 5,7 y 2,5 veces más rápido;
  el proceso usa entre 3 y 6 veces menos RAM, y el modelo exportado es de 4 a 5 veces más pequeño. Con un tope duro de 200 MB de RAM (sin swap) TernaryAI entrenó 11,8 M de pesos,
  mientras que el QAT de PyTorch llegó solo a 307 mil, unas 38 veces menos.
- **Las cifras de una sola conversación están infladas.** El modelo de PyTorch corre en float, en modo eager, y con modelos tan pequeños su tiempo lo dominan Python y el
  lanzamiento de operaciones. La fila del lote de 32 es la más fiable.
- **Calidad:** las pérdidas de evaluación son cercanas (1,833 contra 1,801 en d=96, 2,681 contra 2,376 en d=640); el QAT de PyTorch es algo mejor en ambos.
- **Límites de la medición:** una semilla, entrenamiento corto, modelos pequeños, y el QAT de PyTorch no tiene LayerNorm interna en su BitLinear (a diferencia del BitLinear original).
  Además, las dos reglas de entrenamiento difieren (R2 sin pesos float contra Adam en float32).
- **Hilos** (`tn.set_threads(n)`, `tn.set_engine_threads(n)`): el resultado es idéntico bit a bit con cualquier cantidad.
- **Formato en memoria:** en disco un peso ocupa 1,6 bits (5 valores ternarios por byte); el motor mantiene los pesos a 2 bits y una caché KV compacta.

## Límites conocidos

- Todo se ha validado con modelos pequeños (hasta unos 0,5 M de pesos en los bloques). No hay resultados con modelos grandes.
- A igual número de pesos, un modelo float da mejor calidad que el ternario: en una prueba con novelas en español, a los 25 minutos el float tenía pérdida 4,32 contra 4,51 del ternario.
  La ventaja del ternario está en la memoria, el tamaño en disco y la velocidad de inferencia, no en la calidad por peso.
  Es lo esperado: un modelo ternario (1,58 bits por parámetro) tiene menos capacidad que uno float con el mismo número de parámetros. Los estudios de Microsoft sobre BitNet b1.58
  ([arXiv:2402.17764](https://arxiv.org/abs/2402.17764)) reportan que esa desventaja desaparece a partir de unos 3 mil millones de parámetros (frente a una base FP16, en perplejidad). Es un resultado
  reportado por esos estudios, con una receta de entrenamiento distinta de la nuestra; no se ha reproducido con esta librería, que solo se ha validado con modelos pequeños.
- La atención a larga distancia pide subir `rope_base` (por defecto 10; con 1000 se validó a 512 tokens).

## Pruebas

```bash
make test                # pruebas de C (≈15 min)
make test-python         # pruebas de Python (compila antes la librería)
```


## Licencia

Apache License 2.0. Texto completo en [`LICENSE`](LICENSE) y avisos en [`NOTICE`](NOTICE). Copyright 2026 FDeveloper-hub.
