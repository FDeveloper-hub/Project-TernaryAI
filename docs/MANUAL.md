# Manual de TernaryAI

*English version: [MANUAL.en.md](MANUAL.en.md)*

Este manual describe toda la interfaz pública de Python de TernaryAI (`import ternaryai`). Para qué es la librería, cómo se instala y cuáles son sus límites,
ver [LEEME.md](../LEEME.md). Todos los bloques de código ` ```python ` de este manual se ejecutan, en orden y en un mismo espacio de nombres,
con `tests/test_manual.py`: lo que se muestra aquí funciona con la versión instalada. Los bloques ` ```text ` y ` ```bash ` no se ejecutan.

Convención: `import ternaryai as tn`.

## Índice

1. [Inicio rápido](#1-inicio-rápido)
2. [Ideas básicas](#2-ideas-básicas)
3. [`Model`: construir, entrenar, predecir](#3-model-construir-entrenar-predecir)
4. [Capas](#4-capas)
5. [Tokenizador](#5-tokenizador)
6. [Datasets](#6-datasets)
7. [Modelos de lenguaje de principio a fin](#7-modelos-de-lenguaje-de-principio-a-fin)
8. [Inferencia: `Engine` y `Session`](#8-inferencia-engine-y-session)
9. [Razonamiento: `tn.reasoning`](#9-razonamiento-tnreasoning)
10. [Guardar y exportar](#10-guardar-y-exportar)
11. [Hilos y núcleos de cálculo](#11-hilos-y-núcleos-de-cálculo)
12. [Errores](#12-errores)
13. [Variables de entorno](#13-variables-de-entorno)
14. [Tabla de referencia](#14-tabla-de-referencia)

---

## 1. Inicio rápido

Un clasificador: la entrada son valores -1, 0 o 1 y la salida es una etiqueta -1 o +1.

```python
import numpy as np
import ternaryai as tn

rng = np.random.default_rng(0)
X = rng.choice([-1, 0, 1], size=(64, 8)).astype(np.int8)
y = (X[:, :4].sum(axis=1) > 0).astype(np.float32) * 2 - 1      # +1 si los 4 primeros suman más que 0

model = tn.Model(8, [tn.Dense(16), tn.Dense(1)], seed=1)
history = model.fit(X, y, epochs=200)
print(history)                                                 # <ternaryai.History epochs 0..200 final_loss=...>
print((model.predict(X, sign=True).ravel() == y).mean())       # 1.0
```

Un modelo de lenguaje que conversa está en la [sección 7](#7-modelos-de-lenguaje-de-principio-a-fin).

## 2. Ideas básicas

- **Pesos ternarios.** Cada peso es -1, 0 o +1. No hay una copia float de cada peso: la regla de actualización **R2** guarda por peso un contador int8 con histéresis.
  Cada capa tiene al final una escala float.
- **Activaciones uint8.** Entre capas, los valores viajan como uint8 con una escala y un punto cero por muestra (o por token). Es la única activación disponible y
  es el valor por defecto de todas las capas (`activation="uint8"`).
- **Dos tipos de modelo.**
  - *Entradas ternarias o uint8* (clasificadores, convoluciones): `x` es un arreglo de números, la pérdida es `"mse"`.
  - *Ids de tokens* (modelos de lenguaje): la primera capa es `Embedding`, `x` y `y` son enteros de forma `(n, seq)`, la pérdida es `"xent"`.
- **Determinismo.** El mismo programa da los mismos bits con cualquier número de hilos y con cualquiera de los núcleos (escalar, AVX-512, NEON).
- **Las capas se generan del backend.** `tn.Dense`, `tn.GQA`, etc. salen de la lista de capas de la biblioteca en C; si se añade una capa en C, aparece en Python al recompilar
  (ver `tn.available_layers()`).

## 3. `Model`: construir, entrenar, predecir

### Construcción

```python
model = tn.Model(8, [tn.Dense(16), tn.Dense(1)], seed=1)
model.add(tn.Dense(1))          # también se puede añadir una a una; add devuelve el modelo
print(model.summary())
print(model.input_shape, model.output_shape, model.epoch)
```

```text
Model(inputs, layers=(), *, seed=0, threshold=0.5, input="ternary", backend=None)
```

| Parámetro | Significado |
|---|---|
| `inputs` | Forma de la entrada: un entero, o una tupla como `(28, 28, 1)` (alto, ancho, canales). Para modelos de lenguaje es el largo de la secuencia |
| `layers` | Capas iniciales (ver [capas](#4-capas)) |
| `seed` | Semilla de la inicialización de pesos |
| `threshold` | Umbral para ternarizar la entrada: valores `> threshold` pasan a +1, `< -threshold` a -1, el resto a 0. Datos que ya son -1/0/1 se usan tal cual |
| `input` | `"ternary"` (por defecto) o `"uint8"`: con `"uint8"` la entrada se mantiene como float y la primera capa la lee como uint8 (255 niveles, escala y punto cero por muestra) |

Propiedades: `input_shape`, `output_shape`, `layers`, `epoch` (épocas entrenadas), `r2` (parámetros de la regla R2), `input`, `threshold` (estas dos se pueden cambiar),
`closed`. Las capas solo se pueden añadir antes de empezar a entrenar (`ArchitectureError` si no).

### Entrenar: `fit`

```text
model.fit(x, y, epochs=100, *, initial_epoch=None, verbose=0, loss=None, scale_lr=None, restore_best=None,
          batch_size=None, shuffle=True, shuffle_seed=0, tau=None, beta=None, inc=None,
          zero_target=None, reset=None, head_beta2=None) -> History
```

| Parámetro | Significado |
|---|---|
| `x`, `y` | Para entradas ternarias: `x` de forma `(n, *input_shape)` o `(n, características)`, `y` de forma `(n, salidas)` (float). Para ids: ambos enteros `(n, seq)`; `y[i, j]` es el id esperado en la posición `j` (normalmente el siguiente token) y un valor negativo ignora esa posición |
| `epochs` | Épocas de esta llamada. Llamadas sucesivas continúan el calendario de aprendizaje |
| `initial_epoch` | Fuerza la época desde la que se reanuda (por ejemplo tras `load`) |
| `verbose` | `N`: imprime la pérdida cada `N` épocas (no cambia el resultado) |
| `loss` | `"mse"` (por defecto en modelos de entrada ternaria) o `"xent"` (por defecto, y único, en modelos de ids) |
| `batch_size` | `None` (por defecto) = cada época es un solo paso con todos los datos. Con `B < n`, cada época recorre las muestras en pasos de `B` y cada paso hace su propia actualización |
| `shuffle`, `shuffle_seed` | Con minilotes, baraja las muestras en cada época con una permutación que depende solo de `(shuffle_seed, época)`: es reproducible |
| `scale_lr` | Tasa de aprendizaje de las escalas float (por defecto 0.02 con `"mse"` y 0.1 con `"xent"`) |
| `restore_best` | Solo `"xent"` (por defecto `True`): al final de cada `fit` el modelo vuelve al estado con menor pérdida de entrenamiento. Protege del colapso tardío |
| `tau`, `beta`, `inc`, `zero_target`, `reset`, `head_beta2` | Parámetros de la regla R2 (abajo). Si se dejan en `None` se conserva el valor actual |

Devuelve un `History` con `loss` (lista, una por época), `accuracy` (solo con `"xent"`: fracción de posiciones no ignoradas cuyo logit mayor era el id correcto; `None` con `"mse"`),
`first_epoch` y `last_epoch`.

**Regla R2.** Por fila, el gradiente se normaliza por su RMS y se recorta a ±3. Un contador int8 por peso lo integra; cuando el contador llega al umbral `tau`, el peso se mueve un paso y
el contador reinicia en el lado opuesto (histéresis, a `reset` veces el umbral).

| Parámetro | Por defecto | Significado |
|---|---|---|
| `tau` | 0.7 | Umbral del contador, como fracción de su rango. Rango `(0, 1]` |
| `beta` | 0.999 | Memoria del contador (casi sin decaimiento). Rango `[0, 1)` |
| `inc` | 0.03 | Paso del contador por unidad de gradiente normalizado. Zona estable: 0.02 a 0.03 |
| `zero_target` | 0.2 | Fracción objetivo de ceros por fila; ajusta los umbrales 0→±1 y ±1→0 (negativo = apagado) |
| `reset` | 0.5 | Dónde reinicia el contador tras un cambio, en múltiplos del umbral. Rango `[0, 1]` |
| `head_beta2` | 0.99 | Solo `LMHead`: normaliza su gradiente por peso con la media móvil de g². `0` lo apaga y deja la R2 por fila |

```python
print(model.r2)     # {'tau': 0.7, 'beta': 0.999, 'inc': 0.03, 'zero_target': 0.2, 'reset': 0.5, 'head_beta2': 0.99}
history = model.fit(X, y, epochs=20, batch_size=16, shuffle_seed=3, verbose=10)   # minilotes; imprime cada 10 épocas
print(history.first_epoch, history.last_epoch)
```

Los seis valores viven en el modelo, se guardan con `save()` (checkpoint de entrenamiento) y no con `export()`.

### Predecir

```python
out = model.predict(X)                  # float32, forma (n, *output_shape)
signs = model.predict(X, sign=True)     # int8 -1/0/1
print(out.shape, signs.dtype)
```

En modelos de ids, `predict(x)` recibe ids `(n, seq)` y devuelve los **logits** `(n, seq, vocab)` (sin softmax); el id predicho en cada posición es `argmax` del último eje.
Para una sola muestra use `x[None]`.

### Otros métodos

| Método | Qué hace |
|---|---|
| `summary()` | Texto con una fila por capa (los bloques `Residual` se muestran anidados) |
| `save(carpeta)` | Checkpoint completo: `model.ternary` + `config.json` + `training_state.ternary`. Se puede cargar y seguir entrenando |
| `export(carpeta)` | Solo inferencia: `model.ternary` + `config.json` |
| `export_ternam(ruta, target="auto", tokenizer=None)` | Archivo `.ternam` autocontenido; con `tokenizer=tok` lo incluye (ver [sección 10](#10-guardar-y-exportar)) |
| `engine(tokenizer=None)` | Un `Engine` de inferencia, instantánea del modelo (ver [sección 8](#8-inferencia-engine-y-session)) |
| `ffn_sparsity(reset=False)` | Fracción de la activación de SwiGLU que es exactamente su punto cero, una por capa |
| `attention_probs(layer, head)` | Probabilidades de atención `(seq, seq)` del último `predict` de **una** secuencia |
| `num_gqa()` | Número de capas GQA |
| `set_checkpoint(on=True)` | Menos memoria de activaciones con los mismos bits |
| `prof_enable(on)`, `prof_reset()`, `prof_report()` | Perfilador de ámbitos (árbol de llamadas con tiempos; no cambia resultados) |
| `close()` | Libera el modelo nativo (se puede llamar varias veces). `Model` también funciona con `with` |

### `load`

```python
model.save("checkpoint")           # checkpoint completo
model.export("exportado")          # solo inferencia
again = tn.load("checkpoint")
print(np.array_equal(again.predict(X), model.predict(X)))   # True
```

```text
tn.load(path, *, threshold=0.5, input="ternary", backend=None) -> Model
```

`threshold` e `input` no se guardan en los archivos del modelo: vuelva a pasarlos si usó valores distintos de los de por defecto.

## 4. Capas

Todas las capas aceptan `activation="uint8"` (la única activación). Los parámetros sin valor por defecto son obligatorios.

| Capa | Para qué | Parámetros (por defecto) |
|---|---|---|
| `Dense(units)` | Capa totalmente conectada | `units`, `bias=True` |
| `Conv2D(filters, kernel)` | Convolución 2D sobre `(alto, ancho, canales)`; debe ser la primera | `filters`, `kernel` (entero o `(h, w)`), `stride=1`, `bias=True` |
| `Embedding(vocab_size, embed_dim)` | Tabla de vectores ternarios indexada por id de token; debe ser la primera | `vocab_size`, `embed_dim` |
| `GQA(n_heads)` | Atención causal multicabeza con consultas agrupadas | `n_heads` (debe dividir `embed_dim`), `n_kv_heads=0` (0 = igual que `n_heads`), `rope=1`, `rope_base=10.0`, `score_shift=2`, `out_scale=1.0`, `window=0`, `sinks=0`, `smear=False` |
| `SwiGLU(hidden)` | Bloque feed-forward (sin residual) | `hidden`, `gate_mode=0`, `band=0.5`, `out_scale=1.0` |
| `RMSNorm()` | Normalización RMS por fila, con ganancia ternaria por dimensión y escala float | `eps=1e-05` |
| `LMHead(vocab)` | Cabeza del modelo de lenguaje: proyección ternaria compartida sobre la última dimensión; da logits | `vocab` |
| `Residual(layers)` | Bloque residual `salida = x + f(x)` | `layers`, `activation` |

Detalles de `GQA`:

| Parámetro | Significado |
|---|---|
| `rope` | `0` sin codificación de posición, `1` RoPE con puntajes enteros (recomendado), `2` RoPE con Q/K binarizados |
| `rope_base` | Base de frecuencias de RoPE (`> 1`). Con `10` (por defecto) la rotación es rápida: para atención a larga distancia use una base mayor (`1000` está validada a 512 tokens) |
| `score_shift` | Con `rope=1`: desplazamiento a la derecha del puntaje entero q·k (mayor = atención más plana) |
| `out_scale` | Escala inicial de la proyección de salida Wo (`< 1` hace que la rama empiece pequeña junto al flujo residual; se aprende después) |
| `window`, `sinks` | Atención de ventana deslizante: cada posición ve las últimas `window` (0 = atención causal completa; requiere `rope=1`). Con ventana, las primeras `sinks` posiciones (máximo 48) quedan visibles para todas |
| `smear` | Claves suavizadas, `K'_j = (K_j + K_{j-1}) >> 1` antes de RoPE: permite que una sola capa compare un token con el que precede a cada clave (patrón de *induction head*). Requiere `rope=1` y `window=0` |

Detalles de `SwiGLU`: `gate_mode` es `0` (silu por tabla entera, producto float), `1` (signo ternario de la compuerta, todo entero) o `2` (ReLU² sin compuerta, estilo BitNet; solo dos matrices).
`band` es la banda de ceros de la cuantización interna, relativa a la media de |h| de la fila (0 = ninguna).

Detalles de `Residual`: la entrada del bloque es la salida float de la capa anterior. Al añadirlo a un modelo se comprueba que la primera capa de la cadena tome entrada float (como `RMSNorm`),
que todas pasen gradiente a su entrada, que ninguna tome ids y que la última devuelva tantos valores como `x`. Los bloques se pueden anidar.

### `transformer_block`

```python
blocks = tn.transformer_block(heads=4, hidden=96, kv_heads=2, gate_mode=2)   # lista de dos Residual
print(len(blocks), blocks[0].layers)
```

```text
tn.transformer_block(heads, hidden, *, kv_heads=None, rope=None, rope_base=None, score_shift=None, gate_mode=None,
                     band=None, eps=None, out_scale=None, window=None, sinks=None, smear=None,
                     norm_activation="uint8", backend=None) -> list[Layer]
```

Devuelve `[Residual([RMSNorm, GQA]), Residual([RMSNorm, SwiGLU])]`, es decir `x = x + GQA(norm(x))` y luego `x = x + SwiGLU(norm(x))`. Todo argumento en `None` toma el valor de la capa a la que pertenece.
`out_scale` vale 0.03 por defecto en el bloque (no 1.0): una rama que empieza pequeña hace que los modelos profundos entrenen con más fiabilidad. Con 3 o más bloques use una dimensión de embedding de al menos 32.
Cada elemento de la lista tiene sus propios pesos, así que se puede repetir: `[*tn.transformer_block(...)] * 12`.

### Otras funciones de capas

| Función | Qué hace |
|---|---|
| `tn.layer("dense", 16)` | Construye una capa por nombre de clase o de tipo |
| `tn.available_layers()` | Lista de capas que implementa el backend |

## 5. Tokenizador

BPE a nivel de bytes, implementado en C. No hay tokens fuera de vocabulario: cualquier texto se codifica. Se construye con un tamaño de vocabulario y se entrena con un corpus.

```python
tok = tn.Tokenizer(400, special_tokens=list(tn.CHAT_SPECIAL_TOKENS))
tok.train("hola mundo, hola a todos, hola otra vez " * 20)
ids = tok.encode("hola mundo")
print(ids, tok.decode(ids) == "hola mundo")
print(tok.vocab_size, tok.num_merges, tok.num_special, tok.special_id("<|end|>"))
tok.save("tokenizer.json")
tok2 = tn.Tokenizer.load("tokenizer.json")
```

```text
tn.Tokenizer(vocab_size, special_tokens=None)
```

- `vocab_size` cuenta los 256 bytes, los especiales y las fusiones. Los **tokens especiales** tienen ids fijos (256, 257, … en el orden dado) y el BPE nunca los parte ni los fusiona; se indican al construir, antes de entrenar.
- `train(corpus)` recibe `str` o `bytes` y devuelve el propio tokenizador.
- `encode(text, parse_special=True)`: con `parse_special=False` una cadena como `"<|user|>"` se trata como texto corriente.
- `decode(ids, skip_special=False)`: con `skip_special=True` omite las marcas.
- `special_id(token)` da el id de un especial (`TokenizerError` si no lo es). Propiedades: `vocab_size`, `num_merges`, `num_special`, `special_tokens`.
- `save(ruta)` y `Tokenizer.load(ruta)` usan un archivo JSON.

`tn.CHAT_SPECIAL_TOKENS` es una tupla con los 13 especiales de conversación y razonamiento:

```text
<|user|> <|assistant|> <|end|> <|system|> <|console|> <|terminal|> <|archive|> <|log|> <|result|> <|function_call|> <|tool|> <|think|> <|/think|>
```

## 6. Datasets

`tn.dataset` convierte archivos en ejemplos de texto y luego en arreglos `(x, y)` listos para `Model.fit`. No lee Parquet ni Arrow.

```python
import json
rows = [{"chat": [{"user": "Hola"}, {"model": "Hola, ¿en qué puedo ayudarte?"}]},
        {"chat": [{"user": "¿Cuánto es 2+2?"}, {"model": "<think>2+2=4</think>Es 4."}]},
        {"chat": [{"user": "¿Cuánto es 3+5?"}, {"model": "<think>3+5=8</think>Es 8."}]}] * 8
with open("datos.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

examples, skipped = tn.dataset.load("datos.jsonl")
print(len(examples), skipped)
print(repr(examples[1]))
```

### `dataset.load(path, format=None, strict=True, reasoning="keep")`

Devuelve `(ejemplos, saltados)`: una cadena por conversación, con los turnos escritos como `<|rol|>\ntexto\n` y cerrada con `<|end|>\n`.

- `format`: `tn.dataset.TEXT_LINES`, `TEXT_WHOLE`, `JSON` o `JSONL`. Si se omite, se deduce de la extensión (`.txt` → una línea por ejemplo, `.json`, `.jsonl`). Para un `.txt` como un solo bloque hay que pedir `TEXT_WHOLE`.
- `strict`: con JSON/JSONL, `True` falla en el primer registro que no cumpla ningún esquema conocido; `False` lo salta y `saltados` cuenta cuántos.
- `reasoning`: `"keep"` conserva el razonamiento de los turnos del asistente normalizado a `<|think|>…<|/think|>` (acepta las etiquetas `<think>…</think>` y `<thinking>…</thinking>` y los campos `reasoning_content`, `reasoning` y `thinking`); `"drop"` lo quita todo, para entrenar un modelo que no razona con un dataset que sí. Las etiquetas deben estar balanceadas.

Formatos reconocidos (se detectan por la clave de cada registro; un archivo puede mezclarlos):

| Clave | Formato |
|---|---|
| `"chat"` | Propio: `[{"user": "..."}, {"model": "..."}, ...]` (roles `user`, `model`, `system`, `console`, `terminal`, `archive`, `log`, `result`) |
| `"messages"` | ChatML / OpenAI: `[{"role": ..., "content": ...}, ...]` |
| `"conversations"` | ShareGPT: `[{"from": "human" \| "gpt", "value": ...}, ...]` |
| `"prompt"` + `"completion"` | Pares de prompt y respuesta |
| `"instruction"` [+ `"input"`] + `"output"` | Alpaca |
| `"text"` | Texto plano, sin marcas de turno |

### `dataset.tokenize(examples, tokenizer, seq_len, *, ...)`

```text
tn.dataset.tokenize(examples, tokenizer, seq_len, *, train_roles=("assistant", "function_call"), roles=CHAT_ROLES,
                    train_text=True, overflow="truncate", pad_id=0, drop_empty=True) -> (x, y)
```

Devuelve `x` (ids) e `y` (objetivos de siguiente token), ambos `int32` de forma `(n, seq_len)`. `y[i] = x[i+1]` si ese token se entrena y `-1` donde no se aprende (el relleno y lo que dicen los demás roles).
Se entrena el contenido de los turnos de `train_roles` más el marcador que cierra el turno, para que el modelo aprenda a terminarlo.

- `overflow` decide qué pasa con un ejemplo de más de `seq_len` tokens: `"truncate"` conserva el principio, `"split"` lo parte en ventanas consecutivas (cada una con su propia máscara, sin ver el contexto de las anteriores), `"drop"` lo descarta.
- `train_text`: si el texto fuera de todo turno (registros `text`, corpus planos) se entrena.
- `drop_empty`: descarta las ventanas sin nada que aprender y los ejemplos de menos de 2 tokens.
- `dataset.targets(ids_list, tokenizer, ...)` hace solo la parte de los objetivos, para ejemplos que ya tokenizó usted.

```python
tok = tn.Tokenizer(400, special_tokens=list(tn.CHAT_SPECIAL_TOKENS))
tok.train("\n".join(examples).encode())
x, y = tn.dataset.tokenize(examples, tok, seq_len=64)
print(x.shape, y.shape, x.dtype, int((y >= 0).sum()), "objetivos entrenados")
```

Constantes: `dataset.CHAT_ROLES`, `dataset.DEFAULT_TRAIN_ROLES`, `dataset.TEXT_LINES`, `TEXT_WHOLE`, `JSON`, `JSONL`. Error: `tn.DatasetError`.

## 7. Modelos de lenguaje de principio a fin

Continuando con `tok`, `x` e `y` de la sección anterior:

```python
model = tn.Model(64, [
    tn.Embedding(tok.vocab_size, 48),
    *tn.transformer_block(heads=4, hidden=96, kv_heads=2, gate_mode=2),
    tn.RMSNorm(activation="uint8"),
    tn.LMHead(tok.vocab_size),
])
history = model.fit(x, y, epochs=400)
print(history)

engine = model.engine(tok)                           # motor de inferencia (instantánea del modelo)
r = engine.session().reply("¿Cuánto es 2+2?", max_new_tokens=40)
print(r.text, "|", r.thinking)                       # Es 4. | 2+2=4
print(engine.session().reply("Hola", max_new_tokens=40).text)   # Hola, ¿en qué puedo ayudarte?
```

El resultado exacto depende de la semilla y del tamaño del modelo. La secuencia de un modelo de lenguaje es la primera dimensión de `inputs` (aquí 64) y debe coincidir con `seq_len` al tokenizar.

## 8. Inferencia: `Engine` y `Session`

Un **`Engine`** es un modelo cargado para inferencia (pesos de solo lectura, compartidos entre sesiones). Una **`Session`** es una conversación: caché de claves y valores, historial de tokens y generador aleatorio.

### Crear un `Engine`

| Cómo | Qué obtiene |
|---|---|
| `model.engine(tokenizer=None)` | Instantánea del modelo en entrenamiento; el entrenamiento posterior no la cambia |
| `tn.Engine(ruta)` | Carpeta con el modelo (`model.ternary` + `config.json`, y `tokenizer.json` para funciones de texto) o un archivo `.ternam` |
| `tn.Engine.fused()` | El modelo incrustado en una biblioteca hecha con `export_fused` (ver [sección 10](#10-guardar-y-exportar)) |

Propiedades de `Engine`: `seq_len`, `dim`, `vocab`, `weight_bytes`, `fingerprint`, `has_tokenizer`. Métodos de texto (necesitan tokenizador): `encode(text, parse_special=True)`, `decode(ids, skip_special=True)`,
`special_id(token)` (da `-1` si no existe), `chat_encode(messages, ...)`, `split_thinking(ids, think_inside=False)`. `close()` libera el motor.

```python
print(engine.seq_len, engine.dim, engine.has_tokenizer, engine.special_id("<|end|>"))
print(engine.chat_encode("Hola"))              # prompt de una conversación: [<|user|>, ..., <|assistant|>]
```

### Conversar: `Session.reply`

```text
session.reply(messages, max_new_tokens=256, *, show_thinking=False, think_budget=0, force_think=False,
              after_role="\n", after_msg="\n", stop_on_user=True, **muestreo) -> Reply
```

`messages` es una cadena (un mensaje de usuario) o una lista de diccionarios `{"role", "content"}` o de pares `(rol, contenido)`. El prompt se construye igual que el lector de datasets (`<|rol|>\ntexto\n`)
y la generación termina en `<|end|>` (y en `<|user|>` si `stop_on_user`). Cada llamada reinicia la sesión primero.

```python
reply = engine.session().reply([("user", "¿Cuánto es 3+5?")], max_new_tokens=40)
print(reply.text, "|", reply.thinking, "|", reply.stop)
```

`Reply` tiene `text` (la respuesta, sin el bloque de razonamiento), `thinking` (el razonamiento, vacío si no hubo), `stop` (por qué terminó: `'token'` = el modelo cerró su turno, `'max_tokens'`, …), `ids` y `prompt_ids`.

- `show_thinking=True` deja el bloque dentro de `text`, con las etiquetas, para leer la cadena de pensamiento en su lugar.
- `think_budget=N` limita los tokens de razonamiento: al alcanzarlo se fuerza `<|/think|>`.
- `force_think=True` empieza ya dentro de `<|think|>`.
- `after_role` y `after_msg` cambian los separadores (use `""` para un modelo entrenado sin ellos).

```python
print(engine.session().reply("¿Cuánto es 2+2?", max_new_tokens=40, show_thinking=True).text)
```

### Muestreo

Las funciones que generan (`reply`, `generate`, `sample`, `generate_batch`) aceptan estas opciones:

| Opción | Significado |
|---|---|
| `temperature` | `0` = voraz (siempre el id más probable) |
| `top_k`, `top_p`, `min_p` | Filtros de muestreo |
| `repeat_penalty`, `freq_penalty`, `presence_penalty`, `penalty_last_n` | Penalizaciones de repetición |
| `logit_bias` | Diccionario `{id: sesgo}` |
| `banned` | Ids prohibidos |
| `stop_ids`, `stop_seqs`, `include_stop` | Parada por ids o por secuencias |
| `on_token` | Función llamada con cada token; si devuelve `True`, se detiene |
| `think_budget`, `think_inside` | Razonamiento (`think_inside`: el contexto ya termina dentro del bloque) |

### Control manual de una sesión

```python
s = engine.session()
s.feed(engine.encode("Hola"))                       # procesa ids; devuelve la sesión
print(s.generate(5, temperature=0))                 # continúa desde el estado actual; devuelve ids
print(s.next_position, s.top_probs(3))              # posición siguiente y (id, probabilidad) de los 3 más probables
logprobs, perplexity = engine.session().score(engine.encode(examples[0]))   # qué tan esperado es un texto
print(round(perplexity, 3))
s.reset()
```

| Método de `Session` | Qué hace |
|---|---|
| `feed(ids)` | Procesa ids y actualiza la caché |
| `generate(max_new_tokens=64, **muestreo)` | Genera desde el estado actual y devuelve los ids |
| `reply(messages, ...)` | Una respuesta de conversación completa |
| `argmax()`, `logits()`, `top_probs(n=5)` | Mira la distribución del siguiente token |
| `score(ids)` | Procesa `ids` y devuelve `(log p(ids[i] \| contexto), perplejidad)`; el primer valor es NaN si la sesión estaba vacía |
| `fork()` | Copia la sesión con su estado |
| `reset()`, `seed(valor)`, `close()` | Reinicia, fija la semilla, libera |
| `save(ruta)` / `engine.load_session(ruta)` | Guarda y restaura una sesión (por ejemplo, un prompt de sistema ya procesado) |
| `attention_probs(layer, head)` | Atención del último token (la sesión debe crearse con `record_attn=1`) |
| `next_position` | Posición del siguiente token |

`engine.session(**opciones)` acepta `max_context`, `attn_window`, `sinks`, `ctx_policy`, `keep_tokens`, `record_attn` y `max_history`.
`ctx_policy` decide qué pasa cuando se llena el contexto: `tn.CTX_ERROR` (error), `tn.CTX_TRUNCATE` (trunca) o `tn.CTX_SLIDE` (desliza la ventana).

### Varias respuestas a la vez

```python
replies = engine.sample("¿Cuánto es 3+5?", n=3, max_new_tokens=40, temperature=0.8)
print([r.text for r in replies])

sessions = [engine.session() for _ in range(2)]
for sess in sessions:
    sess.feed(engine.chat_encode("Hola"))
print(engine.generate_batch(sessions, max_new_tokens=5))     # lista de listas de ids
```

`Engine.sample(messages, n=1, max_new_tokens=256, *, seed=0, ...)` genera `n` respuestas independientes a la misma conversación: el prompt se procesa una vez y las candidatas corren como un solo lote.
La candidata `i` usa la semilla `seed + i`: mismos argumentos, mismas respuestas. Use `temperature > 0` para que las candidatas sean distintas.
`Engine.generate_batch(sessions, max_new_tokens=64, **muestreo)` avanza varias sesiones por rondas (cada una ya debe tener un `feed`).

## 9. Razonamiento: `tn.reasoning`

Herramientas para evaluar y mejorar un modelo con respuestas verificables. Los verificadores **solo comparan texto**: nunca ejecutan lo que genera el modelo.

```python
R = tn.reasoning
tasks = [R.Task("¿Cuánto es 2+2?", reference=4), ("¿Cuánto es 3+5?", 8)]
ok = R.all_of(R.numeric(), R.has_thinking())        # correcto Y con un bloque de razonamiento bien cerrado
result = R.evaluate(engine, tasks, ok, n=4, k=(1, 4), temperature=0.8, max_new_tokens=40)
print(result.summary())
print(result.pass_at[1], result.mean_reward)
```

### Tareas y verificadores

`Task(messages, reference=None, meta={})`: un prompt y lo que el verificador necesita para juzgar. `as_task(t)` acepta una `Task`, un diccionario (`messages` o `prompt`, más `answer` o `reference`) o un par `(prompt, referencia)`.

Un **verificador** es una función `verify(reply, task)` que devuelve una recompensa (`1.0` = correcto). Los incluidos:

| Verificador | Acepta si… |
|---|---|
| `exact_match(casefold=True)` | El texto es igual a la referencia, sin espacios ni puntuación en los bordes (y sin distinguir mayúsculas por defecto) |
| `contains(casefold=True)` | La referencia aparece en la respuesta |
| `regex_match(pattern, group=1, casefold=True)` | El grupo de la **última** coincidencia de `pattern` es igual a la referencia |
| `numeric(rel_tol=1e-06, abs_tol=1e-09)` | El último número de la respuesta es igual a `float(reference)` dentro de las tolerancias |
| `has_thinking(min_chars=1, closed=True)` | Hay un bloque de razonamiento de al menos `min_chars` y (si `closed`) el turno terminó solo (`<|end|>`) |
| `all_of(*verificadores)` | Producto de las recompensas: solo `1` si todos aceptan |

Utilidades: `last_number(texto)` (acepta `1,234.5`, `1.234,5` y `3,5`) y `pass_at_k(n, c, k)` (estimador sin sesgo de pass@k).

### Evaluar

`evaluate(policy, tasks, verifier, n=1, k=(1,), *, pass_threshold=1.0, on_task=None, **gen)` toma `n` respuestas por tarea (`policy` es un `Engine`) y devuelve un `EvalResult` con `n_tasks`, `n`,
`pass_at` (diccionario `k → pass@k`), `mean_reward`, `mean_tokens`, `mean_think_tokens`, `stops` y `tasks`; `summary()` lo resume en una línea. `gen` va a `policy.sample`.

### Ajuste fino por muestreo con rechazo (RFT)

```python
rounds = R.rft(model, tok, tasks, ok, rounds=1, n=4, epochs=1, verbose=False, max_new_tokens=40, temperature=1.0)
print(rounds[0].pass_at_1_before, rounds[0].data.accepted, "respuestas verificadas")
```

- `rejection_sample(policy, tasks, verifier, n=8, *, per_task=1, select="shortest", min_reward=1.0, seed=0, end_id=None, **gen)` genera `n` respuestas por tarea y conserva hasta `per_task` de las aceptadas
  (solo las que cerraron su turno con `<|end|>`). `select`: `"shortest"` (menos tokens: favorece razonamientos concisos), `"first"` o `"random"` (reproducible con `seed`). Devuelve un `RFTData`.
- `rft_round(model, tokenizer, tasks, verifier, *, n=8, per_task=1, select="shortest", epochs=1, replay=None, eval_tasks=None, seed=0, fit_kwargs=None, **gen)` hace una pasada: toma una instantánea del modelo, muestrea, conserva lo verificado y entrena `epochs` épocas solo con lo generado. Devuelve un `RoundResult`.
- `rft(model, tokenizer, tasks, verifier, rounds=3, *, verbose=True, **kw)` encadena rondas (cada una muestrea con los pesos actualizados) y para antes si una ronda no encuentra ninguna muestra aceptada.
- `replay=(x, y)` mezcla los datos de entrenamiento originales para limitar el olvido (misma `seq_len`).

## 10. Guardar y exportar

| Formato | Cómo se crea | Qué contiene | Para qué |
|---|---|---|---|
| Checkpoint (carpeta) | `model.save(carpeta)` | `model.ternary` (5 trits por byte) + `config.json` + `training_state.ternary` | Seguir entrenando |
| Modelo (carpeta) | `model.export(carpeta)` | `model.ternary` + `config.json` | Inferencia |
| Tokenizador | `tok.save(carpeta + "/tokenizer.json")` | El BPE | Funciones de texto del motor |
| `.ternam` | `tn.export_ternam(carpeta, "modelo.ternam")` | Configuración, tokenizador (si hay) y pesos ya empaquetados a 2 bits | Un solo archivo que `Engine` abre con `mmap` |
| Biblioteca solo-motor | `tn.build_engine_lib(...)` | Motor de inferencia compilado para una vectorización | Inferencia sin la parte de entrenamiento |
| Biblioteca fusionada | `tn.export_fused(...)` | El modelo incrustado en un motor mínimo | Distribuir un modelo como una sola biblioteca |

```python
import os
model.export("modelo")                              # carpeta de inferencia
tok.save(os.path.join("modelo", "tokenizer.json"))  # el tokenizador va dentro de la carpeta
tn.export_ternam("modelo", "modelo.ternam")
engine2 = tn.Engine("modelo.ternam")
print(engine2.session().reply("Hola", max_new_tokens=40).text)

model.export_ternam("directo.ternam", tokenizer=tok)    # lo mismo, directo desde el modelo
print(tn.Engine("directo.ternam").has_tokenizer)
```

`model.export_ternam(ruta, target="auto", tokenizer=None)` hace lo mismo sin pasar por una carpeta. Sin `tokenizer`, el `.ternam` no tiene funciones de texto (`reply`, `encode`, …).

```text
tn.export_ternam(src, out, target="auto", tokenizer=True) -> str
```

`src` es una carpeta de modelo u otro `.ternam`. `target` es `'scalar'`, `'avx512'`, `'neon'` o `'auto'` (el núcleo activo): el orden de los pesos en el archivo es el de esa familia de núcleos.
Un archivo hecho para otro núcleo funciona igual (los pesos se reordenan en memoria la primera vez).

### Biblioteca solo-motor

```text
tn.build_engine_lib(out=None, target="auto", cc=None, vnni=None, cflags=(), src_dir=None) -> str
tn.use_engine_lib(path=None) -> str
```

`build_engine_lib` compila ahora mismo una biblioteca compartida (`.so`, `.dll` o `.dylib`) con **una sola** vectorización (`'avx512'`, `'neon'`, `'scalar'` o `'auto'`; nunca dos, sin elección en ejecución).
Tiene la API plana de `ternary_infer.h` (`ternaryai_infer_*`) y nada de entrenamiento. `cc` es el compilador (por defecto `$CC` o `gcc`; un compilador cruzado como `aarch64-linux-gnu-gcc` funciona igual) y
`cflags` son banderas extra. Necesita los fuentes C, que vienen dentro del wheel y de la distribución fuente. `use_engine_lib` hace que `Engine` use esa biblioteca (`None` vuelve a `libternaryai`);
llámela **antes** de crear motores; `build_engine_lib` devuelve la ruta absoluta y `use_engine_lib` resuelve una ruta relativa contra la carpeta actual. El sistema carga una biblioteca una sola vez por ruta, así que compile a un nombre nuevo en vez de sobrescribir una ya cargada.

```python
lib = tn.build_engine_lib("motor_lib.so")           # devuelve la ruta absoluta
print(tn.use_engine_lib(lib))                       # nombre de la vectorización compilada
print(tn.Engine("modelo").session().reply("Hola", max_new_tokens=40).text)
tn.use_engine_lib(None)                             # volver a libternaryai
```

### Biblioteca fusionada

```text
tn.export_fused(src, out, expose, target="auto", cc=None, vnni=None, cflags=(), strip=True, src_dir=None) -> str
```

Incrusta **este** modelo en un motor mínimo, compilado ahora mismo para una vectorización. `expose` es la lista, elegida a mano, de funciones de la API que exporta la biblioteca (nombres cortos como `'session_new'`,
`'feed'`, `'generate'`; la lista completa está en `tn.FUSED_API`). No se exporta nada más y el enlazador descarta el resto: no existe `model_load` ni `export_ternam`. El modelo vive en memoria de solo lectura
dentro de la biblioteca (sin archivo ni copia de los pesos). Si no se exponen `encode`, `decode`, `special_id` ni `chat_*`, el tokenizador tampoco se incrusta. Desde Python se usa con `use_engine_lib` y
`Engine.fused()` (solo funcionan las llamadas expuestas).

```python
fused = tn.export_fused("modelo", "modelo_fusionado.so",
                        expose=["session_new", "feed", "generate", "encode", "decode", "special_id"])
print(os.path.exists(fused))
print(tn.FUSED_API[:6])
```

## 11. Hilos y núcleos de cálculo

```python
tn.set_threads(2)                 # hilos de toda la biblioteca (entrenamiento y motor)
print(tn.get_threads(), tn.engine_threads())
tn.set_threads(1)                 # 1 = un solo hilo (por defecto)
```

- `tn.set_threads(n)` / `tn.get_threads()`: número de hilos de cálculo de toda la biblioteca. `n <= 1` es un hilo. El resultado es idéntico con cualquier número de hilos. No lo cambie durante un entrenamiento.
- `tn.set_engine_threads(n)` / `tn.engine_threads()`: hilos del motor de inferencia. En `libternaryai` es el mismo ajuste que `set_threads`; una biblioteca solo-motor o fusionada tiene el suyo. Llámelo sin generación en curso.
- `tn.engine_backend()` da el núcleo activo (`'scalar'`, `'avx512-vnni'`, `'avx512-bw'` o `'neon'`). `tn.set_engine_backend(name)` lo elige (`'auto'`, `'scalar'`, `'avx512'`, `'neon'`); llámelo antes de crear sesiones.

El despacho automático es AVX-512, luego NEON, luego escalar. Los tres núcleos dan los mismos bits.

## 12. Errores

Todas las excepciones de la librería heredan de `tn.TernaryAIError`.

| Excepción | Cuándo |
|---|---|
| `ArchitectureError` | La arquitectura no cumple una regla (por ejemplo, añadir capas tras empezar a entrenar, o una activación desconocida) |
| `DataError` | Datos con forma, tipo o valores inválidos (por ejemplo, NaN, o `x` que no coincide con `input_shape`) |
| `ModelFileError` | Archivo de modelo ilegible o inválido |
| `BackendNotFoundError` | No se encuentra la biblioteca nativa |
| `TokenizerError` | Error del tokenizador (por ejemplo, `special_id` de algo que no es un especial) |
| `DatasetError` | Archivo, formato o esquema de dataset inválido |
| `InferenceError` | Error del motor de inferencia |

Los argumentos de tipo o rango incorrecto lanzan además `TypeError` o `ValueError` con un mensaje que dice cuál es el valor aceptado.

```python
try:
    model.predict(np.zeros((1, 5), dtype=np.int32))
except tn.DataError as e:
    print(type(e).__name__, e)
```

## 13. Variables de entorno

| Variable | Efecto |
|---|---|
| `TERNARYAI_THREADS` | Valor inicial de `set_threads` |
| `TERNARYAI_NO_AVX512=1` | Fuerza que no se use AVX-512 |
| `TERNARYAI_NO_NEON=1` | Fuerza que no se use NEON |
| `TERNARYAI_NO_VNNI=1` | Conserva AVX-512 pero no usa el forward entero con VNNI (mismo resultado, bit a bit) |
| `TERNARYAI_LIB` | Ruta de la biblioteca nativa que se carga |

## 14. Tabla de referencia

Todo lo exportado por `ternaryai` (`tn.__all__`) y las capas generadas.

| Nombre | Qué es |
|---|---|
| `Model`, `History`, `load` | Modelo, resultado de `fit`, carga de carpetas |
| `Dense`, `Conv2D`, `Embedding`, `GQA`, `SwiGLU`, `RMSNorm`, `LMHead` | Capas (generadas del backend) |
| `Layer`, `Residual`, `transformer_block`, `layer`, `available_layers` | Base de las capas, bloque residual, bloque transformer, constructor por nombre, lista de capas |
| `LayerSpec`, `Param`, `LayerDesc`, `ModelDescription`, `Backend`, `get_backend`, `set_backend` | Descripciones y acceso al backend de cálculo (uso avanzado) |
| `Tokenizer`, `CHAT_SPECIAL_TOKENS` | Tokenizador BPE y sus 13 especiales |
| `dataset` | `load`, `tokenize`, `targets` y constantes |
| `reasoning` | Tareas, verificadores, `evaluate`, `rejection_sample`, `rft_round`, `rft` |
| `Engine`, `Session`, `Reply` | Inferencia |
| `CTX_ERROR`, `CTX_TRUNCATE`, `CTX_SLIDE` | Políticas de contexto lleno |
| `export_ternam`, `build_engine_lib`, `export_fused`, `use_engine_lib`, `FUSED_API` | Exportación y bibliotecas de motor |
| `set_threads`, `get_threads`, `set_engine_threads`, `engine_threads`, `set_engine_backend`, `engine_backend` | Hilos y núcleos |
| `TernaryAIError`, `ArchitectureError`, `BackendNotFoundError`, `DataError`, `ModelFileError`, `TokenizerError`, `DatasetError`, `InferenceError` | Excepciones |
