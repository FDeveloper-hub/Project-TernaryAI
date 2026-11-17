# TernaryAI Manual

*Versión en español: [MANUAL.md](MANUAL.md)*

This manual describes the whole public Python interface of TernaryAI (`import ternaryai`). For what the library is for, how to install it and what its limits are,
see [README.md](../README.md). Every ` ```python ` code block in this manual is run, in order and in one shared namespace,
by `tests/test_manual.py`: what is shown here works with the installed version. ` ```text ` and ` ```bash ` blocks are not run.

Convention: `import ternaryai as tn`.

## Contents

1. [Quick start](#1-quick-start)
2. [Basic ideas](#2-basic-ideas)
3. [`Model`: build, train, predict](#3-model-build-train-predict)
4. [Layers](#4-layers)
5. [Tokenizer](#5-tokenizer)
6. [Datasets](#6-datasets)
7. [Language models from start to finish](#7-language-models-from-start-to-finish)
8. [Inference: `Engine` and `Session`](#8-inference-engine-and-session)
9. [Reasoning: `tn.reasoning`](#9-reasoning-tnreasoning)
10. [Saving and exporting](#10-saving-and-exporting)
11. [Threads and compute kernels](#11-threads-and-compute-kernels)
12. [Errors](#12-errors)
13. [Environment variables](#13-environment-variables)
14. [Reference table](#14-reference-table)

---

## 1. Quick start

A classifier: the input is made of -1, 0 or 1 values and the output is a -1 or +1 label.

```python
import numpy as np
import ternaryai as tn

rng = np.random.default_rng(0)
X = rng.choice([-1, 0, 1], size=(64, 8)).astype(np.int8)
y = (X[:, :4].sum(axis=1) > 0).astype(np.float32) * 2 - 1      # +1 if the first 4 columns add up to more than 0

model = tn.Model(8, [tn.Dense(16), tn.Dense(1)], seed=1)
history = model.fit(X, y, epochs=200)
print(history)                                                 # <ternaryai.History epochs 0..200 final_loss=...>
print((model.predict(X, sign=True).ravel() == y).mean())       # 1.0
```

A language model that holds a conversation is in [section 7](#7-language-models-from-start-to-finish).

## 2. Basic ideas

- **Ternary weights.** Every weight is -1, 0 or +1. There is no float copy of each weight: the **R2** update rule keeps, for every weight, an int8 counter with hysteresis.
  Each layer has a final float scale.
- **uint8 activations.** Between layers, values travel as uint8 with a scale and a zero-point per sample (or per token). It is the only activation available and
  the default of every layer (`activation="uint8"`).
- **Two kinds of model.**
  - *Ternary or uint8 inputs* (classifiers, convolutions): `x` is an array of numbers, the loss is `"mse"`.
  - *Token ids* (language models): the first layer is `Embedding`, `x` and `y` are integers of shape `(n, seq)`, the loss is `"xent"`.
- **Determinism.** The same program gives the same bits with any number of threads and with any of the kernels (scalar, AVX-512, NEON).
- **Layers are generated from the backend.** `tn.Dense`, `tn.GQA`, etc. come from the layer list of the C library; if a layer is added in C, it shows up in Python after a rebuild
  (see `tn.available_layers()`).

## 3. `Model`: build, train, predict

### Building

```python
model = tn.Model(8, [tn.Dense(16), tn.Dense(1)], seed=1)
model.add(tn.Dense(1))          # layers can also be added one by one; add returns the model
print(model.summary())
print(model.input_shape, model.output_shape, model.epoch)
```

```text
Model(inputs, layers=(), *, seed=0, threshold=0.5, input="ternary", backend=None)
```

| Parameter | Meaning |
|---|---|
| `inputs` | Input shape: an integer, or a tuple such as `(28, 28, 1)` (height, width, channels). For language models it is the sequence length |
| `layers` | Initial layers (see [layers](#4-layers)) |
| `seed` | Seed of the weight initialization |
| `threshold` | Threshold to ternarize the input: values `> threshold` become +1, `< -threshold` become -1, the rest 0. Data that is already -1/0/1 is used as is |
| `input` | `"ternary"` (default) or `"uint8"`: with `"uint8"` the input is kept as float and the first layer reads it as uint8 (255 levels, scale and zero-point per sample) |

Properties: `input_shape`, `output_shape`, `layers`, `epoch` (epochs trained so far), `r2` (parameters of the R2 rule), `input`, `threshold` (these two can be changed),
`closed`. Layers can only be added before training starts (otherwise `ArchitectureError`).

### Training: `fit`

```text
model.fit(x, y, epochs=100, *, initial_epoch=None, verbose=0, loss=None, scale_lr=None, restore_best=None,
          batch_size=None, shuffle=True, shuffle_seed=0, tau=None, beta=None, inc=None,
          zero_target=None, reset=None, head_beta2=None) -> History
```

| Parameter | Meaning |
|---|---|
| `x`, `y` | For ternary inputs: `x` of shape `(n, *input_shape)` or `(n, features)`, `y` of shape `(n, outputs)` (float). For ids: both integers `(n, seq)`; `y[i, j]` is the id expected at position `j` (normally the next token) and a negative value ignores that position |
| `epochs` | Epochs of this call. Successive calls continue the learning schedule |
| `initial_epoch` | Forces the epoch to resume from (for example after `load`) |
| `verbose` | `N`: prints the loss every `N` epochs (it does not change the result) |
| `loss` | `"mse"` (default for ternary-input models) or `"xent"` (default, and only choice, for id models) |
| `batch_size` | `None` (default) = each epoch is a single step with all the data. With `B < n`, each epoch walks the samples in steps of `B` and every step makes its own update |
| `shuffle`, `shuffle_seed` | With mini-batches, reshuffles the samples every epoch with a permutation that depends only on `(shuffle_seed, epoch)`: it is reproducible |
| `scale_lr` | Learning rate of the float scales (default 0.02 with `"mse"` and 0.1 with `"xent"`) |
| `restore_best` | `"xent"` only (default `True`): at the end of each `fit` the model goes back to the state with the lowest training loss. It guards against late collapse |
| `tau`, `beta`, `inc`, `zero_target`, `reset`, `head_beta2` | Parameters of the R2 rule (below). If left as `None` the current value is kept |

It returns a `History` with `loss` (list, one per epoch), `accuracy` (`"xent"` only: fraction of non-ignored positions whose highest logit was the correct id; `None` with `"mse"`),
`first_epoch` and `last_epoch`.

**The R2 rule.** Per row, the gradient is normalized by its RMS and clipped to ±3. An int8 counter per weight integrates it; when the counter reaches the threshold `tau`, the weight moves one step and
the counter restarts on the opposite side (hysteresis, at `reset` times the threshold).

| Parameter | Default | Meaning |
|---|---|---|
| `tau` | 0.7 | Counter threshold, as a fraction of its range. Range `(0, 1]` |
| `beta` | 0.999 | Counter memory (almost no decay). Range `[0, 1)` |
| `inc` | 0.03 | Counter step per unit of normalized gradient. Stable zone: 0.02 to 0.03 |
| `zero_target` | 0.2 | Target fraction of zeros per row; adjusts the 0→±1 and ±1→0 thresholds (negative = off) |
| `reset` | 0.5 | Where the counter restarts after a change, in multiples of the threshold. Range `[0, 1]` |
| `head_beta2` | 0.99 | `LMHead` only: normalizes its gradient per weight with the moving average of g². `0` turns it off and leaves the per-row R2 |

```python
print(model.r2)     # {'tau': 0.7, 'beta': 0.999, 'inc': 0.03, 'zero_target': 0.2, 'reset': 0.5, 'head_beta2': 0.99}
history = model.fit(X, y, epochs=20, batch_size=16, shuffle_seed=3, verbose=10)   # mini-batches; prints every 10 epochs
print(history.first_epoch, history.last_epoch)
```

The six values live in the model, are saved by `save()` (the training checkpoint) and not by `export()`.

### Predicting

```python
out = model.predict(X)                  # float32, shape (n, *output_shape)
signs = model.predict(X, sign=True)     # int8 -1/0/1
print(out.shape, signs.dtype)
```

In id models, `predict(x)` takes ids `(n, seq)` and returns the **logits** `(n, seq, vocab)` (no softmax); the predicted id at each position is `argmax` of the last axis.
For a single sample use `x[None]`.

### Other methods

| Method | What it does |
|---|---|
| `summary()` | Text with one row per layer (`Residual` blocks are shown nested) |
| `save(folder)` | Full checkpoint: `model.ternary` + `config.json` + `training_state.ternary`. It can be loaded and trained further |
| `export(folder)` | Inference only: `model.ternary` + `config.json` |
| `export_ternam(path, target="auto", tokenizer=None)` | Self-contained `.ternam` file; with `tokenizer=tok` it is included (see [section 10](#10-saving-and-exporting)) |
| `engine(tokenizer=None)` | An inference `Engine`, a snapshot of the model (see [section 8](#8-inference-engine-and-session)) |
| `ffn_sparsity(reset=False)` | Fraction of the SwiGLU activation that is exactly its zero-point, one per layer |
| `attention_probs(layer, head)` | Attention probabilities `(seq, seq)` of the last `predict` of **one** sequence |
| `num_gqa()` | Number of GQA layers |
| `set_checkpoint(on=True)` | Less activation memory with the same bits |
| `prof_enable(on)`, `prof_reset()`, `prof_report()` | Scope profiler (call tree with times; it does not change results) |
| `close()` | Releases the native model (it can be called more than once). `Model` also works with `with` |

### `load`

```python
model.save("checkpoint")           # full checkpoint
model.export("exported")           # inference only
again = tn.load("checkpoint")
print(np.array_equal(again.predict(X), model.predict(X)))   # True
```

```text
tn.load(path, *, threshold=0.5, input="ternary", backend=None) -> Model
```

`threshold` and `input` are not stored in the model files: pass them again if you used non-default values.

## 4. Layers

Every layer accepts `activation="uint8"` (the only activation). Parameters with no default are required.

| Layer | What it is for | Parameters (default) |
|---|---|---|
| `Dense(units)` | Fully connected layer | `units`, `bias=True` |
| `Conv2D(filters, kernel)` | 2D convolution over `(height, width, channels)`; must be the first layer | `filters`, `kernel` (integer or `(h, w)`), `stride=1`, `bias=True` |
| `Embedding(vocab_size, embed_dim)` | Table of ternary vectors indexed by token id; must be the first layer | `vocab_size`, `embed_dim` |
| `GQA(n_heads)` | Causal multi-head grouped-query attention | `n_heads` (must divide `embed_dim`), `n_kv_heads=0` (0 = same as `n_heads`), `rope=1`, `rope_base=10.0`, `score_shift=2`, `out_scale=1.0`, `window=0`, `sinks=0`, `smear=False` |
| `SwiGLU(hidden)` | Feed-forward block (no residual connection) | `hidden`, `gate_mode=0`, `band=0.5`, `out_scale=1.0` |
| `RMSNorm()` | Row-wise RMS normalization, with a ternary gain per dimension and a float scale | `eps=1e-05` |
| `LMHead(vocab)` | Language-model head: one shared ternary projection over the last dimension; outputs logits | `vocab` |
| `Residual(layers)` | Residual block `output = x + f(x)` | `layers`, `activation` |

`GQA` details:

| Parameter | Meaning |
|---|---|
| `rope` | `0` no position encoding, `1` RoPE with integer scores (recommended), `2` RoPE with binarized Q/K |
| `rope_base` | RoPE frequency base (`> 1`). With `10` (default) the rotation is fast: for long-distance attention use a higher base (`1000` was validated at 512 tokens) |
| `score_shift` | With `rope=1`: right shift of the integer q·k score (larger = flatter attention) |
| `out_scale` | Initial scale of the output projection Wo (`< 1` starts the branch small next to the residual stream; it is learned afterwards) |
| `window`, `sinks` | Sliding-window attention: each position sees the last `window` ones (0 = full causal attention; needs `rope=1`). With a window, the first `sinks` positions (at most 48) stay visible to every position |
| `smear` | Smeared keys, `K'_j = (K_j + K_{j-1}) >> 1` before RoPE: lets a single layer match a token against the one that precedes each key (the *induction head* pattern). Needs `rope=1` and `window=0` |

`SwiGLU` details: `gate_mode` is `0` (silu through an integer table, float product), `1` (ternary sign of the gate, all integer) or `2` (ReLU² without a gate, BitNet style; only two matrices).
`band` is the zero band of the inner quantization, relative to the row's mean |h| (0 = none).

`Residual` details: the input of the block is the float output of the previous layer. When it is added to a model, the library checks that the first layer of the chain takes float input (like `RMSNorm`),
that every layer passes gradient to its input, that none takes ids and that the last one returns as many values as `x`. Blocks can be nested.

### `transformer_block`

```python
blocks = tn.transformer_block(heads=4, hidden=96, kv_heads=2, gate_mode=2)   # list of two Residual
print(len(blocks), blocks[0].layers)
```

```text
tn.transformer_block(heads, hidden, *, kv_heads=None, rope=None, rope_base=None, score_shift=None, gate_mode=None,
                     band=None, eps=None, out_scale=None, window=None, sinks=None, smear=None,
                     norm_activation="uint8", backend=None) -> list[Layer]
```

It returns `[Residual([RMSNorm, GQA]), Residual([RMSNorm, SwiGLU])]`, that is `x = x + GQA(norm(x))` and then `x = x + SwiGLU(norm(x))`. Every argument left as `None` takes the default of the layer it belongs to.
`out_scale` is 0.03 by default in the block (not 1.0): a branch that starts small makes deeper models train more reliably. With 3 or more blocks use an embedding dimension of at least 32.
Each element of the list has its own weights, so the list can be repeated: `[*tn.transformer_block(...)] * 12`.

### Other layer functions

| Function | What it does |
|---|---|
| `tn.layer("dense", 16)` | Builds a layer by class name or type name |
| `tn.available_layers()` | List of the layers the backend implements |

## 5. Tokenizer

Byte-level BPE, implemented in C. There are no out-of-vocabulary tokens: any text can be encoded. It is built with a vocabulary size and trained with a corpus.

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

- `vocab_size` counts the 256 bytes, the special tokens and the merges. **Special tokens** have fixed ids (256, 257, … in the given order) and BPE never splits or merges them; they are given when building, before training.
- `train(corpus)` takes `str` or `bytes` and returns the tokenizer itself.
- `encode(text, parse_special=True)`: with `parse_special=False` a string such as `"<|user|>"` is treated as ordinary text.
- `decode(ids, skip_special=False)`: with `skip_special=True` the markers are omitted.
- `special_id(token)` gives the id of a special token (`TokenizerError` if it is not one). Properties: `vocab_size`, `num_merges`, `num_special`, `special_tokens`.
- `save(path)` and `Tokenizer.load(path)` use a JSON file.

`tn.CHAT_SPECIAL_TOKENS` is a tuple with the 13 conversation and reasoning special tokens:

```text
<|user|> <|assistant|> <|end|> <|system|> <|console|> <|terminal|> <|archive|> <|log|> <|result|> <|function_call|> <|tool|> <|think|> <|/think|>
```

## 6. Datasets

`tn.dataset` turns files into text examples and then into `(x, y)` arrays ready for `Model.fit`. It does not read Parquet or Arrow.

```python
import json
rows = [{"chat": [{"user": "Hello"}, {"model": "Hello, how can I help you?"}]},
        {"chat": [{"user": "What is 2+2?"}, {"model": "<think>2+2=4</think>It is 4."}]},
        {"chat": [{"user": "What is 3+5?"}, {"model": "<think>3+5=8</think>It is 8."}]}] * 8
with open("data.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

examples, skipped = tn.dataset.load("data.jsonl")
print(len(examples), skipped)
print(repr(examples[1]))
```

### `dataset.load(path, format=None, strict=True, reasoning="keep")`

It returns `(examples, skipped)`: one string per conversation, with the turns written as `<|role|>\ntext\n` and closed with `<|end|>\n`.

- `format`: `tn.dataset.TEXT_LINES`, `TEXT_WHOLE`, `JSON` or `JSONL`. If omitted, it is guessed from the extension (`.txt` → one example per line, `.json`, `.jsonl`). A `.txt` as a single block must ask for `TEXT_WHOLE`.
- `strict`: with JSON/JSONL, `True` fails at the first record that does not match any known schema; `False` skips it and `skipped` counts how many.
- `reasoning`: `"keep"` keeps the reasoning of the assistant turns normalized to `<|think|>…<|/think|>` (it accepts the `<think>…</think>` and `<thinking>…</thinking>` tags and the fields `reasoning_content`, `reasoning` and `thinking`); `"drop"` removes all of it, to train a model that does not reason with a dataset that does. Tags must be balanced.

Recognized formats (detected by the key of each record; one file can mix them):

| Key | Format |
|---|---|
| `"chat"` | Own: `[{"user": "..."}, {"model": "..."}, ...]` (roles `user`, `model`, `system`, `console`, `terminal`, `archive`, `log`, `result`) |
| `"messages"` | ChatML / OpenAI: `[{"role": ..., "content": ...}, ...]` |
| `"conversations"` | ShareGPT: `[{"from": "human" \| "gpt", "value": ...}, ...]` |
| `"prompt"` + `"completion"` | Prompt and answer pairs |
| `"instruction"` [+ `"input"`] + `"output"` | Alpaca |
| `"text"` | Plain text, no turn markers |

### `dataset.tokenize(examples, tokenizer, seq_len, *, ...)`

```text
tn.dataset.tokenize(examples, tokenizer, seq_len, *, train_roles=("assistant", "function_call"), roles=CHAT_ROLES,
                    train_text=True, overflow="truncate", pad_id=0, drop_empty=True) -> (x, y)
```

It returns `x` (ids) and `y` (next-token targets), both `int32` of shape `(n, seq_len)`. `y[i] = x[i+1]` if that token is trained and `-1` where nothing is learned (padding and what the other roles say).
The content of the `train_roles` turns is trained, plus the marker that closes the turn, so that the model learns to end it.

- `overflow` decides what happens to an example longer than `seq_len` tokens: `"truncate"` keeps the beginning, `"split"` cuts it into consecutive windows (each with its own mask, without seeing the context of the previous ones), `"drop"` discards it.
- `train_text`: whether the text outside any turn (`text` records, plain corpora) is trained.
- `drop_empty`: discards windows with nothing to learn and examples of fewer than 2 tokens.
- `dataset.targets(ids_list, tokenizer, ...)` does only the targets part, for examples that you tokenized yourself.

```python
tok = tn.Tokenizer(400, special_tokens=list(tn.CHAT_SPECIAL_TOKENS))
tok.train("\n".join(examples).encode())
x, y = tn.dataset.tokenize(examples, tok, seq_len=64)
print(x.shape, y.shape, x.dtype, int((y >= 0).sum()), "trained targets")
```

Constants: `dataset.CHAT_ROLES`, `dataset.DEFAULT_TRAIN_ROLES`, `dataset.TEXT_LINES`, `TEXT_WHOLE`, `JSON`, `JSONL`. Error: `tn.DatasetError`.

## 7. Language models from start to finish

Continuing with `tok`, `x` and `y` from the previous section:

```python
model = tn.Model(64, [
    tn.Embedding(tok.vocab_size, 48),
    *tn.transformer_block(heads=4, hidden=96, kv_heads=2, gate_mode=2),
    tn.RMSNorm(activation="uint8"),
    tn.LMHead(tok.vocab_size),
])
history = model.fit(x, y, epochs=400)
print(history)

engine = model.engine(tok)                           # inference engine (a snapshot of the model)
r = engine.session().reply("What is 2+2?", max_new_tokens=40)
print(r.text, "|", r.thinking)                       # It is 4. | 2+2=4
print(engine.session().reply("Hello", max_new_tokens=40).text)   # Hello, how can I help you?
```

The exact result depends on the seed and the model size. The sequence length of a language model is the first dimension of `inputs` (here 64) and it must match `seq_len` when tokenizing.

## 8. Inference: `Engine` and `Session`

An **`Engine`** is a model loaded for inference (read-only weights, shared among sessions). A **`Session`** is one conversation: key/value cache, token history and random generator.

### Creating an `Engine`

| How | What you get |
|---|---|
| `model.engine(tokenizer=None)` | A snapshot of the model being trained; later training does not change it |
| `tn.Engine(path)` | A folder with the model (`model.ternary` + `config.json`, and `tokenizer.json` for text features) or a `.ternam` file |
| `tn.Engine.fused()` | The model embedded in a library made with `export_fused` (see [section 10](#10-saving-and-exporting)) |

`Engine` properties: `seq_len`, `dim`, `vocab`, `weight_bytes`, `fingerprint`, `has_tokenizer`. Text methods (they need a tokenizer): `encode(text, parse_special=True)`, `decode(ids, skip_special=True)`,
`special_id(token)` (gives `-1` if it does not exist), `chat_encode(messages, ...)`, `split_thinking(ids, think_inside=False)`. `close()` releases the engine.

```python
print(engine.seq_len, engine.dim, engine.has_tokenizer, engine.special_id("<|end|>"))
print(engine.chat_encode("Hello"))              # prompt of a conversation: [<|user|>, ..., <|assistant|>]
```

### Chatting: `Session.reply`

```text
session.reply(messages, max_new_tokens=256, *, show_thinking=False, think_budget=0, force_think=False,
              after_role="\n", after_msg="\n", stop_on_user=True, **sampling) -> Reply
```

`messages` is a string (one user message) or a list of `{"role", "content"}` dictionaries or `(role, content)` pairs. The prompt is built like the dataset reader does (`<|role|>\ntext\n`)
and generation stops at `<|end|>` (and at `<|user|>` if `stop_on_user`). Every call resets the session first.

```python
reply = engine.session().reply([("user", "What is 3+5?")], max_new_tokens=40)
print(reply.text, "|", reply.thinking, "|", reply.stop)
```

`Reply` has `text` (the answer, without the reasoning block), `thinking` (the reasoning, empty if there was none), `stop` (why it ended: `'token'` = the model closed its turn, `'max_tokens'`, …), `ids` and `prompt_ids`.

- `show_thinking=True` leaves the block inside `text`, tags included, so the chain of thought can be read in place.
- `think_budget=N` limits the reasoning tokens: when reached, `<|/think|>` is forced.
- `force_think=True` starts already inside `<|think|>`.
- `after_role` and `after_msg` change the separators (use `""` for a model trained without them).

```python
print(engine.session().reply("What is 2+2?", max_new_tokens=40, show_thinking=True).text)
```

### Sampling

The functions that generate (`reply`, `generate`, `sample`, `generate_batch`) accept these options:

| Option | Meaning |
|---|---|
| `temperature` | `0` = greedy (always the most probable id) |
| `top_k`, `top_p`, `min_p` | Sampling filters |
| `repeat_penalty`, `freq_penalty`, `presence_penalty`, `penalty_last_n` | Repetition penalties |
| `logit_bias` | Dictionary `{id: bias}` |
| `banned` | Forbidden ids |
| `stop_ids`, `stop_seqs`, `include_stop` | Stop by ids or by sequences |
| `on_token` | Function called with every token; if it returns `True`, generation stops |
| `think_budget`, `think_inside` | Reasoning (`think_inside`: the context already ends inside the block) |

### Manual control of a session

```python
s = engine.session()
s.feed(engine.encode("Hello"))                       # processes ids; returns the session
print(s.generate(5, temperature=0))                 # continues from the current state; returns ids
print(s.next_position, s.top_probs(3))              # next position and (id, probability) of the 3 most probable
logprobs, perplexity = engine.session().score(engine.encode(examples[0]))   # how expected a text is
print(round(perplexity, 3))
s.reset()
```

| `Session` method | What it does |
|---|---|
| `feed(ids)` | Processes ids and updates the cache |
| `generate(max_new_tokens=64, **sampling)` | Generates from the current state and returns the ids |
| `reply(messages, ...)` | A full conversation answer |
| `argmax()`, `logits()`, `top_probs(n=5)` | Look at the distribution of the next token |
| `score(ids)` | Processes `ids` and returns `(log p(ids[i] \| context), perplexity)`; the first value is NaN if the session was empty |
| `fork()` | Copies the session with its state |
| `reset()`, `seed(value)`, `close()` | Reset, set the seed, release |
| `save(path)` / `engine.load_session(path)` | Save and restore a session (for example an already processed system prompt) |
| `attention_probs(layer, head)` | Attention of the last token (the session must be created with `record_attn=1`) |
| `next_position` | Position of the next token |

`engine.session(**options)` accepts `max_context`, `attn_window`, `sinks`, `ctx_policy`, `keep_tokens`, `record_attn` and `max_history`.
`ctx_policy` decides what happens when the context is full: `tn.CTX_ERROR` (error), `tn.CTX_TRUNCATE` (truncate) or `tn.CTX_SLIDE` (slide the window).

### Several answers at once

```python
replies = engine.sample("What is 3+5?", n=3, max_new_tokens=40, temperature=0.8)
print([r.text for r in replies])

sessions = [engine.session() for _ in range(2)]
for sess in sessions:
    sess.feed(engine.chat_encode("Hello"))
print(engine.generate_batch(sessions, max_new_tokens=5))     # list of lists of ids
```

`Engine.sample(messages, n=1, max_new_tokens=256, *, seed=0, ...)` generates `n` independent answers to the same conversation: the prompt is processed once and the candidates run as one batch.
Candidate `i` uses the seed `seed + i`: same arguments, same answers. Use `temperature > 0` to get different candidates.
`Engine.generate_batch(sessions, max_new_tokens=64, **sampling)` advances several sessions in rounds (each one must already have had a `feed`).

## 9. Reasoning: `tn.reasoning`

Tools to evaluate and improve a model with verifiable answers. The verifiers **only compare text**: they never execute what the model generates.

```python
R = tn.reasoning
tasks = [R.Task("What is 2+2?", reference=4), ("What is 3+5?", 8)]
ok = R.all_of(R.numeric(), R.has_thinking())        # correct AND with a properly closed reasoning block
result = R.evaluate(engine, tasks, ok, n=4, k=(1, 4), temperature=0.8, max_new_tokens=40)
print(result.summary())
print(result.pass_at[1], result.mean_reward)
```

### Tasks and verifiers

`Task(messages, reference=None, meta={})`: a prompt and what the verifier needs to judge it. `as_task(t)` accepts a `Task`, a dictionary (`messages` or `prompt`, plus `answer` or `reference`) or a `(prompt, reference)` pair.

A **verifier** is a function `verify(reply, task)` that returns a reward (`1.0` = correct). The included ones:

| Verifier | Accepts if… |
|---|---|
| `exact_match(casefold=True)` | The text equals the reference, ignoring edge spaces and punctuation (and case, by default) |
| `contains(casefold=True)` | The reference appears in the answer |
| `regex_match(pattern, group=1, casefold=True)` | The group of the **last** match of `pattern` equals the reference |
| `numeric(rel_tol=1e-06, abs_tol=1e-09)` | The last number of the answer equals `float(reference)` within the tolerances |
| `has_thinking(min_chars=1, closed=True)` | There is a reasoning block of at least `min_chars` and (if `closed`) the turn ended on its own (`<|end|>`) |
| `all_of(*verifiers)` | Product of the rewards: `1` only if every one accepts |

Utilities: `last_number(text)` (accepts `1,234.5`, `1.234,5` and `3,5`) and `pass_at_k(n, c, k)` (unbiased estimator of pass@k).

### Evaluating

`evaluate(policy, tasks, verifier, n=1, k=(1,), *, pass_threshold=1.0, on_task=None, **gen)` takes `n` answers per task (`policy` is an `Engine`) and returns an `EvalResult` with `n_tasks`, `n`,
`pass_at` (dictionary `k → pass@k`), `mean_reward`, `mean_tokens`, `mean_think_tokens`, `stops` and `tasks`; `summary()` condenses it into one line. `gen` goes to `policy.sample`.

### Rejection-sampling fine-tuning (RFT)

```python
rounds = R.rft(model, tok, tasks, ok, rounds=1, n=4, epochs=1, verbose=False, max_new_tokens=40, temperature=1.0)
print(rounds[0].pass_at_1_before, rounds[0].data.accepted, "verified answers")
```

- `rejection_sample(policy, tasks, verifier, n=8, *, per_task=1, select="shortest", min_reward=1.0, seed=0, end_id=None, **gen)` generates `n` answers per task and keeps up to `per_task` of the accepted ones
  (only those that closed their turn with `<|end|>`). `select`: `"shortest"` (fewest tokens: favors concise reasoning), `"first"` or `"random"` (reproducible with `seed`). It returns an `RFTData`.
- `rft_round(model, tokenizer, tasks, verifier, *, n=8, per_task=1, select="shortest", epochs=1, replay=None, eval_tasks=None, seed=0, fit_kwargs=None, **gen)` makes one pass: it snapshots the model, samples, keeps what is verified and trains `epochs` epochs only on what was generated. It returns a `RoundResult`.
- `rft(model, tokenizer, tasks, verifier, rounds=3, *, verbose=True, **kw)` chains rounds (each one samples with the updated weights) and stops early if a round finds no accepted sample.
- `replay=(x, y)` mixes in the original training data to limit forgetting (same `seq_len`).

## 10. Saving and exporting

| Format | How it is created | What it holds | What it is for |
|---|---|---|---|
| Checkpoint (folder) | `model.save(folder)` | `model.ternary` (5 trits per byte) + `config.json` + `training_state.ternary` | Continuing to train |
| Model (folder) | `model.export(folder)` | `model.ternary` + `config.json` | Inference |
| Tokenizer | `tok.save(folder + "/tokenizer.json")` | The BPE | The engine's text features |
| `.ternam` | `tn.export_ternam(folder, "model.ternam")` | Configuration, tokenizer (if any) and the weights already packed to 2 bits | A single file that `Engine` opens with `mmap` |
| Engine-only library | `tn.build_engine_lib(...)` | Inference engine compiled for one vectorization | Inference without the training part |
| Fused library | `tn.export_fused(...)` | The model embedded in a minimal engine | Shipping a model as a single library |

```python
import os
model.export("model")                               # inference folder
tok.save(os.path.join("model", "tokenizer.json"))   # the tokenizer goes inside the folder
tn.export_ternam("model", "model.ternam")
engine2 = tn.Engine("model.ternam")
print(engine2.session().reply("Hello", max_new_tokens=40).text)

model.export_ternam("directo.ternam", tokenizer=tok)    # the same, straight from the model
print(tn.Engine("directo.ternam").has_tokenizer)
```

`model.export_ternam(path, target="auto", tokenizer=None)` does the same without going through a folder. Without `tokenizer`, the `.ternam` has no text features (`reply`, `encode`, …).

```text
tn.export_ternam(src, out, target="auto", tokenizer=True) -> str
```

`src` is a model folder or another `.ternam`. `target` is `'scalar'`, `'avx512'`, `'neon'` or `'auto'` (the active kernel): the order of the weights in the file is that of that kernel family.
A file made for another kernel works just the same (the weights are reordered in memory the first time).

### Engine-only library

```text
tn.build_engine_lib(out=None, target="auto", cc=None, vnni=None, cflags=(), src_dir=None) -> str
tn.use_engine_lib(path=None) -> str
```

`build_engine_lib` compiles right now a shared library (`.so`, `.dll` or `.dylib`) with **one** vectorization (`'avx512'`, `'neon'`, `'scalar'` or `'auto'`; never two, no run-time choice).
It has the flat API of `ternary_infer.h` (`ternaryai_infer_*`) and nothing of training. `cc` is the compiler (default `$CC` or `gcc`; a cross compiler such as `aarch64-linux-gnu-gcc` works the same) and
`cflags` are extra flags. It needs the C sources, which come inside the wheel and the source distribution. `use_engine_lib` makes `Engine` use that library (`None` goes back to `libternaryai`);
call it **before** creating engines; `build_engine_lib` returns the absolute path and `use_engine_lib` resolves a relative path against the current directory. The system loads a library only once per path, so build to a new file name instead of overwriting one that is already loaded.

```python
lib = tn.build_engine_lib("motor_lib.so")           # returns the absolute path
print(tn.use_engine_lib(lib))                       # name of the compiled vectorization
print(tn.Engine("model").session().reply("Hello", max_new_tokens=40).text)
tn.use_engine_lib(None)                             # back to libternaryai
```

### Fused library

```text
tn.export_fused(src, out, expose, target="auto", cc=None, vnni=None, cflags=(), strip=True, src_dir=None) -> str
```

It embeds **this** model in a minimal engine, compiled right now for one vectorization. `expose` is the list, chosen by hand, of the API functions the library exports (short names such as `'session_new'`,
`'feed'`, `'generate'`; the full list is in `tn.FUSED_API`). Nothing else is exported and the linker drops the rest: `model_load` and `export_ternam` do not exist. The model lives in read-only memory
inside the library (no file, no copy of the weights). If `encode`, `decode`, `special_id` and `chat_*` are not exposed, the tokenizer is not embedded either. From Python it is used with `use_engine_lib` and
`Engine.fused()` (only the exposed calls work).

```python
fused = tn.export_fused("model", "fused_model.so",
                        expose=["session_new", "feed", "generate", "encode", "decode", "special_id"])
print(os.path.exists(fused))
print(tn.FUSED_API[:6])
```

## 11. Threads and compute kernels

```python
tn.set_threads(2)                 # threads of the whole library (training and engine)
print(tn.get_threads(), tn.engine_threads())
tn.set_threads(1)                 # 1 = a single thread (default)
```

- `tn.set_threads(n)` / `tn.get_threads()`: number of compute threads of the whole library. `n <= 1` is one thread. The result is identical with any number of threads. Do not change it during training.
- `tn.set_engine_threads(n)` / `tn.engine_threads()`: threads of the inference engine. In `libternaryai` it is the same setting as `set_threads`; an engine-only or fused library has its own. Call it with no generation in progress.
- `tn.engine_backend()` gives the active kernel (`'scalar'`, `'avx512-vnni'`, `'avx512-bw'` or `'neon'`). `tn.set_engine_backend(name)` chooses it (`'auto'`, `'scalar'`, `'avx512'`, `'neon'`); call it before creating sessions.

Automatic dispatch is AVX-512, then NEON, then scalar. The three kernels give the same bits.

## 12. Errors

Every exception of the library inherits from `tn.TernaryAIError`.

| Exception | When |
|---|---|
| `ArchitectureError` | The architecture breaks a rule (for example, adding layers after training has started, or an unknown activation) |
| `DataError` | Data with an invalid shape, type or values (for example, NaN, or an `x` that does not match `input_shape`) |
| `ModelFileError` | Unreadable or invalid model file |
| `BackendNotFoundError` | The native library cannot be found |
| `TokenizerError` | Tokenizer error (for example, `special_id` of something that is not a special token) |
| `DatasetError` | Invalid dataset file, format or schema |
| `InferenceError` | Inference engine error |

Arguments of the wrong type or range also raise `TypeError` or `ValueError` with a message that says which value is accepted.

```python
try:
    model.predict(np.zeros((1, 5), dtype=np.int32))
except tn.DataError as e:
    print(type(e).__name__, e)
```

## 13. Environment variables

| Variable | Effect |
|---|---|
| `TERNARYAI_THREADS` | Initial value of `set_threads` |
| `TERNARYAI_NO_AVX512=1` | Forces AVX-512 not to be used |
| `TERNARYAI_NO_NEON=1` | Forces NEON not to be used |
| `TERNARYAI_NO_VNNI=1` | Keeps AVX-512 but skips the VNNI integer forward (same result, bit for bit) |
| `TERNARYAI_LIB` | Path of the native library to load |

## 14. Reference table

Everything exported by `ternaryai` (`tn.__all__`) and the generated layers.

| Name | What it is |
|---|---|
| `Model`, `History`, `load` | Model, result of `fit`, loading folders |
| `Dense`, `Conv2D`, `Embedding`, `GQA`, `SwiGLU`, `RMSNorm`, `LMHead` | Layers (generated from the backend) |
| `Layer`, `Residual`, `transformer_block`, `layer`, `available_layers` | Layer base class, residual block, transformer block, build by name, list of layers |
| `LayerSpec`, `Param`, `LayerDesc`, `ModelDescription`, `Backend`, `get_backend`, `set_backend` | Descriptions and access to the compute backend (advanced use) |
| `Tokenizer`, `CHAT_SPECIAL_TOKENS` | BPE tokenizer and its 13 special tokens |
| `dataset` | `load`, `tokenize`, `targets` and constants |
| `reasoning` | Tasks, verifiers, `evaluate`, `rejection_sample`, `rft_round`, `rft` |
| `Engine`, `Session`, `Reply` | Inference |
| `CTX_ERROR`, `CTX_TRUNCATE`, `CTX_SLIDE` | Full-context policies |
| `export_ternam`, `build_engine_lib`, `export_fused`, `use_engine_lib`, `FUSED_API` | Export and engine libraries |
| `set_threads`, `get_threads`, `set_engine_threads`, `engine_threads`, `set_engine_backend`, `engine_backend` | Threads and kernels |
| `TernaryAIError`, `ArchitectureError`, `BackendNotFoundError`, `DataError`, `ModelFileError`, `TokenizerError`, `DatasetError`, `InferenceError` | Exceptions |
