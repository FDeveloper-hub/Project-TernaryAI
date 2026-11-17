# TernaryAI

*Versión en español: [LEEME.md](https://github.com/FDeveloper-hub/Project-TernaryAI/blob/main/LEEME.md)* · *Full syntax reference: [docs/MANUAL.en.md](https://github.com/FDeveloper-hub/Project-TernaryAI/blob/main/docs/MANUAL.en.md)*

Neural networks whose weights are strictly **−1, 0 or +1**, with the computation in **C11** and a Python interface (`import ternaryai`).
Built for small devices: it trains and generates on a 2018 low-end Android device (32-bit ARMv7, ~1 GB of RAM) and takes advantage of
AVX-512 on x86-64 and NEON on ARM when they are available.

- **Ternary weights without a float copy per weight.** The R2 update rule keeps, for every weight, an 8-bit counter with hysteresis (not a float
  "shadow" weight as in QAT or BitNet). Each layer has a final float scale.
- **Training and inference in the same package.** The inference engine gives logits bit-identical to `Model.predict`, and the scalar, AVX-512 and
  NEON kernels give the same bits.
- **Language models end to end.** Byte-level BPE tokenizer with special tokens, dataset reader (its own format plus the most used Hugging Face ones),
  generation with reasoning (`<|think|>…<|/think|>`), verifiers, pass@k and rejection-sampling fine-tuning.
- **A single Python dependency: NumPy**, and only in the interface (input and output arrays). The computation does not go through NumPy.

## What is inside

| Component | What it does |
|---|---|
| Layers (`tn.Dense`, `Conv2D`, `Embedding`, `GQA`, `SwiGLU`, `RMSNorm`, `LMHead`, `Residual`) | Building blocks for classifiers and transformers (grouped-query attention, integer RoPE, sliding window, SwiGLU or ReLU² activation) |
| `tn.Model` | `fit` (mini-batches, loss mask, checkpoints), `predict`, `save` / `load`, `export` |
| `tn.Tokenizer` | Byte-level BPE in C, with special tokens; `tn.CHAT_SPECIAL_TOKENS` holds the 13 conversation and reasoning tokens |
| `tn.dataset` | `load`, `tokenize`, `targets`: its own `chat` format, ChatML/OpenAI `messages`, ShareGPT `conversations`, `prompt`+`completion`, Alpaca and `text` |
| `tn.Engine` / `Session` | Inference with a KV cache, sampling (`temperature`, `top_k`, `top_p`, …), `reply`, `sample`, batches of conversations, threads |
| Export | `.ternam` (one file, loaded with `mmap`), engine-only library (`build_engine_lib`) and fused library with the model embedded (`export_fused`) |
| `tn.reasoning` | Verifiers, `evaluate` (pass@k), `rejection_sample` and `rft` |

## Installation

You need Python ≥ 3.10 and NumPy ≥ 1.21.6. This is an alpha release (1.0.0a1), so `pip` needs the `--pre` flag. Three ways to install:

- **From PyPI:** `pip install --pre ternaryai`. On Linux x86-64 with glibc ≥ 2.34 it installs a prebuilt wheel (the native library and the C sources it needs are inside; no compiler required). On other platforms, for example Termux on Android, pip builds it from source, which needs a C11 compiler and `make`.
- **Wheel from GitHub:** download `ternaryai-1.0.0a1-py3-none-manylinux_2_34_x86_64.whl` from the [Releases](https://github.com/FDeveloper-hub/Project-TernaryAI/releases) page and install it with `pip install` followed by the file name.
- **From source:** `pip install .` from a clone of this repository. The build runs the Makefile.

To work on the library itself, build it in place and use the package from the source tree:

```bash
make libternaryai.so                 # builds the native library (AVX-512 / NEON are detected at run time)
export PYTHONPATH=$PWD/python        # or: pip install -e .
python3 -c "import ternaryai as tn; print(tn.__name__, 'ok')"
```

Tested on Linux x86-64 with Python 3.10, 3.11, 3.12 and 3.13 and NumPy 1.21.6, 1.24.4, 2.2.6, 2.4.6 and 2.5.5, and on ARMv7 (Termux on a 32-bit ARMv7 device, and under qemu). Not tested on Windows or macOS.
`TERNARYAI_NO_AVX512=1` / `TERNARYAI_NO_NEON=1` force the scalar kernel; `TERNARYAI_NO_VNNI=1` keeps AVX-512 but skips the VNNI integer forward (the result is bit-identical either way).

## Example: from a dataset to a model that talks

A file `data.jsonl` with one conversation per line, in the library's own format (`user` / `model`):

```json
{"chat": [{"user": "Hello"}, {"model": "Hello, how can I help you?"}]}
{"chat": [{"user": "What is 2+2?"}, {"model": "<think>2+2=4</think>It is 4."}]}
```

```python
import ternaryai as tn

examples, skipped = tn.dataset.load("data.jsonl")                # one string per conversation, with the role markers
tok = tn.Tokenizer(400, special_tokens=list(tn.CHAT_SPECIAL_TOKENS))
tok.train("\n".join(examples).encode())
x, y = tn.dataset.tokenize(examples, tok, seq_len=64)            # ids + targets: only what the assistant says is trained

model = tn.Model(64, [
    tn.Embedding(tok.vocab_size, 48),
    *tn.transformer_block(heads=4, hidden=96, kv_heads=2, gate_mode=2),
    tn.RMSNorm(activation="uint8"),
    tn.LMHead(tok.vocab_size),
])
model.fit(x, y, epochs=400)

engine = model.engine(tok)                                       # inference engine (a snapshot of the model)
r = engine.session().reply("What is 2+2?", max_new_tokens=40)
print(r.text, "|", r.thinking)                                   # It is 4. | 2+2=4
print(engine.session().reply("Hello", max_new_tokens=40).text)  # Hello, how can I help you?
```

(With 8 conversations of each kind repeated, as in this test; the exact result depends on the seed and the model size.)

To save the model: `model.save(folder)` (with the counters, to keep training) or `model.export(folder)` (inference only);
`tn.export_ternam(folder, "model.ternam")` writes a self-contained file that `tn.Engine("model.ternam")` opens with `mmap`.

## Conversation and reasoning

`Session.reply(messages, ...)` builds the prompt like the dataset reader does (`<|role|>\ntext\n`) and stops at `<|end|>` and `<|user|>`.
It returns a `Reply` with `.text`, `.thinking`, `.stop` and `.ids`.

- The `<|think|>…<|/think|>` block goes to `.thinking` and is removed from `.text`. With `show_thinking=True` it stays inside `.text`, tags included,
  so you can read the chain of thought in place.
- `think_budget=N` limits the reasoning tokens (when reached, `<|/think|>` is forced); `force_think=True` starts already inside `<|think|>`.
- `dataset.load(..., reasoning="keep" | "drop")` accepts `<think>…</think>` and the fields `reasoning_content`, `reasoning` and `thinking`.
- `Engine.chat_encode`, `Engine.split_thinking` and `Engine.special_id` help build and analyze sequences by hand.

## Evaluation and rejection-sampling fine-tuning

```python
R = tn.reasoning
tasks = [R.Task("What is 2+2?", reference=4), ("What is 3+5?", 8)]
ok = R.all_of(R.numeric(), R.has_thinking())            # correct AND with a properly closed reasoning block
print(R.evaluate(engine, tasks, ok, n=8, k=(1, 4), temperature=0.8, max_new_tokens=64).summary())

R.rft(model, tok, tasks, ok, rounds=3, n=8, epochs=2)   # sample, keep what is verified and train on it
```

- `Engine.sample(messages, n)` generates `n` answers to the same conversation (the prompt is processed once, the candidates run as one batch, candidate
  `i` uses the seed `seed + i`).
- The verifiers (`exact_match`, `contains`, `regex_match`, `numeric`, `has_thinking`, `all_of`, or any function `verify(reply, task)`)
  only compare text: they never execute what the model generates.
- `evaluate` gives pass@k (unbiased estimator), mean reward, lengths and stop reasons. `rft` accepts `replay=(x, y)` to mix in the original
  training data and limit forgetting.

## Performance

Measured against a BitNet b1.58 QAT model written in PyTorch (CPU only, PyTorch 2.14.1), with the same architecture and the same number of weights.
Conditions: 1 thread on 1 core (Xeon 2.1 GHz, AVX-512 VNNI), 30 epochs, batch 8, context 64, byte-level text (`pydoc_data.topics`), a single seed.
The PyTorch model keeps float shadow weights, ternarizes them with absmean and a straight-through estimator, uses int8 activations per token, Adam (lr 0.01),
and ternary embedding and output head. Measured peaks on that core: fp32 215 GFLOP/s, int8 VNNI 677 GOP/s, DRAM 16 GB/s.

| | d=96, TernaryAI | d=96, PyTorch QAT | d=640, TernaryAI | d=640, PyTorch QAT |
|---|---|---|---|---|
| Weights | 307 thousand | 307 thousand | 11.8 M | 11.8 M |
| Training (tokens/s) | 18,798 | 12,289 | 803 | 819 |
| Eval loss | 1.833 | 1.801 | 2.681 | 2.376 |
| Eval accuracy | 53.1 % | 56.1 % | 35.9 % | 40.0 % |
| Prefill (tokens/s) | 150,216 | 26,211 | 8,678 | 3,427 |
| Decoding, 1 conversation (tokens/s) | 115,199 | 990 | 3,869 | 291 |
| Decoding, batch of 32 (tokens/s) | 117,854 | 14,678 | 6,075 | 1,560 |
| Peak process RAM (anonymous) | 34 MB | 196 MB | 193 MB | 616 MB |
| Peak process RSS | 65 MB | 303 MB | 227 MB | 725 MB |
| Exported model | 0.08 MB | 0.31 MB (int8) | 2.4 MB | 11.8 MB (int8) |

How to read it:

- **Training speed is on par.** TernaryAI is 1.5 times faster at d=96 and ties at d=640 (803 against 819 tokens/s).
- **Inference and memory are where the library wins.** With a batch of 32 conversations it is 8 times (d=96) and 3.9 times (d=640) faster; prefill is 5.7 and 2.5 times faster;
  the process uses 3 to 6 times less RAM, and the exported model is 4 to 5 times smaller. Under a hard cap of 200 MB of RAM (no swap) TernaryAI trained 11.8 M weights,
  while the PyTorch QAT model reached only 307 thousand, about 38 times fewer.
- **The single-conversation figures are inflated.** The PyTorch model runs in float, in eager mode, and with models this small its time is dominated by Python and
  operator-launch overhead. The batch-of-32 row is the more reliable one.
- **Quality:** the eval losses are close (1.833 against 1.801 at d=96, 2.681 against 2.376 at d=640); the PyTorch QAT model is somewhat better in both.
- **Limits of the measurement:** one seed, short training, small models, and the PyTorch QAT baseline has no internal LayerNorm in its BitLinear (unlike the original BitLinear).
  The two training rules also differ (R2 without float weights against Adam in float32).
- **Threads** (`tn.set_threads(n)`, `tn.set_engine_threads(n)`): the result is bit-identical with any number of threads.
- **Memory format:** on disk a weight takes 1.6 bits (5 ternary values per byte); the engine keeps the weights at 2 bits and a compact KV cache.

## Known limits

- Everything has been validated with small models (up to about 0.5 M weights in the blocks). There are no results with large models.
- At the same number of weights, a float model is better in quality than the ternary one: in a test with Spanish novels, at 25 minutes the float model had a loss of 4.32 against 4.51 for the ternary one.
  The advantage of the ternary model is in memory, size on disk and inference speed, not in quality per weight.
  This is expected: a ternary model (1.58 bits per parameter) has less capacity than a float one with the same number of parameters. The studies on BitNet b1.58 by Microsoft
  ([arXiv:2402.17764](https://arxiv.org/abs/2402.17764)) report that this disadvantage disappears from about 3 billion parameters (against an FP16 baseline, in perplexity). That is a result reported by those
  studies, with a different training recipe from ours; it has not been reproduced with this library, which has only been validated with small models.
- Long-distance attention needs a higher `rope_base` (default 10; 1000 was validated at 512 tokens).

## Tests

```bash
make test                # C tests (≈15 min)
make test-python         # Python tests (builds the library first)
```


## License

Apache License 2.0. Full text in [`LICENSE`](https://github.com/FDeveloper-hub/Project-TernaryAI/blob/main/LICENSE) and notices in [`NOTICE`](https://github.com/FDeveloper-hub/Project-TernaryAI/blob/main/NOTICE). Copyright 2026 FDeveloper-hub.
