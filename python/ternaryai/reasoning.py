"""Verifiers, evaluation (pass@k) and rejection-sampling fine-tuning for reasoning models.

    import ternaryai as tn
    R = tn.reasoning

    tasks = [R.Task("¿Cuánto es 2+2?", reference=4), ...]            # or dicts {"prompt": ..., "answer": ...}
    ok = R.numeric()                                                  # verifier: last number of the answer == reference
    res = R.evaluate(model.engine(tok), tasks, ok, n=8, k=(1, 4), temperature=0.8, max_new_tokens=128)
    print(res.summary())                                              # pass@1, pass@4, tokens, stop reasons

    # rejection-sampling fine-tuning (STaR / RFT): sample, keep what the verifier accepts, train on it
    R.rft(model, tok, tasks, ok, rounds=3, n=8, epochs=2)

Everything is plain Python over :meth:`ternaryai.Engine.sample` (the candidates of a task run as one batch in the engine).
A *policy* is an :class:`~ternaryai.Engine`, or any object with ``sample(messages, n, **gen) -> list[Reply]`` (or a callable with that
signature), so the same code evaluates models served elsewhere. A *verifier* is ``verify(reply, task) -> bool | float``.
Verifiers never execute model output: they only compare text.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .infer import Reply

__all__ = ["Task", "as_task", "exact_match", "contains", "regex_match", "numeric", "has_thinking", "all_of", "last_number", "pass_at_k",
           "evaluate", "EvalResult", "TaskResult", "rejection_sample", "RFTData", "Sample", "rft_round", "rft"]

Verifier = Callable[[Reply, "Task"], Union[bool, float]]


# ------------------------------------------------------------------ tasks
@dataclass
class Task:
    """A prompt and what the verifier needs to judge the answer. ``messages`` is a string (one user message) or a list of
    ``{"role", "content"}`` dicts / ``(role, content)`` pairs; ``reference`` is the expected answer (any type your verifier understands)."""
    messages: Any
    reference: Any = None
    meta: Dict[str, Any] = field(default_factory=dict)


def as_task(t: Any) -> Task:
    """Accepts a :class:`Task`, a dict (``messages`` or ``prompt``, plus ``answer`` / ``reference``), or a ``(prompt, reference)`` pair."""
    if isinstance(t, Task):
        return t
    if isinstance(t, dict):
        msgs = t["messages"] if "messages" in t else t["prompt"]
        ref = t.get("reference", t.get("answer"))
        meta = {k: v for k, v in t.items() if k not in ("messages", "prompt", "reference", "answer")}
        return Task(msgs, ref, meta)
    if isinstance(t, (tuple, list)) and len(t) == 2:
        return Task(t[0], t[1])
    raise TypeError(f"cannot turn {type(t).__name__} into a Task")


# ------------------------------------------------------------------ verifiers
_NUM = re.compile(r"[-+]?(?:\d{1,3}(?:[ ,.]\d{3})+|\d+)(?:[.,]\d+)?(?:[eE][-+]?\d+)?")


def last_number(text: str) -> Optional[float]:
    """The last number in ``text`` (``1,234.5`` / ``1.234,5`` / ``3,5`` accepted), or None."""
    found = _NUM.findall(text)
    if not found:
        return None
    s = found[-1].strip()
    if re.fullmatch(r"[-+]?\d{1,3}(?:[ ,.]\d{3})+", s):                    # thousands separators only
        s = re.sub(r"[ ,.]", "", s)
    elif "," in s and "." in s:                                           # the last separator is the decimal one
        dec = "," if s.rfind(",") > s.rfind(".") else "."
        s = s.replace("." if dec == "," else ",", "").replace(dec, ".")
    else:
        s = s.replace(",", ".")
        s = re.sub(r"[ ]", "", s)
    try:
        return float(s)
    except ValueError:
        return None


def _norm(s: Any, casefold: bool) -> str:
    s = re.sub(r"\s+", " ", str(s)).strip().strip(".,;:!?¡¿\"'`")
    return s.casefold() if casefold else s


def exact_match(casefold: bool = True) -> Verifier:
    """The answer text equals the reference after trimming spaces and edge punctuation (and case, by default)."""
    return lambda r, t: _norm(r.text, casefold) == _norm(t.reference, casefold)


def contains(casefold: bool = True) -> Verifier:
    """The reference appears in the answer text."""
    return lambda r, t: _norm(t.reference, casefold) in re.sub(r"\s+", " ", r.text.casefold() if casefold else r.text)


def regex_match(pattern: str, group: int = 1, casefold: bool = True) -> Verifier:
    """``pattern`` is searched in the answer (the LAST match wins); ``group`` must equal the reference (as normalized text)."""
    rx = re.compile(pattern, re.S)

    def verify(r: Reply, t: Task) -> bool:
        ms = list(rx.finditer(r.text))
        return bool(ms) and _norm(ms[-1].group(group), casefold) == _norm(t.reference, casefold)
    return verify


def numeric(rel_tol: float = 1e-6, abs_tol: float = 1e-9) -> Verifier:
    """The last number of the answer equals ``float(reference)`` within the tolerances."""
    def verify(r: Reply, t: Task) -> bool:
        v = last_number(r.text)
        return v is not None and math.isclose(v, float(t.reference), rel_tol=rel_tol, abs_tol=abs_tol)
    return verify


def has_thinking(min_chars: int = 1, closed: bool = True) -> Verifier:
    """Format check: the reply has a reasoning block of at least ``min_chars`` and (if ``closed``) the turn ended on its own (``<|end|>``)."""
    return lambda r, t: len(r.thinking.strip()) >= min_chars and (not closed or r.stop == "token")


def all_of(*verifiers: Verifier) -> Verifier:
    """Product of the rewards: 1 only if every verifier accepts (use for 'correct AND well formatted')."""
    def verify(r: Reply, t: Task) -> float:
        out = 1.0
        for v in verifiers:
            out *= float(v(r, t))
            if out == 0.0:
                break
        return out
    return verify


# ------------------------------------------------------------------ metrics
def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k (Chen et al. 2021): probability that at least one of ``k`` samples drawn from ``n`` (``c`` correct) passes."""
    if not (0 <= c <= n) or k < 1 or k > n:
        raise ValueError(f"need 0 <= c <= n and 1 <= k <= n (got n={n}, c={c}, k={k})")
    if n - c < k:
        return 1.0
    i = np.arange(n - c + 1, n + 1, dtype=np.float64)
    return float(1.0 - np.prod(1.0 - k / i))


def _sampler(policy: Any) -> Callable[..., List[Reply]]:
    if hasattr(policy, "sample"):
        return policy.sample
    if callable(policy):
        return policy
    raise TypeError("policy must be an Engine, an object with .sample(messages, n, **gen) or a callable with that signature")


@dataclass
class TaskResult:
    task: Task
    replies: List[Reply]
    rewards: List[float]
    passed: List[bool]


@dataclass
class EvalResult:
    n_tasks: int
    n: int
    pass_at: Dict[int, float]
    mean_reward: float
    mean_tokens: float
    mean_think_tokens: float
    stops: Dict[str, int]
    tasks: List[TaskResult] = field(default_factory=list)

    def summary(self) -> str:
        ps = "  ".join(f"pass@{k} {v:.3f}" for k, v in sorted(self.pass_at.items()))
        st = ", ".join(f"{k}={v}" for k, v in sorted(self.stops.items()))
        return (f"{self.n_tasks} tasks x {self.n} samples | {ps} | reward {self.mean_reward:.3f} | "
                f"tokens {self.mean_tokens:.1f} (thinking {self.mean_think_tokens:.1f}) | stops: {st}")


def evaluate(policy: Any, tasks: Iterable[Any], verifier: Verifier, n: int = 1, k: Sequence[int] = (1,), *, pass_threshold: float = 1.0,
             on_task: Optional[Callable[[int, TaskResult], None]] = None, **gen: Any) -> EvalResult:
    """Sample ``n`` answers per task, verify them and report pass@k (for each ``k <= n``), mean reward, lengths and stop reasons.

    ``gen`` goes to ``policy.sample`` (``max_new_tokens``, ``temperature``, ``think_budget``, ``show_thinking``, ...). With ``n > 1`` use
    ``temperature > 0`` or the candidates are identical. A sample passes when its reward is ``>= pass_threshold``."""
    sample = _sampler(policy)
    ks = sorted(set(int(x) for x in k))
    if not ks or ks[0] < 1 or ks[-1] > n:
        raise ValueError(f"every k must be in 1..n (n={n}, k={list(k)})")
    res: List[TaskResult] = []
    stops: Dict[str, int] = {}
    toks = tot = 0
    rew = 0.0
    per_k = {x: [] for x in ks}
    for i, t in enumerate(tasks):
        t = as_task(t)
        reps = sample(t.messages, n, **gen)
        if len(reps) != n:
            raise ValueError(f"the policy returned {len(reps)} replies, expected {n}")
        rs = [float(verifier(r, t)) for r in reps]
        ok = [x >= pass_threshold for x in rs]
        c = sum(ok)
        for x in ks:
            per_k[x].append(pass_at_k(n, c, x))
        for r in reps:
            stops[r.stop] = stops.get(r.stop, 0) + 1
            toks += len(r.ids)
        rew += sum(rs); tot += n
        tr = TaskResult(t, reps, rs, ok)
        res.append(tr)
        if on_task:
            on_task(i, tr)
    if not res:
        raise ValueError("no tasks")
    think_tokens = _think_tokens(policy, res)
    return EvalResult(len(res), n, {x: float(np.mean(v)) for x, v in per_k.items()}, rew / tot, toks / tot, think_tokens / tot, stops, res)


def _think_tokens(policy: Any, res: List[TaskResult]) -> int:
    split = getattr(policy, "split_thinking", None)
    total = 0
    for tr in res:
        for r in tr.replies:
            if split is not None:
                total += len(split(r.ids)[1])
            else:
                total += len(r.thinking.split())            # no tokenizer access: words as an approximation
    return total


# ------------------------------------------------------------------ rejection sampling
@dataclass
class Sample:
    task: Task
    reply: Reply
    reward: float


@dataclass
class RFTData:
    """Accepted samples of one rejection-sampling pass, with the statistics that tell whether the model is improving."""
    samples: List[Sample]
    end_id: int
    tasks_total: int
    tasks_solved: int
    candidates: int
    accepted: int
    mean_pass_rate: float

    def __len__(self) -> int:
        return len(self.samples)

    def sequences(self) -> List[Tuple[List[int], int]]:
        """``(ids, n_prompt)`` per sample: prompt + generated answer + ``<|end|>``; only the part after ``n_prompt`` is trained."""
        out = []
        for s in self.samples:
            r = s.reply
            out.append((list(r.prompt_ids) + list(r.ids) + [self.end_id], len(r.prompt_ids)))
        return out

    def to_arrays(self, seq_len: int, pad_id: int = 0, overflow: str = "drop") -> Tuple[np.ndarray, np.ndarray]:
        """``(x, y)`` for ``Model.fit``: ids padded to ``seq_len``; ``y[i] = ids[i+1]`` only for the generated part (prompt and padding are -1,
        which the loss ignores). ``overflow``: ``"drop"`` (sequences longer than ``seq_len``) or ``"truncate"``."""
        if overflow not in ("drop", "truncate"):
            raise ValueError("overflow must be 'drop' or 'truncate'")
        seqs = [(ids, npmt) for ids, npmt in self.sequences() if len(ids) <= seq_len or overflow == "truncate"]
        x = np.full((len(seqs), seq_len), pad_id, dtype=np.int32)
        y = np.full((len(seqs), seq_len), -1, dtype=np.int32)
        for j, (ids, npmt) in enumerate(seqs):
            ids = ids[:seq_len]
            L = len(ids)
            x[j, :L] = ids
            lo = max(npmt - 1, 0)                                       # position whose target is the first generated token
            if L - 1 > lo:
                y[j, lo:L - 1] = ids[lo + 1:L]
        return x, y


def rejection_sample(policy: Any, tasks: Iterable[Any], verifier: Verifier, n: int = 8, *, per_task: int = 1, select: str = "shortest",
                     min_reward: float = 1.0, seed: int = 0, end_id: Optional[int] = None, **gen: Any) -> RFTData:
    """Sample ``n`` answers per task and keep up to ``per_task`` that the verifier accepts (reward ``>= min_reward``).

    Only answers where the model closed its own turn with ``<|end|>`` are kept (cut-off answers would teach not to stop).
    ``select``: ``"shortest"`` (fewest tokens: favors concise reasoning), ``"first"`` or ``"random"`` (reproducible with ``seed``).
    Sampling defaults to ``temperature=1.0`` so the candidates differ. ``end_id`` defaults to the policy's ``<|end|>`` id."""
    if select not in ("shortest", "first", "random"):
        raise ValueError("select must be 'shortest', 'first' or 'random'")
    if end_id is None:
        end_id = policy.special_id("<|end|>") if hasattr(policy, "special_id") else -1
    if end_id < 0:
        raise ValueError("the tokenizer has no <|end|> (pass end_id= for a custom policy)")
    gen.setdefault("temperature", 1.0)
    sample = _sampler(policy)
    rng = random.Random(seed)
    kept: List[Sample] = []
    total = solved = cand = 0
    rate = 0.0
    for i, t in enumerate(tasks):
        t = as_task(t)
        reps = sample(t.messages, n, seed=seed + i * n, **gen)
        rs = [float(verifier(r, t)) for r in reps]
        good = [Sample(t, r, x) for r, x in zip(reps, rs) if x >= min_reward]
        rate += len(good) / n
        good = [g for g in good if g.reply.stop == "token" and g.reply.stop_id == end_id]
        total += 1; cand += n
        if good:
            solved += 1
            if select == "shortest":
                good.sort(key=lambda g: len(g.reply.ids))
            elif select == "random":
                rng.shuffle(good)
            kept.extend(good[:per_task])
    if total == 0:
        raise ValueError("no tasks")
    return RFTData(kept, end_id, total, solved, cand, len(kept), rate / total)


# ------------------------------------------------------------------ fine-tuning on accepted samples
@dataclass
class RoundResult:
    round: int
    pass_at_1_before: float
    data: RFTData
    trained_sequences: int
    history: Any = None


def rft_round(model: Any, tokenizer: Any, tasks: Sequence[Any], verifier: Verifier, *, n: int = 8, per_task: int = 1, select: str = "shortest",
              epochs: int = 1, replay: Optional[Tuple[np.ndarray, np.ndarray]] = None, eval_tasks: Optional[Sequence[Any]] = None,
              seed: int = 0, fit_kwargs: Optional[Dict[str, Any]] = None, **gen: Any) -> RoundResult:
    """One pass of rejection-sampling fine-tuning: snapshot the model as an engine, sample ``n`` answers per task, keep the verified ones and
    train ``epochs`` epochs on them (``model.fit``; only the generated part enters the loss).

    ``replay=(x, y)`` (same ``seq_len``) is mixed in, e.g. the original SFT arrays, to limit forgetting. ``eval_tasks`` (default: ``tasks``)
    measure greedy pass@1 before training. ``gen`` goes to the sampling (``max_new_tokens``, ``temperature``, ``think_budget``, ...)."""
    tasks = list(tasks)
    eng = model.engine(tokenizer)
    try:
        before = evaluate(eng, eval_tasks if eval_tasks is not None else tasks, verifier, n=1, k=(1,),
                          **{k: v for k, v in gen.items() if k not in ("temperature", "seed")}).pass_at[1]
        data = rejection_sample(eng, tasks, verifier, n=n, per_task=per_task, select=select, seed=seed, **gen)
        seq_len = eng.seq_len
    finally:
        eng.close()
    hist = None
    trained = 0
    if len(data):
        x, y = data.to_arrays(seq_len)
        if len(x):
            if replay is not None:
                x = np.concatenate([x, np.asarray(replay[0], dtype=np.int32)]); y = np.concatenate([y, np.asarray(replay[1], dtype=np.int32)])
            trained = len(x)
            hist = model.fit(x, y, epochs=epochs, **(fit_kwargs or {}))
    return RoundResult(0, before, data, trained, hist)


def rft(model: Any, tokenizer: Any, tasks: Sequence[Any], verifier: Verifier, rounds: int = 3, *, verbose: bool = True, **kw: Any) -> List[RoundResult]:
    """``rounds`` passes of :func:`rft_round` (the model is the policy and learns from its own verified answers; each round samples with the
    updated weights). Stops early when a round finds no accepted sample. Returns one :class:`RoundResult` per round."""
    out: List[RoundResult] = []
    for r in range(rounds):
        kw["seed"] = kw.get("seed", 0) + (1000 if r else 0)
        res = rft_round(model, tokenizer, tasks, verifier, **kw)
        res.round = r
        out.append(res)
        if verbose:
            d = res.data
            print(f"[rft {r}] pass@1 before {res.pass_at_1_before:.3f} | solved {d.tasks_solved}/{d.tasks_total} | sampled pass rate {d.mean_pass_rate:.3f} | "
                  f"trained on {res.trained_sequences} sequences", flush=True)
        if not len(res.data):
            break
    return out
