"""Paragraph-level extraction for the Indentra demo: B3 and the deterministic TF-IDF baseline, each loaded once.

B3 is the fine-tuned Llama 3.2 3B that labels regex candidates (RESULTS runs 032-033). It runs exactly as in its
evaluation: the candidates prompt (indentra.hybrid), the pinned chat template, greedy mlx_lm.batch_generate with a
256-token cap, and the strict parse, so a malformed answer yields no facts. Loading refuses any base model or adapter
whose digest differs from the one recorded by the B3 evaluation run, so no other model can stand in for it.
Every fact keeps the source spans of the candidates that produced it.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from indentra import baselines, eightk, hybrid, infer, runs
from indentra.task import FIELDS

FIELD_LABELS = {
    "face_amount": ("Principal amount", "The face or par amount of a specific note, bond or loan."),
    "stated_rate": ("Fixed interest rate", "The stated (coupon) rate of a debt instrument."),
    "effective_rate": ("Effective interest rate", "An interest rate the paragraph calls effective."),
    "spread": ("Interest margin", "The margin added to (or taken from) a reference rate such as SOFR or prime."),
    "facility_size": ("Borrowing limit", "The most that can be borrowed under a credit facility or commitment."),
    "commitment_fee": ("Commitment fee", "The fee rate charged on the unused part of a facility."),
}
MAX_CHARS = eightk.MAX_CHARS  # the longest paragraph the 8-K filter (and so the evaluation) admits
MAX_NEW_TOKENS = 256
MODEL_DIR = Path("models/Llama-3.2-3B-Instruct-mlx-q4")
ADAPTER_DIR = Path("adapters/full-neg33-cand")
EVAL_CONFIG = Path("runs/ft-full-neg33-cand-mlx-eightk_fresh/config.json")  # records the digests B3 was evaluated with
MODEL_FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
ADAPTER_FILES = ("adapter_config.json", "adapters.safetensors")
TFIDF_DIR = Path("artifacts/tfidf-lr-det")


class InputError(ValueError):
    """The paragraph cannot be processed as given; the message says why."""


class LoadError(RuntimeError):
    """A model could not be loaded; the message names the missing or mismatched file."""


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class ExtractedFact:
    field: str
    value: str
    label: str
    display: str
    spans: tuple[Span, ...]
    other_occurrences: int  # candidates with the same number that the system did not label


@dataclass(frozen=True)
class Result:
    engine: str
    text: str
    facts: tuple[ExtractedFact, ...]
    seconds: float
    valid: bool = True
    model_called: bool = True
    truncated: bool = False
    note: str | None = None
    raw: str | None = None


def check_paragraph(text: str) -> str:
    """The paragraph with whitespace collapsed, as the evaluation's paragraph builder did; nothing else is changed."""
    paragraph = " ".join(text.split())
    if not paragraph:
        raise InputError("Paste a paragraph first.")
    if len(paragraph) > MAX_CHARS:
        raise InputError(f"This text has {len(paragraph):,} characters. Indentra takes one paragraph of up to {MAX_CHARS:,} "
                         "characters, the longest the evaluation used. Paste a single paragraph; nothing is cut off automatically.")
    return paragraph


def candidate_spans(text: str) -> list[Span]:
    """Where each regex candidate was read from, in the same order as baselines.candidates and hybrid.candidates."""
    spans = []
    for c in baselines.candidates(text):
        match = next(m for pattern in hybrid.PATTERNS[c.kind] if (m := pattern.match(text, c.start)))
        spans.append(Span(c.start, match.end(), text[c.start : match.end()]))
    return spans


def format_value(field_name: str, value: Decimal) -> str:
    return f"${value:,f}" if FIELDS[field_name][1] == "usd" else f"{value:f}%"


def group_facts(text: str, labeled: list[tuple[int, str]]) -> tuple[ExtractedFact, ...]:
    """Facts from (candidate index, field) labels: one fact per (field, value), with every labeled span, in text order."""
    found, spans = baselines.candidates(text), candidate_spans(text)
    indices: dict[tuple[str, Decimal], list[int]] = {}
    for index, field_name in labeled:
        indices.setdefault((field_name, found[index].value), []).append(index)
    facts = []
    for (field_name, value), chosen in indices.items():
        chosen = sorted(set(chosen))
        repeats = sum(1 for i, c in enumerate(found) if c.value == value and c.kind == found[chosen[0]].kind and i not in chosen)
        facts.append(ExtractedFact(field_name, format(value, "f"), FIELD_LABELS[field_name][0], format_value(field_name, value),
                                   tuple(spans[i] for i in chosen), repeats))
    return tuple(sorted(facts, key=lambda f: (f.spans[0].start, f.field)))


@dataclass
class B3Extractor:
    """B3: Llama 3.2 3B (MLX 4-bit) with the full-neg33-cand LoRA adapter, kept in memory.

    MLX binds its GPU stream to the thread that loads the model, so loading and every generation run on one dedicated
    worker thread; requests from other threads queue there, one generation at a time, on the single loaded copy.
    """

    model_dir: Path = MODEL_DIR
    adapter_dir: Path = ADAPTER_DIR
    eval_config: Path = EVAL_CONFIG
    identity: dict = field(default_factory=dict)
    load_seconds: float | None = None
    _generate: Callable[[list[int]], tuple[str, bool]] | None = None
    _render: Callable[[list[dict]], list[int]] | None = None
    _worker: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(max_workers=1, thread_name_prefix="b3-mlx"))

    def load(self) -> None:
        self._worker.submit(self._load).result()

    def _load(self) -> None:
        started = time.monotonic()
        missing = [str(d / name) for d, names in ((self.model_dir, MODEL_FILES), (self.adapter_dir, ADAPTER_FILES))
                   for name in names if not (d / name).is_file()]
        if not self.eval_config.is_file():
            missing.append(str(self.eval_config))
        if missing:
            raise LoadError("B3 cannot load; missing: " + ", ".join(missing))
        recorded = json.loads(self.eval_config.read_text())
        digests = {"model_digest": runs.file_digest(self.model_dir), "adapter_digest": runs.file_digest(self.adapter_dir)}
        for key, value in digests.items():
            if value != recorded.get(key):
                raise LoadError(f"{key} {value} does not match {recorded.get(key)} recorded in {self.eval_config}; "
                                "refusing to present another model as B3.")
        from mlx_lm import load
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        model, mlx_tokenizer = load(str(self.model_dir), adapter_path=str(self.adapter_dir))

        def generate(ids: list[int]) -> tuple[str, bool]:
            response = infer.mlx_batch(model, mlx_tokenizer, [ids], MAX_NEW_TOKENS)
            return response.texts[0], infer.mlx_truncated(response, MAX_NEW_TOKENS)[0]

        weights = hashlib.sha256((self.adapter_dir / "adapters.safetensors").read_bytes()).hexdigest()
        self.use_generator(generate, lambda messages: infer.render_ids(tokenizer, messages), {
            "name": "B3", "description": "Llama 3.2 3B Instruct, 4-bit MLX, with the B3 LoRA adapter (labels regex candidates)",
            "base_model": str(self.model_dir), "adapter": str(self.adapter_dir), "adapter_weights_sha256": weights,
            **digests, "matches_evaluation_run": str(self.eval_config), "decoding": f"greedy, up to {MAX_NEW_TOKENS} new tokens"})
        self.load_seconds = round(time.monotonic() - started, 1)

    def use_generator(self, generate: Callable[[list[int]], tuple[str, bool]], render: Callable[[list[dict]], list[int]], identity: dict) -> None:
        self._generate, self._render, self.identity = generate, render, identity

    def extract(self, text: str) -> Result:
        if self._generate is None or self._render is None:
            raise LoadError("B3 is not loaded.")
        started = time.monotonic()
        found = hybrid.candidates(text)
        if not found:
            return Result("b3", text, (), round(time.monotonic() - started, 3), model_called=False,
                          note="No dollar amounts, percentages or basis points were found, so there is nothing for B3 to label.")
        ids = self._render(hybrid.messages(text, found))
        raw, truncated = self._worker.submit(self._generate, ids).result()
        pairs = hybrid.labeled_pairs(raw, found)
        seconds = round(time.monotonic() - started, 3)
        if pairs is None:
            return Result("b3", text, (), seconds, valid=False, truncated=truncated, raw=raw,
                          note="B3's answer did not follow the expected format, so no facts are shown (its evaluation counts this as no facts).")
        return Result("b3", text, group_facts(text, [(n - 1, f) for n, f in pairs]), seconds, truncated=truncated, raw=raw)


@dataclass
class TfidfExtractor:
    """The deterministic TF-IDF + logistic regression baseline (RESULTS run 034), checked against its saved manifest."""

    artifact_dir: Path = TFIDF_DIR
    identity: dict = field(default_factory=dict)
    load_seconds: float | None = None
    _model: baselines.TfidfBaseline | None = None

    def load(self) -> None:
        started = time.monotonic()
        manifest_path, model_path = self.artifact_dir / "manifest.json", self.artifact_dir / "model.pkl"
        missing = [str(p) for p in (manifest_path, model_path) if not p.is_file()]
        if missing:
            raise LoadError("TF-IDF cannot load; missing: " + ", ".join(missing)
                            + '. Restore the original files (see "Recovering the TF-IDF model file" in docs/REPRODUCING.md);'
                            " extraction without --compare does not need them.")
        manifest, data = json.loads(manifest_path.read_text()), model_path.read_bytes()
        if hashlib.sha256(data).hexdigest() != manifest["model_pkl_sha256"]:
            raise LoadError(f"{model_path} does not match the sha256 in {manifest_path}; refusing to load it.")
        model = pickle.loads(data)  # the project's own artifact, verified by hash above
        if model.digest() != manifest["model_digest"]:
            raise LoadError(f"the model in {model_path} does not reproduce the digest in {manifest_path}.")
        self._model = model
        self.identity = {"name": "TF-IDF + logistic regression", "description": "Deterministic baseline that classifies the same regex candidates",
                         "artifact": str(self.artifact_dir), "model_digest": manifest["model_digest"]}
        self.load_seconds = round(time.monotonic() - started, 1)

    def extract(self, text: str) -> Result:
        if self._model is None:
            raise LoadError("TF-IDF is not loaded.")
        started = time.monotonic()
        labels = self._model.label_candidates(text)
        facts = group_facts(text, [(i, label) for i, label in enumerate(labels) if label != baselines.NONE])
        note = None if labels else "No dollar amounts, percentages or basis points were found."
        return Result("tfidf", text, facts, round(time.monotonic() - started, 4), model_called=bool(labels), note=note)
