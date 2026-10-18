"""Run prompts through llama.cpp (GGUF) or MLX (4-bit base plus LoRA adapter) with identical token ids.

Prompts are tokenized once with the model's chat template, so both engines and training see the same tokens.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.request
from pathlib import Path

from indentra.task import FIELDS

LLAMA_SERVER = "llama-server"
SERVER_START_TIMEOUT = 180
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"field": {"enum": sorted(FIELDS)}, "value": {"type": "number"}},
                "required": ["field", "value"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["facts"],
    "additionalProperties": False,
}


def render_ids(tokenizer, messages: list[dict[str, str]]) -> list[int]:
    return list(tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=False))


class LlamaServer:
    """One llama-server process with a single slot, so results do not depend on batch composition."""

    def __init__(self, gguf: Path, port: int = 8091, context: int = 4096, lora: Path | None = None) -> None:
        self._command = [LLAMA_SERVER, "-m", str(gguf), "--port", str(port), "-c", str(context),
                         "-np", "1", "-ngl", "99", "--no-webui", "--seed", "0"] + (["--lora", str(lora)] if lora else [])
        self._url = f"http://127.0.0.1:{port}"
        self._process: subprocess.Popen | None = None

    def __enter__(self) -> LlamaServer:
        self._process = subprocess.Popen(self._command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + SERVER_START_TIMEOUT
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {self._process.returncode}")
            try:
                with urllib.request.urlopen(f"{self._url}/health", timeout=2) as response:
                    if json.load(response).get("status") == "ok":
                        return self
            except OSError:
                pass
            time.sleep(0.5)
        raise TimeoutError("llama-server did not become healthy")

    def __exit__(self, *exc) -> None:
        if self._process is not None:
            self._process.terminate()
            self._process.wait(timeout=30)

    def complete(self, ids: list[int], max_tokens: int, schema: dict | None = None) -> dict:
        body = {"prompt": ids, "n_predict": max_tokens, "temperature": 0.0, "seed": 0, "cache_prompt": True}
        if schema is not None:
            body["json_schema"] = schema
        request = urllib.request.Request(
            f"{self._url}/completion", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.load(response)


def mlx_generate(model_path: Path, adapter_path: Path | None, prompts: list[list[int]], max_tokens: int, batch_size: int = 16):
    """Greedy batched generation; returns decoded texts, per-output truncation flags, and throughput stats."""
    from mlx_lm import load

    model, tokenizer = load(str(model_path), adapter_path=str(adapter_path) if adapter_path else None)
    texts, truncated, prompt_tps, generation_tps, peak_memory = [], [], [], [], 0.0
    for start in range(0, len(prompts), batch_size):
        response = mlx_batch(model, tokenizer, prompts[start : start + batch_size], max_tokens)
        texts.extend(response.texts)
        truncated.extend(mlx_truncated(response, max_tokens))
        prompt_tps.append(response.stats.prompt_tps)
        generation_tps.append(response.stats.generation_tps)
        peak_memory = max(peak_memory, response.stats.peak_memory)
    return texts, truncated, {
        "prompt_tps_mean": sum(prompt_tps) / len(prompt_tps),
        "generation_tps_mean": sum(generation_tps) / len(generation_tps),
        "peak_memory_gb": peak_memory,
    }


def mlx_batch(model, tokenizer, prompts: list[list[int]], max_tokens: int):
    """One greedy mlx_lm.batch_generate call on an already loaded model (the decoding used by every MLX evaluation)."""
    from mlx_lm import batch_generate

    return batch_generate(model, tokenizer, prompts, max_tokens=max_tokens, return_token_ids=True)


def mlx_truncated(response, max_tokens: int) -> list[bool]:
    # batch_generate drops the stop token, so a full-length id list means generation hit max_tokens
    return [len(ids) >= max_tokens for ids in response.token_ids]
