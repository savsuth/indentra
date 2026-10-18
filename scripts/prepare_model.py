"""Prepare local model copies: pin the chat template's date, then build the MLX 4-bit base used for QLoRA.

Llama 3.2's chat template writes "Today Date: <today>" via strftime_now, so the same example would
tokenize differently on different days. We pin it to the template's own fallback, "26 Jul 2024".

    .venv/bin/python -m scripts.prepare_model
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HF_MODEL = Path("models/Llama-3.2-3B-Instruct")
MLX_Q4 = Path("models/Llama-3.2-3B-Instruct-mlx-q4")
PINNED_DATE = '{%- set date_string = "26 Jul 2024" %}\n'


def pin_template_date(model_dir: Path) -> None:
    """The template lives in tokenizer_config.json (transformers 4) or chat_template.jinja (transformers 5)."""
    jinja = model_dir / "chat_template.jinja"
    if jinja.exists():
        template = jinja.read_text()
        if not template.startswith(PINNED_DATE):
            jinja.write_text(PINNED_DATE + template)
        return
    path = model_dir / "tokenizer_config.json"
    config = json.loads(path.read_text())
    if not config["chat_template"].startswith(PINNED_DATE):
        config["chat_template"] = PINNED_DATE + config["chat_template"]
        path.write_text(json.dumps(config, indent=2, ensure_ascii=False))


def main() -> None:
    pin_template_date(HF_MODEL)
    if not MLX_Q4.exists():
        subprocess.run(
            [sys.executable, "-m", "mlx_lm", "convert", "--hf-path", str(HF_MODEL), "--mlx-path", str(MLX_Q4),
             "-q", "--q-bits", "4", "--q-group-size", "64", "--q-mode", "affine"],
            check=True,
        )
    pin_template_date(MLX_Q4)


if __name__ == "__main__":
    main()
