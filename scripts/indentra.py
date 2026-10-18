"""Indentra: structured debt-term extraction from corporate filings, on the command line (indentra.cli).

    .venv/bin/python -m scripts.indentra extract examples/works.txt
    .venv/bin/python -m scripts.indentra extract examples/mistake.txt --compare --json result.json
    .venv/bin/python -m scripts.indentra batch examples/batch.jsonl --output results.jsonl --compare
"""

import os
import sys

# Everything stays on this machine: no model hub lookups, no telemetry. Set before transformers or mlx_lm load.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")  # library notices only; does not change how the model runs

from indentra import cli  # noqa: E402

if __name__ == "__main__":
    sys.exit(cli.main())
