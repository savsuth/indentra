# Indentra

Indentra extracts debt terms from corporate filings using a fine-tuned Llama 3.2 model. It runs locally and returns structured results with references to the source text.

Indentra reads one paragraph of filing text at a time. It reports the debt terms that the paragraph states, in six fields: principal amount, fixed interest rate, effective interest rate, interest margin, borrowing limit and commitment fee. Every reported value is a figure that appears in the paragraph. It is returned with its exact normalized value, the quoted source text and that text's character position. Results are printed in readable form or saved as JSON.

Built with Llama.

## Overview

**Purpose.** US public companies describe new borrowing in prose. A common source is Form 8-K, the current report a company files after a significant event; its Item 2.03 covers new direct financial obligations such as note offerings and credit agreements. Unlike the figures in annual reports, the debt terms in these paragraphs carry no XBRL tags, the machine-readable labels that companies attach to reported numbers. Indentra converts such a paragraph into structured fields.

**Input.** One paragraph of UTF-8 text, from a file or standard input, or many paragraphs in a JSON Lines batch file. A paragraph may have at most 3,000 characters, the maximum paragraph length admitted to the evaluation data. Runs of whitespace are collapsed to single spaces. Empty text, longer text and text that contains a blank line (more than one paragraph) are rejected with an explanation; nothing is truncated or merged.

**Output.** Each extracted fact has one of these fields:

| Field | Label in the output | Meaning |
|---|---|---|
| `face_amount` | Principal amount | The face or par amount of a specific note, bond or loan. |
| `stated_rate` | Fixed interest rate | The stated (coupon) rate of a debt instrument. |
| `effective_rate` | Effective interest rate | An interest rate the paragraph calls effective. |
| `spread` | Interest margin | The margin added to, or subtracted from, a reference rate such as SOFR (the Secured Overnight Financing Rate) or the prime rate. |
| `facility_size` | Borrowing limit | The most that can be borrowed under a credit facility or commitment. |
| `commitment_fee` | Commitment fee | The fee rate charged on the unused part of a credit facility. |

**Models.** Extraction uses Llama 3.2 3B Instruct with a fine-tuned LoRA adapter, a small set of additional trained weights ([How it works](#how-it-works)). Indentra's output identifies this fine-tuned model as **B3**, its experiment identifier. For comparison, Indentra can also run a conventional baseline on the same paragraph: a logistic-regression classifier over TF-IDF features, which are word counts weighted by how distinctive each word is.

## Example

Input: [examples/works.txt](examples/works.txt), a paragraph from an 8-K that Gladstone Investment filed on 2026-02-18.

> On February 18, 2026, in connection with a previously announced public offering, Gladstone Investment Corporation (the “Company”) and UMB Bank, National Association, as trustee (the “Trustee”), entered into a Seventh Supplemental Indenture (the “Seventh Supplemental Indenture”) to the Indenture, dated May 22, 2020, between the Company and the Trustee (together with the Seventh Supplemental Indenture, the “Indenture”). The Seventh Supplemental Indenture relates to Company’s issuance, offer and sale of up to $115.0 million aggregate principal amount of its 7.125% Notes due 2031 (the “Notes”).

Command:

```bash
.venv/bin/python -m scripts.indentra extract examples/works.txt \
  --json examples/output/works.json
```

Recorded output, from [examples/output/works.terminal.txt](examples/output/works.terminal.txt). Loading messages are written to standard error and are not part of this file.

```text
Paragraph: 597 characters

B3, fine-tuned Llama 3.2 3B (1.20 s, adapter digest 5d33f539ad3b394c)
  Principal amount         $115,000,000       from "$115.0 million"
  Fixed interest rate      7.125%             from "7.125%"
```

The first fact in the saved JSON file, [examples/output/works.json](examples/output/works.json):

```json
{
  "field": "face_amount",
  "value": "115000000",
  "label": "Principal amount",
  "display": "$115,000,000",
  "spans": [
    {
      "start": 512,
      "end": 526,
      "text": "$115.0 million"
    }
  ],
  "other_occurrences": 0
}
```

Errors are recorded as well. On [examples/mistake.txt](examples/mistake.txt), the fine-tuned model finds one of five expected facts and labels an interest margin as a commitment fee ([examples/output/mistake.terminal.txt](examples/output/mistake.terminal.txt)). The expected answers for all examples are in [examples/expected.json](examples/expected.json).

## Setup

### Supported environment

- **macOS on Apple silicon.** The fine-tuned model runs on MLX, Apple's machine-learning framework, which requires Apple silicon. Every Indentra command loads this model, including runs with `--compare`.
- **Python 3.13.** All recorded results used Python 3.13.

On other platforms, `pip install -e ".[dev]"` installs the library, tests and linter without MLX, but extraction is not available.

### Dependencies

From the repository root:

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.lock -e ".[mlx,dev]"
```

`requirements.lock` pins every dependency to the version used for the recorded results. The Python package is `indentra`, in `src/indentra/`.

### Model artifacts

Model weights are excluded from Git. Indentra expects these files at the paths shown, relative to the repository root:

| Artifact | Path | Availability |
|---|---|---|
| Base model (Llama 3.2 3B Instruct, 4-bit MLX) | `models/Llama-3.2-3B-Instruct-mlx-q4/` | Created locally with the commands below. |
| Fine-tuned adapter (B3), 56 MB | `adapters/full-neg33-cand/adapters.safetensors` | Not distributed. |
| TF-IDF baseline, for `--compare` only | `artifacts/tfidf-lr-det/model.pkl` | Not distributed. |

Because the adapter is not distributed, a clone of this repository cannot run extraction on its own. The other files in the adapter folder are included. The TF-IDF file cannot be regenerated byte for byte: scikit-learn stores a run-specific memory address in the pickled file, so only the original file matches the hash recorded in `artifacts/tfidf-lr-det/manifest.json`. Folder names such as `full-neg33-cand` and `tfidf-lr-det` are the project's internal experiment identifiers, which the code and the recorded metadata refer to.

The base model is created from `unsloth/Llama-3.2-3B-Instruct`, a public, ungated copy of Meta's Llama 3.2 3B Instruct ([NOTICE](NOTICE) describes its provenance):

```bash
# Download the base weights
.venv/bin/hf download unsloth/Llama-3.2-3B-Instruct \
  --include "*.safetensors" --include "*.json" \
  --local-dir models/Llama-3.2-3B-Instruct

# Fix the chat template's date and write the 4-bit MLX copy
.venv/bin/python -m scripts.prepare_model
```

Indentra refuses to use a base model, adapter or TF-IDF file that differs from the evaluated one (see "Identity checks" under [How it works](#how-it-works)). Whether a new download and conversion reproduce the evaluated base model exactly has not been tested.

## Usage

Run all commands from the repository root; Indentra locates its model files relative to it.

### Extract one paragraph

```bash
# From a text file
.venv/bin/python -m scripts.indentra extract examples/works.txt

# From standard input (omit the file name, or pass -)
cat examples/works.txt | .venv/bin/python -m scripts.indentra extract
```

Results go to standard output. Loading and progress messages, including the model-loading time, go to standard error. Each result also reports the time taken for its paragraph.

### Save structured output

```bash
.venv/bin/python -m scripts.indentra extract examples/works.txt --json result.json
```

The JSON file contains `paragraph`, the input text after whitespace normalization, and `results`, with one entry per model. Each entry contains:

- `facts`: each with `field`, `label`, the exact `value`, a formatted `display`, the source `spans` (character `start` and `end`, and the quoted `text`) and `other_occurrences` (how many other occurrences of the same figure in the paragraph were not given this label);
- `seconds`: the extraction time;
- `valid`: whether the answer followed the required format;
- `raw`: the fine-tuned model's unprocessed answer;
- `model`: the model's identity, including the digests checked at load time.

### Compare with the TF-IDF baseline

```bash
.venv/bin/python -m scripts.indentra extract examples/mistake.txt \
  --compare --json result.json
```

`--compare` also runs the TF-IDF baseline on the identical paragraph. The readable output lists each model's facts, then the facts found by both models and the facts found by only one of them.

### Process a batch

```bash
.venv/bin/python -m scripts.indentra batch examples/batch.jsonl \
  --output results.jsonl --compare
```

Batch mode processes each record individually while keeping the model loaded between records. The input is a JSON Lines file, with one JSON object per line. Each line needs a unique, non-empty string `id` and a string `text`; blank lines are skipped. A malformed file is rejected before any model loads, and every invalid line is listed. A record that breaks the input rules is reported in the output and does not stop the others.

Each line of the output file holds one record: `id`, `paragraph`, `results` (as with `--json`) and `error`, which is `null` when the record was extracted.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Success. |
| 1 | A model could not be loaded (missing or mismatched files), or extraction failed on a paragraph. |
| 2 | Invalid input; nothing was extracted. |
| 3 | A batch finished, but some records were rejected; their errors are in the output file. |

For all options, run `.venv/bin/python -m scripts.indentra extract --help` or `.venv/bin/python -m scripts.indentra batch --help`.

## How it works

1. **Candidate detection.** Regular expressions find every dollar amount, percentage and basis-point figure in the paragraph, and code converts each to an exact value. For example, "$115.0 million" becomes 115000000 and "175 basis points" (1.75 percentage points) becomes 1.75.
2. **Model classification.** The fine-tuned model receives the paragraph and its numbered candidates. It answers in JSON, naming the candidates that are debt terms and the field of each, for example `{"facts": [[1, "face_amount"], [2, "stated_rate"]]}` (`src/indentra/hybrid.py`). The model never writes a number itself, so it cannot report a figure that is not in the text.
3. **Structured output.** The answer is parsed strictly; an answer that does not follow the required format yields no facts and is marked invalid. Labeled candidates with the same field and value become one fact that lists the location of each labeled occurrence.

The model can label only the figures that candidate detection finds. In the training data, the candidates covered 91.7% of the labeled facts. Negative margins, such as "prime minus 0.50%", cannot be represented.

**Fine-tuned model.** The base model is Llama 3.2 3B Instruct, quantized to 4 bits for MLX. Fine-tuning used LoRA (low-rank adaptation), which trains a small set of added weights while the base model stays fixed; training such an adapter on a quantized base model is known as QLoRA. The adapter was trained on 7,608 debt paragraphs from 10-K annual reports filed from 2022 to 2025, labeled automatically from the reports' XBRL tags. Decoding is greedy: the model always takes its most likely next token, for at most 256 new tokens. The model gives no confidence scores.

**TF-IDF baseline.** The baseline labels the same candidates. A logistic-regression classifier assigns each candidate a field, or none, from the words around it. Those words are represented as TF-IDF features: counts of words and word pairs, weighted by how distinctive each term is across the training data. Indentra uses a deterministic version of this baseline (`artifacts/tfidf-lr-det`), which yields the same model each time it is trained.

**Identity checks.** Before loading, Indentra computes a digest (a shortened SHA-256 fingerprint) of the base-model folder and of the adapter folder. It refuses to load unless both equal the digests recorded by the fine-tuned model's evaluation run, in [runs/ft-full-neg33-cand-mlx-eightk_fresh/config.json](runs/ft-full-neg33-cand-mlx-eightk_fresh/config.json). The adapter digest covers every file in the adapter folder, including its training log, so a retrained adapter is refused rather than presented as the evaluated model. The TF-IDF file is checked against the hash and digest in `artifacts/tfidf-lr-det/manifest.json` before it is used.

**One paragraph at a time.** Indentra processes paragraphs one at a time in both `extract` and `batch`, so a paragraph receives the same answer in either mode. The evaluation ran the model on 16 paragraphs at a time. On 25 stored test paragraphs, Indentra reproduced the evaluation's answer exactly for 24. Processing several paragraphs together changes the floating-point arithmetic slightly and can change a closely balanced answer, which is the likely cause of the one difference.

**Local operation.** Indentra reads its models from disk. Its launcher switches the Hugging Face libraries to offline mode and disables their telemetry, so extraction does not send text over the network.

## Evaluation

The datasets, predictions and run records behind these results are not included in this repository.

### Test set and metric

The test set contains 450 paragraphs from Item 2.03 8-K filings dated 2025-01-01 to 2026-10-14. The paragraphs come from 201 companies that appear in neither the training data nor the development data. Their expected facts were labeled and frozen before any model ran on them. The model to deploy and the rule for declaring a difference were fixed in a written protocol before the test, which makes this a pre-registered test.

Results are reported as F1. A predicted fact counts as correct only when both its field and its value match an expected fact. Precision is the share of predicted facts that are correct, recall is the share of expected facts that are predicted, and F1 is their harmonic mean. The 95% confidence intervals (CI) come from resampling companies, so that paragraphs from the same company are treated together.

### Results

On 450 held-out 8-K paragraphs, the fine-tuned model achieved an F1 score of 0.767, compared with 0.739 for TF-IDF with logistic regression. The estimated difference was 0.028 (95% CI: −0.006 to 0.066); the experiment did not establish an improvement over the baseline.

All fine-tuned systems in this table are Llama 3.2 3B. "Writes facts" means the model writes each fact itself; "labels candidates" means it labels the figures found by candidate detection, as Indentra does.

| System | F1 | 95% CI | Precision | Recall |
|---|---|---|---|---|
| Rules (regular expressions and keywords) | 0.680 | 0.637 to 0.720 | 0.585 | 0.813 |
| TF-IDF with logistic regression | 0.739 | 0.702 to 0.775 | 0.824 | 0.670 |
| Fine-tuned, 4,000 paragraphs, writes facts | 0.699 | 0.652 to 0.742 | 0.769 | 0.640 |
| Fine-tuned, 7,801 paragraphs, writes facts | 0.763 | 0.728 to 0.797 | 0.750 | 0.776 |
| Fine-tuned, 7,608 paragraphs, labels candidates (B3, deployed) | 0.767 | 0.727 to 0.802 | 0.816 | 0.725 |

Interpretation:

- The confidence interval for the difference includes zero, so the result does not show that the fine-tuned model is better than the baseline. It also does not show that the two are equivalent.
- On every field present in the test set, the fine-tuned model's measured F1 is higher than the baseline's, mainly through higher recall at similar precision.
- The baseline is much cheaper to run: it needs no GPU and was more than a thousand times faster per paragraph in the evaluation runs.
- A test able to detect a difference of this size with 80% probability would need about 700 companies, roughly 1,550 paragraphs.
- Fine-tuning made a large difference for the 3B model. On a separate 120-paragraph development set, a model fine-tuned on 2,000 paragraphs scored 0.752, against 0.383 for the same base model prompted with three examples, a difference of 0.369 (95% CI: 0.194 to 0.508).

### Exploratory comparisons

These comparisons were made after the results above were known. They are reported for completeness and do not replace the pre-registered result.

Each compares the fine-tuned model with the deterministic TF-IDF baseline that `--compare` uses, which scores 0.735 (95% CI: 0.697 to 0.773) on the test set. A positive difference favors the fine-tuned model. The corrected labels are a second label version that settled 20 open labeling cases after the results had been seen.

| Data set and labels | Difference in F1 (95% CI) |
|---|---|
| Test set, original labels | 0.032 (95% CI: −0.005 to 0.071) |
| Test set, corrected labels | 0.038 (95% CI: 0.003 to 0.078) |
| Development set (120 paragraphs), corrected labels | −0.017 (95% CI: −0.095 to 0.045) |

### Labels and human review

- **Training labels** come from companies' XBRL tags, without manual labeling. An AI-authored label audit of 100 test paragraphs, drafted by Claude, estimated the tags' precision at 0.841 and completeness at 0.777. The audit has not been reviewed by a person.
- **Test labels** were drafted by Claude following written labeling rules and checked by a second AI system, Codex, which agreed on 447 of the 450 paragraphs. The project owner decided the 3 disagreements and approved the rules. No person reviewed every label.
- **Development labels** for the 120-paragraph set were drafted by Claude and have not been reviewed by a person.

### Software checks and extraction accuracy

The automated test suite (`.venv/bin/python -m pytest -q`, 130 tests) checks software behavior: input validation, parsing, exit codes and model identity checks. It uses stand-ins where a model would be needed and does not measure extraction accuracy.

The recorded examples distinguish software defects from model errors. A software defect is output the tool must never produce, such as a quoted source span that is not in the paragraph or a value that differs from its candidate. A model error is valid output that differs from the expected facts. The expected answers in [examples/expected.json](examples/expected.json) were written by AI (Claude, some also checked by Codex) and have not been reviewed by a person.

The examples contain five valid paragraphs and three invalid inputs ([examples/batch.jsonl](examples/batch.jsonl), with recorded output in [examples/output/batch.jsonl](examples/output/batch.jsonl)). The fine-tuned model matched the expected facts on two of the valid paragraphs and made errors on three. All three invalid inputs were rejected as expected. No software defects were found.

## Repository layout

- `src/indentra/`: the extraction library: candidate detection, model loading and identity checks, output formatting and the command-line interface.
- `scripts/indentra.py`: the command-line entry point. `scripts/prepare_model.py`: builds the 4-bit MLX base model.
- `tests/`: unit tests; models are replaced by stand-ins.
- `examples/`: example inputs, expected answers and recorded outputs.
- `adapters/full-neg33-cand/`, `runs/ft-full-neg33-cand-mlx-eightk_fresh/config.json` and `artifacts/tfidf-lr-det/manifest.json`: metadata for the model identity and integrity checks. The weights themselves are not included.

## License and attribution

- **Code.** MIT License; see [LICENSE](LICENSE).
- **Model.** The fine-tuned adapter is a derivative of Llama 3.2 and is subject to the Llama 3.2 Community License and its Acceptable Use Policy. Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright (c) Meta Platforms, Inc. All Rights Reserved. Any distributed copy of the model must include the license, and its name must begin with "Llama". See [NOTICE](NOTICE).
- **Data.** Filings come from SEC EDGAR, whose information is public. This project is not affiliated with or endorsed by the SEC.

Indentra reports what a filing states. It is not financial advice.
