"""Indentra command line: extract debt terms from one paragraph (a file or standard input) or a JSON Lines batch with
B3, optionally beside the deterministic TF-IDF baseline on the identical text. Extraction is indentra.extraction,
unchanged: B3's identity checks, prompt, candidates, greedy decoding and strict parse.

Exit codes: 0 success; 1 a model could not load or failed; 2 invalid input (nothing was extracted); 3 a batch
finished but rejected one or more records (their errors are in the output file).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import TextIO

from indentra import extraction

EXIT_OK, EXIT_MODEL, EXIT_INPUT, EXIT_REJECTED = 0, 1, 2, 3
BLANK_LINE = re.compile(r"\n[ \t]*\n")
ENGINE_TITLES = {"b3": "B3, fine-tuned Llama 3.2 3B", "tfidf": "TF-IDF baseline"}
SHORT_NAMES = {"b3": "B3", "tfidf": "TF-IDF"}


class Failure(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def main(argv: list[str] | None = None, make_b3: Callable = extraction.B3Extractor, make_tfidf: Callable = extraction.TfidfExtractor,
         stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout, stderr: TextIO = sys.stderr) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "extract":
            return _extract(args, make_b3, make_tfidf, stdin, stdout, stderr)
        return _batch(args, make_b3, make_tfidf, stderr)
    except Failure as failure:
        print(f"error: {failure}", file=stderr)
        return failure.code


def check_input(text: str) -> str:
    """One paragraph, whitespace collapsed (extraction.check_paragraph); a blank line means several paragraphs."""
    if BLANK_LINE.search(text.strip()):
        raise extraction.InputError("The input contains more than one paragraph (separated by a blank line). Indentra extracts "
                                    "one paragraph at a time: put each paragraph in its own file, or use batch mode.")
    return extraction.check_paragraph(text)


def parse_batch(path: Path) -> list[tuple[str, str]]:
    """(id, text) pairs from a JSON Lines file of {"id": "...", "text": "..."}; every format problem is reported at once."""
    text = _read(path)
    records, problems, seen = [], [], set()
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except ValueError:
            problems.append(f"line {number}: not valid JSON")
            continue
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"].strip():
            problems.append(f'line {number}: needs a non-empty string "id"')
        elif not isinstance(item.get("text"), str):
            problems.append(f'line {number}: needs a string "text"')
        elif item["id"] in seen:
            problems.append(f'line {number}: duplicate id "{item["id"]}"')
        else:
            seen.add(item["id"])
            records.append((item["id"], item["text"]))
    if not records and not problems:
        problems.append("contains no records")
    if problems:
        raise Failure(EXIT_INPUT, f"{path} is not a valid batch file:\n  " + "\n  ".join(problems))
    return records


def _extract(args, make_b3, make_tfidf, stdin, stdout, stderr) -> int:
    raw = stdin.read() if args.input in (None, "-") else _read(Path(args.input))
    try:
        paragraph = check_input(raw)
    except extraction.InputError as error:
        raise Failure(EXIT_INPUT, str(error)) from None
    models = _load(make_b3, make_tfidf, args.compare, stderr)
    results = [_run(engine, model, paragraph) for engine, model in models.items()]
    print(_render(paragraph, results), file=stdout)
    if args.json:
        Path(args.json).write_text(json.dumps({"paragraph": paragraph, "results": results}, indent=2) + "\n")
        print(f"Saved JSON to {args.json}", file=stderr)
    return EXIT_OK


def _batch(args, make_b3, make_tfidf, stderr) -> int:
    source, target = Path(args.input), Path(args.output)
    if target.exists() and source.resolve() == target.resolve():
        raise Failure(EXIT_INPUT, "--output is the same file as the input; choose another output path.")
    records = parse_batch(source)
    models = _load(make_b3, make_tfidf, args.compare, stderr)
    counts = {"extracted": 0, "rejected": 0, "failed": 0}
    with target.open("w") as out:
        for number, (record_id, text) in enumerate(records, 1):
            line = _batch_record(record_id, text, models)
            out.write(json.dumps(line) + "\n")
            out.flush()
            status = "rejected" if line["error"] and line["error"]["kind"] == "input" else "failed" if line["error"] else "extracted"
            counts[status] += 1
            count = len(line["results"][0]["facts"]) if line["results"] else 0
            detail = line["error"]["message"] if line["error"] else f"{count} fact{'' if count == 1 else 's'} ({line['results'][0]['seconds']:.2f} s)"
            print(f"[{number}/{len(records)}] {record_id}: {status}: {detail}", file=stderr)
    summary = f"{len(records)} records: {counts['extracted']} extracted, {counts['rejected']} rejected"
    print(summary + (f", {counts['failed']} failed" if counts["failed"] else "") + f". Output: {target}", file=stderr)
    return EXIT_MODEL if counts["failed"] else EXIT_REJECTED if counts["rejected"] else EXIT_OK


def _batch_record(record_id: str, text: str, models: dict) -> dict:
    try:
        paragraph = check_input(text)
    except extraction.InputError as error:
        return {"id": record_id, "paragraph": None, "results": [], "error": {"kind": "input", "message": str(error)}}
    try:
        results = [_run(engine, model, paragraph) for engine, model in models.items()]
    except Failure as failure:
        return {"id": record_id, "paragraph": paragraph, "results": [], "error": {"kind": "model", "message": str(failure)}}
    return {"id": record_id, "paragraph": paragraph, "results": results, "error": None}


def _load(make_b3, make_tfidf, compare: bool, stderr) -> dict:
    models = {"b3": make_b3()}
    if compare:
        models["tfidf"] = make_tfidf()
    for engine, model in models.items():
        print(f"Loading {SHORT_NAMES[engine]}...", file=stderr, flush=True)
        try:
            model.load()
        except extraction.LoadError as error:
            raise Failure(EXIT_MODEL, str(error)) from None
        print(f"{SHORT_NAMES[engine]} ready in {model.load_seconds} s", file=stderr, flush=True)
    return models


def _run(engine: str, model, paragraph: str) -> dict:
    try:
        result = model.extract(paragraph)
    except Exception as error:  # reported, never replaced by other output
        raise Failure(EXIT_MODEL, f"{SHORT_NAMES[engine]} failed on this paragraph: {error}") from None
    return {**asdict(result), "model": model.identity}


def _render(paragraph: str, results: list[dict]) -> str:
    lines = [f"Paragraph: {len(paragraph):,} characters"]
    for result in results:
        lines += ["", _title(result)]
        if not result["valid"]:
            lines.append(f"  {result['note']}")
        elif not result["facts"]:
            lines.append("  No supported debt facts found." + (f" {result['note']}" if result["note"] else ""))
        for fact in result["facts"]:
            sources = ", ".join(f'"{span["text"]}"' for span in fact["spans"])
            lines.append(f"  {fact['label']:<24} {fact['display']:<18} from {sources}")
            if fact["other_occurrences"]:
                times = "1 more time" if fact["other_occurrences"] == 1 else f"{fact['other_occurrences']} more times"
                lines.append(f"  {'':<24} (the same figure appears {times}; shown where {SHORT_NAMES[result['engine']]} labeled it)")
        if result["truncated"]:
            lines.append("  Note: the answer reached the 256-token limit.")
    if len(results) == 2:
        lines += ["", _comparison(*results)]
    return "\n".join(lines)


def _title(result: dict) -> str:
    seconds = result["seconds"]
    timing = f"{seconds * 1000:.1f} ms" if seconds < 0.1 else f"{seconds:.2f} s"
    model = result["model"]
    digest = f", adapter digest {model['adapter_digest']}" if model.get("adapter_digest") else ""
    return f"{ENGINE_TITLES[result['engine']]} ({timing}{digest})"


def _comparison(b3: dict, tfidf: dict) -> str:
    def facts(result):
        return {(f["field"], f["value"]): f"{f['label']} {f['display']}" for f in result["facts"]}
    a, b = facts(b3), facts(tfidf)
    parts = [f"Both: {', '.join(a[k] for k in a if k in b) or 'none'}",
             f"Only B3: {', '.join(a[k] for k in a if k not in b) or 'none'}",
             f"Only TF-IDF: {', '.join(b[k] for k in b if k not in a) or 'none'}"]
    return "Comparison on this paragraph. " + ". ".join(parts) + "."


def _read(path: Path) -> str:
    try:
        return path.read_bytes().decode("utf-8")
    except OSError as error:
        raise Failure(EXIT_INPUT, f"cannot read {path}: {error.strerror}") from None
    except UnicodeDecodeError:
        raise Failure(EXIT_INPUT, f"cannot read {path} as UTF-8 text; save it as UTF-8 and try again.") from None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="indentra", description="Structured debt-term extraction from corporate filings (B3).")
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("extract", help="extract one paragraph from a text file, or from standard input")
    one.add_argument("input", nargs="?", help="UTF-8 text file holding one paragraph; omit or use - for standard input")
    one.add_argument("--compare", action="store_true", help="also run the TF-IDF baseline on the identical paragraph")
    one.add_argument("--json", metavar="PATH", help="also save the structured result as JSON")
    many = commands.add_parser("batch", help='extract every record of a JSON Lines file of {"id": ..., "text": ...}')
    many.add_argument("input", help="JSON Lines input file")
    many.add_argument("--output", required=True, metavar="PATH", help="JSON Lines output file, one result per record")
    many.add_argument("--compare", action="store_true", help="also run the TF-IDF baseline on every record")
    return parser
