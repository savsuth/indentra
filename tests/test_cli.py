import io
import json

import pytest

from indentra import cli, extraction

TEXT = "The Borrower entered into a $300 million term loan that bears interest at Term SOFR plus 175 basis points."


class FakeModel:
    """Stands in for B3 or TF-IDF: labels the first two regex candidates, counts loads and calls."""

    def __init__(self, engine, labels=("face_amount", "spread"), fail_load=None, fail_extract=False):
        self.engine, self.labels, self.fail_load, self.fail_extract = engine, labels, fail_load, fail_extract
        self.loads, self.texts = 0, []
        self.identity = {"name": engine.upper(), "description": f"fake {engine}"}
        self.load_seconds = 0.1

    def load(self):
        self.loads += 1
        if self.fail_load:
            raise extraction.LoadError(self.fail_load)

    def extract(self, text):
        self.texts.append(text)
        if self.fail_extract:
            raise RuntimeError("device lost")
        facts = extraction.group_facts(text, list(enumerate(self.labels)))
        return extraction.Result(self.engine, text, facts, 0.5, raw='{"facts": [[1, "face_amount"], [2, "spread"]]}')


@pytest.fixture
def models():
    return {"b3": FakeModel("b3"), "tfidf": FakeModel("tfidf", labels=("face_amount",))}


def run(argv, models, stdin=""):
    out, err = io.StringIO(), io.StringIO()
    code = cli.main(argv, make_b3=lambda: models["b3"], make_tfidf=lambda: models["tfidf"], stdin=io.StringIO(stdin), stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def test_extract_reads_a_file_and_prints_readable_facts(tmp_path, models):
    path = tmp_path / "p.txt"
    path.write_text(TEXT)
    code, out, err = run(["extract", str(path)], models)
    assert code == cli.EXIT_OK
    assert "Principal amount" in out and "$300,000,000" in out and '"$300 million"' in out
    assert "Interest margin" in out and "1.75%" in out
    assert models["b3"].loads == 1 and models["tfidf"].loads == 0


def test_extract_reads_standard_input_when_no_file_is_given(models):
    code, out, _ = run(["extract"], models, stdin=TEXT + "\n")
    assert code == cli.EXIT_OK and "$300,000,000" in out and models["b3"].texts == [TEXT]


def test_json_option_saves_the_structured_result(tmp_path, models):
    path, out_path = tmp_path / "p.txt", tmp_path / "out.json"
    path.write_text(TEXT)
    assert run(["extract", str(path), "--json", str(out_path)], models)[0] == cli.EXIT_OK
    saved = json.loads(out_path.read_text())
    assert saved["paragraph"] == TEXT and [r["engine"] for r in saved["results"]] == ["b3"]
    fact = saved["results"][0]["facts"][0]
    assert (fact["field"], fact["value"], fact["display"]) == ("face_amount", "300000000", "$300,000,000")
    assert TEXT[fact["spans"][0]["start"]:fact["spans"][0]["end"]] == "$300 million"
    assert saved["results"][0]["model"]["name"] == "B3"


def test_compare_runs_the_baseline_on_the_identical_paragraph(tmp_path, models):
    path = tmp_path / "p.txt"
    path.write_text("  " + TEXT.replace(" ", "\n", 3) + "  ")
    code, out, _ = run(["extract", str(path), "--compare"], models)
    assert code == cli.EXIT_OK and models["b3"].texts == models["tfidf"].texts == [TEXT]
    assert "TF-IDF" in out and "Only B3: Interest margin 1.75%" in out


@pytest.mark.parametrize("content, message", [
    ("   \n  ", "Paste a paragraph"),
    ("x" * 3001, "3,001 characters"),
    ("First paragraph with $5 million.\n\nSecond paragraph with 5%.", "more than one paragraph"),
], ids=["empty", "too-long", "two-paragraphs"])
def test_invalid_input_exits_2_without_loading_any_model(tmp_path, models, content, message):
    path = tmp_path / "p.txt"
    path.write_text(content)
    code, out, err = run(["extract", str(path)], models)
    assert code == cli.EXIT_INPUT and message in err and out == ""
    assert models["b3"].loads == 0


def test_unreadable_input_file_exits_2(tmp_path, models):
    code, _, err = run(["extract", str(tmp_path / "missing.txt")], models)
    assert code == cli.EXIT_INPUT and "cannot read" in err
    bad = tmp_path / "latin1.txt"
    bad.write_bytes("caf\xe9 debt of $5 million".encode("latin-1"))
    code, _, err = run(["extract", str(bad)], models)
    assert code == cli.EXIT_INPUT and "UTF-8" in err


def test_missing_model_files_exit_1_with_the_loaders_message(tmp_path, models):
    path = tmp_path / "p.txt"
    path.write_text(TEXT)
    models["b3"] = FakeModel("b3", fail_load="B3 cannot load; missing: adapters/full-neg33-cand/adapters.safetensors")
    code, out, err = run(["extract", str(path)], models)
    assert code == cli.EXIT_MODEL and "adapters.safetensors" in err and out == ""


def test_extraction_failure_exits_1_and_writes_no_json(tmp_path, models):
    path, out_path = tmp_path / "p.txt", tmp_path / "out.json"
    path.write_text(TEXT)
    models["b3"] = FakeModel("b3", fail_extract=True)
    code, _, err = run(["extract", str(path), "--json", str(out_path)], models)
    assert code == cli.EXIT_MODEL and "device lost" in err and not out_path.exists()


def write_batch(tmp_path, records):
    path = tmp_path / "in.jsonl"
    path.write_text("".join((r if isinstance(r, str) else json.dumps(r)) + "\n" for r in records))
    return path


def test_batch_loads_the_model_once_and_writes_one_line_per_record(tmp_path, models):
    path = write_batch(tmp_path, [{"id": "a", "text": TEXT}, "", {"id": "b", "text": TEXT.replace("300", "450")}])
    out_path = tmp_path / "out.jsonl"
    code, _, err = run(["batch", str(path), "--output", str(out_path), "--compare"], models)
    lines = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert code == cli.EXIT_OK and models["b3"].loads == 1 and len(models["b3"].texts) == 2
    assert [line["id"] for line in lines] == ["a", "b"] and all(line["error"] is None for line in lines)
    assert [r["engine"] for r in lines[1]["results"]] == ["b3", "tfidf"] and "$450,000,000" in json.dumps(lines[1])
    assert "2 records: 2 extracted" in err


def test_batch_rejects_bad_records_individually_and_exits_3(tmp_path, models):
    path = write_batch(tmp_path, [{"id": "ok", "text": TEXT}, {"id": "empty", "text": "  "}, {"id": "long", "text": "x" * 3001}])
    out_path = tmp_path / "out.jsonl"
    code, _, err = run(["batch", str(path), "--output", str(out_path)], models)
    lines = {line["id"]: line for line in map(json.loads, out_path.read_text().splitlines())}
    assert code == cli.EXIT_REJECTED and len(models["b3"].texts) == 1
    assert lines["ok"]["error"] is None and lines["empty"]["error"]["kind"] == "input" and "3,001" in lines["long"]["error"]["message"]
    assert "1 extracted, 2 rejected" in err


@pytest.mark.parametrize("records, message", [
    (["{not json"], "line 1: not valid JSON"),
    ([{"text": TEXT}], "line 1: needs a non-empty string \"id\""),
    ([{"id": "a", "text": 5}], "line 1: needs a string \"text\""),
    ([{"id": "a", "text": TEXT}, {"id": "a", "text": TEXT}], "line 2: duplicate id \"a\""),
    ([], "contains no records"),
], ids=["bad-json", "missing-id", "non-string-text", "duplicate-id", "empty-file"])
def test_malformed_batch_file_exits_2_before_loading_the_model(tmp_path, models, records, message):
    path, out_path = write_batch(tmp_path, records), tmp_path / "out.jsonl"
    code, _, err = run(["batch", str(path), "--output", str(out_path)], models)
    assert code == cli.EXIT_INPUT and message in err and models["b3"].loads == 0 and not out_path.exists()


def test_batch_refuses_to_overwrite_its_input(tmp_path, models):
    path = write_batch(tmp_path, [{"id": "a", "text": TEXT}])
    code, _, err = run(["batch", str(path), "--output", str(path)], models)
    assert code == cli.EXIT_INPUT and "same file" in err
