"""Protocol tests for the CPU encoder worker, without model weights."""
import io
import json
import os
import queue
import sys
import threading

import pytest

from paper_evidence_rag.embeddings import DOCUMENT_PROMPT, MODEL_ID, MODEL_REVISION, QUERY_PROMPT
from paper_evidence_rag.isolated_encoder import (
    IsolatedLocalEncoder,
    MAX_RESPONSE_CHARS,
    MAX_TEXTS,
    _STOP,
    _serve_worker,
    _validate_vectors,
)


def metadata():
    return {"id": MODEL_ID, "revision": MODEL_REVISION, "dimension": 768,
        "query_prompt": QUERY_PROMPT, "document_prompt": DOCUMENT_PROMPT,
        "dtype": "float32", "normalize_embeddings": True, "max_length": 512,
        "file_sha256": {"model.safetensors": "a" * 64}}


class FakeStdout:
    def __init__(self):
        self.lines = queue.Queue()
        self.closed = False

    def readline(self, limit):
        line = self.lines.get(timeout=5)
        if line and len(line) > limit:
            return line[:limit]
        return line

    def close(self):
        self.closed = True


class FakeProcess:
    def __init__(self):
        self.stdout = FakeStdout()
        self.stdin = FakeStdin(self)
        self.returncode = None
        self.killed = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def kill(self):
        self.killed = True
        self.returncode = -9
        self.stdout.lines.put("")


class FakeStdin:
    def __init__(self, process):
        self.process = process
        self.pending = ""
        self.closed = False

    def write(self, value):
        self.pending += value
        return len(value)

    def flush(self):
        request, self.pending = json.loads(self.pending), ""
        if request["op"] == "close":
            self.process.stdout.lines.put("")
            return
        rows = len(request["texts"])
        vector = [0.25] * 768
        response = {"ok": True, "vectors": [vector for _ in range(rows)]}
        if request["op"] == "split_documents":
            response = {"ok": True, "splits": [[{"text": text, "start": 0, "end": len(text)}]
                for text in request["texts"]]}
        self.process.stdout.lines.put(json.dumps(response) + "\n")

    def close(self):
        self.closed = True


def test_encoder_keeps_persistent_child_and_does_not_pass_secrets(tmp_path, monkeypatch):
    calls = []

    def fake_popen(args, **kwargs):
        process = FakeProcess()
        calls.append((args, kwargs, process))
        return process

    monkeypatch.setenv("OPENROUTER_API_KEY", "must-not-cross-process-boundary")
    torch_was_loaded = "torch" in sys.modules
    encoder = IsolatedLocalEncoder(tmp_path, metadata(), popen=fake_popen)
    try:
        assert ("torch" in sys.modules) == torch_was_loaded
        assert len(encoder.encode_question("What did they measure?")) == 768
        assert len(encoder.encode_documents(["passage one", "passage two"])) == 2
        assert encoder.split_documents(["all text"])[0] == [{"text": "all text", "start": 0, "end": 8}]
        assert len(calls) == 1
        args, options, _ = calls[0]
        assert args[1:4] == ["-m", "paper_evidence_rag.isolated_encoder", "--worker"]
        assert options["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert options["env"]["HF_HUB_OFFLINE"] == "1"
        assert "OPENROUTER_API_KEY" not in options["env"]
        assert options["stderr"] == __import__("subprocess").DEVNULL
    finally:
        encoder.close()


def test_worker_protocol_uses_existing_encoder_contract_and_hides_stdout(monkeypatch):
    initial = os.environ.get("CUDA_VISIBLE_DEVICES")
    calls = []

    class FakeEncoder:
        def __init__(self, directory, model_metadata):
            calls.append((directory, model_metadata))

        def encode_question(self, text):
            print("worker library output is not protocol")
            return [0.5] * 768

        def encode_documents(self, texts):
            return [[0.75] * 768 for _ in texts]
        def split_documents(self, texts):
            return [[{"text": text, "start": 0, "end": len(text)}] for text in texts]

    request = {"op": "question", "directory": "/offline/model", "metadata": metadata(),
        "texts": ["question"]}
    lines = io.StringIO(json.dumps(request) + "\n" + json.dumps({**request, "op": "documents",
        "texts": ["first", "second"]}) + "\n" + json.dumps({**request, "op": "split_documents",
        "texts": ["complete text"]}) + "\n")
    output = io.StringIO()
    try:
        _serve_worker(lines, output, encoder_factory=FakeEncoder)
        responses = [json.loads(line) for line in output.getvalue().splitlines()]
        assert [len(item.get("vectors", item.get("splits"))) for item in responses] == [1, 2, 1]
        assert responses[0]["vectors"][0][0] == 0.5
        assert responses[1]["vectors"][1][0] == 0.75
        assert responses[2]["splits"][0][0] == {"text": "complete text", "start": 0, "end": 13}
        assert len(calls) == 1
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    finally:
        if initial is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = initial


def test_encoder_rejects_bad_vectors_and_oversized_input():
    with pytest.raises(RuntimeError, match="invalid vectors"):
        _validate_vectors([[float("nan")] * 768], 1)
    encoder = IsolatedLocalEncoder(".", metadata(), popen=lambda *args, **kwargs: pytest.fail("started worker"))
    try:
        with pytest.raises(ValueError, match="invalid number"):
            encoder.encode_documents(["x"] * (MAX_TEXTS + 1))
        with pytest.raises(ValueError, match="invalid text"):
            encoder.encode_question("x" * 10_001)
    finally:
        encoder.close()


def test_tokenizer_splits_without_dropping_characters_or_exceeding_budget():
    from types import SimpleNamespace
    from paper_evidence_rag.app_models import LocalEncoder

    class Tokenizer:
        def __call__(self, text, truncation=False):
            return {"input_ids": text.split()}

    encoder = LocalEncoder.__new__(LocalEncoder)
    budget = len(DOCUMENT_PROMPT.split()) + 8
    encoder.model = SimpleNamespace(tokenizer=Tokenizer(), max_seq_length=budget)
    text = "alpha beta gamma delta. epsilon zeta eta theta. iota kappa lambda mu."
    pieces = encoder.split_documents([text])[0]
    assert "".join(piece["text"] for piece in pieces) == text
    assert pieces[0]["start"] == 0 and pieces[-1]["end"] == len(text)
    assert all(len((DOCUMENT_PROMPT + piece["text"]).split()) <= budget for piece in pieces)
    assert all(left["end"] == right["start"] for left, right in zip(pieces, pieces[1:]))


def test_worker_split_validation_rejects_missing_or_fabricated_text():
    from paper_evidence_rag.isolated_encoder import _validate_splits
    assert _validate_splits([[{"text": "abc", "start": 0, "end": 3}]], ["abc"])
    with pytest.raises(RuntimeError, match="document splits"):
        _validate_splits([[{"text": "ab", "start": 0, "end": 2}]], ["abc"])


@pytest.mark.parametrize("text", ["equation123=" * 100, "数式λ→∑。" * 100, "A sentence. " * 100])
def test_token_heavy_inputs_preserve_all_characters_and_prompt_overhead(text):
    from types import SimpleNamespace
    from paper_evidence_rag.app_models import LocalEncoder

    class ByteTokenizer:
        def __call__(self, value, **kwargs):
            assert kwargs["truncation"] is False
            return {"input_ids": [0] * (len(value.encode("utf-8")) + 2)}

    encoder = LocalEncoder.__new__(LocalEncoder)
    encoder.model = SimpleNamespace(tokenizer=ByteTokenizer(), max_seq_length=64)
    pieces = encoder.split_documents([text])[0]
    assert len(pieces) > 1 and "".join(p["text"] for p in pieces) == text
    assert all(len((DOCUMENT_PROMPT + p["text"]).encode("utf-8")) + 2 <= 64 for p in pieces)


def test_worker_rejects_bad_requests_without_loading_encoder(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "visible-before-test")
    made = []

    def factory(*args):
        made.append(args)
        raise AssertionError("invalid request must not load the model")

    request = {"op": "question", "directory": ".", "metadata": metadata(),
        "texts": ["x"] * (MAX_TEXTS + 1)}
    output = io.StringIO()
    _serve_worker(io.StringIO(json.dumps(request) + "\n"), output, encoder_factory=factory)
    assert json.loads(output.getvalue())["error"] == "invalid_request"
    assert made == []
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_timeout_kills_worker_and_does_not_retry(tmp_path):
    calls = []

    class SilentStdin(FakeStdin):
        def flush(self):
            self.pending = ""

    def fake_popen(*args, **kwargs):
        process = FakeProcess()
        process.stdin = SilentStdin(process)
        calls.append(process)
        return process

    encoder = IsolatedLocalEncoder(tmp_path, metadata(), timeout=1, popen=fake_popen)
    try:
        with pytest.raises(TimeoutError, match="exceeded its time budget"):
            encoder.encode_question("How was the evaluation measured?")
        assert len(calls) == 1
        process = calls[0]
        assert process.killed is True
        assert process.stdin.closed is True
        assert process.stdout.closed is True
    finally:
        encoder.close()
    assert len(calls) == 1


def test_response_reader_does_not_block_on_oversized_protocol_line(tmp_path):
    class OversizedStdout:
        def readline(self, limit):
            return "x" * (MAX_RESPONSE_CHARS + 1)

    class Process:
        stdout = OversizedStdout()

    encoder = IsolatedLocalEncoder(tmp_path, metadata(), popen=lambda *args, **kwargs: None)
    responses = queue.Queue(maxsize=1)
    reader = threading.Thread(target=encoder._read_responses, args=(Process(), responses), daemon=True)
    try:
        reader.start()
        reader.join(timeout=1)
        assert not reader.is_alive()
        assert responses.get_nowait() is _STOP
    finally:
        encoder.close()
