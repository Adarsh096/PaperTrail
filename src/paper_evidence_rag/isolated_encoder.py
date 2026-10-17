"""Run the lazy CPU query encoder outside ZeroGPU's forked web process."""
from contextlib import redirect_stdout
import atexit
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading

MAX_TEXTS = 300
MAX_TEXT_CHARS = 10_000
MAX_TOTAL_CHARS = 300_000
MAX_RESPONSE_CHARS = 16 * 1024 * 1024
DEFAULT_TIMEOUT = 180
_STOP = object()


def _validate_texts(texts):
    if not isinstance(texts, list) or not 1 <= len(texts) <= MAX_TEXTS:
        raise ValueError("CPU encoder input has an invalid number of texts.")
    total = 0
    for text in texts:
        if not isinstance(text, str) or len(text) > MAX_TEXT_CHARS:
            raise ValueError("CPU encoder input contains an invalid text.")
        total += len(text)
    if total > MAX_TOTAL_CHARS:
        raise ValueError("CPU encoder input exceeds the text budget.")


def _validate_vectors(vectors, expected):
    if not isinstance(vectors, list) or len(vectors) != expected:
        raise RuntimeError("CPU encoder worker returned invalid vectors.")
    clean = []
    for vector in vectors:
        if not isinstance(vector, list) or len(vector) != 768:
            raise RuntimeError("CPU encoder worker returned invalid vectors.")
        row = []
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise RuntimeError("CPU encoder worker returned invalid vectors.")
            row.append(float(value))
        clean.append(row)
    return clean


def _validate_splits(splits, texts):
    if not isinstance(splits, list) or len(splits) != len(texts):
        raise RuntimeError("CPU encoder worker returned invalid document splits.")
    clean = []
    for pieces, text in zip(splits, texts, strict=True):
        if not isinstance(pieces, list) or not pieces:
            raise RuntimeError("CPU encoder worker returned invalid document splits.")
        cursor, row = 0, []
        for piece in pieces:
            if (not isinstance(piece, dict) or set(piece) != {"text", "start", "end"}
                    or type(piece["start"]) is not int or type(piece["end"]) is not int
                    or piece["start"] != cursor or piece["end"] <= cursor
                    or piece["end"] > len(text) or piece["text"] != text[piece["start"]:piece["end"]]):
                raise RuntimeError("CPU encoder worker returned invalid document splits.")
            row.append(piece)
            cursor = piece["end"]
        if cursor != len(text):
            raise RuntimeError("CPU encoder worker returned incomplete document splits.")
        clean.append(row)
    return clean


class IsolatedLocalEncoder:
    """Keep one CPU encoder in a plain child process, isolated from ZeroGPU forks."""
    def __init__(self, directory, metadata, *, timeout=DEFAULT_TIMEOUT, popen=subprocess.Popen):
        from .embeddings import DOCUMENT_PROMPT, MODEL_ID, MODEL_REVISION, QUERY_PROMPT
        if (metadata.get("id"), metadata.get("revision"), metadata.get("dimension")) != (MODEL_ID, MODEL_REVISION, 768):
            raise ValueError("Encoder metadata does not match the selected model")
        if metadata.get("query_prompt") != QUERY_PROMPT or metadata.get("document_prompt") != DOCUMENT_PROMPT:
            raise ValueError("Encoder prompts do not match the stored vectors")
        if metadata.get("dtype") != "float32" or metadata.get("normalize_embeddings") is not True:
            raise ValueError("Encoder normalization and precision must match the corpus")
        if type(metadata.get("max_length")) is not int or not 32 <= metadata["max_length"] <= 2048:
            raise ValueError("Invalid encoder token budget")
        if not isinstance(metadata.get("file_sha256"), dict):
            raise ValueError("Encoder file manifest is missing")
        if len(metadata["file_sha256"]) > 128 or len(json.dumps(metadata, separators=(",", ":"))) > 65_536:
            raise ValueError("Encoder metadata exceeds its size limit")
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 1 <= timeout <= 600:
            raise ValueError("Invalid encoder worker timeout")
        self.directory = str(Path(directory).resolve())
        self.metadata = metadata
        self.timeout = float(timeout)
        self._popen = popen
        self._process = None
        self._responses = None
        self._reader = None
        self._lock = threading.Lock()
        atexit.register(self.close)

    def _read_responses(self, process, responses):
        def signal_stop():
            try:
                responses.put_nowait(_STOP)
            except queue.Full:
                pass

        try:
            while True:
                line = process.stdout.readline(MAX_RESPONSE_CHARS + 1)
                if not line:
                    break
                if len(line) > MAX_RESPONSE_CHARS or not line.endswith("\n"):
                    signal_stop()
                    return
                responses.put(line)
        except Exception:
            pass
        finally:
            signal_stop()

    def _start(self):
        if self._process is not None and self._process.poll() is None:
            return
        self._stop()
        allowed_environment = ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PYTHONPATH",
            "PYTHONHOME", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH")
        env = {name: os.environ[name] for name in allowed_environment if name in os.environ}
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        process = self._popen([sys.executable, "-m", "paper_evidence_rag.isolated_encoder", "--worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", bufsize=1, env=env, cwd=self.directory)
        self._process = process
        self._responses = queue.Queue(maxsize=1)
        self._reader = threading.Thread(target=self._read_responses,
            args=(process, self._responses), daemon=True)
        self._reader.start()

    def _stop(self):
        process, self._process = self._process, None
        self._responses = None
        self._reader = None
        if process is None:
            return
        try:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
        except Exception:
            pass
        for pipe in (process.stdin, process.stdout):
            try:
                if pipe:
                    pipe.close()
            except Exception:
                pass

    def close(self):
        with self._lock:
            if self._process is not None:
                try:
                    if self._process.poll() is None and self._process.stdin:
                        self._process.stdin.write('{"op":"close"}\n')
                        self._process.stdin.flush()
                        self._process.wait(timeout=2)
                except Exception:
                    pass
                self._stop()

    def _request(self, texts, operation):
        _validate_texts(texts)
        with self._lock:
            self._start()
            process, responses = self._process, self._responses
            request = {"op": operation, "texts": texts, "directory": self.directory,
                       "metadata": self.metadata}
            try:
                process.stdin.write(json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n")
                process.stdin.flush()
                line = responses.get(timeout=self.timeout)
                if line is _STOP:
                    raise RuntimeError("CPU encoder worker stopped without a response.")
                response = json.loads(line)
                if not isinstance(response, dict) or response.get("ok") is not True:
                    error = response.get("error") if isinstance(response, dict) else None
                    if error == "question_too_long":
                        raise ValueError("Question exceeds the encoder token budget. Please ask a shorter question.")
                    if error == "text_too_long":
                        raise ValueError("Text exceeds the encoder token budget.")
                    raise RuntimeError("CPU encoder worker failed (" + str(error or "invalid response") + ").")
                if operation == "split_documents":
                    return _validate_splits(response.get("splits"), texts)
                return _validate_vectors(response.get("vectors"), len(texts))
            except queue.Empty:
                self._stop()
                raise TimeoutError("CPU encoder worker exceeded its time budget.") from None
            except (BrokenPipeError, OSError, UnicodeError, json.JSONDecodeError, RuntimeError):
                self._stop()
                raise

    def encode_question(self, text):
        return self._request([text], "question")[0]

    def encode_documents(self, texts):
        return self._request(texts, "documents")

    def split_documents(self, texts):
        return self._request(texts, "split_documents")


def _serve_worker(source, destination, encoder_factory=None):
    """Serve bounded JSON-lines requests. `encoder_factory` is injectable in tests."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    encoder = None
    for line in source:
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError
            if request.get("op") == "close":
                return
            operation = request.get("op")
            texts = request.get("texts")
            _validate_texts(texts)
            if operation not in {"question", "documents", "split_documents"} or (operation == "question" and len(texts) != 1):
                raise ValueError
            if encoder is None:
                if encoder_factory is None:
                    from .app_models import LocalEncoder
                    encoder_factory = LocalEncoder
                encoder = encoder_factory(request["directory"], request["metadata"])
            with redirect_stdout(sys.stderr):
                if operation == "split_documents":
                    splits = encoder.split_documents(texts)
                    response = {"ok": True, "splits": _validate_splits(splits, texts)}
                else:
                    vectors = (encoder.encode_question(texts[0]) if operation == "question"
                               else encoder.encode_documents(texts))
                    vectors = _validate_vectors(vectors if operation == "documents" else [vectors], len(texts))
                    response = {"ok": True, "vectors": vectors}
        except ValueError as error:
            error_code = (("question_too_long" if operation == "question" else "text_too_long")
                          if "token budget" in str(error).lower()
                          else "invalid_request")
            response = {"ok": False, "error": error_code}
        except Exception as error:
            response = {"ok": False, "error": type(error).__name__}
        destination.write(json.dumps(response, separators=(",", ":")) + "\n")
        destination.flush()


if __name__ == "__main__" and sys.argv[1:] == ["--worker"]:
    _serve_worker(sys.stdin, sys.stdout)
