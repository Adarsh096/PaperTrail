"""Explicit inference placement; requesting CUDA never falls back to CPU."""
import os


def configure(torch, device):
    if device not in {"cpu", "cuda"}:
        raise ValueError("Inference device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; no CPU fallback")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    if device == "cuda":
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    return {"device": device, "dtype": "float32", "threads": 4, "seed": 0,
        "hardware": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
        "cuda_version": torch.version.cuda if device == "cuda" else None,
        "tf32": False if device == "cuda" else None,
        "deterministic_algorithms": device == "cuda"}


def synchronize(torch, device):
    if device == "cuda":
        torch.cuda.synchronize()
