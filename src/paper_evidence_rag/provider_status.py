"""Sanitized provider and local quota failures shared by model adapters."""
import math


class LocalCallLimit(RuntimeError):
    def __init__(self, kind):
        if kind not in ("daily", "cooldown"):
            raise ValueError("Unknown local call limit")
        self.kind = kind
        super().__init__("Local provider call limit reached")


class ProviderRateLimit(RuntimeError):
    def __init__(self, stage, retry_after=None):
        if stage not in ("screening", "safety", "reranking", "generation"):
            raise ValueError("Unknown provider stage")
        if type(retry_after) not in (int, float) or not math.isfinite(retry_after) or not 0 < retry_after <= 3600:
            retry_after = None
        self.stage, self.retry_after = stage, float(retry_after) if retry_after is not None else None
        super().__init__("Provider rate limit reached")


class ProviderUnavailable(RuntimeError):
    def __init__(self, stage):
        if stage not in ("screening", "safety", "reranking", "generation"):
            raise ValueError("Unknown provider stage")
        self.stage = stage
        super().__init__("Provider unavailable")


class PipelineStageFailure(RuntimeError):
    def __init__(self, stage):
        if stage not in ("reranking", "generation"):
            raise ValueError("Unknown pipeline stage")
        self.stage = stage
        super().__init__("Hosted pipeline stage failed")
