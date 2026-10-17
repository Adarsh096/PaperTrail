import pytest

from paper_evidence_rag.provider_status import (
    LocalCallLimit, PipelineStageFailure, ProviderRateLimit, ProviderUnavailable,
)


@pytest.mark.parametrize("value", [None, 0, -1, float("inf"), float("nan"), 3601, "12"])
def test_invalid_retry_after_is_discarded(value):
    assert ProviderRateLimit("generation", value).retry_after is None


def test_retry_after_is_numeric_and_bounded():
    assert ProviderRateLimit("reranking", 12).retry_after == 12.0
    assert LocalCallLimit("daily").kind == "daily"
    assert LocalCallLimit("cooldown").kind == "cooldown"
    assert ProviderUnavailable("safety").stage == "safety"
    assert PipelineStageFailure("generation").stage == "generation"
