"""Direct Gradio handler checks for public-mode admission gates."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("gradio")
pytest.importorskip("pypdf")

from test_app import app_inputs
from paper_evidence_rag.admission import Admission, AdmissionDenied
from paper_evidence_rag.bucket_admission import AdmissionDenied as BucketAdmissionDenied
from paper_evidence_rag.app_service import DemoService
from paper_evidence_rag.gradio_app import build_app, visitor_identity
from paper_evidence_rag.gradio_app import upload_admission_message
from paper_evidence_rag.guardrails import QueryGate


def api_handler(app, name):
    event = next(item for item in app.config["dependencies"] if item.get("api_name") == name)
    return app.fns[event["id"]].fn


@pytest.fixture
def service_and_app(app_inputs, tmp_path):
    root, requests, generated, _, _ = app_inputs
    service = DemoService(root, requests, generated)
    admission = Admission(tmp_path / "public.sqlite", initialize=True, daily=1, visitor_daily=1)
    app = build_app(service, admission=admission, public_mode=True)
    try:
        yield service, app, admission
    finally:
        service.close()


def admission_rows(admission):
    with admission._connect() as db:
        return db.execute("SELECT COUNT(*) FROM admissions").fetchone()[0]


def test_pdf_admission_messages_distinguish_capacity_from_backend_failure():
    assert upload_admission_message("The demo has reached its current processing capacity.") == \
        "PDF processing is temporarily busy. Try again later."
    assert upload_admission_message("No PDF processing allowance remains right now.") == \
        "No PDF processing attempts remain right now."
    assert upload_admission_message("This session has no remaining PDF processing allowance right now.") == \
        "No PDF processing attempts remain right now."
    assert upload_admission_message("The admission reservation could not be stored durably. Processing is paused.") == \
        "PDF processing is temporarily unavailable while usage tracking recovers. Please try again later."
    assert upload_admission_message("The usage limiter is unavailable. New processing is paused.") == \
        "PDF processing is temporarily unavailable while usage tracking recovers. Please try again later."
    assert BucketAdmissionDenied is AdmissionDenied
    assert upload_admission_message(str(BucketAdmissionDenied("The private admission bucket is unavailable."))) == \
        "PDF processing is temporarily unavailable while usage tracking recovers. Please try again later."


def test_missing_visitor_fails_closed_before_query_or_upload(service_and_app):
    service, app, _ = service_and_app
    service.ask = Mock(wraps=service.ask)
    service.upload = Mock(wraps=service.upload)
    request = SimpleNamespace(session_hash="arbitrary", client=None)

    query_result = api_handler(app, "ask_paper")("p", "new question", "bm25", False, request)
    upload_result = api_handler(app, "process_pdf")(b"would-be-pdf", request)

    assert "Cannot identify this visitor" in query_result[1]
    assert "Cannot identify this visitor" in upload_result[1]
    service.ask.assert_not_called()
    service.upload.assert_not_called()


@pytest.mark.parametrize("host,expected", [
    ("192.0.2.1", "ipv4:192.0.2.1"),
    ("::ffff:192.0.2.1", "ipv4:192.0.2.1"),
    ("2001:db8:abcd:1234:1111::1", "ipv6:2001:db8:abcd:1234::/64"),
    ("2001:db8:abcd:1234:ffff::2", "ipv6:2001:db8:abcd:1234::/64"),
])
def test_network_identity_normalizes_addresses_and_ignores_browser_state(host, expected):
    request = SimpleNamespace(client=SimpleNamespace(host=host), session_hash="one")
    request.headers = {"x-forwarded-for": "198.51.100.44", "forwarded": "for=198.51.100.45"}
    assert visitor_identity(request) == expected


@pytest.mark.parametrize("request_value", [None, SimpleNamespace(client=None),
    SimpleNamespace(client=SimpleNamespace(host="not-an-ip")),
    SimpleNamespace(client=SimpleNamespace(host="")),
    SimpleNamespace(client=SimpleNamespace(host=123)),
    SimpleNamespace(client=SimpleNamespace(host="fe80::1%eth0"))])
def test_missing_or_invalid_network_identity_fails_closed(service_and_app, request_value):
    service, app, _ = service_and_app
    service.ask = Mock()
    result = api_handler(app, "ask_paper")("p", "question", "bm25", False, request_value)
    assert "Cannot identify" in result[1]
    service.ask.assert_not_called()


def test_rotated_gradio_sessions_share_query_upload_and_refresh_quota(app_inputs, tmp_path):
    root, requests, generated, _, _ = app_inputs
    service = DemoService(root, requests, generated)
    admission = Admission(tmp_path / "network-identity.sqlite", initialize=True,
        daily=10, visitor_daily=1, upload_daily=10, upload_visitor_daily=1)
    uploaded = SimpleNamespace(paper_id="uploaded", pages=1, passages=[])
    service.upload = Mock(return_value=uploaded)
    service.browse = Mock(return_value=[])
    service.ask = Mock(return_value={"state": "live_answer", "answer": {"status": "abstained",
        "reason": "fixture", "claims": [], "answer": ""}, "hits": [], "origin": "live",
        "guardrail": {"action": "allow"}})
    app = build_app(service, admission=admission, public_mode=True)
    request1 = SimpleNamespace(session_hash="tab-a", client=SimpleNamespace(host="203.0.113.9"))
    request2 = SimpleNamespace(session_hash="tab-b", client=SimpleNamespace(host="203.0.113.9"),
                               cookies={}, headers={"x-forwarded-for": "198.51.100.7"})
    try:
        api_handler(app, "ask_paper")("p", "question", "bm25", False, request1)
        denied = api_handler(app, "ask_paper")("p", "question", "bm25", False, request2)
        assert "No question allowance remains" in denied[1]
        assert service.ask.call_count == 1

        api_handler(app, "process_pdf")(b"pdf", request1)
        denied_upload = api_handler(app, "process_pdf")(b"pdf", request2)
        assert "No PDF processing attempts remain" in denied_upload[1]
        assert service.upload.call_count == 1

        refresh = next(entry.fn for entry in app.fns.values()
                       if getattr(entry.fn, "__name__", "") == "refresh_usage")
        rendered = refresh(request2)
        assert "Questions:** 0 remaining" in rendered
        assert "PDF processing:** 0 remaining" in rendered
    finally:
        service.close()


def test_displayed_allowance_uses_configured_backend_caps(app_inputs, tmp_path):
    root, requests, generated, _, _ = app_inputs
    service = DemoService(root, requests, generated)
    admission = Admission(tmp_path / "custom-limits.sqlite", initialize=True,
                          daily=11, visitor_daily=4, cooldown=17)
    try:
        app = build_app(service, admission=admission, public_mode=True)
        markdown = "\n".join(str(component.get("props", {}).get("value", ""))
                              for component in app.config["components"])
        assert "4 live questions" in markdown
        assert "2 PDFs" in markdown
        assert "successful action expires from the count after 24 hours" in markdown
        assert "provider failures do not" in markdown
        assert "Saved examples and browsing are free." in markdown
        assert "11 across this demo" not in markdown
        assert "17-second cooldown" not in markdown
        info_button = next(component for component in app.config["components"]
                           if component.get("props", {}).get("elem_id") == "admission-info-button")
        assert info_button["props"].get("value") == "ⓘ"
        popup_html = "\n".join(str(component.get("props", {}).get("value", ""))
                                for component in app.config["components"])
        assert '<dialog id="admission-info-dialog"' in popup_html
        assert not any(component.get("props", {}).get("elem_id") == "admission-policy"
                       for component in app.config["components"])
        usage_component = next(component for component in app.config["components"]
                               if component.get("props", {}).get("elem_id") == "admission-usage")
        refresh = next(entry.fn for entry in app.fns.values()
                       if getattr(entry.fn, "__name__", "") == "refresh_usage")
        rendered = refresh(SimpleNamespace(session_hash="session-for-display", client=SimpleNamespace(host="192.0.2.10")))
        assert "Questions:** 4 remaining" in rendered
        assert "PDF processing:** 2 remaining" in rendered
        assert "shared" not in rendered
        # Initial page load and both query/PDF success or failure chains target
        # the visible usage component, so failed attempts refresh after release.
        usage_id = usage_component["id"]
        assert any(usage_id in event.get("outputs", []) and event.get("trigger_after") is None
                   for event in app.config["dependencies"])
        assert sum(usage_id in event.get("outputs", []) and "trigger_after" in event
                   for event in app.config["dependencies"]) >= 3
    finally:
        service.close()


def test_query_and_upload_shared_denials_are_concise(app_inputs, tmp_path):
    root, requests, generated, _, _ = app_inputs
    service = DemoService(root, requests, generated)
    admission = Admission(tmp_path / "denials.sqlite", initialize=True, daily=3,
                          visitor_daily=2, upload_daily=3, upload_visitor_daily=2)
    service.ask = Mock()
    service.upload = Mock()
    app = build_app(service, admission=admission, public_mode=True)
    try:
        for kind in ("query", "upload"):
            for visitor in (f"{kind}-1", f"{kind}-2", f"{kind}-3"):
                with admission.admit(visitor, kind) as reservation:
                    reservation.commit()
        request = SimpleNamespace(session_hash="new-session", client=SimpleNamespace(host="192.0.2.10"))

        query = api_handler(app, "ask_paper")("p", "question", "bm25", False, request)
        upload = api_handler(app, "process_pdf")(b"pdf", request)

        assert "current processing capacity" in query[1]
        assert upload[1] == "PDF processing is temporarily busy. Try again later."
        assert "shared" not in query[1] + upload[1]
        assert "limit" not in query[1] + upload[1]
        service.ask.assert_not_called()
        service.upload.assert_not_called()
    finally:
        service.close()


@pytest.mark.parametrize("data", [None, b""])
def test_empty_pdf_replacement_clears_session_without_admission(service_and_app, data):
    service, app, admission = service_and_app
    service.upload = Mock()
    admission.admit = Mock(wraps=admission.admit)
    before = admission_rows(admission)

    result = api_handler(app, "process_pdf")(data, SimpleNamespace(session_hash="visitor-1", client=SimpleNamespace(host="192.0.2.10")))

    assert result[0] is None
    assert result[1] == "Choose a PDF first."
    assert result[2]["notice"] == "No uploaded paper."
    assert result[3] == "No uploaded paper."
    assert result[4] == ""
    service.upload.assert_not_called()
    admission.admit.assert_not_called()
    assert admission_rows(admission) == before


def test_failed_pdf_replacement_clears_session_and_result(service_and_app):
    service, app, admission = service_and_app
    service.upload = Mock(side_effect=ValueError("PDF could not be read."))
    before = admission_rows(admission)

    result = api_handler(app, "process_pdf")(b"invalid-pdf", SimpleNamespace(session_hash="visitor-1", client=SimpleNamespace(host="192.0.2.10")))

    assert result[0] is None
    assert result[1] == "PDF could not be read."
    assert result[2]["notice"] == "Upload not processed."
    assert result[3] == "Upload not processed."
    assert result[4] == ""
    assert result[5] == []
    assert result[-2].value is None
    assert result[-1].value == ""
    service.upload.assert_called_once_with(b"invalid-pdf")
    assert admission_rows(admission) == before


def test_blank_questions_are_rejected_before_admission_and_services(service_and_app):
    service, app, admission = service_and_app
    uploaded = SimpleNamespace(paper_id="uploaded-paper", pages=1, passages=[])
    service.upload = Mock(return_value=uploaded)
    service.browse = Mock(return_value=[])
    service.ask = Mock()
    service.gate = Mock()
    service.encoder = Mock()
    service.answerer = Mock()
    service.cloud_answerer = Mock()
    request = SimpleNamespace(session_hash="visitor-1", client=SimpleNamespace(host="192.0.2.10"))
    processed = api_handler(app, "process_pdf")(b"pdf-bytes", request)
    admitted_before = admission_rows(admission)
    admission.admit = Mock(wraps=admission.admit)

    paper_result = api_handler(app, "ask_paper")("p", " \t\n ", "bm25", False, request)
    upload_result = api_handler(app, "ask_upload")("p", "  ", "bm25", False, processed[0], request)

    for result in (paper_result, upload_result):
        assert result[0]["notice"] == "Enter a question."
        assert result[1] == "Enter a question."
        assert result[2] == ""
        assert result[3] == []
    service.ask.assert_not_called()
    service.gate.assert_not_called()
    service.gate.screen.assert_not_called()
    service.encoder.assert_not_called()
    service.encoder.encode_question.assert_not_called()
    service.answerer.assert_not_called()
    service.cloud_answerer.assert_not_called()
    admission.admit.assert_not_called()
    assert admission_rows(admission) == admitted_before


def test_missing_upload_fails_before_admission_or_service_calls(service_and_app):
    service, app, admission = service_and_app
    service.ask = Mock()
    admission.admit = Mock(wraps=admission.admit)
    before = admission_rows(admission)

    result = api_handler(app, "ask_upload")(
        "p", "a valid question", "bm25", False, None,
        SimpleNamespace(session_hash="visitor-1", client=SimpleNamespace(host="192.0.2.10")))

    assert result[0]["notice"] == "Process a PDF in this session first."
    assert result[1] == "Process a PDF in this session first."
    service.ask.assert_not_called()
    admission.admit.assert_not_called()
    assert admission_rows(admission) == before


def test_daily_denial_precedes_service_and_provider_calls(service_and_app):
    service, app, admission = service_and_app
    provider = Mock(return_value={"scores": {"allow": 1.0}})
    service.gate = QueryGate(provider)
    service.cloud_answerer = Mock()
    service.ask = Mock(wraps=service.ask)
    with admission.admit("visitor-1") as reservation:
        reservation.commit()

    result = api_handler(app, "ask_paper")(
        "p", "new question", "bm25", False, SimpleNamespace(session_hash="visitor-2", client=SimpleNamespace(host="192.0.2.10")))

    assert "current processing capacity" in result[1]
    service.ask.assert_not_called()
    provider.assert_not_called()
    service.cloud_answerer.assert_not_called()


def test_saved_example_and_browsing_do_not_consume_admission(service_and_app):
    service, app, admission = service_and_app
    before = admission_rows(admission)

    example = api_handler(app, "open_example")("p", "How many documents?", "bm25")
    browsed = api_handler(app, "browse_paper")("p")

    assert example[0]["origin"] == "saved"
    assert browsed[0]["origin"] == "browse"
    assert admission_rows(admission) == before == 0


def test_successful_live_answer_commits_and_failed_provider_result_releases(service_and_app):
    service, app, admission = service_and_app
    request = SimpleNamespace(session_hash="private-session-id", client=SimpleNamespace(host="192.0.2.10"))
    service.ask = Mock(return_value={"state": "provider_failure", "notice": "No fresh answer."})
    before = admission_rows(admission)
    api_handler(app, "ask_paper")("p", "question", "bm25", False, request)
    assert admission_rows(admission) == before

    service.ask.return_value = {"state": "live_answer", "answer": {"status": "abstained",
        "reason": "no_supported_claims", "claims": [], "answer": ""}, "hits": [],
        "origin": "live", "guardrail": {"action": "allow"}}
    api_handler(app, "ask_paper")("p", "question", "bm25", False, request)
    assert admission_rows(admission) == before + 1
    usage = admission.usage(visitor_identity(request))
    assert usage["query"]["session_remaining"] == 0
    assert usage["query"]["shared_remaining"] == 0
    assert "private-session-id" not in repr(usage)


def test_public_mode_requires_initialized_admission_backend(app_inputs, tmp_path):
    root, requests, generated, _, _ = app_inputs
    service = DemoService(root, requests, generated)
    try:
        with pytest.raises(ValueError, match="durable admission backend"):
            build_app(service, public_mode=True)
        Admission(tmp_path / "public.sqlite", initialize=True)
    finally:
        service.close()
