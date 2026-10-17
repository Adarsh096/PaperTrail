"""Gradio interface over the reusable service. No hosting-specific GPU decorators."""
import argparse
from contextlib import nullcontext
from importlib.resources import files
import json
import ipaddress
import os
from pathlib import Path

from .admission import AdmissionDenied
from .app_service import DemoService

THEME_JS = """() => {
  const root = document.documentElement;
  const body = document.body;
  const saved = localStorage.getItem('paper-evidence-theme');
  const dark = saved ? saved === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  root.classList.toggle('dark', dark);
  body.classList.toggle('dark', dark);
  const button = document.querySelector('#theme-toggle button, button#theme-toggle');
  if (button) { button.setAttribute('aria-label', 'Toggle light and dark theme'); button.title = 'Toggle light and dark theme'; }
}"""
TOGGLE_JS = """() => {
  const dark = !document.documentElement.classList.contains('dark');
  document.documentElement.classList.toggle('dark', dark);
  document.body.classList.toggle('dark', dark);
  localStorage.setItem('paper-evidence-theme', dark ? 'dark' : 'light');
}"""
POLICY_INFO_JS = """() => {
  const dialog = document.getElementById('admission-info-dialog');
  if (!dialog) return;
  if (dialog.open) { dialog.close(); return; }
  const close = dialog.querySelector('[data-close]');
  close.onclick = () => dialog.close();
  dialog.onclick = event => { if (event.target === dialog) dialog.close(); };
  dialog.showModal();
}"""


def visitor_identity(request):
    """Return a stable network key from Gradio's server-populated client address.

    Forwarded headers and Gradio session identifiers are intentionally ignored.
    IPv6 addresses share a /64 key to tolerate privacy-address rotation.
    """
    try:
        raw_host = request.client.host
        if not isinstance(raw_host, str):
            return None
        address = ipaddress.ip_address(raw_host)
        if isinstance(address, ipaddress.IPv6Address) and address.scope_id is not None:
            return None
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
            address = address.ipv4_mapped
        if isinstance(address, ipaddress.IPv4Address):
            return f"ipv4:{address.compressed}"
        return f"ipv6:{ipaddress.ip_network(f'{address}/64', strict=False).network_address.compressed}/64"
    except (AttributeError, TypeError, ValueError):
        return None


def _source_choices(result):
    sources = {p["passage_id"]: p for p in result.get("presented", [])}
    for hit in result.get("hits", []):
        sources.setdefault(hit["passage"]["passage_id"], hit["passage"])
    return [(f"{p['section_name']} / {key}", key) for key, p in sources.items()]


def _result_text(result):
    if "guardrail" in result and result["guardrail"]["action"] != "allow":
        decision = result["guardrail"]
        fallback = " Saved examples and paper browsing remain available." if decision["status"] in ("unavailable", "timeout", "malformed_output") else ""
        return "Query not processed. " + decision["reason"] + fallback, ""
    answer = result.get("answer")
    if answer:
        origin = ("Saved Qwen3 answer, replayed from cache after local CPU retrieval; no Groq or OpenRouter request was made"
                  if result["origin"] == "saved" else "Live " + result.get("answer_origin", "Qwen3") + " answer")
        if answer.get("answer_format") == "hosted-grounded-v1":
            status = {"answered": "Supporting quotes match the supplied passages",
                      "partial": "Draft shown; some supporting quotes could not be verified",
                      "unverified": "Draft shown without verified supporting quotes"}.get(answer["status"], "Unverified draft shown")
            limits = (" Limits: " + answer["limitations"]) if answer.get("limitations") else ""
            return origin + ". Question allowed. " + status + ". Review the answer against its sources." + limits, answer["answer"]
        if answer["status"] == "answered":
            return origin + ". Quotation checks passed; review relevance and completeness.", answer["answer"]
        return origin + " withheld: " + str(answer["reason"]) + ". You can inspect its raw draft below.", ""
    return result.get("notice", "Choose a saved question or inspect the paper's sources."), ""


def _admission_policy_notice(admission):
    ending = (" Only successfully completed processing uses an allowance. Failed processing and "
              "provider failures do not. Saved examples and browsing are free. External service "
              "availability can affect live answers.")
    if admission is None:
        return "This app instance has no server-side usage limits configured. External service availability can affect live answers."
    try:
        limits = getattr(admission, "limits", None)
        if limits is not None and callable(getattr(limits, "get", None)):
            query = limits.get("query")
            upload = limits.get("upload")
        else:
            daily = getattr(admission, "daily", None)
            visitor_daily = getattr(admission, "visitor_daily", None)
            query = upload = (daily, visitor_daily)
        pairs = (query, upload)
        if all(isinstance(pair, (tuple, list)) and len(pair) == 2
               and all(type(value) is int and value > 0 for value in pair)
               for pair in pairs):
            q_total, q_session = query
            u_total, u_session = upload
            return (f"You can process {u_session} PDFs and submit {q_session} live questions "
                    "during a rolling 24-hour period. Changing papers does not reset these counts. "
                    "Each successful action expires from the count after 24 hours." + ending)
    except Exception:
        pass
    return ("PDF processing and live questions may have separate rolling 24-hour visitor limits. "
            "The exact limits are shown above." + ending)


def upload_admission_message(message):
    """Keep quota exhaustion distinct from an unavailable admission backend."""
    normalized = str(message).casefold()
    if "current processing capacity" in normalized:
        return "PDF processing is temporarily busy. Try again later."
    if ("no remaining pdf processing allowance" in normalized
            or "no pdf processing allowance remains" in normalized):
        return "No PDF processing attempts remain right now."
    if any(term in normalized for term in ("bucket", "ledger", "reservation", "durable", "admission", "limiter")):
        return "PDF processing is temporarily unavailable while usage tracking recovers. Please try again later."
    return str(message)


def build_app(service, sample_pdfs=None, *, admission=None, public_mode=False):
    """Construct an app without launching it. All uploaded state is per session."""
    import gradio as gr
    if public_mode and admission is None:
        raise ValueError("Public mode requires an initialized durable admission backend")
    catalog = service.catalog()
    corpus_questions = service.viewer.contents["questions.json"]
    corpus_question_prompts = [
        "What is the paper's main contribution?",
        "How did the authors evaluate their approach?",
        "What limitations do the authors discuss?",
    ]
    sample_pdfs = {name: Path(path).resolve() for name, path in (sample_pdfs or {}).items()}
    first = catalog["papers"][0]["paper_id"]
    default_mode = "dense" if service.dense else "bm25"

    def render(result):
        status, answer = _result_text(result)
        rows = [[h["rank"], h["passage"]["passage_id"], round(h["score"], 6),
                 h["passage"]["section_name"], h["passage"]["text"][:160]] for h in result.get("hits", [])]
        choices = _source_choices(result)
        return (result, status, answer, rows, gr.Dropdown(choices=choices, value=choices[0][1] if choices else None),
                "", "", gr.Checkbox(value=False), gr.Textbox(value="", visible=False))

    def fail(message):
        return render({"notice": message})

    def refresh_usage(request: gr.Request = None):
        if admission is None:
            return "**Current allowances**\n\nUsage limits are not configured for this app instance."
        try:
            usage = getattr(admission, "usage", None)
            if not callable(usage):
                return "### Your available requests\n\nUsage counts are unavailable."
            values = usage(visitor_identity(request))
            lines = ["### Your available requests"]
            for key, label in (("query", "Questions"), ("upload", "PDF processing")):
                item = values[key]
                if item["session_remaining"] is None:
                    lines.append(f"- **{label}:** temporarily unavailable")
                else:
                    lines.append(f"- **{label}:** {item['session_remaining']} remaining")
            return "\n".join(lines)
        except Exception:
            return "### Your available requests\n\nTemporarily unavailable. Refresh this page to check again."

    def open_example(paper, question, mode):
        try:
            return render(service.example(paper, question, mode))
        except (ValueError, LookupError, RuntimeError) as error:
            return fail(str(error))

    def ask(paper, question, mode, use_gpu, uploaded, request=None):
        if not isinstance(question, str) or not question.strip():
            return fail("Enter a question.")
        try:
            visitor = visitor_identity(request)
            with admission.admit(visitor) if admission else nullcontext() as reservation:
                result = service.ask(uploaded.paper_id if uploaded else paper, question, mode,
                                     uploaded=uploaded, generate=True, execution_mode="gpu" if use_gpu else "cpu")
                if result.get("state") == "live_answer" and reservation is not None:
                    reservation.commit()
                return render(result)
        except (ValueError, LookupError, RuntimeError) as error:
            return fail(str(error))

    def inspect(paper, source, result, uploaded):
        if not source:
            return "", "Choose a source."
        # Selections must belong to this view, not merely to some known paper.
        valid = {key for _, key in _source_choices(result)}
        if source not in valid:
            return "", "Choose a source from the current result."
        try:
            active_upload = uploaded if uploaded and result.get("paper_id") == uploaded.paper_id else None
            p = service.passage(result.get("paper_id", paper), source, active_upload)
            location = {"source_id": p["passage_id"], "location": p["section_name"], "kind": p["kind"]}
            if active_upload:
                mapping = active_upload.dense_locations if p["passage_id"] in active_upload.dense_locations else active_upload.locations
                location.update(mapping[p["passage_id"]])
            supplied = next((x for x in result.get("presented", []) if x["passage_id"] == source), None)
            if supplied:
                location["characters_presented_to_answer_model"] = len(supplied["text"])
            return p["text"], json.dumps(location, ensure_ascii=False, indent=2)
        except (ValueError, LookupError):
            return "", "Source is not available for the current paper."

    def raw_output(show, result):
        raw = ""
        if show and result.get("origin") == "saved":
            row = service.example(result["paper_id"], result["question"], result["method"], include_raw=True)
            raw = row.get("diagnostic", {}).get("raw_output", "")
        elif show:
            raw = result.get("_raw", "")
        return gr.Textbox(value=raw, visible=bool(show))

    def browse(paper, uploaded):
        paper_id = uploaded.paper_id if uploaded else paper
        try:
            passages = service.browse(paper_id, uploaded)
            return render({"notice": "Paper text only. No model call was made.",
                           "presented": passages, "origin": "browse", "paper_id": paper_id})
        except ValueError as error:
            return fail(str(error))

    def change_paper(paper):
        p = service.paper(paper)
        questions = [q["question"] for item in catalog["papers"] if item["paper_id"] == paper for q in item["questions"]]
        suggested = [q["question"] for q in corpus_questions if q["paper_id"] == paper]
        suggested.extend(q for q in corpus_question_prompts if q not in suggested)
        return (gr.Dropdown(choices=questions, value=questions[0] if questions else None), p["abstract"],
                gr.Dropdown(choices=suggested, value=None), gr.Textbox(value=""),
                *fail("Choose a saved question, or browse the paper text."))

    def suggest_pdf_question(question):
        return question or ""

    generic_pdf_questions = [
        "What research question does this paper address?",
        "What method did the authors use, and why?",
        "What data or participants did the study use?",
        "What were the main findings and limitations?",
    ]
    qasper_questions = [
        "How many questions and research papers does QASPER contain?",
        "How were the questions and answers in QASPER collected?",
        "How are supporting evidence passages annotated in QASPER?",
        "What answer types does the QASPER dataset support?",
    ]

    def pdf_suggestions(name):
        # This name comes only from the configured sample selector. Arbitrary
        # uploaded filenames never select a corpus-specific prompt list.
        values = qasper_questions if name and "qasper" in name.casefold() else generic_pdf_questions
        return gr.Dropdown(choices=values, value=None), gr.Textbox(value="")

    def upload_pdf(data, suggestion_choices=None, request=None):
        suggestion_choices = suggestion_choices or generic_pdf_questions
        if data is None or data == b"":
            return None, "Choose a PDF first.", *fail("No uploaded paper." ), gr.Dropdown(choices=suggestion_choices, value=None), gr.Textbox(value="")
        try:
            with admission.admit(visitor_identity(request), "upload") if admission else nullcontext() as reservation:
                uploaded = service.upload(data)
                if reservation is not None:
                    reservation.commit()
            return (uploaded, f"{uploaded.pages} pages; {len(uploaded.passages)} chunks. Stored only in this session.",
                    *browse(None, uploaded), gr.Dropdown(choices=suggestion_choices, value=None), gr.Textbox(value=""))
        except (ValueError, RuntimeError) as error:
            # Clear the old paper after a rejected replacement. Never query it
            # accidentally while displaying a different filename in the uploader.
            message = str(error)
            if isinstance(error, AdmissionDenied):
                message = upload_admission_message(message)
            return None, message, *fail("Upload not processed."), gr.Dropdown(choices=generic_pdf_questions, value=None), gr.Textbox(value="")

    def use_sample(name, request: gr.Request):
        path = sample_pdfs.get(name)
        if path is None:
            return (None, "Select an available sample PDF.", *fail("No sample selected."),
                    gr.Dropdown(choices=generic_pdf_questions, value=None), gr.Textbox(value=""))
        choices = qasper_questions if "qasper" in name.casefold() else generic_pdf_questions
        return upload_pdf(path.read_bytes(), choices, request)

    def process_pdf(data, request: gr.Request):
        return upload_pdf(data, request=request)

    def ask_paper(paper, question, mode, gpu, request: gr.Request):
        return ask(paper, question, mode, gpu, None, request)

    def ask_upload(paper, question, mode, gpu, uploaded, request: gr.Request):
        return ask(paper, question, mode, gpu, uploaded, request) if uploaded else fail("Process a PDF in this session first.")

    with gr.Blocks(title="Paper evidence", analytics_enabled=False, delete_cache=(300, 3600)) as app:
        with gr.Row(elem_id="masthead"):
            gr.HTML('<div><h1>Paper evidence</h1><p>Ask a paper. Inspect the sources behind its answer.</p></div>')
            theme = gr.Button("◐", size="sm", min_width=44, scale=0, elem_id="theme-toggle")
        theme.click(fn=None, js=TOGGLE_JS, queue=False, api_name=False)
        gr.Markdown("Choose a paper and ask a question. We find supporting passages and prepare a cited answer automatically.", elem_id="intro")
        with gr.Row(elem_id="admission-usage-row"):
            usage_display = gr.Markdown("### Your available requests\n\nLoading…",
                elem_id="admission-usage", elem_classes="admission-usage", scale=1)
            policy_info = gr.Button("ⓘ", size="sm", elem_id="admission-info-button", scale=0, min_width=38)
        gr.HTML(f'<dialog id="admission-info-dialog" aria-labelledby="admission-policy-title"><button type="button" data-close aria-label="Close">×</button><h2 id="admission-policy-title">About your usage</h2><p id="admission-policy">{_admission_policy_notice(admission)} Refreshing or reopening the app does not reset the rolling 24-hour allowance. People using the same internet connection may share it.</p></dialog>')
        policy_info.click(fn=None, js=POLICY_INFO_JS, queue=False, api_name=False)
        gr.Markdown("**Privacy notice:** Do not upload sensitive personal information, confidential documents or unpublished research. Live questions and paper excerpts may be sent to external AI providers.", elem_id="privacy-notice")
        with gr.Accordion("How to use this app", open=True):
            gr.Markdown("1. Start with an example paper and click **View saved example**, or process a sample PDF.\n2. Read the answer, then select a source to inspect the original text.\n3. Ask your own question with **Get answer**. Safety and relevance checks run before retrieval.\n\nSaved examples work without model calls. New answers need the configured model services. Live questions and paper excerpts may be sent to external providers.")
        gpu_available = getattr(service.answerer, "device", None) == "cuda"
        use_gpu = gr.Checkbox(label="Use HF GPU", value=gpu_available, interactive=gpu_available,
            info="Off uses CPU vector search and hosted reranking/answers. It still uses the internet. GPU mode is available only when its models are configured.")
        with gr.Tabs():
            with gr.Tab("Example papers"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=300):
                        paper = gr.Dropdown([(p["title"], p["paper_id"]) for p in catalog["papers"]], value=first, label="Research paper")
                        with gr.Accordion("About this paper", open=False):
                            abstract = gr.Textbox(service.paper(first)["abstract"], show_label=False, interactive=False, lines=7)
                        question = gr.Dropdown([q["question"] for q in catalog["papers"][0]["questions"]],
                            value=catalog["papers"][0]["questions"][0]["question"] if catalog["papers"][0]["questions"] else None,
                            label="Saved example question")
                        mode = gr.Radio([("Vector search", "dense"), ("Keyword search", "bm25")] if service.dense else [("Keyword search", "bm25")], value=default_mode, label="Retrieve sources with", visible=False)
                        example_button = gr.Button("View saved example", variant="primary")
                        browse_button = gr.Button("Browse paper text")
                    with gr.Column(scale=2, min_width=420):
                        gr.Markdown("### Your question")
                        initial_suggestions = [q["question"] for q in corpus_questions if q["paper_id"] == first]
                        initial_suggestions.extend(q for q in corpus_question_prompts if q not in initial_suggestions)
                        paper_suggestion = gr.Dropdown(choices=initial_suggestions, value=None, label="Suggested question")
                        new_question = gr.Textbox(label="Ask about the selected paper", lines=3,
                            placeholder="What data did the authors use, and how did they evaluate their method?")
                        ask_button = gr.Button("Get answer", variant="primary")
                        gr.Markdown("New questions require query screening. If screening or model inference is unavailable, saved examples and paper browsing still work.", elem_classes="small-note")
            with gr.Tab("Your PDF"):
                gr.Markdown("**Research-focused demo.** Any successfully processed PDF can be used to ask questions. Live questions and paper excerpts may be sent to external AI providers. Do not upload sensitive personal information, confidential documents or unpublished research. Text PDFs are limited to 5 MB and 20 pages; scans and encrypted PDFs are unsupported. Files are temporary; remove your paper when finished.")
                upload = gr.File(label="Research paper PDF", file_types=[".pdf"], type="binary")
                upload_status = gr.Textbox(label="Paper status", value="No uploaded paper.", interactive=False)
                with gr.Row():
                    process = gr.Button("Process PDF", variant="primary")
                    remove = gr.Button("Remove my paper")
                if sample_pdfs:
                    with gr.Accordion("Try a sample PDF", open=True):
                        sample = gr.Dropdown(list(sample_pdfs), label="Sample paper", value=next(iter(sample_pdfs)))
                        use = gr.Button("Process this sample as an upload")
                        gr.Markdown("These sample papers are provided under CC BY 4.0. They use the same extraction and chunking as your own file. [QASPER paper](https://aclanthology.org/2021.naacl-main.365/) by Dasigi et al., 2021; [Sentence-BERT](https://aclanthology.org/D19-1410/) by Reimers and Gurevych, 2019.")
                uploaded_suggestion = gr.Dropdown(choices=generic_pdf_questions, value=None, label="Suggested question")
                uploaded_question = gr.Textbox(label="Ask about your PDF", lines=2)
                uploaded_mode = gr.Radio([("Vector search", "dense"), ("Keyword search", "bm25")], value="dense", label="Retrieve sources with", visible=False)
                uploaded_ask = gr.Button("Get answer", variant="primary")
        uploaded = gr.State(value=None, time_to_live=3600)
        current = gr.State(value={})
        gr.Markdown("## Answer and evidence")
        status = gr.Textbox(value="Choose a saved example or browse a paper.", label="Result status", interactive=False, lines=2)
        answer = gr.Textbox(label="Answer", interactive=False, lines=4, placeholder="A saved or live answer appears here. Retrieval-only results remain below.")
        table = gr.Dataframe(headers=["Rank", "Passage ID", "Search score", "Location", "Excerpt"], datatype=["number", "str", "number", "str", "str"],
            type="array", interactive=False, wrap=True, label="Retrieved sources", max_height=330)
        gr.Markdown("Scores order passages for this question. Vector, keyword and reranker scores use different scales and are not probabilities that an answer is correct.", elem_classes="small-note")
        with gr.Row():
            with gr.Column(scale=1):
                source = gr.Dropdown(choices=[], label="Inspect a source", interactive=True)
                location = gr.Textbox(label="Source location", interactive=False, lines=5)
            with gr.Column(scale=2):
                source_text = gr.Textbox(label="Full original source passage", interactive=False, lines=10)
        raw_toggle = gr.Checkbox(label="Show unvalidated model output", value=False)
        raw = gr.Textbox(label="Raw model draft, not an accepted answer", visible=False, interactive=False, lines=6)
        gr.Markdown("Saved answers were generated earlier with Qwen3-1.7B and BGE over dense candidates. Changing the retrieval view does not regenerate them. Exact quotations establish provenance; they do not guarantee a correct or complete answer.", elem_classes="small-note")
        outputs = [current, status, answer, table, source, source_text, location, raw_toggle, raw]
        policy = dict(concurrency_limit=1, concurrency_id="paper-pipeline")
        example_button.click(open_example, [paper, question, mode], outputs, api_name="open_example", **policy)
        paper.change(change_paper, [paper], [question, abstract, paper_suggestion, new_question, *outputs], api_name="select_paper", **policy)
        paper_suggestion.change(suggest_pdf_question, [paper_suggestion], [new_question], api_name="select_question_suggestion", queue=False)
        uploaded_suggestion.change(suggest_pdf_question, [uploaded_suggestion], [uploaded_question], api_name="select_pdf_question_suggestion", queue=False)
        if sample_pdfs:
            sample.change(pdf_suggestions, [sample], [uploaded_suggestion, uploaded_question], api_name="select_sample_suggestion", queue=False)
        browse_button.click(lambda p: browse(p, None), [paper], outputs, api_name="browse_paper", **policy)
        ask_button.click(ask_paper, [paper, new_question, mode, use_gpu], outputs, api_name="ask_paper", **policy).then(
            refresh_usage, None, usage_display, api_name=False, queue=False)
        source.change(inspect, [paper, source, current, uploaded], [source_text, location], api_name="inspect_source", **policy)
        raw_toggle.change(raw_output, [raw_toggle, current], [raw], api_name="raw_output", **policy)
        process.click(process_pdf, [upload], [uploaded, upload_status, *outputs, uploaded_suggestion, uploaded_question], api_name="process_pdf", **policy).then(
            refresh_usage, None, usage_display, api_name=False, queue=False)
        if sample_pdfs:
            use.click(use_sample, [sample], [uploaded, upload_status, *outputs, uploaded_suggestion, uploaded_question], api_name="process_sample", **policy).then(
                refresh_usage, None, usage_display, api_name=False, queue=False)
        uploaded_ask.click(ask_upload,
            [paper, uploaded_question, uploaded_mode, use_gpu, uploaded], outputs, api_name="ask_upload", **policy).then(
                refresh_usage, None, usage_display, api_name=False, queue=False)
        remove.click(lambda: (None, None, "Uploaded paper removed.", *fail("Uploaded paper removed from this session."),
                gr.Dropdown(choices=generic_pdf_questions, value=None), gr.Textbox(value="")),
            [], [uploaded, upload, upload_status, *outputs, uploaded_suggestion, uploaded_question], api_name="remove_upload", **policy)
        app.load(fn=None, js=THEME_JS, api_name=False)
        app.load(fn=refresh_usage, inputs=None, outputs=usage_display, api_name=False, queue=False)
    app.queue(max_size=8, default_concurrency_limit=1)
    return app


def launch_options(service, samples=None):
    import gradio as gr
    # File inputs use Gradio's bounded temporary cache; corpus, model and local
    # project files must never become downloadable via the file-serving route.
    blocked = [str(Path.cwd().resolve()), str(Path.home() / ".codex")]
    allowed = [str(Path(p).resolve()) for p in (samples or {}).values()]
    return dict(server_name="127.0.0.1", share=False, inbrowser=False, show_error=False,
        max_file_size="5mb", max_threads=4, state_session_capacity=32, strict_cors=True,
        allowed_paths=allowed, blocked_paths=blocked, run_history=False, enable_monitoring=False,
        footer_links=[], theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate",
            font=["Aptos", "Segoe UI", "sans-serif"]),
        css=files("paper_evidence_rag").joinpath("assets/app.css").read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Run the paper evidence Gradio app locally")
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--requests", type=Path)
    parser.add_argument("--generated", type=Path)
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--samples", type=Path)
    parser.add_argument("--encoder-directory", type=Path)
    parser.add_argument("--reranker-directory", type=Path)
    parser.add_argument("--generator-directory", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--budget-db", type=Path, default=Path("data/app-usage.sqlite"))
    parser.add_argument("--enable-groq", action="store_true", help="Opt in to calls using GROQ_API_KEY")
    parser.add_argument("--enable-openrouter", action="store_true", help="Opt in to public-data safety/reranking/answers using OPENROUTER_API_KEY")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    from .hosted_guardrail import configured_gate
    from .app_models import LocalEncoder, SnapshotAnswerer
    if bool(args.reranker_directory) != bool(args.generator_directory):
        parser.error("Provide both reranker and generator snapshots, or neither")
    if args.encoder_directory and not args.embeddings:
        parser.error("The encoder needs matching corpus embedding metadata")
    encoder = LocalEncoder(args.encoder_directory, json.loads(args.embeddings.read_text())["model"]) if args.encoder_directory else None
    answerer = SnapshotAnswerer(args.reranker_directory, args.generator_directory, args.device) if args.generator_directory else None
    samples = {p.stem: p for p in sorted(args.samples.glob("*.pdf"))} if args.samples else {}
    gate = configured_gate(os.environ.get("GROQ_API_KEY") if args.enable_groq else None, args.budget_db)
    cloud_answerer = None
    if args.enable_openrouter:
        from .cloud_pipeline import OpenRouterClient, PacedBudget, SafetyFirstGate, CloudAnswerer
        client = OpenRouterClient(os.environ.get("OPENROUTER_API_KEY"),
            PacedBudget(args.budget_db.with_name("openrouter-usage.sqlite")))
        gate, cloud_answerer = SafetyFirstGate(client, gate), CloudAnswerer(client)
    from .ingestion import digest
    service = DemoService(args.artifacts, args.requests, args.generated, args.embeddings,
        gate=gate, encoder=encoder, answerer=answerer, cloud_answerer=cloud_answerer,
        public_pdf_hashes=[digest(p.read_bytes()) for p in samples.values()])
    app = build_app(service, samples)
    try:
        app.launch(server_port=args.port, **launch_options(service, samples))
    finally:
        app.close()
        service.close()


if __name__ == "__main__":
    main()
