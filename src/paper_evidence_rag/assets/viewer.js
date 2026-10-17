/* Sources and drafts are untrusted text. Build DOM nodes without HTML parsing. */
const el = (tag, text, cls) => {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (cls) node.className = cls;
  return node;
};
const app = document.querySelector('#app');
const header = el('header');
header.append(el('p', 'Paper evidence workbench', 'eyebrow'), el('h1', 'Read the answer. Check the source.'));
const theme = el('button', '◐', 'theme');
theme.type = 'button'; theme.setAttribute('aria-label', 'Toggle light and dark mode');
const applyTheme = dark => {
  document.documentElement.dataset.theme = dark ? 'dark' : 'light';
  theme.setAttribute('aria-pressed', String(dark));
};
let stored;
try { stored = localStorage.getItem('paper-theme'); } catch (_) { /* Storage may be disabled. */ }
applyTheme(['light', 'dark'].includes(stored) ? stored === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches);
theme.onclick = () => {
  const dark = document.documentElement.dataset.theme !== 'dark'; applyTheme(dark);
  try { localStorage.setItem('paper-theme', dark ? 'dark' : 'light'); } catch (_) { /* Keep the toggle usable. */ }
};
header.append(theme); app.append(header);
const form = el('form', undefined, 'query-form');
const paper = el('select'); paper.id = 'paper';
const question = el('textarea'); question.id = 'question'; question.rows = 3; question.maxLength = 2000; question.required = true;
const saved = el('select'); saved.id = 'saved';
const labeled = (name, control) => { const wrap = el('div'); const label = el('label', name); label.htmlFor = control.id; wrap.append(label, control); return wrap; };
const controls = el('div', undefined, 'selectors'); controls.append(labeled('Paper', paper), labeled('Saved question', saved));
const submit = el('button', 'Find evidence', 'primary'); submit.type = 'submit';
submit.disabled = true;
form.append(controls, labeled('Your question', question), submit);
const meta = el('p', '', 'meta');
const status = el('p', 'Loading the local corpus…', 'status'); status.setAttribute('role', 'status');
app.append(meta, form, status);
const workspace = el('div', undefined, 'workspace');
const results = el('section'); results.setAttribute('aria-label', 'Answer and evidence');
const reader = el('aside'); reader.setAttribute('aria-label', 'Full source passage');
reader.append(el('h2', 'Source reader'), el('p', 'Open a citation or source passage to read its original text.', 'muted'));
workspace.append(results, reader); app.append(workspace);
let catalog, current, pending;
const api = async (path, options = {}) => {
  const controller = new AbortController(); pending?.abort(); pending = controller;
  const timer = setTimeout(() => controller.abort(), 8000);
  try {
    const response = await fetch(path, {...options, signal: controller.signal});
    const value = await response.json();
    if (!response.ok) throw new Error(value.error || 'The request failed.');
    return value;
  } finally { clearTimeout(timer); }
};
const locationText = p => `${p.section_name} · ${p.kind} ${p.position}${p.section_index === null ? '' : ` · section ${p.section_index}`}`;
async function openSource(id, quote = '') {
  status.textContent = 'Opening source…';
  try {
    const {passage: p} = await api(`/api/passage?id=${encodeURIComponent(id)}`);
    reader.replaceChildren(el('h2', 'Source reader'), el('p', locationText(p), 'meta'), el('code', p.passage_id));
    const text = el('p', undefined, 'source-text'); const offset = quote ? p.text.indexOf(quote) : -1;
    if (offset >= 0) text.append(document.createTextNode(p.text.slice(0, offset)), el('mark', quote), document.createTextNode(p.text.slice(offset + quote.length)));
    else text.textContent = p.text;
    reader.append(text); status.textContent = 'Full original passage opened.';
  } catch (error) { status.textContent = error.name === 'AbortError' ? 'Source request stopped or timed out. Try again.' : error.message; }
}
function sourceButton(id, quote = '') {
  const button = el('button', id, 'source-link'); button.type = 'button'; button.onclick = () => openSource(id, quote); return button;
}
function render(value) {
  if (value.guardrail && value.guardrail.action !== 'allow') {
    results.replaceChildren(el('h2', value.guardrail.action === 'reject' ? 'Query declined' : 'Clarification needed'), el('p', value.guardrail.reason, 'notice'));
    return;
  }
  results.replaceChildren(); reader.replaceChildren(el('h2', 'Source reader'), el('p', 'Choose a source below to inspect the whole passage.', 'muted'));
  if (value.state === 'saved_answer') {
    const answer = value.answer; results.append(el('p', 'Saved Qwen3 run', 'eyebrow'), el('h2', answer.status === 'answered' ? 'Answer with source checks' : 'No accepted answer'));
    if (answer.status === 'answered') {
      for (const claim of answer.claims) { const block = el('blockquote'); block.append(el('p', claim.text), sourceButton(claim.passage_id, claim.text)); results.append(block); }
      results.append(el('p', 'Quotes match the sources. Check whether they answer your question fully.', 'notice'));
    } else results.append(el('p', `Validation reason: ${answer.reason}. The original draft is available below.`, 'notice'));
    const toggle = el('input'); toggle.type = 'checkbox'; toggle.id = 'show-raw'; toggle.checked = false;
    const label = el('label', 'Show model output'); label.htmlFor = toggle.id;
    const diagnostic = el('div', undefined, 'diagnostic'); const check = el('div', undefined, 'check'); check.append(toggle, label); results.append(check, diagnostic);
    toggle.onchange = async () => {
      diagnostic.replaceChildren();
      if (!toggle.checked) return;
      const selected = current;
      try {
        const raw = await api('/api/query', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({...selected, include_raw: true})});
        if (!toggle.checked || current !== selected) return;
        diagnostic.append(el('h3', raw.diagnostic.label), el('p', `Validation reason: ${raw.diagnostic.validation_reason || 'No quotation validation failure. Semantic review still required.'}`), el('pre', raw.diagnostic.raw_output));
      } catch (error) { diagnostic.append(el('p', error.name === 'AbortError' ? 'Draft request stopped or timed out. Toggle again to retry.' : error.message)); }
    };
    const details = el('details'); details.append(el('summary', 'Run identity'));
    details.append(el('p', `${value.lineage.model.id} · ${value.lineage.prompt_version}`), el('code', `Cache ${value.lineage.sha256}`), el('code', `Corpus ${value.lineage.corpus_revision}`)); results.append(details);
  } else {
    results.append(el('h2', value.state === 'no_evidence' ? 'No matching evidence' : 'Evidence found; answer needs a batch run'));
    results.append(el('p', value.state === 'no_evidence' ? 'Try words used in this paper or choose a saved question.' : 'This question has no saved Qwen3 answer. Keyword search works here; new embeddings, reranking and generation run separately on the model host.', 'notice'));
  }
  const sources = value.presented || value.hits.map(h => h.passage);
  if (sources.length) results.append(el('h2', value.presented ? 'Evidence shown to the model' : 'Keyword evidence'));
  sources.forEach((p, i) => {
    const block = el('article', undefined, 'passage'); block.append(el('p', `${i + 1}. ${locationText(p)}`, 'meta'), sourceButton(p.passage_id));
    if (value.hits) {
      const score = el('p', `Search score ${value.hits[i].score.toFixed(4)}`, 'meta');
      const info = el('button', 'i', 'info'); info.type = 'button'; info.title = 'BM25 combines query-term matches with inverse document frequency and passage-length normalization. Higher scores rank first within this paper. Scores are not probabilities and are not comparable to cosine or reranker scores.'; info.setAttribute('aria-label', info.title); score.append(info); block.append(score);
    }
    block.append(el('p', p.text, 'excerpt')); results.append(block);
  });
}
function selectPaper() {
  current = null; pending?.abort(); results.replaceChildren();
  reader.replaceChildren(el('h2', 'Source reader'), el('p', 'Choose a question and open its evidence.', 'muted'));
  const p = catalog.papers.find(p => p.paper_id === paper.value); saved.replaceChildren();
  saved.append(new Option('Choose a saved question', ''));
  for (const q of p.questions) saved.append(new Option(q.question, q.question));
  question.value = ''; status.textContent = p.questions.length ? 'Choose a saved question or search with your own.' : 'No saved answers for this paper. Enter a question to search its text.';
}
paper.onchange = selectPaper;
function clearResult() {
  current = null; pending?.abort(); results.replaceChildren();
  reader.replaceChildren(el('h2', 'Source reader'), el('p', 'Find evidence to update this source view.', 'muted'));
  status.textContent = 'Question changed. Find evidence to update the result.';
}
saved.onchange = () => { question.value = saved.value; clearResult(); };
question.oninput = () => { saved.value = ''; clearResult(); };
form.onsubmit = async event => {
  event.preventDefault(); submit.disabled = true; results.replaceChildren(); status.textContent = 'Finding evidence…';
  reader.replaceChildren(el('h2', 'Source reader'), el('p', 'Waiting for evidence for this question.', 'muted'));
  const selected = {paper_id: paper.value, question: question.value}; current = selected;
  try {
    const value = await api('/api/query', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(selected)});
    if (current !== selected) return;
    render(value); status.textContent = value.state === 'saved_answer' ? 'Saved answer loaded. Check its sources.' : 'Keyword search finished.';
  } catch (error) { if (current === selected) status.textContent = error.name === 'AbortError' ? 'Request stopped or exceeded eight seconds. Try again.' : error.message; }
  finally { submit.disabled = false; }
};
api('/api/catalog').then(value => {
  catalog = value; for (const p of catalog.papers) paper.append(new Option(p.title, p.paper_id));
  meta.textContent = `${catalog.paper_count} papers · ${catalog.passage_count} source passages · ${catalog.cached_question_count} saved questions · ${catalog.split} debugging corpus`;
  selectPaper(); submit.disabled = false;
}).catch(error => { status.textContent = `Cannot load the corpus: ${error.message}. Restart the local workbench and reload.`; submit.disabled = true; });
