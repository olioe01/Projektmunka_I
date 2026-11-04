const input = document.querySelector('#text');
const feedback = document.querySelector('#feedback');
const button = document.querySelector('#predict');
let resultsStale = false;
let models = [];
const escapeHtml = value => String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
function render(results = [], loading = false) {
  document.querySelector('#results').innerHTML = models.map(model => {
    const result = results.find(item => item.id === model.id);
    const message = loading ? 'Feldolgozás…' : result?.status === 'ok' ? result.prediction : result?.message || (model.ready ? 'Az eredmény futtatás után itt jelenik meg.' : model.message);
    const estimate = result?.probability != null
      ? `<div class="probability">Becsült osztályvalószínűség<strong>${(result.probability * 100).toLocaleString('hu-HU', {maximumFractionDigits: 1})}%</strong></div>`
      : result?.decision_margin != null
      ? `<div class="probability">Döntési pontszám<strong>${result.decision_margin.toLocaleString('hu-HU', {minimumFractionDigits: 2, maximumFractionDigits: 2})}</strong><small>Nagyobb érték: határozottabb besorolás. Nem százalék.</small></div>` : '';
    return `<article class="model-card ${result?.status === 'ok' ? 'success' : result?.status === 'error' ? 'error' : ''}"><h3>${escapeHtml(model.name)}</h3><p class="model-output">${escapeHtml(message || '')}</p>${estimate}</article>`;
  }).join('');
}

function updateCount() { document.querySelector('#counter').textContent = `${input.value.length.toLocaleString('hu-HU')} / 10 000`; }
input.addEventListener('input', () => { updateCount(); if (resultsStale) { render(); feedback.textContent = 'A bemenet megváltozott. Futtasd újra a predikciókat.'; resultsStale = false; } });
const exampleSelect = document.querySelector('#example-select');
let examples = [];
function useExample() {
  const example = examples[Number(exampleSelect.value)];
  if (exampleSelect.value === '' || !example) { feedback.textContent = 'Először válassz egy példamondatot.'; return; }
  input.value = example.text; updateCount(); render(); resultsStale = false; input.focus(); feedback.textContent = '';
}
function setExamples(data) {
  if (!Array.isArray(data) || !data.length || !data.every(item => item && typeof item.text === 'string' && item.text.trim() && item.text.length <= 10000 && (item.title == null || typeof item.title === 'string'))) throw new Error('A JSON nem üres lista legyen, minden elemben 1–10 000 karakteres text mezővel és opcionális title mezővel.');
  examples = data;
  exampleSelect.innerHTML = '<option value="">Válassz egy mondatot…</option>' + examples.map((item, i) => `<option value="${i}">${escapeHtml(item.title || item.text)}</option>`).join('');
}
document.querySelector('#example').addEventListener('click', useExample);
exampleSelect.addEventListener('change', () => { if (exampleSelect.value !== '') useExample(); });
document.querySelector('#examples-file').addEventListener('change', async event => {
  const file = event.target.files[0]; if (!file) return;
  const status = document.querySelector('#examples-feedback');
  try {
    if (file.size > 1000000) throw new Error('Legfeljebb 1 MB méretű JSON tölthető be.');
    setExamples(JSON.parse(await file.text())); status.textContent = `${examples.length} példamondat betöltve.`;
  } catch (error) { status.textContent = error instanceof SyntaxError ? 'Érvénytelen JSON-fájl.' : error.message; }
  finally { event.target.value = ''; }
});
fetch('/examples.json').then(response => { if (!response.ok) throw new Error(); return response.json(); }).then(setExamples).catch(() => { document.querySelector('#examples-feedback').textContent = 'Az alapértelmezett példák nem tölthetők be. Válassz saját JSON-fájlt.'; });
const tabs = [...document.querySelectorAll('[role="tab"]')];
function activateTab(tab) {
  tabs.forEach(item => { const selected = item === tab; item.setAttribute('aria-selected', String(selected)); item.tabIndex = selected ? 0 : -1; document.getElementById(item.getAttribute('aria-controls')).hidden = !selected; });
}
tabs.forEach((tab, index) => {
  tab.addEventListener('click', () => activateTab(tab));
  tab.addEventListener('keydown', event => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault(); const next = event.key === 'Home' ? tabs[0] : event.key === 'End' ? tabs.at(-1) : tabs[(index + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length]; activateTab(next); next.focus();
  });
});
document.querySelector('#clear').addEventListener('click', () => {input.value = ''; updateCount(); render(); feedback.textContent = ''; input.focus();});
button.addEventListener('click', async () => {
  if (!input.value.trim()) { feedback.textContent = 'Először adj meg egy elemezni kívánt szöveget.'; input.focus(); return; }
  button.disabled = true; input.disabled = true; exampleSelect.disabled = true; document.querySelector('#examples-file').disabled = true;
  document.querySelector('#clear').disabled = true; document.querySelector('#example').disabled = true;
  feedback.textContent = 'Predikciók lekérése…'; render([], true);
  try {
    const response = await fetch('/api/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text:input.value})});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'A kérés sikertelen.');
    render(data.results); resultsStale = true;
    const count = data.results.filter(item => item.status === 'ok').length;
    feedback.textContent = count ? `${count} modell predikciója elkészült.` : 'Nem készült predikció. A részletek az eredménykártyákon láthatók.';
  } catch (error) {render(); feedback.textContent = error.message || 'A szerver nem érhető el.';}
  finally {button.disabled = false; input.disabled = false; exampleSelect.disabled = false; document.querySelector('#examples-file').disabled = false; document.querySelector('#clear').disabled = false; document.querySelector('#example').disabled = false;}
});
async function init() {
  try {
    const response = await fetch('/api/models');
    if (!response.ok) throw new Error();
    const data = await response.json(); models = data.models; render();
    document.querySelector('#model-info').innerHTML = models.map(model => `<article class="model-card"><h3>${escapeHtml(model.name)}</h3><p class="hint">${escapeHtml(model.description || '')}</p><div class="probability">Tesztpontosság<strong>${model.metrics?.accuracy == null ? 'Nincs mérési adat' : (model.metrics.accuracy * 100).toLocaleString('hu-HU', {maximumFractionDigits: 1}) + '%'}</strong></div></article>`).join('');
  } catch {
    feedback.textContent = 'Nem sikerült betölteni a modellek állapotát.'; button.disabled = true;
  }
}
init();
