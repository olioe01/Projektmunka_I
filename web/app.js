const input = document.querySelector('#text');
const feedback = document.querySelector('#feedback');
const button = document.querySelector('#predict');
let resultsStale = false;
let models = [];
const escapeHtml = value => String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
function render(results = [], loading = false) {
  document.querySelector('#results').innerHTML = models.map((model, index) => {
    const result = results.find(item => item.id === model.id);
    const state = loading ? 'Feldolgozás' : result?.status === 'ok' ? 'Elkészült' : result?.status === 'error' ? 'Sikertelen' : model.ready ? 'Csatlakoztatva' : 'Nem érhető el';
    const message = loading ? 'A predikció feldolgozása folyamatban…' : result?.status === 'ok' ? result.prediction : result?.message || (model.ready ? 'A predikció futtatás után itt jelenik meg.' : model.message || 'A predikcióhoz szükséges modellfájl még nem érhető el.');
    const probability = result?.probability == null ? '' : `<div class="probability">Becsült osztályvalószínűség <strong>${(result.probability * 100).toLocaleString('hu-HU', {maximumFractionDigits: 1})}%</strong></div>`;
    const metrics = model.metrics ? `<div class="model-metric">Tesztpontosság <strong>${(model.metrics.accuracy * 100).toLocaleString('hu-HU', {maximumFractionDigits: 1})}%</strong></div>` : '';
    return `<article class="model-card ${result?.status === 'ok' ? 'success' : result?.status === 'error' ? 'error' : ''}"><div class="card-top"><span class="model-number">0${index + 1}</span><span class="model-state">${state}</span></div><h3>${escapeHtml(model.name)}</h3><p class="model-output">${escapeHtml(message)}</p>${probability}${metrics}</article>`;
  }).join('');
}
function updateCount() { document.querySelector('#counter').textContent = `${input.value.length.toLocaleString('hu-HU')} / 10 000`; }
input.addEventListener('input', () => { updateCount(); if (resultsStale) { render(); feedback.textContent = 'A bemenet megváltozott. Futtasd újra a predikciókat.'; resultsStale = false; } });
document.querySelector('#example').addEventListener('click', () => { input.value = 'Muscle cramps are a common problem characterized by a sudden, painful, involuntary contraction of muscle.'; updateCount(); render(); resultsStale = false; input.focus(); feedback.textContent = ''; });
document.querySelector('#clear').addEventListener('click', () => {input.value = ''; updateCount(); render(); feedback.textContent = ''; input.focus();});
button.addEventListener('click', async () => {
  if (!input.value.trim()) { feedback.textContent = 'Először adj meg egy elemezni kívánt szöveget.'; input.focus(); return; }
  button.disabled = true; input.disabled = true;
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
  finally {button.disabled = false; input.disabled = false; document.querySelector('#clear').disabled = false; document.querySelector('#example').disabled = false;}
});
async function init() {
  try {
    const response = await fetch('/api/models');
    if (!response.ok) throw new Error();
    const data = await response.json(); models = data.models; render();
    const ready = models.filter(m => m.ready).length;
    document.querySelector('#connection').textContent = `${ready} / ${models.length} modell csatlakoztatva${ready === 0 ? ' · A betanított modellek feltöltésére vár' : ''}`;
  } catch {
    document.querySelector('#connection').textContent = 'A szerver nem érhető el. Indítás: python3 server.py';
    feedback.textContent = 'Nem sikerült betölteni a modellek állapotát.'; button.disabled = true;
  }
}
init();
