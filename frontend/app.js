document.addEventListener('DOMContentLoaded', () => {
  const input = document.getElementById('symptom-input');
  const submit = document.getElementById('submit-btn');
  const loading = document.getElementById('loading-indicator');
  const results = document.getElementById('results-section');
  const error = document.getElementById('form-error');
  const causes = document.getElementById('res-causes');
  const alternatives = document.getElementById('res-alternatives');
  const steps = document.getElementById('res-steps');
  const care = document.getElementById('res-care');
  const urgency = document.getElementById('urgency-banner');
  const fields = ['age', 'hr', 'bp', 'spo2', 'temp', 'glucose'];

  const selectedSymptoms = () => [...document.querySelectorAll('.symptom-pill.selected')].map(pill => pill.dataset.symptom);
  const showError = message => { error.textContent = message; error.classList.remove('hidden'); };
  const renderList = (node, values) => { node.replaceChildren(...values.map(value => { const item = document.createElement('li'); item.textContent = value; return item; })); };

  document.querySelectorAll('.symptom-pill').forEach(pill => {
    pill.addEventListener('click', () => {
      pill.classList.toggle('selected');
      pill.setAttribute('aria-pressed', pill.classList.contains('selected'));
    });
  });

  async function analyze() {
    const symptoms = input.value.trim();
    const selected = selectedSymptoms();
    if (!symptoms && !selected.length) return showError('Describe a symptom or select at least one option to continue.');
    error.classList.add('hidden');
    results.classList.add('hidden');
    loading.classList.remove('hidden');
    submit.disabled = true;

    const payload = { symptoms: symptoms || selected.join(', '), selected_symptoms: selected };
    fields.forEach(key => { payload[key] = Number(document.getElementById(key).value); });
    try {
      const response = await fetch('/api/analyze', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(body.detail || 'We could not complete the symptom check. Please try again.');
      causes.textContent = `${body.primary_condition} (${body.confidence}% model confidence). ${body.possible_causes}`;
      alternatives.textContent = body.alternatives.length ? `Other model possibilities: ${body.alternatives.join(', ')}.` : '';
      renderList(steps, body.next_steps);
      renderList(care, body.when_to_seek_care);
      urgency.textContent = body.safety_signals.length ? `Urgent safety signals: ${body.safety_signals.join(' · ')}` : body.model_notice;
      urgency.className = `urgency-banner ${body.urgency === 'urgent' ? 'urgent' : 'routine'}`;
      results.classList.remove('hidden');
      results.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (requestError) {
      showError(requestError.message);
    } finally {
      loading.classList.add('hidden');
      submit.disabled = false;
    }
  }
  submit.addEventListener('click', analyze);
  input.addEventListener('keydown', event => { if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') analyze(); });
});
