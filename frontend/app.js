/* ==========================================================================
   Aarogya — client
   Flow: intake -> (adaptive questions) -> assessment -> grounded chat.

   Two behaviours are deliberate:
   * A client that has a warning sign never sees the follow-up questions. The
     server decides this, so a slow or modified client cannot ask an
     emergency patient a list of ordinary questions first.
   * Nothing is rendered as a percentage. The API returns likelihood bands and
     the UI shows exactly what it was given.
   ========================================================================== */

const API = "/api/v1";

const SYMPTOMS = [
  "fever", "cough", "headache", "sore throat", "shortness of breath",
  "chest pain", "fatigue", "body ache", "nausea", "vomiting", "diarrhea",
  "abdominal pain", "rash", "dizziness", "joint pain", "loss of appetite",
  "night sweats", "runny nose", "sneezing",
];

const TRIAGE_TONES = ["emergency", "urgent", "routine", "self-care"];

const state = {
  sessionId: (crypto.randomUUID ? crypto.randomUUID() : `s-${Date.now()}`).slice(0, 64),
  selected: new Set(),
  symptomText: "",
  vitals: {},
  displayName: "",
  questions: [],
  assessment: null,
  history: [],
};

/* ------------------------------------------------------------- utilities -- */

const $ = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[character]);
}

async function api(path, options = {}) {
  const response = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = payload && payload.detail;
    throw new Error(
      typeof detail === "string" ? detail
        : Array.isArray(detail) && detail.length ? detail[0].msg
        : `Request failed (${response.status})`
    );
  }
  return payload;
}

function show(stage) {
  ["intake-stage", "questions-stage", "results-stage"].forEach((id) => {
    $(id).classList.toggle("hidden", id !== `${stage}-stage`);
  });
  if (stage !== "intake") $("hero").classList.add("hidden");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function notice(message) {
  const box = $("form-error");
  if (!message) { box.classList.add("hidden"); box.textContent = ""; return; }
  box.textContent = message;
  box.classList.remove("hidden");
}

/* ---------------------------------------------------------------- health -- */

async function checkHealth() {
  const pill = $("status-pill");
  const text = $("status-text");
  try {
    const health = await api("/health");
    const ok = health.status === "ok";
    pill.classList.add(ok ? "is-ok" : "is-warn");
    text.textContent = ok
      ? `Engine ready · ${health.database} · ${health.cache}`
      : `Degraded · ${health.detail || "dependency issue"}`;
  } catch (error) {
    pill.classList.add("is-down");
    text.textContent = "Service unreachable";
  }
}

/* ---------------------------------------------------------------- intake -- */

function buildChips() {
  const container = $("symptom-chips");
  SYMPTOMS.forEach((symptom) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.textContent = symptom;
    chip.setAttribute("aria-pressed", "false");
    chip.addEventListener("click", () => {
      const active = chip.getAttribute("aria-pressed") === "true";
      chip.setAttribute("aria-pressed", String(!active));
      if (active) state.selected.delete(symptom); else state.selected.add(symptom);
    });
    container.appendChild(chip);
  });
}

function readNumber(id) {
  const raw = $(id).value.trim();
  if (raw === "") return null;
  const value = Number(raw);
  return Number.isFinite(value) ? value : null;
}

function collectVitals() {
  const vitals = {};
  const age = readNumber("age");
  const heartRate = readNumber("heart-rate");
  const systolic = readNumber("systolic-bp");
  const spo2 = readNumber("spo2");
  const temperature = readNumber("temperature");
  const glucose = readNumber("glucose");

  if (age !== null) vitals.age = age;
  if (heartRate !== null) vitals.heart_rate = heartRate;
  if (systolic !== null) vitals.systolic_bp = systolic;
  if (spo2 !== null) vitals.spo2 = spo2;
  if (temperature !== null) vitals.temperature_c = temperature;
  if (glucose !== null) vitals.glucose_mgdl = glucose;
  return vitals;
}

async function submitIntake(event) {
  event.preventDefault();
  notice(null);

  const text = $("symptom-text").value.trim();
  const selected = Array.from(state.selected);
  if (!text && !selected.length) {
    notice("Describe your symptoms or select at least one before continuing.");
    return;
  }

  state.symptomText = text || selected.join(", ");
  state.selected = new Set(selected);
  state.vitals = collectVitals();
  state.displayName = $("display-name").value.trim();

  const button = $("intake-submit");
  button.disabled = true;
  button.innerHTML = '<span class="spinner"></span>Checking for warning signs…';

  try {
    const questionnaire = await api("/assessments/questionnaire", {
      method: "POST",
      body: JSON.stringify({
        symptom_text: state.symptomText,
        selected_symptoms: selected,
        vitals: state.vitals,
      }),
    });

    // A warning sign means the ordinary questions are inappropriate.
    if (questionnaire.red_flags.length) {
      await runAssessment({});
      return;
    }

    state.questions = questionnaire.follow_up_questions;
    if (!state.questions.length) {
      await runAssessment({});
      return;
    }
    renderQuestions();
    show("questions");
  } catch (error) {
    notice(error.message);
  } finally {
    button.disabled = false;
    button.textContent = "Continue";
  }
}

/* ------------------------------------------------------------- questions -- */

function renderQuestions() {
  const body = $("questions-body");
  body.innerHTML = "";
  state.questions.forEach((question) => {
    const field = document.createElement("div");
    field.className = "field";
    const legend = document.createElement("label");
    legend.textContent = question.question;
    field.appendChild(legend);

    const chips = document.createElement("div");
    chips.className = "chips";
    question.options.forEach((option) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "chip";
      chip.textContent = option;
      chip.setAttribute("aria-pressed", "false");
      chip.addEventListener("click", () => {
        Array.from(chips.children).forEach((sibling) => sibling.setAttribute("aria-pressed", "false"));
        chip.setAttribute("aria-pressed", "true");
        state.answers = state.answers || {};
        state.answers[question.id] = option;
      });
      chips.appendChild(chip);
    });

    field.appendChild(chips);
    body.appendChild(field);
  });
}

async function submitQuestions(event) {
  event.preventDefault();
  const button = $("questions-submit");
  button.disabled = true;
  button.innerHTML = '<span class="spinner"></span>Assessing…';
  try {
    await runAssessment(state.answers || {});
  } catch (error) {
    notice(error.message);
  } finally {
    button.disabled = false;
    button.textContent = "See my assessment";
  }
}

/* ------------------------------------------------------------ assessment -- */

async function runAssessment(followUpAnswers) {
  const button = $("intake-submit");
  button.disabled = true;

  const payload = {
    symptom_text: state.symptomText,
    selected_symptoms: Array.from(state.selected),
    vitals: state.vitals,
    follow_up_answers: followUpAnswers,
    session_id: state.sessionId,
    include_narrative: true,
  };
  if (state.displayName) payload.patient = { display_name: state.displayName };

  const assessment = await api("/assessments", {
    method: "POST",
    body: JSON.stringify(payload),
  });

  state.assessment = assessment;
  state.history = [];
  $("chat-log").innerHTML = "";
  renderResults(assessment);
  show("results");
}

/* ---------------------------------------------------------------- render -- */

function bandClass(band) {
  return `band--${String(band).toLowerCase().replace(/\s+/g, "-")}`;
}

function toneClass(tone) {
  return TRIAGE_TONES.includes(tone) ? tone : "routine";
}

function renderEvidence(items, emptyText, className = "") {
  if (!items.length) return `<ul class="evidence"><li>${escapeHtml(emptyText)}</li></ul>`;
  const pills = items.map((item) => `<li class="${className}">${escapeHtml(item)}</li>`).join("");
  return `<ul class="evidence">${pills}</ul>`;
}

function renderResults(data) {
  const { triage, entities, differential, red_flags: redFlags, uncertainty } = data;
  const blocks = [];

  /* 1. Urgency first. */
  blocks.push(`
    <div class="triage triage--${toneClass(triage.tone)}">
      <div class="triage-top">
        <span class="triage-level">Level ${triage.level} of 5 · ${escapeHtml(triage.label)}</span>
      </div>
      <h2 class="triage-headline">${escapeHtml(triage.headline)}</h2>
      <p class="triage-summary">${escapeHtml(triage.summary)}</p>
    </div>
  `);

  if (redFlags.length) {
    const items = redFlags.map((flag) => `<li><strong>${escapeHtml(flag.label)}</strong> — ${escapeHtml(flag.message)}</li>`).join("");
    blocks.push(`
      <div class="alert" role="alert">
        <p class="alert-title">Warning signs detected — normal assessment stopped</p>
        <ul class="alert-list">${items}</ul>
        <div class="alert-actions">
          <a class="link-btn" href="tel:112">Call emergency services</a>
          <a class="link-btn" href="https://www.google.com/maps/search/emergency+department+near+me" target="_blank" rel="noopener">Find nearest emergency department</a>
        </div>
      </div>
    `);
  }

  if (data.narrative) {
    blocks.push(`
      <div class="panel" style="margin-top:22px">
        <p class="panel-label">Summary</p>
        <p style="margin:0;color:var(--ink-soft)">${escapeHtml(data.narrative)}</p>
      </div>
    `);
  }

  /* 2. What was extracted. */
  let findings = "";
  if (entities.symptoms.length) {
    findings += `<div class="diff-block"><h4>Symptoms recorded</h4>${renderEvidence(entities.symptoms, "")}</div>`;
  } else {
    findings += `<div class="diff-block"><h4>Symptoms recorded</h4><p class="note">Nothing specific could be identified. Try describing your symptoms in more detail.</p></div>`;
  }
  if (entities.explicitly_denied.length) {
    findings += `<div class="diff-block"><h4>You reported as absent</h4>${renderEvidence(entities.explicitly_denied, "", "denied")}</div>`;
  }

  const context = [];
  if (entities.duration) context.push(`Duration reported: ${escapeHtml(entities.duration)}`);
  if (entities.severity !== null && entities.severity !== undefined) context.push(`Severity reported: ${escapeHtml(entities.severity)} / 10`);

  blocks.push(`
    <div class="section-head"><span>What we found</span></div>
    <div class="panel">${findings}${context.length ? `<p class="note">${context.join(" · ")}</p>` : ""}</div>
  `);

  if (uncertainty) {
    const next = entities.missing_information && entities.missing_information.length
      ? ` One more answer would help most: ${entities.missing_information[0]}`
      : "";
    blocks.push(`<div class="notice">${escapeHtml(uncertainty)}${escapeHtml(next)}</div>`);
  }

  /* 3. Differential. */
  blocks.push(`<div class="section-head"><span>Possible causes</span></div>`);

  if (!differential.length) {
    blocks.push(`
      <div class="panel">
        <p class="note" style="margin:0">
          No common condition in the reference set matches the symptoms recorded. That is a
          limitation of the reference set, not a clean bill of health — if you feel unwell,
          speak to a clinician.
        </p>
      </div>
    `);
  } else {
    const cards = differential.map((item) => `
      <article class="diff">
        <div class="diff-rank">${String(item.rank).padStart(2, "0")}</div>
        <div>
          <div class="diff-head">
            <h3 class="diff-name">${escapeHtml(item.condition)}</h3>
            <span class="band ${bandClass(item.band)}">${escapeHtml(item.band)}</span>
          </div>
          <div class="diff-block"><h4>Supporting</h4>${renderEvidence(item.supporting, "No supporting symptoms recorded")}</div>
          <div class="diff-block"><h4>Against</h4>${renderEvidence(item.against, "Nothing recorded against this")}</div>
          <div class="diff-block"><h4>Not reported</h4>${renderEvidence(item.missing, "No notable gaps")}</div>
        </div>
      </article>
    `).join("");
    blocks.push(`
      <div class="diff-grid">${cards}</div>
      <p class="note">
        Likelihood is shown as a band, never a percentage. Aarogya measures how well each
        condition explains what you reported; it does not estimate how likely a condition
        actually is, because it has no prevalence data to do so honestly.
      </p>
    `);
  }

  /* 4. Next steps. */
  const steps = triage.actions;
  const column = (title, items) => `
    <div class="step-col">
      <h3>${title}</h3>
      <ul>${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
    </div>
  `;
  blocks.push(`
    <div class="section-head"><span>What to do next</span></div>
    <div class="panel">
      <div class="steps">
        ${column("For now", steps.for_now)}
        ${column("Consider medical evaluation if", steps.consider)}
        ${column("Seek urgent help if", steps.urgent)}
      </div>
    </div>
  `);

  /* 5. Transparency. */
  const vitals = Object.entries(state.vitals);
  blocks.push(`
    <details class="disclosure">
      <summary>How Aarogya reached this assessment</summary>
      <div class="disclosure-body">
        <h4>1. Symptoms considered</h4>
        <p>${entities.symptoms.length ? escapeHtml(entities.symptoms.join(", ")) : "None reliably identified"}</p>
        <h4>2. Measurements supplied</h4>
        <p>${vitals.length ? escapeHtml(vitals.map(([key, value]) => `${key.replace(/_/g, " ")}: ${value}`).join(" · ")) : "None supplied — treated as not measured"}</p>
        <h4>3. Warning signs checked</h4>
        <p>${redFlags.length ? escapeHtml(redFlags.map((flag) => flag.label).join(", ")) : "None detected"}</p>
        <h4>4. Where the ranking comes from</h4>
        <p>
          A curated reference set of common presentations, scored on how much of each
          condition's usual picture your reported symptoms account for. Urgency is decided
          by a separate deterministic rule set, so a ranking can never lower the urgency level.
          Engine version ${escapeHtml(data.engine_version)}.
        </p>
        <h4>5. Language layer</h4>
        <p>
          The written summary comes from ${escapeHtml(data.narrative_source === "deterministic"
            ? "a deterministic template, not a language model"
            : `an optional language model (${data.narrative_source}); it rephrases findings and cannot change them`)}.
        </p>
      </div>
    </details>
  `);

  blocks.push(`<p class="footnote">${escapeHtml(data.disclaimer)}</p>`);

  $("results-content").innerHTML = blocks.join("");
}

/* ------------------------------------------------------------------ chat -- */

function appendTurn(role, text, sources, isNotice) {
  const log = $("chat-log");
  const turn = document.createElement("div");
  turn.className = `turn ${role}${isNotice ? " notice" : ""}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  if (sources && sources.length) {
    const box = document.createElement("div");
    box.className = "chat-sources";
    box.innerHTML = `<h5>Reference material used</h5><ol>${sources.map((item) =>
      `<li><strong>${escapeHtml(item.title)}</strong>${item.source ? ` — ${escapeHtml(item.source)}` : ""}<br>${escapeHtml(item.snippet)}</li>`
    ).join("")}</ol>`;
    bubble.appendChild(box);
  }

  turn.appendChild(bubble);
  log.appendChild(turn);
  log.scrollTop = log.scrollHeight;
}

async function sendChat(event) {
  event.preventDefault();
  const input = $("chat-input");
  const message = input.value.trim();
  if (!message) return;

  input.value = "";
  appendTurn("user", message);
  state.history.push({ role: "user", content: message });

  $("chat-typing").classList.remove("hidden");
  $("chat-submit").disabled = true;

  try {
    const response = await api("/chat", {
      method: "POST",
      body: JSON.stringify({
        message,
        session_id: state.sessionId,
        assessment_id: state.assessment ? state.assessment.id : null,
        history: state.history.slice(-8),
      }),
    });
    appendTurn("assistant", response.reply, response.sources, Boolean(response.safety_notice));
    state.history.push({ role: "assistant", content: response.reply });
  } catch (error) {
    appendTurn("assistant", `Something went wrong: ${error.message}`);
  } finally {
    $("chat-typing").classList.add("hidden");
    $("chat-submit").disabled = false;
    input.focus();
  }
}

/* ----------------------------------------------------------------- wiring -- */

function reset() {
  state.selected = new Set();
  state.answers = {};
  state.questions = [];
  state.assessment = null;
  state.history = [];
  state.sessionId = (crypto.randomUUID ? crypto.randomUUID() : `s-${Date.now()}`).slice(0, 64);

  $("intake-form").reset();
  document.querySelectorAll("#symptom-chips .chip").forEach((chip) => chip.setAttribute("aria-pressed", "false"));
  $("chat-log").innerHTML = "";
  notice(null);
  $("hero").classList.remove("hidden");
  show("intake");
}

buildChips();
checkHealth();

$("intake-form").addEventListener("submit", submitIntake);
$("questions-form").addEventListener("submit", submitQuestions);
$("questions-back").addEventListener("click", () => show("intake"));
$("chat-form").addEventListener("submit", sendChat);
$("start-over").addEventListener("click", reset);
