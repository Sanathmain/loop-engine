const form = document.getElementById("start-form");
const problemEl = document.getElementById("problem");
const maxRoundsEl = document.getElementById("max-rounds");
const runBtn = document.getElementById("run-btn");
const stopBtn = document.getElementById("stop-btn");
const continueBtn = document.getElementById("continue-btn");
const statusLine = document.getElementById("status-line");
const roundPill = document.getElementById("round-pill");
const scoreTrend = document.getElementById("score-trend");
const writerCol = document.getElementById("writer-col");
const criticCol = document.getElementById("critic-col");
const finalEl = document.getElementById("final");
const sessionList = document.getElementById("session-list");
const refreshSessionsBtn = document.getElementById("refresh-sessions");

let abortController = null;
let scoreByRound = {};
let bestRound = null;
let activeSessionId = null;

function setStatus(text) {
  statusLine.textContent = text;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function listBlock(title, items) {
  if (!items || !items.length) return "";
  const lis = items.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  return `<p><strong>${escapeHtml(title)}</strong></p><ul>${lis}</ul>`;
}

function costFoot(payload) {
  const bits = [];
  if (payload && payload.duration_ms != null) bits.push(`${payload.duration_ms} ms`);
  if (payload && payload.tokens != null) bits.push(`${payload.tokens} tokens`);
  return bits.length ? `<span class="foot">${escapeHtml(bits.join(" · "))}</span>` : "";
}

function addMessage(column, className, html) {
  const node = document.createElement("article");
  node.className = className;
  node.innerHTML = html;
  column.appendChild(node);
  node.scrollIntoView({ behavior: "smooth", block: "end" });
  return node;
}

function showThinking(side, text) {
  clearThinking();
  const column = side === "writer" ? writerCol : criticCol;
  const node = addMessage(
    column,
    "msg thinking",
    `<span class="tag">${side}</span>${escapeHtml(text)}`
  );
  node.id = "thinking";
}

function clearThinking() {
  const existing = document.getElementById("thinking");
  if (existing) existing.remove();
}

function resetBoard() {
  writerCol.innerHTML = "";
  criticCol.innerHTML = "";
  finalEl.classList.add("hidden");
  finalEl.innerHTML = "";
  scoreTrend.innerHTML = "";
  scoreByRound = {};
  bestRound = null;
  roundPill.textContent = "No round yet";
}

function renderScoreTrend() {
  const rounds = Object.keys(scoreByRound)
    .map(Number)
    .sort((a, b) => a - b);
  scoreTrend.innerHTML = rounds
    .map((round) => {
      const score = scoreByRound[round];
      const isBest = bestRound != null && Number(bestRound) === round;
      return `<span class="score-chip${isBest ? " best" : ""}">R${round}: ${escapeHtml(
        score
      )}${isBest ? " · best" : ""}</span>`;
    })
    .join("");
}

function setRunning(isRunning) {
  runBtn.disabled = isRunning;
  stopBtn.disabled = !isRunning;
  continueBtn.disabled = isRunning || !activeSessionId;
}

function setContinueEnabled(enabled) {
  continueBtn.disabled = !enabled || !!abortController;
}

function handleEvent(event) {
  if (event.type === "round") {
    roundPill.textContent = `Round ${event.round_number}`;
    showThinking("writer", "Drafting a solution…");
    setStatus(`Round ${event.round_number}: Writer is talking`);
    return;
  }

  if (event.type === "writer") {
    clearThinking();
    const p = event.payload || {};
    addMessage(
      writerCol,
      "msg",
      `<span class="tag">Round ${event.round_number} · confidence ${p.confidence ?? "—"}</span>
       ${escapeHtml(p.solution || "")}
       <details>
         <summary>Notes</summary>
         <p>${escapeHtml(p.analysis || "")}</p>
         ${p.confidence_rationale ? `<p><strong>Confidence rationale</strong></p><p>${escapeHtml(p.confidence_rationale)}</p>` : ""}
         ${listBlock("Assumptions", p.assumptions)}
         ${listBlock("Risks", p.risks)}
       </details>
       ${costFoot(p)}`
    );
    showThinking("critic", "Reading the draft…");
    setStatus(`Round ${event.round_number}: Critic is talking`);
    return;
  }

  if (event.type === "critic") {
    clearThinking();
    const p = event.payload || {};
    scoreByRound[event.round_number] = p.score;
    renderScoreTrend();
    addMessage(
      criticCol,
      "msg",
      `<span class="tag">Round ${event.round_number} · score ${p.score ?? "—"} · ${p.verdict || "unchanged"}</span>
       ${escapeHtml((p.weaknesses && p.weaknesses[0]) || (p.blocking_issues && p.blocking_issues[0]) || "Review complete.")}
       <details>
         <summary>Full critique</summary>
         ${listBlock("Strengths", p.strengths)}
         ${listBlock("Weaknesses", p.weaknesses)}
         ${listBlock("Blocking issues", p.blocking_issues)}
         ${listBlock("Resolved points", p.resolved_points)}
         ${listBlock("Regressions", p.regressions)}
         ${listBlock("Unsupported claims", p.unsupported_claims)}
         ${listBlock("Missing information", p.missing_information)}
         ${listBlock("Risks", p.risks)}
         ${listBlock("Alternatives", p.alternative_approaches)}
         ${listBlock("Recommended changes", p.recommended_changes)}
       </details>
       ${costFoot(p)}`
    );
    return;
  }

  if (event.type === "completed") {
    clearThinking();
    const session = event.payload || {};
    const answer = session.final_answer || {};
    bestRound = session.best_round;
    activeSessionId = session.id || activeSessionId;
    renderScoreTrend();
    finalEl.classList.remove("hidden");
    finalEl.innerHTML = `<h2>Best output${
      bestRound != null ? ` · round ${escapeHtml(bestRound)}` : ""
    }</h2><p>${escapeHtml(answer.solution || "")}</p>
    <p class="foot">Not satisfied? Click Continue 3 more rounds.</p>`;
    setStatus("Debate finished — you can continue for 3 more rounds");
    roundPill.textContent = "Complete";
    setRunning(false);
    setContinueEnabled(true);
    loadSessions();
    return;
  }

  if (event.type === "error") {
    clearThinking();
    const detail = (event.payload && event.payload.detail) || "Loop failed";
    addMessage(
      criticCol,
      "msg error",
      `<span class="tag">Error</span>${escapeHtml(detail)}`
    );
    setStatus("Failed — you can try Continue if some rounds finished");
    setRunning(false);
    setContinueEnabled(Boolean(activeSessionId));
    loadSessions();
  }
}

function parseSseChunk(buffer, onEvent) {
  const parts = buffer.split("\n\n");
  const rest = parts.pop() || "";
  for (const part of parts) {
    const line = part
      .split("\n")
      .filter((row) => row.startsWith("data:"))
      .map((row) => row.slice(5).trim())
      .join("");
    if (!line) continue;
    onEvent(JSON.parse(line));
  }
  return rest;
}

async function streamSession(sessionId, signal, options = {}) {
  const mode = options.continueMode ? "?mode=continue" : "";
  const response = await fetch(`/sessions/${sessionId}/events${mode}`, { signal });
  if (!response.ok || !response.body) {
    throw new Error(await response.text());
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    buffer = parseSseChunk(buffer, handleEvent);
  }
  if (buffer.trim()) {
    parseSseChunk(`${buffer}\n\n`, handleEvent);
  }
}

function renderSession(session) {
  resetBoard();
  activeSessionId = session.id;
  problemEl.value = session.problem || "";
  if (session.max_rounds) {
    maxRoundsEl.value = String(Math.min(Number(session.max_rounds) || 3, 10));
  }
  bestRound = session.best_round;
  roundPill.textContent =
    session.status === "completed"
      ? "Complete"
      : session.current_round
        ? `Round ${session.current_round}`
        : "No round yet";

  for (const item of session.history || []) {
    scoreByRound[item.round_number] = item.critic_response.score;
    const writer = item.revised_writer_response || item.writer_response;
    addMessage(
      writerCol,
      "msg",
      `<span class="tag">Round ${item.round_number} · confidence ${writer.confidence ?? "—"}</span>
       ${escapeHtml(writer.solution || "")}
       ${costFoot(item)}`
    );
    const critic = item.critic_response;
    addMessage(
      criticCol,
      "msg",
      `<span class="tag">Round ${item.round_number} · score ${critic.score ?? "—"} · ${critic.verdict || "unchanged"}</span>
       ${escapeHtml((critic.weaknesses && critic.weaknesses[0]) || "Review complete.")}
       <details>
         <summary>Full critique</summary>
         ${listBlock("Strengths", critic.strengths)}
         ${listBlock("Weaknesses", critic.weaknesses)}
         ${listBlock("Blocking issues", critic.blocking_issues)}
         ${listBlock("Resolved points", critic.resolved_points)}
         ${listBlock("Regressions", critic.regressions)}
         ${listBlock("Recommended changes", critic.recommended_changes)}
       </details>
       ${costFoot(item)}`
    );
  }
  renderScoreTrend();

  if (session.final_answer) {
    finalEl.classList.remove("hidden");
    finalEl.innerHTML = `<h2>Best output${
      bestRound != null ? ` · round ${escapeHtml(bestRound)}` : ""
    }</h2><p>${escapeHtml(session.final_answer.solution || "")}</p>
    <p class="foot">Not satisfied? Click Continue 3 more rounds.</p>`;
  }
  setStatus(`Loaded session (${session.status})`);
  setContinueEnabled(
    Boolean(session.history && session.history.length) &&
      session.status !== "running"
  );
}

async function loadSessions() {
  try {
    const response = await fetch("/sessions");
    if (!response.ok) return;
    const sessions = await response.json();
    sessionList.innerHTML = "";
    for (const session of sessions.slice(0, 20)) {
      const li = document.createElement("li");
      const btn = document.createElement("button");
      btn.type = "button";
      const preview = (session.problem || "").slice(0, 72);
      btn.innerHTML = `${escapeHtml(preview)}${
        session.problem && session.problem.length > 72 ? "…" : ""
      }<span class="meta">${escapeHtml(session.status)} · best ${
        session.best_score != null ? escapeHtml(session.best_score) : "—"
      }</span>`;
      btn.addEventListener("click", async () => {
        const full = await fetch(`/sessions/${session.id}`);
        if (!full.ok) return;
        renderSession(await full.json());
      });
      li.appendChild(btn);
      sessionList.appendChild(li);
    }
  } catch (_err) {
    // Ignore list failures in the UI shell.
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const problem = problemEl.value.trim();
  if (!problem) return;
  const maxRounds = Number(maxRoundsEl.value) || 3;

  if (abortController) abortController.abort();
  abortController = new AbortController();
  resetBoard();
  activeSessionId = null;
  setRunning(true);
  setStatus("Opening the room…");

  try {
    const created = await fetch("/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ problem, max_rounds: maxRounds }),
      signal: abortController.signal,
    });
    if (!created.ok) {
      throw new Error(await created.text());
    }
    const session = await created.json();
    activeSessionId = session.id;
    setStatus("Debate in progress…");
    await streamSession(session.id, abortController.signal);
    if (runBtn.disabled) {
      setStatus("Done");
      setRunning(false);
      setContinueEnabled(Boolean(activeSessionId));
    }
  } catch (err) {
    clearThinking();
    if (err.name === "AbortError") {
      setStatus("Stopped — click Continue to keep going");
      setRunning(false);
      setContinueEnabled(Boolean(activeSessionId));
      loadSessions();
      return;
    }
    addMessage(
      criticCol,
      "msg error",
      `<span class="tag">Error</span>${escapeHtml(err.message || err)}`
    );
    setStatus("Failed");
    setRunning(false);
    setContinueEnabled(Boolean(activeSessionId));
  } finally {
    abortController = null;
  }
});

continueBtn.addEventListener("click", async () => {
  if (!activeSessionId) return;
  const extraRounds = 3;

  if (abortController) abortController.abort();
  abortController = new AbortController();
  setRunning(true);
  finalEl.classList.add("hidden");
  setStatus(`Continuing for ${extraRounds} more round(s)…`);

  try {
    const continued = await fetch(`/sessions/${activeSessionId}/continue`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ extra_rounds: extraRounds }),
      signal: abortController.signal,
    });
    if (!continued.ok) {
      throw new Error(await continued.text());
    }
    const session = await continued.json();
    activeSessionId = session.id;
    roundPill.textContent = `Continuing to ${session.max_rounds}`;
    await streamSession(session.id, abortController.signal, { continueMode: true });
    if (runBtn.disabled) {
      setRunning(false);
      setContinueEnabled(Boolean(activeSessionId));
    }
  } catch (err) {
    clearThinking();
    if (err.name === "AbortError") {
      setStatus("Stopped — click Continue to keep going");
      setRunning(false);
      setContinueEnabled(Boolean(activeSessionId));
      loadSessions();
      return;
    }
    addMessage(
      criticCol,
      "msg error",
      `<span class="tag">Error</span>${escapeHtml(err.message || err)}`
    );
    setStatus("Continue failed");
    setRunning(false);
    setContinueEnabled(Boolean(activeSessionId));
  } finally {
    abortController = null;
  }
});

stopBtn.addEventListener("click", () => {
  if (abortController) abortController.abort();
});

refreshSessionsBtn.addEventListener("click", loadSessions);
loadSessions();
setContinueEnabled(false);
