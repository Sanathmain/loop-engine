const form = document.getElementById("start-form");
const problemEl = document.getElementById("problem");
const runBtn = document.getElementById("run-btn");
const statusLine = document.getElementById("status-line");
const roundPill = document.getElementById("round-pill");
const writerCol = document.getElementById("writer-col");
const criticCol = document.getElementById("critic-col");
const finalEl = document.getElementById("final");

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
         ${listBlock("Assumptions", p.assumptions)}
         ${listBlock("Risks", p.risks)}
       </details>`
    );
    showThinking("critic", "Reading the draft…");
    setStatus(`Round ${event.round_number}: Critic is talking`);
    return;
  }

  if (event.type === "critic") {
    clearThinking();
    const p = event.payload || {};
    addMessage(
      criticCol,
      "msg",
      `<span class="tag">Round ${event.round_number} · score ${p.score ?? "—"}</span>
       ${escapeHtml((p.weaknesses && p.weaknesses[0]) || "Review complete.")}
       <details>
         <summary>Full critique</summary>
         ${listBlock("Strengths", p.strengths)}
         ${listBlock("Weaknesses", p.weaknesses)}
         ${listBlock("Unsupported claims", p.unsupported_claims)}
         ${listBlock("Missing information", p.missing_information)}
         ${listBlock("Risks", p.risks)}
         ${listBlock("Alternatives", p.alternative_approaches)}
         ${listBlock("Recommended changes", p.recommended_changes)}
       </details>`
    );
    return;
  }

  if (event.type === "completed") {
    clearThinking();
    const answer = (event.payload && event.payload.final_answer) || {};
    finalEl.classList.remove("hidden");
    finalEl.innerHTML = `<h2>Agreed output</h2><p>${escapeHtml(answer.solution || "")}</p>`;
    setStatus("Debate finished");
    roundPill.textContent = "Complete";
    runBtn.disabled = false;
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
    setStatus("Failed");
    runBtn.disabled = false;
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

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const problem = problemEl.value.trim();
  if (!problem) return;

  writerCol.innerHTML = "";
  criticCol.innerHTML = "";
  finalEl.classList.add("hidden");
  finalEl.innerHTML = "";
  runBtn.disabled = true;
  setStatus("Opening the room…");

  try {
    const created = await fetch("/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ problem }),
    });
    if (!created.ok) {
      throw new Error(await created.text());
    }
    const session = await created.json();
    setStatus("Debate in progress…");

    const response = await fetch(`/sessions/${session.id}/events`);
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
    if (runBtn.disabled) {
      setStatus("Done");
      runBtn.disabled = false;
    }
  } catch (err) {
    clearThinking();
    addMessage(
      criticCol,
      "msg error",
      `<span class="tag">Error</span>${escapeHtml(err.message || err)}`
    );
    setStatus("Failed");
    runBtn.disabled = false;
  }
});
