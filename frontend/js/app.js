const API_BASE = "";
/** Client cap: must exceed server graph + fallback time on Render (see backend SEARCH_MAX_SECONDS). */
const BUILD_FETCH_TIMEOUT_MS = 130_000;

const fetchNoStore = (url, opts = {}) =>
  fetch(url, {
    ...opts,
    cache: "no-store",
    headers: { ...opts.headers, "Cache-Control": "no-cache", Pragma: "no-cache" },
  });

async function loadOptions() {
  const res = await fetchNoStore(`${API_BASE}/api/options`);
  if (!res.ok) throw new Error("Failed to load options");
  return res.json();
}

function money(n) {
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(n);
}

function componentCard(title, row) {
  const keys = Object.keys(row).filter((k) => row[k] !== null && row[k] !== undefined && row[k] !== "");
  const meta = keys
    .map((k) => `<div><span class="k">${k}</span>: ${row[k]}</div>`)
    .join("");
  return `
    <article class="card">
      <h3>${title}</h3>
      <div class="title">${row.name ?? row.cpu_id ?? row.mb_id ?? row.ram_id ?? row.storage_id ?? row.gpu_id ?? row.psu_id ?? ""}</div>
      <div class="meta">${meta}</div>
    </article>
  `;
}

function renderResults(data) {
  const empty = document.getElementById("empty");
  const results = document.getElementById("results");
  const summary = document.getElementById("summary");

  if (!data.found || !data.build) {
    empty.classList.remove("hidden");
    results.classList.add("hidden");
    summary.classList.add("hidden");
    empty.textContent = data.message || "No build returned.";
    return;
  }

  const b = data.build;
  empty.classList.add("hidden");
  results.classList.remove("hidden");
  summary.classList.remove("hidden");

  const capPct =
    b.prefilter_component_price_cap_usd != null && b.request_budget_usd
      ? Math.round((100 * b.prefilter_component_price_cap_usd) / b.request_budget_usd)
      : null;
  const cap =
    b.prefilter_component_price_cap_usd != null && capPct != null
      ? `<span>Pre-search max single-part price (~${capPct}% of budget): <strong>${money(
          b.prefilter_component_price_cap_usd
        )}</strong></span>`
      : "";
  const budgetHint =
    b.request_budget_usd != null &&
    b.total_price != null &&
    Number(b.total_price) < Number(b.request_budget_usd) * 0.92
      ? `<div class="budget-hint">Your budget is an <strong>upper limit</strong>. The search returns a compatible build under that amount; it does not try to spend every dollar.</div>`
      : "";
  const util =
    b.budget_utilization != null && b.request_budget_usd != null
      ? `<span>Uses <strong>${b.budget_utilization}%</strong> of ${money(b.request_budget_usd)} limit</span>`
      : "";
  const eng =
    b.engine_fingerprint != null
      ? `<span title="If this never changes after you edit search_engine.py, the server is not reloading that file.">Loaded engine <code>${b.engine_fingerprint}</code></span>`
      : "";
  summary.innerHTML = `
    <span>Total: <strong>${money(b.total_price)}</strong></span>
    ·
    <span>PSU target ${b.required_psu_watts}W (+20W buffer)</span>
    ·
    <span>Headroom ${b.psu_headroom_watts}W</span>
    ${cap ? "· " + cap : ""}
    ${util ? "· " + util : ""}
    ${eng ? "<br/>" + eng : ""}
    ${budgetHint}
  `;

  const gpuHtml = b.gpu
    ? componentCard("GPU", b.gpu)
    : `<article class="card"><h3>GPU</h3><div class="title">Integrated graphics only</div><div class="meta">Office-style build without a discrete graphics card.</div></article>`;

  results.innerHTML = `
    ${componentCard("CPU", b.cpu)}
    ${componentCard("Motherboard", b.motherboard)}
    ${componentCard("RAM", b.ram)}
    ${componentCard("Storage", b.storage)}
    ${gpuHtml}
    ${componentCard("PSU", b.psu)}
    <div class="badge-row">
      <span class="badge ok">Compatibility: OK</span>
      <span class="badge">Algorithm: ${b.algorithm.toUpperCase()}</span>
      <span class="badge">Purpose: ${b.purpose}</span>
    </div>
    <div class="notes">${(b.notes || []).map((n) => `<div>${n}</div>`).join("")}</div>
  `;
}

async function init() {
  const purposeSel = document.getElementById("purpose");
  const algoSel = document.getElementById("algorithm");
  const opts = await loadOptions();
  purposeSel.innerHTML = opts.purposes
    .map((p) => `<option value="${p}">${p.replaceAll("_", " ")}</option>`)
    .join("");
  algoSel.innerHTML = opts.algorithms.map((a) => `<option value="${a}">${a.toUpperCase()}</option>`).join("");
}

document.getElementById("build-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const status = document.getElementById("status");
  const btn = document.getElementById("submit-btn");
  const budget = Number(document.getElementById("budget").value);
  const purpose = document.getElementById("purpose").value;
  const algorithm = document.getElementById("algorithm").value;

  btn.disabled = true;
  status.textContent = "Searching state space…";
  const ctrl = new AbortController();
  const kill = setTimeout(() => ctrl.abort(), BUILD_FETCH_TIMEOUT_MS);
  try {
    const res = await fetchNoStore(`${API_BASE}/api/build`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ budget, purpose, algorithm }),
      signal: ctrl.signal,
    });
    const data = await res.json();
    if (!res.ok) {
      status.textContent = data.detail || "Request failed.";
      return;
    }
    status.textContent = data.found ? "Build found." : data.message || "No build found.";
    renderResults(data);
  } catch (err) {
    if (err && err.name === "AbortError") {
      status.textContent =
        "Request timed out. Try algorithm A* or BFS, or a simpler purpose. If this is Render, redeploy after the latest backend fix.";
    } else {
      status.textContent =
        "Could not reach the API. Start the backend: uvicorn main:app --reload (from the backend folder).";
    }
    console.error(err);
  } finally {
    clearTimeout(kill);
    btn.disabled = false;
  }
});

init().catch((err) => {
  document.getElementById("status").textContent = "Failed to load /api/options. Is the backend running?";
  console.error(err);
});
