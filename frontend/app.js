/**
 * Explainable TGAT Financial Fraud Ring Investigator
 * Frontend Single-Page Application Controller
 * Pure Vanilla ES6 / Zero External Build Tools
 */

// Application State
const state = {
  currentSessionId: null,
  activeTab: "view-overview",
  uploadedCsvContent: null,
  uploadedFilename: null,
  resultsPage: 1,
  resultsLimit: 25,
  resultsSort: "risk_score:desc",
  resultsSearch: "",
  resultsFlagFilter: "",
  selectedTransactionId: null,
  graphNodes: [],
  graphEdges: [],
  selectedNode: null,
};

// ============================================================================
// INITIALIZATION & TAB ROUTING
// ============================================================================

document.addEventListener("DOMContentLoaded", () => {
  initNavigation();
  initOverview();
  initAnalysisStepper();
  initResultsTable();
  initGraphCanvas();
  initPerformanceView();

  // Load initial health & check for existing session
  checkInitialSession();
});

function initNavigation() {
  const navButtons = document.querySelectorAll(".nav-item-btn");
  navButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetView = btn.getAttribute("data-target");
      switchTab(targetView);
    });
  });

  const quickNewAnalysisBtn = document.getElementById("btn-quick-new-analysis");
  if (quickNewAnalysisBtn) {
    quickNewAnalysisBtn.addEventListener("click", () => {
      switchTab("view-analysis");
      resetStepperToStep1();
    });
  }

  const emptyStartBtn = document.getElementById("btn-empty-start-analysis");
  if (emptyStartBtn) {
    emptyStartBtn.addEventListener("click", () => {
      switchTab("view-analysis");
      resetStepperToStep1();
    });
  }

  const viewAllResultsBtn = document.getElementById("btn-view-all-results");
  if (viewAllResultsBtn) {
    viewAllResultsBtn.addEventListener("click", () => switchTab("view-results"));
  }

  const backFromExpBtn = document.getElementById("btn-back-to-results-from-exp");
  if (backFromExpBtn) {
    backFromExpBtn.addEventListener("click", () => switchTab("view-results"));
  }
}

function switchTab(viewId) {
  state.activeTab = viewId;

  // Update nav buttons
  document.querySelectorAll(".nav-item-btn").forEach((btn) => {
    const isTarget = btn.getAttribute("data-target") === viewId;
    btn.classList.toggle("active", isTarget);
    btn.setAttribute("aria-selected", isTarget ? "true" : "false");
  });

  // Update view visibility
  document.querySelectorAll(".tab-view").forEach((view) => {
    view.classList.toggle("active", view.id === viewId);
  });

  // Contextual refresh on tab entry
  if (viewId === "view-overview") {
    loadOverview();
  } else if (viewId === "view-results") {
    loadResultsTable(1);
  } else if (viewId === "view-investigation") {
    loadGraph();
  } else if (viewId === "view-performance") {
    loadPerformance();
  }
}

async function checkInitialSession() {
  try {
    const res = await fetch("/api/analysis/overview");
    if (!res.ok) return;
    const data = await res.json();
    if (data.has_data && data.session_id) {
      state.currentSessionId = data.session_id;
      updateHeaderSession(data.session_id, data.threshold);
      loadOverview();
    }
  } catch (err) {
    console.warn("No active initial session:", err);
  }
}

function updateHeaderSession(sessionId, threshold) {
  const pill = document.getElementById("header-session-id");
  if (pill) pill.textContent = sessionId;

  const threshVal = document.getElementById("header-threshold-val");
  if (threshVal && threshold) threshVal.textContent = Number(threshold).toFixed(4);
}

// ============================================================================
// 1. OVERVIEW VIEW
// ============================================================================

function initOverview() {
  const loadSampleBtn = document.getElementById("btn-overview-load-sample");
  if (loadSampleBtn) {
    loadSampleBtn.addEventListener("click", async () => {
      await loadDemoSampleBatch(50);
    });
  }
}

async function loadOverview() {
  const emptyState = document.getElementById("overview-empty-state");
  const activeContent = document.getElementById("overview-active-content");

  try {
    const url = state.currentSessionId
      ? `/api/analysis/overview?session_id=${encodeURIComponent(state.currentSessionId)}`
      : `/api/analysis/overview`;
    const res = await fetch(url);
    if (!res.ok) throw new Error("Overview API error");
    const data = await res.json();

    if (!data.has_data) {
      if (emptyState) emptyState.style.display = "flex";
      if (activeContent) activeContent.style.display = "none";
      return;
    }

    if (emptyState) emptyState.style.display = "none";
    if (activeContent) activeContent.style.display = "block";

    state.currentSessionId = data.session_id;
    updateHeaderSession(data.session_id, data.threshold);

    // KPIs
    document.getElementById("kpi-total-analysed").textContent = data.total_transactions.toLocaleString();
    document.getElementById("kpi-flagged-count").textContent = data.flagged_transactions.toLocaleString();
    document.getElementById("kpi-flag-rate").textContent = `${data.flagged_rate}%`;
    document.getElementById("kpi-flag-rate-denom").textContent = `Denominator: ${data.flagged_transactions} / ${data.total_transactions} txs`;
    document.getElementById("kpi-distinct-accounts").textContent = data.distinct_accounts.toLocaleString();

    // Execution Record
    document.getElementById("meta-filename").textContent = data.filename || "upload.csv";
    document.getElementById("meta-session-id").textContent = data.session_id;
    document.getElementById("meta-duration").textContent = `${data.duration_seconds}s`;

    // Histogram
    renderHistogram(data.score_distribution || []);

    // Recent Flagged Table
    renderRecentFlagged(data.recent_flagged || []);
  } catch (err) {
    console.error("Failed to load overview:", err);
  }
}

function renderHistogram(bins) {
  const container = document.getElementById("overview-histogram");
  if (!container) return;
  container.innerHTML = "";

  const maxCount = Math.max(1, ...bins.map((b) => b.count));

  bins.forEach((b, idx) => {
    const col = document.createElement("div");
    col.className = "hist-col";

    const heightPct = Math.max(4, Math.round((b.count / maxCount) * 100));
    const isAboveThreshold = idx >= 7; // >= 0.70

    col.innerHTML = `
      <div style="font-size: 10px; font-weight: 600; color: var(--text-muted); margin-bottom: 2px;">${b.count}</div>
      <div class="hist-bar ${isAboveThreshold ? "flagged-bar" : ""}" style="height: ${heightPct}%;" title="${b.range}: ${b.count} txs (${b.percentage}%)"></div>
      <div class="hist-label">${b.range.split("–")[0]}</div>
    `;
    container.appendChild(col);
  });
}

function renderRecentFlagged(flaggedTxs) {
  const tbody = document.getElementById("overview-flagged-tbody");
  if (!tbody) return;

  if (flaggedTxs.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--text-muted); padding: 24px;">No transactions exceeded the 0.7372 threshold in this batch.</td></tr>`;
    return;
  }

  tbody.innerHTML = flaggedTxs
    .map(
      (tx) => `
    <tr>
      <td class="mono"><strong>${tx.id}</strong></td>
      <td class="mono">${formatTimestamp(tx.timestamp)}</td>
      <td class="mono">${tx.from_account}</td>
      <td class="mono">${tx.to_account}</td>
      <td class="mono" style="font-weight: 600;">$${Number(tx.amount).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
      <td>
        <div class="score-bar-wrapper">
          <div class="score-bar"><div class="score-fill flagged" style="width: ${Math.round(tx.risk_score * 100)}%;"></div></div>
          <span class="score-num text-danger">${Number(tx.risk_score).toFixed(4)}</span>
        </div>
      </td>
      <td><span class="risk-badge flagged">FLAGGED</span></td>
      <td><button class="btn btn-secondary btn-sm" onclick="inspectTransaction('${tx.id}')">Investigate</button></td>
    </tr>
  `
    )
    .join("");
}

// ============================================================================
// 2. NEW ANALYSIS STEPPER
// ============================================================================

function initAnalysisStepper() {
  const dropzone = document.getElementById("csv-dropzone");
  const fileInput = document.getElementById("csv-file-input");
  const browseBtn = document.getElementById("btn-browse-file");
  const demoBatchBtn = document.getElementById("btn-load-demo-batch");
  const proceedValidateBtn = document.getElementById("btn-proceed-to-validate");
  const backUploadBtn = document.getElementById("btn-back-to-upload");
  const proceedAnalyseBtn = document.getElementById("btn-proceed-to-analyse");
  const startScoringBtn = document.getElementById("btn-start-model-scoring");
  const step4ResultsBtn = document.getElementById("btn-step4-view-results");
  const step4GraphBtn = document.getElementById("btn-step4-view-graph");

  if (dropzone && fileInput) {
    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzone.classList.add("drag-over");
    });
    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag-over"));
    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.classList.remove("drag-over");
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFileSelected(e.dataTransfer.files[0]);
      }
    });
  }

  if (browseBtn && fileInput) {
    browseBtn.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", () => {
      if (fileInput.files && fileInput.files[0]) {
        handleFileSelected(fileInput.files[0]);
      }
    });
  }

  if (demoBatchBtn) {
    demoBatchBtn.addEventListener("click", async () => {
      await loadDemoSampleBatch(50);
    });
  }

  if (proceedValidateBtn) {
    proceedValidateBtn.addEventListener("click", () => runValidationStep());
  }

  if (backUploadBtn) {
    backUploadBtn.addEventListener("click", () => setStepperStep(1));
  }

  if (proceedAnalyseBtn) {
    proceedAnalyseBtn.addEventListener("click", () => setStepperStep(3));
  }

  if (startScoringBtn) {
    startScoringBtn.addEventListener("click", () => executeBatchScoring());
  }

  if (step4ResultsBtn) {
    step4ResultsBtn.addEventListener("click", () => switchTab("view-results"));
  }

  if (step4GraphBtn) {
    step4GraphBtn.addEventListener("click", () => switchTab("view-investigation"));
  }
}

function handleFileSelected(file) {
  if (!file.name.endsWith(".csv")) {
    alert("Please select a valid .csv file.");
    return;
  }

  state.uploadedFilename = file.name;
  const reader = new FileReader();
  reader.onload = (e) => {
    state.uploadedCsvContent = e.target.result;
    document.getElementById("file-name-display").textContent = file.name;
    document.getElementById("file-size-display").textContent = `${(file.size / 1024).toFixed(1)} KB`;
    document.getElementById("selected-file-summary").style.display = "flex";
  };
  reader.readAsText(file);
}

async function loadDemoSampleBatch(count = 50) {
  try {
    const res = await fetch(`/api/analysis/sample?count=${count}`);
    if (!res.ok) throw new Error("Sample generation failed");
    const data = await res.json();

    state.uploadedFilename = data.filename;
    state.uploadedCsvContent = data.csv_content;

    document.getElementById("file-name-display").textContent = data.filename;
    document.getElementById("file-size-display").textContent = `${(data.csv_content.length / 1024).toFixed(1)} KB (${data.row_count} rows)`;
    document.getElementById("selected-file-summary").style.display = "flex";

    switchTab("view-analysis");
    await runValidationStep();
  } catch (err) {
    alert(`Could not load benchmark sample: ${err.message}`);
  }
}

function resetStepperToStep1() {
  setStepperStep(1);
  state.uploadedCsvContent = null;
  state.uploadedFilename = null;
  const summary = document.getElementById("selected-file-summary");
  if (summary) summary.style.display = "none";
}

function setStepperStep(stepNumber) {
  for (let i = 1; i <= 4; i++) {
    const nav = document.getElementById(`step-nav-${i}`);
    const container = document.getElementById(`step-container-${i}`);
    if (nav) {
      nav.classList.toggle("active", i === stepNumber);
      nav.classList.toggle("completed", i < stepNumber);
    }
    if (container) {
      container.style.display = i === stepNumber ? "block" : "none";
    }
  }
}

async function runValidationStep() {
  if (!state.uploadedCsvContent) {
    alert("Please upload a CSV file or load a sample first.");
    return;
  }

  setStepperStep(2);
  const output = document.getElementById("validation-diagnostic-output");
  const proceedBtn = document.getElementById("btn-proceed-to-analyse");
  output.innerHTML = `<div style="text-align: center; padding: 24px; color: var(--text-muted);">Validating schema and transaction causal structure...</div>`;
  proceedBtn.disabled = true;

  try {
    const res = await fetch("/api/analysis/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        filename: state.uploadedFilename || "upload.csv",
        csv_content: state.uploadedCsvContent,
      }),
    });
    if (!res.ok) throw new Error("Validation service error");
    const val = await res.json();

    let html = "";
    if (val.valid) {
      proceedBtn.disabled = false;
      html += `
        <div style="background: var(--color-success-bg); border: 1px solid var(--color-success-border); padding: 14px 18px; border-radius: var(--radius-sm); margin-bottom: var(--space-4); display: flex; align-items: center; gap: var(--space-3);">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#059669" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
          <div>
            <strong style="color: #065F46;">Schema Validation Succeeded</strong>
            <div style="font-size: 12px; color: #047857;">Format: ${val.format_detected.toUpperCase()} &bull; Total valid rows: ${val.valid_rows.toLocaleString()}</div>
          </div>
        </div>
      `;
    } else {
      html += `
        <div style="background: var(--color-flagged-bg); border: 1px solid var(--color-flagged-border); padding: 14px 18px; border-radius: var(--radius-sm); margin-bottom: var(--space-4);">
          <strong style="color: var(--color-flagged);">Validation Errors Detected</strong>
          <ul style="font-size: 13px; color: var(--color-flagged); margin-top: 6px; padding-left: 18px;">
            ${val.errors.map((e) => `<li>${e.column ? `<strong>[${e.column}]</strong> ` : ""}${e.message}</li>`).join("")}
          </ul>
        </div>
      `;
    }

    if (val.warnings && val.warnings.length > 0) {
      html += `
        <div style="background: var(--color-warning-bg); border: 1px solid var(--color-warning-border); padding: 12px 16px; border-radius: var(--radius-sm); margin-bottom: var(--space-4);">
          <strong style="color: var(--color-warning); font-size: 12px;">Operational Warnings:</strong>
          <ul style="font-size: 12px; color: #92400E; margin-top: 4px; padding-left: 18px;">
            ${val.warnings.map((w) => `<li>${w}</li>`).join("")}
          </ul>
        </div>
      `;
    }

    // Sample preview
    if (val.sample_preview && val.sample_preview.length > 0) {
      const keys = Object.keys(val.sample_preview[0]);
      html += `
        <div style="margin-top: var(--space-4);">
          <div style="font-size: 12px; font-weight: 600; color: var(--text-muted); margin-bottom: 6px;">Data Preview (First ${val.sample_preview.length} Rows):</div>
          <div class="table-responsive" style="max-height: 200px;">
            <table class="data-table">
              <thead><tr>${keys.map((k) => `<th>${k}</th>`).join("")}</tr></thead>
              <tbody>
                ${val.sample_preview
                  .map(
                    (row) => `
                  <tr>${keys.map((k) => `<td class="mono">${row[k] !== undefined ? row[k] : ""}</td>`).join("")}</tr>
                `
                  )
                  .join("")}
              </tbody>
            </table>
          </div>
        </div>
      `;
    }

    output.innerHTML = html;
  } catch (err) {
    output.innerHTML = `<div style="color: var(--color-flagged); padding: 16px;">Validation failed: ${err.message}</div>`;
  }
}

async function executeBatchScoring() {
  const readyDiv = document.getElementById("analysis-ready-state");
  const loadingDiv = document.getElementById("analysis-loading-state");

  readyDiv.style.display = "none";
  loadingDiv.style.display = "block";

  try {
    const res = await fetch("/api/analysis/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        filename: state.uploadedFilename || "upload.csv",
        csv_content: state.uploadedCsvContent,
      }),
    });

    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.detail || "Analysis execution failed.");
    }

    const data = await res.json();
    state.currentSessionId = data.session_id;
    updateHeaderSession(data.session_id, data.threshold);

    // Populate Step 4
    document.getElementById("step4-kpi-total").textContent = data.total_transactions.toLocaleString();
    document.getElementById("step4-kpi-flagged").textContent = data.flagged_transactions.toLocaleString();
    document.getElementById("step4-kpi-rate").textContent = `${data.flagged_rate}%`;
    document.getElementById("step4-kpi-duration").textContent = `${data.duration_seconds}s`;

    setStepperStep(4);
  } catch (err) {
    alert(`Batch analysis error: ${err.message}`);
    readyDiv.style.display = "block";
    loadingDiv.style.display = "none";
  }
}

// ============================================================================
// 3. DETECTION RESULTS TABLE
// ============================================================================

function initResultsTable() {
  const searchInput = document.getElementById("table-search-input");
  const flagFilter = document.getElementById("table-flag-filter");
  const sortFilter = document.getElementById("table-sort-filter");
  const prevBtn = document.getElementById("btn-prev-page");
  const nextBtn = document.getElementById("btn-next-page");
  const exportBtn = document.getElementById("btn-export-results-csv");

  let debounceTimer;
  if (searchInput) {
    searchInput.addEventListener("input", () => {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        state.resultsSearch = searchInput.value.trim();
        loadResultsTable(1);
      }, 300);
    });
  }

  if (flagFilter) {
    flagFilter.addEventListener("change", () => {
      state.resultsFlagFilter = flagFilter.value;
      loadResultsTable(1);
    });
  }

  if (sortFilter) {
    sortFilter.addEventListener("change", () => {
      state.resultsSort = sortFilter.value;
      loadResultsTable(1);
    });
  }

  if (prevBtn) {
    prevBtn.addEventListener("click", () => {
      if (state.resultsPage > 1) loadResultsTable(state.resultsPage - 1);
    });
  }

  if (nextBtn) {
    nextBtn.addEventListener("click", () => {
      loadResultsTable(state.resultsPage + 1);
    });
  }

  if (exportBtn) {
    exportBtn.addEventListener("click", () => {
      if (!state.currentSessionId) {
        alert("No active analysis session to export.");
        return;
      }
      window.location.href = `/api/analysis/${encodeURIComponent(state.currentSessionId)}/export`;
    });
  }
}

async function loadResultsTable(page = 1) {
  state.resultsPage = page;
  const tbody = document.getElementById("results-table-tbody");
  if (!tbody) return;

  if (!state.currentSessionId) {
    tbody.innerHTML = `<tr><td colspan="9" style="text-align: center; color: var(--text-muted); padding: 36px;">No analysis loaded. Upload transactions to view results.</td></tr>`;
    return;
  }

  tbody.innerHTML = `<tr><td colspan="9" style="text-align: center; color: var(--text-muted); padding: 24px;">Loading paginated results...</td></tr>`;

  try {
    const [sortCol, sortDir] = state.resultsSort.split(":");
    const params = new URLSearchParams({
      page: page,
      limit: state.resultsLimit,
      sort_by: sortCol || "risk_score",
      sort_dir: sortDir || "desc",
    });

    if (state.resultsSearch) params.append("search", state.resultsSearch);
    if (state.resultsFlagFilter) params.append("flag_filter", state.resultsFlagFilter);

    const res = await fetch(`/api/analysis/${encodeURIComponent(state.currentSessionId)}/results?${params.toString()}`);
    if (!res.ok) throw new Error("Results query failed");
    const data = await res.json();

    const items = data.items || [];
    if (items.length === 0) {
      tbody.innerHTML = `<tr><td colspan="9" style="text-align: center; color: var(--text-muted); padding: 36px;">No matching transactions found for filter criteria.</td></tr>`;
      document.getElementById("pagination-info").textContent = "Showing 0 of 0 records";
      document.getElementById("btn-prev-page").disabled = true;
      document.getElementById("btn-next-page").disabled = true;
      return;
    }

    tbody.innerHTML = items
      .map((tx) => {
        const isFlagged = tx.classification === "FLAGGED";
        const scorePct = Math.min(100, Math.round(tx.risk_score * 100));

        let groundTruthBadge = `<span class="mono" style="color: var(--text-muted); font-size: 11px;">Unlabeled</span>`;
        if (tx.ground_truth === 1) {
          groundTruthBadge = `<span class="risk-badge flagged" title="IBM AML Dataset Verified Label">LAUNDERING (1)</span>`;
        } else if (tx.ground_truth === 0) {
          groundTruthBadge = `<span class="risk-badge normal">CLEAN (0)</span>`;
        }

        return `
        <tr>
          <td class="mono"><strong>${tx.id}</strong></td>
          <td class="mono" style="font-size: 11px;">${tx.timestamp_str || formatTimestamp(tx.timestamp)}</td>
          <td class="mono">${tx.from_account}</td>
          <td class="mono">${tx.to_account}</td>
          <td class="mono" style="font-weight: 600;">$${Number(tx.amount).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
          <td>
            <div class="score-bar-wrapper">
              <div class="score-bar"><div class="score-fill ${isFlagged ? "flagged" : ""}" style="width: ${scorePct}%;"></div></div>
              <span class="score-num ${isFlagged ? "text-danger" : ""}">${Number(tx.risk_score).toFixed(4)}</span>
            </div>
          </td>
          <td>
            <span class="risk-badge ${isFlagged ? "flagged" : "normal"}">
              ${isFlagged ? "FLAGGED" : "BELOW THRESHOLD"}
            </span>
          </td>
          <td>${groundTruthBadge}</td>
          <td>
            <button class="btn btn-secondary btn-sm" onclick="inspectTransaction('${tx.id}')">Investigate</button>
          </td>
        </tr>
      `;
      })
      .join("");

    // Update Pagination Toolbar
    const startIdx = (data.page - 1) * data.limit + 1;
    const endIdx = Math.min(data.total, data.page * data.limit);
    document.getElementById("pagination-info").textContent = `Showing ${startIdx}–${endIdx} of ${data.total} records`;
    document.getElementById("current-page-display").textContent = `Page ${data.page} of ${data.pages}`;
    document.getElementById("btn-prev-page").disabled = data.page <= 1;
    document.getElementById("btn-next-page").disabled = data.page >= data.pages;
    document.getElementById("results-count-summary").textContent = `Analysis total: ${data.total} records`;
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="9" style="text-align: center; color: var(--color-flagged); padding: 24px;">Failed to load results: ${err.message}</td></tr>`;
  }
}

// Global inspect function wired to Investigate button
window.inspectTransaction = function (txId) {
  state.selectedTransactionId = txId;
  switchTab("view-explanation");
  loadTransactionExplanation(txId);
};

// ============================================================================
// 4. CONNECTED ACCOUNTS GRAPH (CANVAS FORCE-DIRECTED)
// ============================================================================

let canvas, ctx, simulationTimer;

function initGraphCanvas() {
  canvas = document.getElementById("graph-canvas");
  if (!canvas) return;
  ctx = canvas.getContext("2d");

  const scoreSlider = document.getElementById("graph-score-slider");
  const scoreVal = document.getElementById("graph-score-val");
  const resetBtn = document.getElementById("btn-reset-graph");

  if (scoreSlider && scoreVal) {
    scoreSlider.addEventListener("input", () => {
      scoreVal.textContent = Number(scoreSlider.value).toFixed(2);
      loadGraph(parseFloat(scoreSlider.value));
    });
  }

  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      if (scoreSlider) scoreSlider.value = "0";
      if (scoreVal) scoreVal.textContent = "0.00";
      loadGraph(0);
    });
  }

  // Handle canvas clicks
  canvas.addEventListener("click", (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Find clicked node
    for (const node of state.graphNodes) {
      const dist = Math.hypot(node.x - x, node.y - y);
      if (dist <= 15) {
        state.selectedNode = node;
        inspectAccount(node.id);
        drawGraph();
        return;
      }
    }
  });
}

async function loadGraph(minScore = 0) {
  if (!canvas) return;

  // Responsive sizing
  const wrapper = canvas.parentElement;
  canvas.width = wrapper.clientWidth || 800;
  canvas.height = 540;

  if (!state.currentSessionId) {
    drawEmptyGraph("No active analysis. Upload transactions to view connected accounts network.");
    return;
  }

  try {
    const res = await fetch(`/api/analysis/${encodeURIComponent(state.currentSessionId)}/graph?min_score=${minScore}`);
    if (!res.ok) throw new Error("Graph query failed");
    const data = await res.json();

    const rawNodes = data.nodes || [];
    const rawEdges = data.edges || [];

    if (rawNodes.length === 0) {
      drawEmptyGraph("No accounts match the selected edge score threshold.");
      return;
    }

    // Initialize layout positions
    const width = canvas.width;
    const height = canvas.height;
    const cx = width / 2;
    const cy = height / 2;

    state.graphNodes = rawNodes.map((n, i) => {
      const angle = (i / rawNodes.length) * 2 * Math.PI;
      const radius = 100 + Math.random() * 140;
      return {
        ...n,
        x: cx + Math.cos(angle) * radius,
        y: cy + Math.sin(angle) * radius,
        vx: 0,
        vy: 0,
      };
    });

    state.graphEdges = rawEdges;

    // Run short relaxation simulation
    runForceSimulation();
  } catch (err) {
    drawEmptyGraph(`Failed to load graph: ${err.message}`);
  }
}

function runForceSimulation() {
  let iterations = 40;
  clearInterval(simulationTimer);

  simulationTimer = setInterval(() => {
    if (iterations-- <= 0) {
      clearInterval(simulationTimer);
      drawGraph();
      return;
    }

    const width = canvas.width;
    const height = canvas.height;
    const cx = width / 2;
    const cy = height / 2;

    // Repulsion
    for (let i = 0; i < state.graphNodes.length; i++) {
      for (let j = i + 1; j < state.graphNodes.length; j++) {
        const n1 = state.graphNodes[i];
        const n2 = state.graphNodes[j];
        const dx = n2.x - n1.x;
        const dy = n2.y - n1.y;
        const dist = Math.max(15, Math.hypot(dx, dy));
        const force = 400 / (dist * dist);
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;
        n1.vx -= fx;
        n1.vy -= fy;
        n2.vx += fx;
        n2.vy += fy;
      }
    }

    // Edge attraction
    const nodeMap = new Map(state.graphNodes.map((n) => [n.id, n]));
    for (const e of state.graphEdges) {
      const n1 = nodeMap.get(e.source);
      const n2 = nodeMap.get(e.target);
      if (n1 && n2) {
        const dx = n2.x - n1.x;
        const dy = n2.y - n1.y;
        const dist = Math.max(1, Math.hypot(dx, dy));
        const force = (dist - 80) * 0.04;
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;
        n1.vx += fx;
        n1.vy += fy;
        n2.vx -= fx;
        n2.vy -= fy;
      }
    }

    // Apply & gravity
    for (const n of state.graphNodes) {
      n.vx += (cx - n.x) * 0.02;
      n.vy += (cy - n.y) * 0.02;
      n.x += Math.max(-12, Math.min(12, n.vx));
      n.y += Math.max(-12, Math.min(12, n.vy));
      n.vx *= 0.6;
      n.vy *= 0.6;
      n.x = Math.max(30, Math.min(width - 30, n.x));
      n.y = Math.max(30, Math.min(height - 30, n.y));
    }

    drawGraph();
  }, 20);
}

function drawGraph() {
  if (!ctx || !canvas) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const nodeMap = new Map(state.graphNodes.map((n) => [n.id, n]));

  // Draw Edges
  for (const e of state.graphEdges) {
    const s = nodeMap.get(e.source);
    const t = nodeMap.get(e.target);
    if (!s || !t) continue;

    const isFlagged = e.max_risk_score >= 0.7372;
    ctx.strokeStyle = isFlagged ? "#DC2626" : "rgba(37, 99, 235, 0.35)";
    ctx.lineWidth = isFlagged ? 2.5 : 1.2;

    ctx.beginPath();
    ctx.moveTo(s.x, s.y);
    ctx.lineTo(t.x, t.y);
    ctx.stroke();

    // Directional Arrow Head
    const angle = Math.atan2(t.y - s.y, t.x - s.x);
    const arrowDist = 18;
    const ax = t.x - Math.cos(angle) * arrowDist;
    const ay = t.y - Math.sin(angle) * arrowDist;

    ctx.fillStyle = ctx.strokeStyle;
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(ax - 7 * Math.cos(angle - Math.PI / 6), ay - 7 * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(ax - 7 * Math.cos(angle + Math.PI / 6), ay - 7 * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.fill();
  }

  // Draw Nodes
  for (const n of state.graphNodes) {
    const isSelected = state.selectedNode && state.selectedNode.id === n.id;
    const isFlagged = n.is_flagged;

    ctx.beginPath();
    ctx.arc(n.x, n.y, isSelected ? 12 : 9, 0, 2 * Math.PI);
    ctx.fillStyle = isFlagged ? "#DC2626" : "#2563EB";
    ctx.fill();

    ctx.strokeStyle = isSelected ? "#0F172A" : "#FFFFFF";
    ctx.lineWidth = isSelected ? 3 : 2;
    ctx.stroke();

    // Label
    ctx.font = "10px Inter, sans-serif";
    ctx.fillStyle = "#0F172A";
    ctx.textAlign = "center";
    ctx.fillText(n.id, n.x, n.y + 20);
  }
}

function drawEmptyGraph(msg) {
  if (!ctx || !canvas) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#64748B";
  ctx.font = "13px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(msg, canvas.width / 2, canvas.height / 2);
}

async function inspectAccount(accountId) {
  const title = document.getElementById("panel-entity-title");
  const content = document.getElementById("panel-entity-content");
  if (!title || !content) return;

  title.textContent = `Account: ${accountId}`;
  content.innerHTML = `<div style="text-align: center; color: var(--text-muted); padding: 16px;">Loading account history...</div>`;

  try {
    const res = await fetch(`/api/analysis/${encodeURIComponent(state.currentSessionId)}/account/${encodeURIComponent(accountId)}`);
    if (!res.ok) throw new Error("Account query failed");
    const acc = await res.json();

    content.innerHTML = `
      <div style="display: flex; flex-direction: column; gap: 12px; font-size: 13px;">
        <div style="background: var(--bg-card-subtle); padding: 12px; border-radius: var(--radius-sm); border: 1px solid var(--border-card);">
          <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
            <span style="color: var(--text-muted); font-size: 11px;">ACTIVITY SUMMARY</span>
            <span class="risk-badge ${acc.flagged_transactions > 0 ? "flagged" : "normal"}">${acc.flagged_transactions} Flagged</span>
          </div>
          <div><strong>Total Transactions:</strong> ${acc.total_transactions}</div>
          <div><strong>Total Volume Sent:</strong> $${Number(acc.total_sent).toLocaleString(undefined, { minimumFractionDigits: 2 })}</div>
          <div><strong>Total Volume Received:</strong> $${Number(acc.total_received).toLocaleString(undefined, { minimumFractionDigits: 2 })}</div>
        </div>

        <div>
          <div style="font-size: 11px; font-weight: 600; color: var(--text-muted); margin-bottom: 6px;">CONNECTED COUNTERPARTIES (${acc.counterparties.length})</div>
          <div style="display: flex; flex-wrap: wrap; gap: 4px;">
            ${acc.counterparties.map((c) => `<span class="session-pill mono">${c}</span>`).join("")}
          </div>
        </div>

        <div>
          <div style="font-size: 11px; font-weight: 600; color: var(--text-muted); margin-bottom: 6px;">TRANSACTION LOG</div>
          <div style="display: flex; flex-direction: column; gap: 6px; max-height: 240px; overflow-y: auto;">
            ${acc.transactions
              .map(
                (t) => `
              <div style="padding: 8px; border: 1px solid var(--border-subtle); border-radius: 4px; display: flex; justify-content: space-between; align-items: center; font-size: 12px;">
                <div>
                  <div class="mono" style="font-weight: 600;">${t.id} &bull; $${Number(t.amount).toLocaleString()}</div>
                  <div style="font-size: 10px; color: var(--text-muted);">${t.from_account === accountId ? `To &rarr; ${t.to_account}` : `From &larr; ${t.from_account}`}</div>
                </div>
                <div style="text-align: right;">
                  <span class="mono ${t.classification === "FLAGGED" ? "text-danger" : ""}" style="font-weight: 600;">${Number(t.risk_score).toFixed(4)}</span>
                  <div><button class="btn btn-secondary btn-sm" style="padding: 2px 6px; font-size: 10px;" onclick="inspectTransaction('${t.id}')">View</button></div>
                </div>
              </div>
            `
              )
              .join("")}
          </div>
        </div>
      </div>
    `;
  } catch (err) {
    content.innerHTML = `<div style="color: var(--color-flagged); font-size: 12px;">Failed to inspect account: ${err.message}</div>`;
  }
}

// ============================================================================
// 5. TRANSACTION EXPLANATION & ATTENTION
// ============================================================================

async function loadTransactionExplanation(txId) {
  const expTxId = document.getElementById("exp-tx-id");
  const expTime = document.getElementById("exp-tx-time");
  const expSender = document.getElementById("exp-tx-sender");
  const expReceiver = document.getElementById("exp-tx-receiver");
  const expAmount = document.getElementById("exp-tx-amount");
  const expScore = document.getElementById("exp-tx-score");
  const attnBody = document.getElementById("exp-attention-body");
  const evBody = document.getElementById("exp-evidence-body");

  try {
    const res = await fetch(`/api/analysis/${encodeURIComponent(state.currentSessionId)}/transaction/${encodeURIComponent(txId)}`);
    if (!res.ok) throw new Error("Transaction fetch error");
    const tx = await res.json();

    // Populate Header
    expTxId.textContent = tx.id;
    expTime.textContent = tx.timestamp_str || formatTimestamp(tx.timestamp);
    expSender.textContent = tx.from_account;
    expReceiver.textContent = tx.to_account;
    expAmount.textContent = `$${Number(tx.amount).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    expScore.textContent = Number(tx.risk_score).toFixed(4);
    expScore.style.color = tx.classification === "FLAGGED" ? "var(--color-flagged)" : "var(--text-primary)";

    // Section A: Model Attribution
    if (!tx.has_history) {
      attnBody.innerHTML = `
        <div style="padding: 16px; background: var(--color-warning-bg); border: 1px solid var(--color-warning-border); border-radius: var(--radius-sm); font-size: 13px; color: #92400E;">
          <strong>Model attribution not available (no prior temporal neighbors).</strong>
          <p style="margin-top: 4px; font-size: 12px;">This transaction involves isolated accounts with zero prior interactions recorded before this timestamp. Score relies entirely on tabular transaction properties.</p>
        </div>
      `;
    } else {
      const senderNeighbors = tx.sender_attention.neighbors || [];
      const receiverNeighbors = tx.receiver_attention.neighbors || [];

      attnBody.innerHTML = `
        <div class="attention-grid">
          <!-- Sender Attention -->
          <div class="attention-card">
            <div style="font-size: 12px; font-weight: 600; margin-bottom: 8px;">Sender Account Counterparties (${senderNeighbors.length})</div>
            ${
              senderNeighbors.length === 0
                ? `<div style="font-size: 12px; color: var(--text-muted);">No prior sender interactions.</div>`
                : senderNeighbors
                    .map(
                      (n) => `
              <div class="neighbor-attention-row">
                <div>
                  <div class="neighbor-partner">${n.partner}</div>
                  <div style="font-size: 10px; color: var(--text-muted);">&Delta;t: ${formatDelta(n.delta_seconds)}</div>
                </div>
                <div style="text-align: right;">
                  <span class="attention-pill">&alpha;<sub>L2</sub>: ${(n.attention_l2 * 100).toFixed(1)}%</span>
                  <div style="font-size: 10px; color: var(--text-muted);">&alpha;<sub>L1</sub>: ${(n.attention_l1 * 100).toFixed(1)}%</div>
                </div>
              </div>
            `
                    )
                    .join("")
            }
          </div>

          <!-- Receiver Attention -->
          <div class="attention-card">
            <div style="font-size: 12px; font-weight: 600; margin-bottom: 8px;">Receiver Account Counterparties (${receiverNeighbors.length})</div>
            ${
              receiverNeighbors.length === 0
                ? `<div style="font-size: 12px; color: var(--text-muted);">No prior receiver interactions.</div>`
                : receiverNeighbors
                    .map(
                      (n) => `
              <div class="neighbor-attention-row">
                <div>
                  <div class="neighbor-partner">${n.partner}</div>
                  <div style="font-size: 10px; color: var(--text-muted);">&Delta;t: ${formatDelta(n.delta_seconds)}</div>
                </div>
                <div style="text-align: right;">
                  <span class="attention-pill">&alpha;<sub>L2</sub>: ${(n.attention_l2 * 100).toFixed(1)}%</span>
                  <div style="font-size: 10px; color: var(--text-muted);">&alpha;<sub>L1</sub>: ${(n.attention_l1 * 100).toFixed(1)}%</div>
                </div>
              </div>
            `
                    )
                    .join("")
            }
          </div>
        </div>
        <div style="font-size: 11px; color: var(--text-muted); margin-top: 10px;">
          * &alpha;<sub>L1</sub> and &alpha;<sub>L2</sub> denote learned softmax attention weights over past counterparty edges in TGAT multi-head attention.
        </div>
      `;
    }

    // Section B: Empirical Evidence
    const evidenceList = tx.explainability.evidence || [];
    evBody.innerHTML = `
      <div style="margin-bottom: 12px;">
        <div style="font-size: 12px; font-weight: 600; color: var(--text-muted);">SUMMARY ATTRIBUTION:</div>
        <div style="font-size: 13px; font-weight: 500; color: var(--text-primary); margin-top: 2px;">${tx.explainability.summary}</div>
      </div>
      <div style="font-size: 12px; font-weight: 600; color: var(--text-muted); margin-bottom: 6px;">FACTUAL TRANSACTION CONTEXT:</div>
      <div class="evidence-list">
        ${evidenceList.map((e) => `<div class="evidence-item"><span class="evidence-bullet"></span><span>${e}</span></div>`).join("")}
      </div>
      <div style="font-size: 11px; color: var(--text-muted); margin-top: 14px; font-style: italic;">
        Note: Empirical attributes are factual behavioral markers extracted from chronological transaction logs, provided strictly as investigator context.
      </div>
    `;
  } catch (err) {
    attnBody.innerHTML = `<div style="color: var(--color-flagged); font-size: 12px;">Failed to load explanation: ${err.message}</div>`;
  }
}

// ============================================================================
// 6. MODEL PERFORMANCE VIEW
// ============================================================================

function initPerformanceView() {
  // Handled dynamically on tab entry
}

async function loadPerformance() {
  try {
    const [metricsRes, ablRes] = await Promise.all([
      fetch("/api/metrics"),
      fetch("/api/ablations"),
    ]);

    if (!metricsRes.ok || !ablRes.ok) return;

    const metricsData = await metricsRes.json();
    const ablData = await ablRes.json();

    // Baseline Table
    const baselinesTbody = document.getElementById("perf-baselines-tbody");
    if (baselinesTbody) {
      const b = metricsData.baselines || {};
      const tgatFrozen = metricsData.tgat_frozen_test || {};
      const tgatOfficial = metricsData.tgat_official_metrics || {};

      baselinesTbody.innerHTML = `
        <tr style="background-color: var(--accent-blue-subtle);">
          <td><strong>TGAT (Frozen &tau;* = 0.7372)</strong></td>
          <td><span class="mono" style="font-weight: 700; color: var(--color-success);">0.9022</span></td>
          <td><span class="mono" style="font-weight: 700; color: var(--color-success);">0.0184</span></td>
          <td><span class="mono" style="font-weight: 700; color: var(--color-success);">0.0504</span></td>
          <td class="mono"><strong>4.55%</strong></td>
          <td class="mono">5.65%</td>
          <td class="mono" style="color: var(--color-success); font-weight: 700;">357</td>
          <td><span class="risk-badge" style="background: #E0E7FF; color: #3730A3;">Dynamic Edge Attention</span></td>
        </tr>
        <tr>
          <td>TGAT (Default Threshold 0.50)</td>
          <td class="mono">0.9022</td>
          <td class="mono">0.0184</td>
          <td class="mono">0.0364</td>
          <td class="mono">1.91%</td>
          <td class="mono">42.86%</td>
          <td class="mono">2,198</td>
          <td><span class="risk-badge" style="background: #E0E7FF; color: #3730A3;">Dynamic Edge Attention</span></td>
        </tr>
        <tr>
          <td>Temporal XGBoost</td>
          <td class="mono">0.9175</td>
          <td class="mono">0.0280</td>
          <td class="mono">0.0781</td>
          <td class="mono">4.16%</td>
          <td class="mono">58.47%</td>
          <td class="mono">4,040</td>
          <td>Global Feature Importance (Black-box)</td>
        </tr>
        <tr>
          <td>Tabular XGBoost</td>
          <td class="mono">0.8901</td>
          <td class="mono">0.0305</td>
          <td class="mono">0.0180</td>
          <td class="mono">0.91%</td>
          <td class="mono">80.40%</td>
          <td class="mono">26,399</td>
          <td>Global Feature Importance (Black-box)</td>
        </tr>
        <tr>
          <td>Logistic Regression</td>
          <td class="mono">0.8270</td>
          <td class="mono">0.0049</td>
          <td class="mono">0.0023</td>
          <td class="mono">0.11%</td>
          <td class="mono">99.67%</td>
          <td class="mono">263,840</td>
          <td>Linear Feature Weights</td>
        </tr>
        <tr style="background-color: #FEF2F2;">
          <td><strong style="color: #991B1B;">Static GCN (FAILED)</strong></td>
          <td class="mono" style="color: #DC2626;">0.6064</td>
          <td class="mono" style="color: #DC2626;">0.0017</td>
          <td class="mono" style="color: #DC2626;">0.0038</td>
          <td class="mono">0.19%</td>
          <td class="mono">7.64%</td>
          <td class="mono">1,177</td>
          <td style="color: #991B1B; font-size: 11px;">Failed: Collapsing time causes future leakage & topological smearing</td>
        </tr>
      `;
    }

    // Ablations Table
    const ablTbody = document.getElementById("perf-ablations-tbody");
    if (ablTbody && ablData.experiments) {
      ablTbody.innerHTML = ablData.experiments
        .map(
          (r) => `
        <tr>
          <td class="mono"><strong>${r.id}</strong></td>
          <td>${getAblationDescription(r.name)}</td>
          <td class="mono">${r.val_roc_auc ? r.val_roc_auc.toFixed(4) : "—"}</td>
          <td class="mono" style="font-weight: 600;">${r.test_roc_auc ? r.test_roc_auc.toFixed(4) : "—"}</td>
          <td class="mono">${r.test_pr_auc ? r.test_pr_auc.toFixed(4) : "—"}</td>
          <td class="mono">${r.test_f1 !== undefined ? r.test_f1.toFixed(4) : "—"}</td>
          <td class="mono">${r.test_p100 !== undefined ? (r.test_p100 * 100).toFixed(1) + "%" : "—"}</td>
          <td class="mono">${r.training_seconds ? Math.round(r.training_seconds) + "s" : "—"}</td>
        </tr>
      `
        )
        .join("");
    }
  } catch (err) {
    console.warn("Failed to load performance metrics:", err);
  }
}

function getAblationDescription(name) {
  if (name.includes("a0_baseline")) return "Full baseline control (K=10, 2-layer, 2-head, d=32)";
  if (name.includes("no_temporal")) return "Drop all 15 temporal state features (-0.0574 ROC-AUC drop)";
  if (name.includes("b_no_neighbors")) return "Isolated accounts (No historical graph neighbors; F1 collapses)";
  if (name.includes("k_1")) return "Query neighborhood budget K=1";
  if (name.includes("k_5")) return "Query neighborhood budget K=5";
  if (name.includes("k_10")) return "Reference neighborhood budget K=10 (Sweet spot)";
  if (name.includes("k_20")) return "Expanded neighborhood budget K=20";
  if (name.includes("no_time_encoding")) return "Remove Bochner harmonic time encoding (F1 collapses)";
  if (name.includes("one_layer")) return "Single attention layer (L=1)";
  if (name.includes("one_head")) return "Single attention head (H=1)";
  if (name.includes("hidden_16")) return "Reduced hidden dimension (d=16)";
  if (name.includes("hidden_64")) return "Expanded hidden dimension (d=64)";
  if (name.includes("dropout_0")) return "Zero dropout regularization (0.9116 ROC-AUC)";
  if (name.includes("dropout_04")) return "Heavy dropout (0.4)";
  return "Architecture variant";
}

// ============================================================================
// HELPER UTILITIES
// ============================================================================

function formatTimestamp(tsSec) {
  if (!tsSec) return "—";
  const d = new Date(tsSec * 1000);
  return d.toISOString().replace("T", " ").substring(0, 19);
}

function formatDelta(deltaSec) {
  if (deltaSec < 60) return `${Math.round(deltaSec)}s`;
  if (deltaSec < 3600) return `${Math.round(deltaSec / 60)}m`;
  return `${(deltaSec / 3600).toFixed(1)}h`;
}
