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
  resultsRiskLevelFilter: "",
  selectedTransactionId: null,
  activeTransactionData: null,
  activeAccountData: null,
  activeSidepanelTab: "reason",
  activeExpTab: "factors",
  graphNodes: [],
  graphEdges: [],
  particles: [],
  isAnimating: true,
  transform: { x: 0, y: 0, scale: 1 },
  selectedNode: null,
  dragNode: null,
  hoveredNode: null,
  isPanning: false,
  panStart: { x: 0, y: 0 },
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
    
    const highRiskAccEl = document.getElementById("kpi-high-risk-accounts");
    if (highRiskAccEl) {
      highRiskAccEl.textContent = (data.high_risk_accounts !== undefined ? data.high_risk_accounts : data.flagged_transactions).toLocaleString();
    }

    // Plain-English Executive Summary Callout
    const execCallout = document.getElementById("overview-exec-callout");
    const execText = document.getElementById("overview-exec-text");
    if (execCallout && execText && data.executive_summary) {
      execCallout.style.display = "flex";
      execCallout.className = `layman-callout ${data.flagged_transactions > 0 ? "high" : "low"}`;
      execText.textContent = data.executive_summary;
    }

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
  const riskFilter = document.getElementById("table-risk-level-filter");
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

  if (riskFilter) {
    riskFilter.addEventListener("change", () => {
      state.resultsRiskLevelFilter = riskFilter.value;
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
    if (state.resultsRiskLevelFilter) params.append("risk_level", state.resultsRiskLevelFilter);

    const res = await fetch(`/api/analysis/${encodeURIComponent(state.currentSessionId)}/results?${params.toString()}`);
    if (!res.ok) throw new Error("Results query failed");
    const data = await res.json();

    // Populate Top Summary Pills (FraudLens Style)
    const sm = data.summary || {};
    const pillTotal = document.getElementById("pill-total-count");
    const pillFlagged = document.getElementById("pill-flagged-count");
    const pillHigh = document.getElementById("pill-high-count");
    const pillMed = document.getElementById("pill-med-count");
    const pillLow = document.getElementById("pill-low-count");
    if (pillTotal) pillTotal.textContent = (sm.total !== undefined ? sm.total : data.total).toLocaleString();
    if (pillFlagged) pillFlagged.textContent = (sm.flagged || 0).toLocaleString();
    if (pillHigh) pillHigh.textContent = (sm.high_risk || 0).toLocaleString();
    if (pillMed) pillMed.textContent = (sm.medium_risk || 0).toLocaleString();
    if (pillLow) pillLow.textContent = (sm.low_risk || 0).toLocaleString();

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
        const isFlagged = tx.classification === "FLAGGED" || tx.risk_score >= 0.7372;
        const scorePct = Math.min(100, Math.round(tx.risk_score * 100));

        let groundTruthBadge = `<span class="mono" style="color: var(--text-muted); font-size: 11px;">Unlabeled</span>`;
        if (tx.ground_truth === 1) {
          groundTruthBadge = `<span class="risk-badge flagged" title="IBM AML Dataset Verified Label">LAUNDERING (1)</span>`;
        } else if (tx.ground_truth === 0) {
          groundTruthBadge = `<span class="risk-badge normal">CLEAN (0)</span>`;
        }

        // FraudLens Card 3 Risk Level Badge
        let riskBadge;
        if (tx.risk_level_code === "high" || tx.risk_score >= 0.7372) {
          riskBadge = `<span class="risk-badge high">High (${Number(tx.risk_score).toFixed(2)})</span>`;
        } else if (tx.risk_level_code === "medium" || tx.risk_score >= 0.40) {
          riskBadge = `<span class="risk-badge medium">Medium (${Number(tx.risk_score).toFixed(2)})</span>`;
        } else {
          riskBadge = `<span class="risk-badge low">Low (${Number(tx.risk_score).toFixed(2)})</span>`;
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
          <td>${riskBadge}</td>
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
// 4. CONNECTED ACCOUNTS GRAPH (CANVAS FORCE-DIRECTED & ANIMATED FLOW)
// ============================================================================

let canvas, ctx, animFrameId, simulationTimer;

function initGraphCanvas() {
  canvas = document.getElementById("graph-canvas");
  if (!canvas) return;
  ctx = canvas.getContext("2d");

  const scoreSlider = document.getElementById("graph-score-slider");
  const scoreVal = document.getElementById("graph-score-val");
  const resetBtn = document.getElementById("btn-reset-graph");
  const zoomInBtn = document.getElementById("btn-graph-zoom-in");
  const zoomOutBtn = document.getElementById("btn-graph-zoom-out");
  const toggleFlowBtn = document.getElementById("btn-toggle-flow");

  if (scoreSlider && scoreVal) {
    scoreSlider.addEventListener("input", () => {
      scoreVal.textContent = Number(scoreSlider.value).toFixed(2);
      loadGraph(parseFloat(scoreSlider.value));
    });
  }

  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      state.transform = { x: 0, y: 0, scale: 1 };
      if (scoreSlider) scoreSlider.value = "0";
      if (scoreVal) scoreVal.textContent = "0.00";
      loadGraph(0);
    });
  }

  if (zoomInBtn) {
    zoomInBtn.addEventListener("click", () => {
      state.transform.scale = Math.min(3.0, state.transform.scale * 1.25);
    });
  }

  if (zoomOutBtn) {
    zoomOutBtn.addEventListener("click", () => {
      state.transform.scale = Math.max(0.4, state.transform.scale * 0.8);
    });
  }

  if (toggleFlowBtn) {
    toggleFlowBtn.addEventListener("click", () => {
      state.isAnimating = !state.isAnimating;
      toggleFlowBtn.classList.toggle("active", state.isAnimating);
      const icon = document.getElementById("flow-btn-icon");
      const text = document.getElementById("flow-btn-text");
      if (icon) icon.textContent = state.isAnimating ? "⏸" : "▶";
      if (text) text.textContent = state.isAnimating ? "Pause Flow" : "Animate Flow";
    });
  }

  // Mouse Interactivity: Pan, Drag, Hover, Zoom
  canvas.addEventListener("mousedown", handleCanvasMouseDown);
  canvas.addEventListener("mousemove", handleCanvasMouseMove);
  window.addEventListener("mouseup", handleCanvasMouseUp);
  canvas.addEventListener("wheel", handleCanvasWheel, { passive: false });

  // Animation Loop for Particle Flow and Halo Pulses
  cancelAnimationFrame(animFrameId);
  function renderLoop() {
    drawGraph();
    animFrameId = requestAnimationFrame(renderLoop);
  }
  renderLoop();
}

function getCanvasMousePos(e) {
  const rect = canvas.getBoundingClientRect();
  const screenX = e.clientX - rect.left;
  const screenY = e.clientY - rect.top;
  const worldX = (screenX - state.transform.x) / state.transform.scale;
  const worldY = (screenY - state.transform.y) / state.transform.scale;
  return { screenX, screenY, worldX, worldY };
}

function handleCanvasMouseDown(e) {
  const { screenX, screenY, worldX, worldY } = getCanvasMousePos(e);

  // Check if clicked a node
  for (let i = state.graphNodes.length - 1; i >= 0; i--) {
    const node = state.graphNodes[i];
    const radius = node.is_flagged ? 18 : 14;
    const dist = Math.hypot(node.x - worldX, node.y - worldY);
    if (dist <= radius + 6) {
      state.dragNode = node;
      state.selectedNode = node;
      inspectAccount(node.id);
      return;
    }
  }

  // Otherwise start canvas panning
  state.isPanning = true;
  state.panStart = { x: screenX - state.transform.x, y: screenY - state.transform.y };
}

function handleCanvasMouseMove(e) {
  const { screenX, screenY, worldX, worldY } = getCanvasMousePos(e);

  if (state.dragNode) {
    state.dragNode.x = worldX;
    state.dragNode.y = worldY;
    return;
  }

  if (state.isPanning) {
    state.transform.x = screenX - state.panStart.x;
    state.transform.y = screenY - state.panStart.y;
    return;
  }

  // Hover detection for cursor
  let hovering = false;
  for (const node of state.graphNodes) {
    const dist = Math.hypot(node.x - worldX, node.y - worldY);
    if (dist <= (node.is_flagged ? 18 : 14) + 6) {
      hovering = true;
      state.hoveredNode = node;
      break;
    }
  }
  if (!hovering) state.hoveredNode = null;
  canvas.style.cursor = hovering ? "pointer" : (state.isPanning ? "grabbing" : "grab");
}

function handleCanvasMouseUp() {
  state.dragNode = null;
  state.isPanning = false;
}

function handleCanvasWheel(e) {
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;

  const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
  const newScale = Math.max(0.4, Math.min(3.0, state.transform.scale * zoomFactor));

  // Zoom centered on cursor position
  state.transform.x = mouseX - (mouseX - state.transform.x) * (newScale / state.transform.scale);
  state.transform.y = mouseY - (mouseY - state.transform.y) * (newScale / state.transform.scale);
  state.transform.scale = newScale;
}

function initParticles() {
  state.particles = [];
  state.graphEdges.forEach((edge, edgeIdx) => {
    const count = edge.is_flagged || edge.max_risk_score >= 0.7372 ? 4 : 2;
    for (let i = 0; i < count; i++) {
      state.particles.push({
        edgeIdx: edgeIdx,
        progress: Math.random(),
        speed: 0.005 + Math.min(0.015, (edge.max_risk_score || 0.2) * 0.012),
      });
    }
  });
}

async function loadGraph(minScore = 0) {
  if (!canvas) return;

  const wrapper = canvas.parentElement;
  canvas.width = wrapper.clientWidth || 800;
  canvas.height = 560;

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
    initParticles();
    runForceSimulation();

    // Auto-select first flagged node or first node if none selected
    if (!state.selectedNode && state.graphNodes.length > 0) {
      const topFlagged = state.graphNodes.find((n) => n.is_flagged || n.max_risk_score >= 0.7372) || state.graphNodes[0];
      state.selectedNode = topFlagged;
      inspectAccount(topFlagged.id);
    }
  } catch (err) {
    drawEmptyGraph(`Failed to load graph: ${err.message}`);
  }
}

function runForceSimulation() {
  let iterations = 45;
  clearInterval(simulationTimer);

  simulationTimer = setInterval(() => {
    if (iterations-- <= 0) {
      clearInterval(simulationTimer);
      return;
    }

    const width = canvas.width;
    const height = canvas.height;
    const cx = width / 2;
    const cy = height / 2;

    // Node-Node Repulsion
    for (let i = 0; i < state.graphNodes.length; i++) {
      for (let j = i + 1; j < state.graphNodes.length; j++) {
        const n1 = state.graphNodes[i];
        const n2 = state.graphNodes[j];
        const dx = n2.x - n1.x;
        const dy = n2.y - n1.y;
        const dist = Math.max(20, Math.hypot(dx, dy));
        const force = 450 / (dist * dist);
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;
        n1.vx -= fx;
        n1.vy -= fy;
        n2.vx += fx;
        n2.vy += fy;
      }
    }

    // Edge Attraction
    const nodeMap = new Map(state.graphNodes.map((n) => [n.id, n]));
    for (const e of state.graphEdges) {
      const n1 = nodeMap.get(e.source);
      const n2 = nodeMap.get(e.target);
      if (n1 && n2) {
        const dx = n2.x - n1.x;
        const dy = n2.y - n1.y;
        const dist = Math.max(1, Math.hypot(dx, dy));
        const force = (dist - 90) * 0.045;
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;
        n1.vx += fx;
        n1.vy += fy;
        n2.vx -= fx;
        n2.vy -= fy;
      }
    }

    // Gravity to center & update positions
    for (const n of state.graphNodes) {
      if (state.dragNode && state.dragNode.id === n.id) continue;
      n.vx += (cx - n.x) * 0.02;
      n.vy += (cy - n.y) * 0.02;
      n.x += Math.max(-14, Math.min(14, n.vx));
      n.y += Math.max(-14, Math.min(14, n.vy));
      n.vx *= 0.65;
      n.vy *= 0.65;
      n.x = Math.max(40, Math.min(width - 40, n.x));
      n.y = Math.max(40, Math.min(height - 40, n.y));
    }
  }, 20);
}

function drawGraph() {
  if (!ctx || !canvas) return;

  const width = canvas.width;
  const height = canvas.height;

  ctx.clearRect(0, 0, width, height);

  if (state.graphNodes.length === 0) {
    drawEmptyGraph("No accounts match the selected edge score threshold.");
    return;
  }

  ctx.save();
  ctx.translate(state.transform.x, state.transform.y);
  ctx.scale(state.transform.scale, state.transform.scale);

  const nodeMap = new Map(state.graphNodes.map((n) => [n.id, n]));
  const now = Date.now();

  // 1. Draw Edges
  state.graphEdges.forEach((e) => {
    const s = nodeMap.get(e.source);
    const t = nodeMap.get(e.target);
    if (!s || !t) return;

    const isFlagged = e.is_flagged || e.max_risk_score >= 0.7372;
    const isMedium = !isFlagged && e.max_risk_score >= 0.40;
    const isEdgeSelected = state.selectedNode && (state.selectedNode.id === s.id || state.selectedNode.id === t.id);

    ctx.beginPath();
    ctx.moveTo(s.x, s.y);
    ctx.lineTo(t.x, t.y);

    if (isFlagged) {
      ctx.strokeStyle = isEdgeSelected ? "rgba(220, 38, 38, 0.95)" : "rgba(220, 38, 38, 0.65)";
      ctx.lineWidth = isEdgeSelected ? 3.5 : 2.5;
    } else if (isMedium) {
      ctx.strokeStyle = isEdgeSelected ? "rgba(245, 158, 11, 0.9)" : "rgba(245, 158, 11, 0.5)";
      ctx.lineWidth = isEdgeSelected ? 2.5 : 1.8;
    } else {
      ctx.strokeStyle = isEdgeSelected ? "rgba(37, 99, 235, 0.8)" : "rgba(148, 163, 184, 0.35)";
      ctx.lineWidth = isEdgeSelected ? 2.0 : 1.2;
    }
    ctx.stroke();

    // Directional Arrow Head
    const angle = Math.atan2(t.y - s.y, t.x - s.x);
    const radius = t.is_flagged || t.max_risk_score >= 0.7372 ? 18 : 14;
    const ax = t.x - Math.cos(angle) * (radius + 3);
    const ay = t.y - Math.sin(angle) * (radius + 3);

    ctx.fillStyle = ctx.strokeStyle;
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(ax - 7 * Math.cos(angle - Math.PI / 6), ay - 7 * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(ax - 7 * Math.cos(angle + Math.PI / 6), ay - 7 * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.fill();
  });

  // 2. Animate Money Flow Particles
  if (state.isAnimating && state.particles.length > 0) {
    state.particles.forEach((p) => {
      const edge = state.graphEdges[p.edgeIdx];
      if (!edge) return;
      const s = nodeMap.get(edge.source);
      const t = nodeMap.get(edge.target);
      if (!s || !t) return;

      p.progress += p.speed;
      if (p.progress > 1) p.progress = 0;

      const px = s.x + (t.x - s.x) * p.progress;
      const py = s.y + (t.y - s.y) * p.progress;

      const isFlagged = edge.is_flagged || edge.max_risk_score >= 0.7372;

      ctx.beginPath();
      ctx.arc(px, py, isFlagged ? 4.0 : 3.0, 0, Math.PI * 2);
      ctx.fillStyle = isFlagged ? "#EF4444" : "#38BDF8";
      ctx.shadowColor = isFlagged ? "#DC2626" : "#0284C7";
      ctx.shadowBlur = 6;
      ctx.fill();
      ctx.shadowBlur = 0;
    });
  }

  // 3. Draw Nodes
  state.graphNodes.forEach((n) => {
    const isSelected = state.selectedNode && state.selectedNode.id === n.id;
    const isHovered = state.hoveredNode && state.hoveredNode.id === n.id;
    const isFlagged = n.is_flagged || n.max_risk_score >= 0.7372;
    const isMedium = !isFlagged && n.max_risk_score >= 0.40;
    const radius = isFlagged ? 18 : (isMedium ? 15 : 13);

    // Pulsing Halo Glow for High-Risk and Medium-Risk Nodes
    if (isFlagged) {
      const pulse = Math.sin(now / 220) * 4;
      ctx.beginPath();
      ctx.arc(n.x, n.y, radius + 5 + pulse, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(220, 38, 38, 0.22)";
      ctx.fill();
    } else if (isMedium) {
      const pulse = Math.sin(now / 300) * 2.5;
      ctx.beginPath();
      ctx.arc(n.x, n.y, radius + 3 + pulse, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(245, 158, 11, 0.18)";
      ctx.fill();
    }

    // Node Core Circle
    ctx.beginPath();
    ctx.arc(n.x, n.y, radius, 0, Math.PI * 2);
    ctx.fillStyle = isFlagged ? "#DC2626" : (isMedium ? "#F59E0B" : "#2563EB");
    ctx.fill();

    // Node Stroke / Selection
    ctx.strokeStyle = isSelected ? "#0F172A" : (isHovered ? "#38BDF8" : "#FFFFFF");
    ctx.lineWidth = isSelected ? 3.5 : (isHovered ? 2.5 : 2);
    ctx.stroke();

    // Node Label
    ctx.font = isSelected ? "bold 11px Inter, sans-serif" : "10px Inter, sans-serif";
    ctx.fillStyle = "#0F172A";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(n.id, n.x, n.y + radius + 5);
  });

  ctx.restore();
}

function drawEmptyGraph(msg) {
  if (!ctx || !canvas) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#64748B";
  ctx.font = "13px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(msg, canvas.width / 2, canvas.height / 2);
}

// Sidepanel Subtab switcher
window.switchSidepanelTab = function (tabName) {
  state.activeSidepanelTab = tabName;
  document.querySelectorAll("#sidepanel-tabs .subnav-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.id === `btn-sidetab-${tabName}`);
  });
  renderSidepanelContent();
};

window.filterTableByAccount = function (accountId) {
  state.resultsSearch = accountId;
  const searchInput = document.getElementById("table-search-input");
  if (searchInput) searchInput.value = accountId;
  switchTab("view-results");
};

async function inspectAccount(accountId) {
  const title = document.getElementById("panel-entity-title");
  const badge = document.getElementById("panel-entity-badge");
  const content = document.getElementById("panel-entity-content");
  const subtabs = document.getElementById("sidepanel-tabs");
  if (!title || !content) return;

  title.textContent = `Account: ${accountId}`;
  content.innerHTML = `<div style="text-align: center; color: var(--text-muted); padding: 24px;">Loading account history...</div>`;

  try {
    const res = await fetch(`/api/analysis/${encodeURIComponent(state.currentSessionId)}/account/${encodeURIComponent(accountId)}`);
    if (!res.ok) throw new Error("Account query failed");
    const acc = await res.json();
    state.activeAccountData = acc;

    if (subtabs) subtabs.style.display = "flex";
    if (badge) {
      const code = acc.risk_level_code || "low";
      badge.innerHTML = `<span class="risk-badge ${code}">${acc.risk_level} (${acc.max_risk_score.toFixed(2)})</span>`;
    }

    renderSidepanelContent();
  } catch (err) {
    content.innerHTML = `<div style="color: var(--color-flagged); font-size: 12px; padding: 12px;">Failed to inspect account: ${err.message}</div>`;
  }
}

function renderSidepanelContent() {
  const content = document.getElementById("panel-entity-content");
  const acc = state.activeAccountData;
  if (!content || !acc) return;

  const tab = state.activeSidepanelTab || "reason";

  if (tab === "reason") {
    // TAB 1: Plain-English Layman Reason & Risk Factors Breakdown
    const factorsHtml = (acc.risk_factors || [])
      .map(
        (f) => `
      <div class="risk-factor-card">
        <div class="risk-factor-header">
          <span class="risk-factor-title">${f.title}</span>
          <span class="impact-tag ${f.level}">${f.impact}</span>
        </div>
        <div class="risk-factor-desc">${f.description}</div>
      </div>
    `
      )
      .join("");

    content.innerHTML = `
      <div style="display: flex; flex-direction: column; gap: 10px; font-size: 13px;">
        <!-- Layman Callout Banner -->
        <div class="layman-callout ${acc.risk_level_code || "low"}">
          <div class="layman-callout-title">
            ${
              acc.risk_level_code === "high"
                ? "🚨 High-Risk Account Flag"
                : acc.risk_level_code === "medium"
                ? "⚠️ Elevated Risk Notice"
                : "✅ Normal Account Baseline"
            }
          </div>
          <div>${acc.plain_english_summary}</div>
        </div>

        <!-- Key Risk Factors -->
        <div>
          <div style="font-size: 11px; font-weight: 700; color: var(--text-muted); margin-bottom: 6px; letter-spacing: 0.3px;">RISK FACTOR BREAKDOWN</div>
          ${factorsHtml}
        </div>

        <!-- Account Metrics Grid -->
        <div style="background: var(--bg-card-subtle); padding: 12px; border-radius: var(--radius-sm); border: 1px solid var(--border-card); margin-top: 4px;">
          <div style="font-size: 11px; font-weight: 700; color: var(--text-muted); margin-bottom: 8px;">ACCOUNT OVERVIEW</div>
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 12px;">
            <div><span style="color: var(--text-muted);">Total Txs:</span> <strong>${acc.total_transactions}</strong></div>
            <div><span style="color: var(--text-muted);">Flagged Txs:</span> <strong class="${acc.flagged_transactions > 0 ? "text-danger" : ""}">${acc.flagged_transactions}</strong></div>
            <div><span style="color: var(--text-muted);">Total Sent:</span> <strong class="mono">$${Number(acc.total_sent).toLocaleString(undefined, { minimumFractionDigits: 2 })}</strong></div>
            <div><span style="color: var(--text-muted);">Total Received:</span> <strong class="mono">$${Number(acc.total_received).toLocaleString(undefined, { minimumFractionDigits: 2 })}</strong></div>
            <div><span style="color: var(--text-muted);">Unique Counterparties:</span> <strong>${acc.counterparties.length}</strong></div>
            <div><span style="color: var(--text-muted);">Peak Score:</span> <strong class="mono ${acc.risk_level_code === "high" ? "text-danger" : ""}">${acc.max_risk_score.toFixed(4)}</strong></div>
          </div>
        </div>

        <button class="btn btn-primary btn-sm" onclick="filterTableByAccount('${acc.account_id}')" style="width: 100%; margin-top: 4px;">
          Filter Detection Results by This Account &rarr;
        </button>
      </div>
    `;
  } else if (tab === "txs") {
    // TAB 2: Transaction History
    const txRows = (acc.transactions || [])
      .map(
        (t) => `
      <div style="padding: 10px; border: 1px solid var(--border-subtle); border-radius: 6px; display: flex; justify-content: space-between; align-items: center; font-size: 12px; background: #FFFFFF;">
        <div>
          <div class="mono" style="font-weight: 600;">${t.id} &bull; $${Number(t.amount).toLocaleString()}</div>
          <div style="font-size: 11px; color: var(--text-muted); margin-top: 2px;">
            ${t.from_account === acc.account_id ? `To &rarr; <span class="mono">${t.to_account}</span>` : `From &larr; <span class="mono">${t.from_account}</span>`}
          </div>
          <div style="font-size: 10px; color: var(--text-muted);">${t.timestamp_str || formatTimestamp(t.timestamp)}</div>
        </div>
        <div style="text-align: right; display: flex; flex-direction: column; align-items: flex-end; gap: 4px;">
          <span class="risk-badge ${t.classification === "FLAGGED" ? "high" : "low"}" style="font-size: 10px;">${Number(t.risk_score).toFixed(2)}</span>
          <button class="btn btn-secondary btn-sm" style="padding: 2px 8px; font-size: 10px;" onclick="inspectTransaction('${t.id}')">Investigate</button>
        </div>
      </div>
    `
      )
      .join("");

    content.innerHTML = `
      <div style="display: flex; flex-direction: column; gap: 8px;">
        <div style="font-size: 11px; font-weight: 600; color: var(--text-muted);">TRANSACTION LEDGER (${acc.transactions.length})</div>
        <div style="display: flex; flex-direction: column; gap: 6px; max-height: 480px; overflow-y: auto;">
          ${txRows}
        </div>
      </div>
    `;
  } else if (tab === "counterparties") {
    // TAB 3: Counterparties
    const cpDetails = acc.counterparties_details || [];
    const cpRows = cpDetails
      .map(
        (cp) => `
      <div style="padding: 10px; border: 1px solid var(--border-subtle); border-radius: 6px; display: flex; justify-content: space-between; align-items: center; font-size: 12px; background: #FFFFFF;">
        <div>
          <div class="mono" style="font-weight: 700; color: var(--accent-blue);">${cp.account_id}</div>
          <div style="font-size: 11px; color: var(--text-muted); margin-top: 2px;">
            ${cp.transaction_count} transaction(s) &bull; $${Number(cp.total_amount).toLocaleString()}
          </div>
        </div>
        <div style="text-align: right; display: flex; flex-direction: column; align-items: flex-end; gap: 4px;">
          <span class="risk-badge ${cp.risk_level_code}">${cp.risk_level} (${cp.max_risk_score.toFixed(2)})</span>
          <button class="btn btn-secondary btn-sm" style="padding: 2px 8px; font-size: 10px;" onclick="inspectAccount('${cp.account_id}')">Inspect</button>
        </div>
      </div>
    `
      )
      .join("");

    content.innerHTML = `
      <div style="display: flex; flex-direction: column; gap: 8px;">
        <div style="font-size: 11px; font-weight: 600; color: var(--text-muted);">CONNECTED COUNTERPARTIES (${cpDetails.length})</div>
        <div style="display: flex; flex-direction: column; gap: 6px; max-height: 480px; overflow-y: auto;">
          ${cpRows}
        </div>
      </div>
    `;
  }
}

// ============================================================================
// 5. TRANSACTION EXPLANATION & ATTENTION (FRAUDLENS CARD 5 STYLE)
// ============================================================================

window.switchExpTab = function (tabName) {
  state.activeExpTab = tabName;
  document.querySelectorAll("#view-explanation .subnav-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.id === `btn-exptab-${tabName}`);
  });

  const tabFactors = document.getElementById("tab-content-factors");
  const tabRelated = document.getElementById("tab-content-related");
  const tabBehavior = document.getElementById("tab-content-behavior");
  const tabModel = document.getElementById("tab-content-model");

  if (tabFactors) tabFactors.style.display = tabName === "factors" ? "block" : "none";
  if (tabRelated) tabRelated.style.display = tabName === "related" ? "block" : "none";
  if (tabBehavior) tabBehavior.style.display = tabName === "behavior" ? "block" : "none";
  if (tabModel) tabModel.style.display = tabName === "model" ? "block" : "none";
};

async function loadTransactionExplanation(txId) {
  const expTxId = document.getElementById("exp-tx-id");
  const expTime = document.getElementById("exp-tx-time");
  const expSender = document.getElementById("exp-tx-sender");
  const expReceiver = document.getElementById("exp-tx-receiver");
  const expAmount = document.getElementById("exp-tx-amount");
  const expScore = document.getElementById("exp-tx-score");
  const expBadge = document.getElementById("exp-tx-badge");
  const expDecision = document.getElementById("exp-tx-decision");
  const expCallout = document.getElementById("exp-layman-callout");
  const expCalloutTitle = document.getElementById("exp-layman-title");
  const expCalloutText = document.getElementById("exp-layman-text");

  const factorsContainer = document.getElementById("exp-factors-container");
  const relatedContainer = document.getElementById("exp-related-container");
  const behaviorContainer = document.getElementById("exp-behavior-container");
  const attnBody = document.getElementById("exp-attention-body");
  const evBody = document.getElementById("exp-evidence-body");

  try {
    const res = await fetch(`/api/analysis/${encodeURIComponent(state.currentSessionId)}/transaction/${encodeURIComponent(txId)}`);
    if (!res.ok) throw new Error("Transaction fetch error");
    const tx = await res.json();
    state.activeTransactionData = tx;

    const isFlagged = tx.classification === "FLAGGED" || tx.risk_score >= (tx.threshold || 0.7372);
    const scoreVal = Number(tx.risk_score);

    // Populate Header Card (FraudLens Card 5)
    expTxId.textContent = tx.id;
    expTime.textContent = tx.timestamp_str || formatTimestamp(tx.timestamp);
    expSender.textContent = tx.from_account;
    expReceiver.textContent = tx.to_account;
    expAmount.textContent = `$${Number(tx.amount).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

    expScore.textContent = scoreVal.toFixed(4);
    expScore.style.color = isFlagged ? "var(--color-flagged)" : "var(--text-primary)";

    if (expBadge) {
      expBadge.className = `risk-badge ${tx.risk_level_code || (isFlagged ? "high" : "low")}`;
      expBadge.textContent = tx.risk_level || (isFlagged ? "High Risk" : "Low Risk");
    }

    if (expDecision) {
      expDecision.innerHTML = isFlagged
        ? `<strong style="color: var(--color-flagged);">Suspicious (Flagged)</strong>`
        : `<strong style="color: var(--color-success);">Cleared (Normal)</strong>`;
    }

    // Layman Callout Banner
    if (expCallout && expCalloutText) {
      expCallout.style.display = "flex";
      expCallout.className = `layman-callout ${tx.risk_level_code || (isFlagged ? "high" : "low")}`;
      if (expCalloutTitle) {
        expCalloutTitle.innerHTML = isFlagged
          ? `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg> Why Was This Transaction Flagged?`
          : `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg> Why Is This Transaction Low Risk?`;
      }
      expCalloutText.textContent = tx.plain_english_summary || tx.explainability?.summary;
    }

    // TAB 1: Key Risk Factors Breakdown (FraudLens Card 5)
    if (factorsContainer) {
      const factors = tx.plain_english_factors || [];
      if (factors.length === 0) {
        factorsContainer.innerHTML = `<div style="color: var(--text-muted); font-size: 13px;">No explicit risk factors recorded.</div>`;
      } else {
        factorsContainer.innerHTML = factors
          .map(
            (f) => `
          <div class="risk-factor-card">
            <div class="risk-factor-header">
              <span class="risk-factor-title">${f.title}</span>
              <span class="impact-tag ${f.level}">${f.impact}</span>
            </div>
            <div class="risk-factor-desc">${f.description}</div>
          </div>
        `
          )
          .join("");
      }
    }

    // TAB 2: Related Transactions in Batch
    if (relatedContainer) {
      // Find other transactions involving sender or receiver
      try {
        const relatedRes = await fetch(`/api/analysis/${encodeURIComponent(state.currentSessionId)}/results?search=${encodeURIComponent(tx.from_account)}&limit=10`);
        const relatedData = await relatedRes.json();
        const relatedTxs = (relatedData.items || []).filter((t) => t.id !== tx.id);

        if (relatedTxs.length === 0) {
          relatedContainer.innerHTML = `<div style="color: var(--text-muted); font-size: 13px; text-align: center; padding: 20px 0;">No other transactions involving these counterparties were found in this batch.</div>`;
        } else {
          relatedContainer.innerHTML = `
            <div style="display: flex; flex-direction: column; gap: 8px;">
              <div style="font-size: 12px; font-weight: 600; color: var(--text-muted); margin-bottom: 4px;">CONCURRENT TRANSACTIONS INVOLVING ${tx.from_account} (${relatedTxs.length})</div>
              ${relatedTxs
                .map(
                  (t) => `
                <div style="padding: 10px; border: 1px solid var(--border-subtle); border-radius: 6px; display: flex; justify-content: space-between; align-items: center; font-size: 12px; background: #FFFFFF;">
                  <div>
                    <div class="mono" style="font-weight: 600;">${t.id} &bull; $${Number(t.amount).toLocaleString()}</div>
                    <div style="font-size: 11px; color: var(--text-muted); margin-top: 2px;">
                      From <span class="mono">${t.from_account}</span> &rarr; To <span class="mono">${t.to_account}</span>
                    </div>
                  </div>
                  <div style="text-align: right; display: flex; flex-direction: column; align-items: flex-end; gap: 4px;">
                    <span class="risk-badge ${t.classification === "FLAGGED" ? "high" : "low"}">${Number(t.risk_score).toFixed(2)}</span>
                    <button class="btn btn-secondary btn-sm" style="padding: 2px 8px; font-size: 10px;" onclick="inspectTransaction('${t.id}')">Inspect</button>
                  </div>
                </div>
              `
                )
                .join("")}
            </div>
          `;
        }
      } catch (err) {
        relatedContainer.innerHTML = `<div style="color: var(--text-muted); font-size: 12px;">Could not load related transactions.</div>`;
      }
    }

    // TAB 3: Account Behavior Summary
    if (behaviorContainer) {
      behaviorContainer.innerHTML = `
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-4);">
          <div style="background: var(--bg-card-subtle); padding: 14px; border-radius: var(--radius-sm); border: 1px solid var(--border-card);">
            <div style="font-size: 11px; font-weight: 700; color: var(--text-muted); margin-bottom: 6px;">SENDER PROFILE</div>
            <div class="mono" style="font-size: 16px; font-weight: 700; color: var(--accent-blue); margin-bottom: 8px;">${tx.from_account}</div>
            <button class="btn btn-secondary btn-sm" onclick="switchTab('view-investigation'); inspectAccount('${tx.from_account}')" style="width: 100%;">
              Open Account in Network Graph &rarr;
            </button>
          </div>

          <div style="background: var(--bg-card-subtle); padding: 14px; border-radius: var(--radius-sm); border: 1px solid var(--border-card);">
            <div style="font-size: 11px; font-weight: 700; color: var(--text-muted); margin-bottom: 6px;">RECEIVER PROFILE</div>
            <div class="mono" style="font-size: 16px; font-weight: 700; color: var(--accent-blue); margin-bottom: 8px;">${tx.to_account}</div>
            <button class="btn btn-secondary btn-sm" onclick="switchTab('view-investigation'); inspectAccount('${tx.to_account}')" style="width: 100%;">
              Open Account in Network Graph &rarr;
            </button>
          </div>
        </div>
      `;
    }

    // TAB 4: TGAT Model Attention (Layer 1 & 2 Attention Weights)
    if (!tx.has_history) {
      attnBody.innerHTML = `
        <div style="padding: 16px; background: var(--color-warning-bg); border: 1px solid var(--color-warning-border); border-radius: var(--radius-sm); font-size: 13px; color: #92400E;">
          <strong>Model attribution not available (no prior temporal neighbors).</strong>
          <p style="margin-top: 4px; font-size: 12px;">This transaction involves isolated accounts with zero prior interactions recorded before this timestamp. Score relies entirely on tabular transaction properties.</p>
        </div>
      `;
    } else {
      const senderNeighbors = tx.sender_attention?.neighbors || [];
      const receiverNeighbors = tx.receiver_attention?.neighbors || [];

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

    // Factual Evidence list in Tab 4
    const evidenceList = tx.explainability?.evidence || [];
    evBody.innerHTML = `
      <div style="margin-bottom: 12px;">
        <div style="font-size: 12px; font-weight: 600; color: var(--text-muted);">SUMMARY ATTRIBUTION:</div>
        <div style="font-size: 13px; font-weight: 500; color: var(--text-primary); margin-top: 2px;">${tx.explainability?.summary || "N/A"}</div>
      </div>
      <div style="font-size: 12px; font-weight: 600; color: var(--text-muted); margin-bottom: 6px;">FACTUAL TRANSACTION CONTEXT:</div>
      <div class="evidence-list">
        ${evidenceList.map((e) => `<div class="evidence-item"><span class="evidence-bullet"></span><span>${e}</span></div>`).join("")}
      </div>
      <div style="font-size: 11px; color: var(--text-muted); margin-top: 14px; font-style: italic;">
        Note: Empirical attributes are factual behavioral markers extracted from chronological transaction logs, provided strictly as investigator context.
      </div>
    `;

    // Make sure first tab is active
    switchExpTab("factors");
  } catch (err) {
    if (attnBody) attnBody.innerHTML = `<div style="color: var(--color-flagged); font-size: 12px;">Failed to load explanation: ${err.message}</div>`;
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
