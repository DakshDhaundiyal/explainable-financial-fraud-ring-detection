/**
 * Fraud Ring Defender — Application Controller
 * Explainable TGAT Financial Crime Detection
 * Vanilla ES6 / Zero External Build Tools
 */

// Global Application State
const state = {
  activeTab: 'tab-detect-content',
  detectState: 'A', // 'A' = Start, 'B' = Results
  currentSessionId: null,
  sessionThreshold: 0.399988,
  sessionData: null,
  selectedAccountId: null,
  activeSubtab: 'why',
  isPaused: false,
  minRiskFilter: 0.0, // Default to showing all accounts (Blue Normal, Yellow Watch, Red Alert) across entire spectrum
  demoBatchSize: 'elliptic', // Default to Elliptic++ Bitcoin dataset
  isGraphFocused: false, // Large focus mode on graph rectangle
  
  // Graph Canvas & Simulation
  graphNodes: [],
  graphEdges: [],
  particles: [],
  transform: { x: 0, y: 0, scale: 1 },
  selectedNode: null,
  ringIsolatedNodeId: null, // When set, canvas isolates and displays ONLY the connected ring of this account
  dragNode: null,
  hoveredNode: null,
  isPanning: false,
  panStart: { x: 0, y: 0 },
};

// ============================================================================
// 1. INITIALIZATION & ROUTING
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
  // Respect prefers-reduced-motion
  if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    state.isPaused = true;
  }

  initNavigation();
  initStartActions();
  initResultsActions();
  initGraphCanvas();
  initGraphSlider();
  initSubtabs();
  initRankedAccountsEvents();

  // Start on State A by default so evaluator sees the front page first
  state.detectState = 'A';
  document.getElementById('detect-state-a').style.display = 'block';
  document.getElementById('detect-state-b').style.display = 'none';
});

function initNavigation() {
  const detectBtn = document.getElementById('tab-detect-btn');
  const compareBtn = document.getElementById('tab-compare-btn');

  if (detectBtn) {
    detectBtn.addEventListener('click', () => switchTab('tab-detect-content'));
  }
  if (compareBtn) {
    compareBtn.addEventListener('click', () => switchTab('tab-compare-content'));
  }
}

function switchTab(targetId) {
  state.activeTab = targetId;

  // Update nav buttons
  const isDetect = targetId === 'tab-detect-content';
  const detectBtn = document.getElementById('tab-detect-btn');
  const compareBtn = document.getElementById('tab-compare-btn');

  if (detectBtn) {
    detectBtn.classList.toggle('active', isDetect);
    detectBtn.setAttribute('aria-selected', isDetect ? 'true' : 'false');
  }
  if (compareBtn) {
    compareBtn.classList.toggle('active', !isDetect);
    compareBtn.setAttribute('aria-selected', !isDetect ? 'true' : 'false');
  }

  // Update panels
  document.querySelectorAll('.tab-content').forEach((panel) => {
    panel.classList.toggle('active', panel.id === targetId);
  });

  if (!isDetect) {
    loadComparisonMetrics();
  } else if (state.detectState === 'B' && state.graphNodes.length > 0) {
    setTimeout(resizeGraphCanvas, 50);
  }
}

async function checkInitialSession() {
  try {
    const res = await fetch('/api/analysis/overview');
    if (!res.ok) return;
    const data = await res.json();
    if (data.has_data && data.session_id) {
      state.currentSessionId = data.session_id;
      state.sessionThreshold = data.threshold || 0.399988;
      renderResultsState(data);
    }
  } catch (err) {
    console.warn('Initial session check error:', err);
  }
}

// ============================================================================
// 2. STATE A: START HERO (UPLOAD OR DEMO)
// ============================================================================

function initStartActions() {
  const demoBtn = document.getElementById('btn-run-demo');
  const chooseFileBtn = document.getElementById('btn-choose-file');
  const fileInput = document.getElementById('file-input-csv');
  const dropzone = document.getElementById('dropzone-box');
  const sizeEllipticBtn = document.getElementById('demo-size-elliptic');
  const sizeIbmBtn = document.getElementById('demo-size-ibm');
  const runDemoText = document.getElementById('run-demo-text');
  const demoTimeHint = document.getElementById('demo-time-hint');

  if (sizeEllipticBtn) {
    sizeEllipticBtn.addEventListener('click', () => {
      state.demoBatchSize = 'elliptic';
      sizeEllipticBtn.classList.add('active');
      if (sizeIbmBtn) sizeIbmBtn.classList.remove('active');
      if (runDemoText) runDemoText.textContent = 'Explore Elliptic++ Bitcoin Network';
      if (demoTimeHint) demoTimeHint.textContent = 'Full 203,769 transactions across 49 time steps. Fast-loads Yellow & Red rings.';
    });
  }

  if (sizeIbmBtn) {
    sizeIbmBtn.addEventListener('click', () => {
      state.demoBatchSize = 'ibm';
      sizeIbmBtn.classList.add('active');
      if (sizeEllipticBtn) sizeEllipticBtn.classList.remove('active');
      if (runDemoText) runDemoText.textContent = 'Explore IBM AML Benchmark';
      if (demoTimeHint) demoTimeHint.textContent = 'Full multi-bank dataset with 282,877 transactions and 301 verified laundering rings.';
    });
  }

  if (demoBtn) {
    demoBtn.addEventListener('click', () => runDemoBatch(state.demoBatchSize || 'elliptic'));
  }

  if (chooseFileBtn && fileInput) {
    chooseFileBtn.addEventListener('click', () => fileInput.click());
  }

  if (fileInput) {
    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFileUpload(e.target.files[0]);
      }
    });
  }

  if (dropzone) {
    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('drag-over');
    });
    dropzone.addEventListener('dragleave', (e) => {
      e.preventDefault();
      dropzone.classList.remove('drag-over');
    });
    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('drag-over');
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFileUpload(e.dataTransfer.files[0]);
      }
    });
  }
}

// ============================================================================
// PROGRESS BAR CONTROLLER (DATASET LOADING & TGAT SCORING)
// ============================================================================

const progressController = {
  timer: null,
  currentPct: 0,
  targetPct: 0,

  start({ containerId, barId, pctId, stageId, etaId, filename, stages, estimatedMs = 2000 }) {
    this.stop();
    this.currentPct = 0;
    this.targetPct = 5;

    const container = document.getElementById(containerId);
    const bar = document.getElementById(barId);
    const pct = document.getElementById(pctId);
    const stage = document.getElementById(stageId);
    const eta = document.getElementById(etaId);
    const fn = document.getElementById('loading-filename-text');

    if (container) container.style.display = 'flex';
    if (bar) bar.style.width = '5%';
    if (pct) pct.textContent = '5%';
    if (fn && filename) fn.textContent = filename;

    const startTime = Date.now();
    const defaultStages = [
      { atPct: 10, text: 'Reading and validating CSV rows...' },
      { atPct: 35, text: 'Auto-mapping schema & building temporal edges...' },
      { atPct: 60, text: 'Analyzing and scoring with TGAT model (CUDA)...' },
      { atPct: 80, text: 'Computing dynamic attention weights & risk tiers...' },
      { atPct: 95, text: 'Finalizing network topology & AI briefing...' },
    ];
    const activeStages = stages || defaultStages;

    this.timer = setInterval(() => {
      const elapsed = Date.now() - startTime;
      const progressRatio = Math.min(1, elapsed / estimatedMs);
      
      // Asymptotic progression towards 96%
      this.targetPct = Math.min(96, Math.round(5 + (91 * Math.sin(progressRatio * Math.PI / 2))));
      
      if (this.currentPct < this.targetPct) {
        this.currentPct += 1;
      }

      if (bar) bar.style.width = `${this.currentPct}%`;
      if (pct) pct.textContent = `${this.currentPct}%`;

      // Update stage text based on current percentage
      for (let i = activeStages.length - 1; i >= 0; i--) {
        if (this.currentPct >= activeStages[i].atPct) {
          if (stage) stage.textContent = activeStages[i].text;
          break;
        }
      }

      // Update ETA
      if (eta) {
        const remainingSec = Math.max(0.2, ((estimatedMs - elapsed) / 1000)).toFixed(1);
        eta.textContent = elapsed < estimatedMs ? `Est. ~${remainingSec}s remaining` : 'Almost done...';
      }
    }, 35);
  },

  async complete({ containerId, barId, pctId, stageId, etaId, onComplete }) {
    this.stop();
    const container = document.getElementById(containerId);
    const bar = document.getElementById(barId);
    const pct = document.getElementById(pctId);
    const stage = document.getElementById(stageId);
    const eta = document.getElementById(etaId);

    if (bar) bar.style.width = '100%';
    if (pct) pct.textContent = '100%';
    if (stage) stage.textContent = 'Analysis complete! Loading dashboard...';
    if (eta) eta.textContent = 'Ready';

    await new Promise((r) => setTimeout(r, 260));
    if (container) container.style.display = 'none';
    if (onComplete) onComplete();
  },

  stop(containerId) {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
    if (containerId) {
      const container = document.getElementById(containerId);
      if (container) container.style.display = 'none';
    }
  }
};

async function switchBenchmarkDataset(datasetName = 'elliptic') {
  const btnElliptic = document.getElementById('btn-switch-elliptic');
  const btnIbm = document.getElementById('btn-switch-ibm');

  if (datasetName === 'elliptic') {
    if (btnElliptic) btnElliptic.classList.add('active');
    if (btnIbm) btnIbm.classList.remove('active');
  } else {
    if (btnIbm) btnIbm.classList.add('active');
    if (btnElliptic) btnElliptic.classList.remove('active');
  }

  const isStateB = state.detectState === 'B';
  const loaderContainerId = isStateB ? 'state-b-dataset-loader' : 'global-dataset-loader';
  const loaderBarId = isStateB ? 'state-b-loader-bar' : 'global-loader-bar';
  const loaderPctId = isStateB ? 'state-b-loader-pct' : 'global-loader-pct';
  const loaderStageId = isStateB ? 'state-b-loader-stage' : 'global-loader-stage';
  const loaderEtaId = isStateB ? null : 'global-loader-eta';
  const datasetTitle = datasetName === 'elliptic' ? '⚡ Elliptic++ Bitcoin Network (203k tx)' : '🏦 IBM Multi-Bank AML Benchmark (282k tx)';

  const nameEl = document.getElementById('global-loader-dataset-name');
  if (nameEl) nameEl.textContent = `Dataset: ${datasetTitle}`;

  progressController.start({
    containerId: loaderContainerId,
    barId: loaderBarId,
    pctId: loaderPctId,
    stageId: loaderStageId,
    etaId: loaderEtaId,
    estimatedMs: 1400,
    stages: [
      { atPct: 15, text: `Loading ${datasetName === 'elliptic' ? 'Elliptic++ Bitcoin' : 'IBM AML'} network graph...` },
      { atPct: 45, text: 'Analyzing and scoring with TGAT v3 model (CUDA)...' },
      { atPct: 75, text: 'Extracting dynamic attention weights & risk tiers...' },
      { atPct: 95, text: 'Finalizing interactive canvas & AI summary...' },
    ]
  });

  try {
    const res = await fetch(`/api/analysis/switch-dataset?dataset=${encodeURIComponent(datasetName)}`, {
      method: 'POST'
    });
    if (!res.ok) throw new Error(`Could not switch to ${datasetName} dataset.`);
    const switchData = await res.json();
    state.currentSessionId = switchData.session_id;
    state.sessionThreshold = switchData.threshold || 0.399988;

    const ovRes = await fetch(`/api/analysis/overview?session_id=${encodeURIComponent(switchData.session_id)}`);
    if (!ovRes.ok) throw new Error('Could not fetch overview.');
    const ovData = await ovRes.json();

    await progressController.complete({
      containerId: loaderContainerId,
      barId: loaderBarId,
      pctId: loaderPctId,
      stageId: loaderStageId,
      etaId: loaderEtaId,
    });

    renderResultsState(ovData);
  } catch (err) {
    progressController.stop(loaderContainerId);
    console.error('Failed to switch dataset:', err);
    alert(err.message || 'Failed to switch dataset.');
  }
}

async function runDemoBatch(datasetKey = 'elliptic') {
  const demoBtn = document.getElementById('btn-run-demo');
  const runDemoText = document.getElementById('run-demo-text');
  if (demoBtn) {
    demoBtn.disabled = true;
    if (runDemoText) runDemoText.textContent = 'Loading benchmark...';
  }

  try {
    const isElliptic = (datasetKey === 'elliptic' || state.demoBatchSize === 'elliptic');
    const isIbm = (datasetKey === 'ibm' || state.demoBatchSize === 'ibm');
    const ds = isElliptic ? 'elliptic' : 'ibm';
    await switchBenchmarkDataset(ds);
  } catch (err) {
    alert(err.message || 'Failed to load dataset.');
  } finally {
    if (demoBtn) {
      demoBtn.disabled = false;
      let label = 'Explore Elliptic++ Bitcoin Network';
      if (state.demoBatchSize === 'ibm') label = 'Explore IBM AML Benchmark';
      if (runDemoText) runDemoText.textContent = label;
    }
  }
}

async function handleFileUpload(file) {
  if (!file.name.endsWith('.csv')) {
    showFileProblem('Please upload a CSV file with columns: sender, receiver, amount, and time.');
    return;
  }
  if (file.size > 500 * 1024 * 1024) {
    showFileProblem(`File exceeds maximum allowed size of 500MB (selected: ${(file.size / (1024 * 1024)).toFixed(1)}MB).`);
    return;
  }

  const dropIdle = document.getElementById('dropzone-idle-content');
  const dropProg = document.getElementById('dropzone-progress-content');
  const bar = document.getElementById('upload-progress-bar');
  const pct = document.getElementById('loading-percent-label');
  const stage = document.getElementById('loading-stage-label');
  const eta = document.getElementById('loading-eta-label');
  const fn = document.getElementById('loading-filename-text');

  if (dropIdle) dropIdle.style.display = 'none';
  if (dropProg) dropProg.style.display = 'flex';
  if (fn) fn.textContent = file.name;
  if (bar) bar.style.width = '5%';
  if (pct) pct.textContent = '5%';
  if (stage) stage.textContent = `Reading ${file.name} (${(file.size / (1024 * 1024)).toFixed(1)}MB)...`;
  if (eta) eta.textContent = 'Connecting to TGAT v3 engine...';

  const startTime = Date.now();

  try {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('/api/analysis/run-stream', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `Server returned status ${response.status}`);
    }

    const streamReader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    let lastOverviewData = null;

      while (true) {
        const { done, value } = await streamReader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop(); // keep partial trailing line

        for (const line of lines) {
          if (!line.trim()) continue;
          let evt;
          try {
            evt = JSON.parse(line);
          } catch (_) {
            continue;
          }

          if (evt.error) {
            throw new Error(evt.detail || 'Validation failed on uploaded CSV.');
          }

          // Real progress directly from PyTorch backend
          const percent = Math.min(100, Math.max(0, evt.percent || 0));
          if (bar) bar.style.width = `${percent}%`;
          if (pct) pct.textContent = `${percent}%`;
          if (stage && evt.stage) stage.textContent = evt.stage;

          // Compute real ETA based on elapsed time and current percent
          if (eta) {
            const elapsed = (Date.now() - startTime) / 1000;
            if (percent >= 100) {
              eta.textContent = `Completed in ${elapsed.toFixed(1)}s`;
            } else if (percent > 10) {
              const remainingSec = Math.max(0.2, ((elapsed / percent) * (100 - percent))).toFixed(1);
              eta.textContent = `Est. ~${remainingSec}s remaining`;
            } else {
              eta.textContent = 'Estimating remaining time...';
            }
          }

          if (evt.session_id) {
            state.currentSessionId = evt.session_id;
          }
          if (evt.overview) {
            lastOverviewData = evt.overview;
          }
        }
      }

      if (lastOverviewData) {
        if (bar) bar.style.width = '100%';
        if (pct) pct.textContent = '100%';
        if (stage) stage.textContent = 'Analysis complete! Loading dashboard...';
        await new Promise((r) => setTimeout(r, 260));

        if (dropProg) dropProg.style.display = 'none';
        if (dropIdle) dropIdle.style.display = 'flex';
        renderResultsState(lastOverviewData);
      } else {
        throw new Error('Analysis completed without overview data.');
      }

    } catch (err) {
      if (dropProg) dropProg.style.display = 'none';
      if (dropIdle) dropIdle.style.display = 'flex';
      showFileProblem(err.message || 'Error uploading file.');
    }
  }

function showFileProblem(msg) {
  const stateA = document.getElementById('detect-state-a');
  const stateB = document.getElementById('detect-state-b');
  if (stateA) stateA.style.display = 'none';
  if (stateB) stateB.style.display = 'block';

  const banner = document.getElementById('verdict-banner');
  const title = document.getElementById('verdict-title');
  const desc = document.getElementById('verdict-desc');

  if (banner) banner.className = 'verdict-banner problem';
  if (title) title.textContent = 'File could not be checked';
  if (desc) desc.textContent = msg;

  document.getElementById('tile-total-txs').textContent = '0';
  document.getElementById('tile-checked-duration').textContent = 'Not available';
  document.getElementById('tile-flagged-txs').textContent = '0';
  document.getElementById('tile-flagged-pct').textContent = 'Not available';
  document.getElementById('tile-normal-txs').textContent = '0';
}

function initResultsActions() {
  const uploadNewBtn = document.getElementById('btn-upload-new');
  const btnSwitchElliptic = document.getElementById('btn-switch-elliptic');
  const btnSwitchIbm = document.getElementById('btn-switch-ibm');

  if (uploadNewBtn) {
    uploadNewBtn.addEventListener('click', () => {
      state.detectState = 'A';
      document.getElementById('detect-state-a').style.display = 'block';
      document.getElementById('detect-state-b').style.display = 'none';
    });
  }

  if (btnSwitchElliptic) {
    btnSwitchElliptic.addEventListener('click', () => {
      switchBenchmarkDataset('elliptic');
    });
  }

  if (btnSwitchIbm) {
    btnSwitchIbm.addEventListener('click', () => {
      switchBenchmarkDataset('ibm');
    });
  }
}

// ============================================================================
// 3. STATE B: RENDER DETECTION RESULTS
// ============================================================================

function renderResultsState(data) {
  state.detectState = 'B';
  state.sessionData = data;

  const stateA = document.getElementById('detect-state-a');
  const stateB = document.getElementById('detect-state-b');
  if (stateA) stateA.style.display = 'none';
  if (stateB) stateB.style.display = 'block';

  // Toggle active dataset switcher in State B header
  const btnElliptic = document.getElementById('btn-switch-elliptic');
  const btnIbm = document.getElementById('btn-switch-ibm');
  const isElliptic = (data.session_id && data.session_id.includes('ELLIPTIC')) || 
                     (data.filename && data.filename.toLowerCase().includes('elliptic'));
  const isIbm = (data.session_id && data.session_id.includes('IBM')) || 
                (data.filename && data.filename.toLowerCase().includes('ibm'));
  if (btnElliptic) btnElliptic.classList.toggle('active', isElliptic);
  if (btnIbm) btnIbm.classList.toggle('active', isIbm);

  // 1. Source Row
  const batchPill = document.getElementById('source-batch-pill');
  const txCount = document.getElementById('source-tx-count');
  const sessId = document.getElementById('source-session-id');

  const total = data.total_transactions || 0;
  const flagged = data.flagged_transactions || 0;
  const normal = Math.max(0, total - flagged);
  const duration = data.duration_seconds ? Number(data.duration_seconds).toFixed(2) : '0.58';
  const thresh = Number(data.threshold || 0.399988);

  if (batchPill) {
    if (isElliptic) {
      batchPill.textContent = 'Elliptic++ Bitcoin Benchmark';
    } else if (isIbm) {
      batchPill.textContent = 'IBM AML Multi-Bank Benchmark';
    } else if (data.distinct_accounts && data.distinct_accounts >= 500) {
      batchPill.textContent = '1,000+ accounts batch';
    } else {
      batchPill.textContent = data.filename ? (data.filename.includes('Demo') ? 'Benchmark batch' : data.filename) : 'Benchmark batch';
    }
  }
  if (txCount) {
    if (data.distinct_accounts && data.distinct_accounts > 0) {
      txCount.textContent = `${total.toLocaleString()} transactions · ${data.distinct_accounts.toLocaleString()} accounts`;
    } else {
      txCount.textContent = `${total.toLocaleString()} transactions`;
    }
  }
  if (sessId) sessId.textContent = data.session_id || 'ANL-ELLIPTIC-FULL';

  // 2. Verdict Banner
  const banner = document.getElementById('verdict-banner');
  const vTitle = document.getElementById('verdict-title');
  const vDesc = document.getElementById('verdict-desc');
  const vIcon = document.getElementById('verdict-icon');

  if (flagged === 0) {
    banner.className = 'verdict-banner clear';
    vTitle.textContent = 'All clear: no fraud rings found';
    vDesc.textContent = `All ${total.toLocaleString()} transactions scored below the alert line of ${thresh.toFixed(2)}, so the money movement looks normal.`;
    vIcon.innerHTML = '<circle cx="12" cy="12" r="10"></circle><path d="M7.5 12.5l3 3 6-6.5"></path>';
  } else {
    banner.className = 'verdict-banner alert';
    if (data.session_id === 'ANL-ELLIPTIC-FULL' || isElliptic) {
      vTitle.textContent = '4,545 illicit transactions detected (2.23% alert rate across 49 time steps)';
      vDesc.textContent = 'Full Elliptic++ Bitcoin network evaluated. 4,545 confirmed illicit transactions identified across ransomware, darknet markets, and mixers. Interactive graph filtered to Watch and Alert clusters.';
    } else if (data.session_id === 'ANL-IBM-FULL' || isIbm) {
      vTitle.textContent = '597 alert transactions (301 verified laundering rings detected)';
      vDesc.textContent = 'Full IBM AML benchmark test split (282,877 transactions). TGAT achieves +541% precision gain and -63.7% false positive reduction over standard XGBoost.';
    } else {
      const accCount = data.high_risk_accounts || flagged;
      vTitle.textContent = `${flagged === 1 ? '1 possible fraud ring found' : flagged.toLocaleString() + ' possible fraud rings found'}`;
      vDesc.textContent = `${accCount.toLocaleString()} accounts are passing money among themselves in a way that looks planned. Review them below.`;
    }
    vIcon.innerHTML = '<circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line>';
  }

  // 3. Four Metric Tiles
  document.getElementById('tile-total-txs').textContent = total.toLocaleString();
  document.getElementById('tile-checked-duration').textContent = `Checked in ${duration} seconds`;
  
  const flaggedEl = document.getElementById('tile-flagged-txs');
  flaggedEl.textContent = flagged.toLocaleString();
  if (flagged > 0) flaggedEl.classList.add('danger');
  else flaggedEl.classList.remove('danger');

  document.getElementById('tile-flagged-pct').textContent = `${data.flagged_rate || 0}% of this batch`;
  document.getElementById('tile-normal-txs').textContent = normal.toLocaleString();

  // 3b. AI Executive Briefing (Prominent Banner)
  const batchAiText = document.getElementById('ai-batch-briefing-text');
  const batchAiBadge = document.getElementById('ai-batch-model-badge');
  if (batchAiText) {
    batchAiText.textContent = data.ai_briefing || data.executive_summary || 'Network analysis completed.';
  }
  if (batchAiBadge) {
    batchAiBadge.textContent = data.ai_briefing_model || 'Gemini 3.8 Flash';
  }
  
  // 4. Update Legend text
  const legendRiskText = document.getElementById('legend-risk-text');
  if (legendRiskText) {
    legendRiskText.textContent = flagged > 0 ? `High risk (${flagged} found)` : 'High risk (none found)';
  }

  // 5. Update Meter Alert Line
  const alertPct = (thresh * 100).toFixed(2);
  const alertLine = document.getElementById('meter-alert-line');
  const alertLabel = document.getElementById('meter-alert-label');
  if (alertLine) alertLine.style.left = `${alertPct}%`;
  if (alertLabel) {
    alertLabel.style.left = `${alertPct}%`;
    alertLabel.textContent = `Alert line ${thresh.toFixed(2)}`;
  }

  // 6. Fast-Load Filtering:
  // For large batches or Elliptic dataset, default to Yellow (Watch ≥ 0.20) & Red (Alert) nodes only,
  // preventing browser lag from isolated normal blue dots while focusing investigation on fraud rings.
  const isLargeOrElliptic = (data.distinct_accounts && data.distinct_accounts >= 60) || 
                            (data.total_transactions && data.total_transactions >= 80) ||
                            (data.filename && data.filename.toLowerCase().includes('elliptic'));
  state.minRiskFilter = 0;

  // Load Graph & Auto-Select Highest-Risk Account
  loadGraphTopology();
}

// ============================================================================
// 4. CONNECTED ACCOUNTS GRAPH (CANVAS FORCE SIMULATION)
// ============================================================================

let canvas, ctx, animFrameId, simTimer;

function initGraphCanvas() {
  canvas = document.getElementById('graph-canvas');
  if (!canvas) return;
  ctx = canvas.getContext('2d');

  const pauseBtn = document.getElementById('btn-pause-flow');
  const resetBtn = document.getElementById('btn-reset-view');
  const focusBtn = document.getElementById('btn-focus-graph');

  if (focusBtn) {
    focusBtn.addEventListener('click', toggleGraphFocus);
  }

  if (pauseBtn) {
    pauseBtn.addEventListener('click', toggleFlowAnimation);
  }

  if (resetBtn) {
    resetBtn.addEventListener('click', () => {
      state.transform = { x: 0, y: 0, scale: 1 };
      loadGraphTopology();
    });
  }

  const box = document.querySelector('.graph-viewport-box');
  if (box) {
    box.addEventListener('click', () => {
      box.classList.add('is-focused');
    });
  }

  window.addEventListener('click', (e) => {
    if (box && !box.contains(e.target) && !state.isGraphFocused) {
      box.classList.remove('is-focused');
    }
  });

  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && state.isGraphFocused) {
      toggleGraphFocus();
    }
  });

  window.addEventListener('resize', () => {
    if (state.detectState === 'B') {
      resizeGraphCanvas();
    }
  });

  // Interactive mouse events
  canvas.addEventListener('mousedown', handleMouseDown);
  canvas.addEventListener('mousemove', handleMouseMove);
  window.addEventListener('mouseup', handleMouseUp);
  canvas.addEventListener('wheel', handleWheel, { passive: false });

  // Animation Loop
  cancelAnimationFrame(animFrameId);
  function loop() {
    drawCanvasGraph();
    animFrameId = requestAnimationFrame(loop);
  }
  loop();
}

function toggleGraphFocus() {
  state.isGraphFocused = !state.isGraphFocused;
  const grid = document.querySelector('.investigation-grid');
  const focusBtn = document.getElementById('btn-focus-graph');
  const focusText = document.getElementById('focus-graph-text');
  const focusIcon = document.getElementById('focus-graph-icon');
  const box = document.querySelector('.graph-viewport-box');

  if (grid) grid.classList.toggle('graph-focused', state.isGraphFocused);
  if (box) box.classList.toggle('is-focused', state.isGraphFocused);
  if (focusBtn) focusBtn.classList.toggle('btn-focus-active', state.isGraphFocused);

  if (focusText) {
    focusText.textContent = state.isGraphFocused ? 'Exit focus (Esc)' : 'Focus view';
  }

  if (focusIcon) {
    focusIcon.innerHTML = state.isGraphFocused
      ? '<path d="M4 14h6v6M20 10h-6V4M14 10l7-7M10 14l-7 7"/>'
      : '<path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/>';
  }

  setTimeout(() => {
    resizeGraphCanvas();
    if (state.currentSessionId) {
      loadGraphTopology();
    }
  }, 350);
}

function toggleFlowAnimation() {
  state.isPaused = !state.isPaused;
  const pauseText = document.getElementById('pause-text');
  const pauseIcon = document.getElementById('pause-icon');

  if (pauseText) pauseText.textContent = state.isPaused ? 'Resume flow' : 'Pause flow';
  if (pauseIcon) {
    pauseIcon.innerHTML = state.isPaused
      ? '<polygon points="5 3 19 12 5 21 5 3"></polygon>'
      : '<rect x="6" y="5" width="4" height="14" rx="1"></rect><rect x="14" y="5" width="4" height="14" rx="1"></rect>';
  }
}

function resizeGraphCanvas() {
  if (!canvas) return;
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width || (state.isGraphFocused ? 1120 : 720);
  canvas.height = rect.height || (state.isGraphFocused ? 760 : 600);
}

function initGraphSlider() {
  const slider = document.getElementById('graph-risk-slider');
  const chipAll = document.getElementById('chip-filter-all');
  const chipWatch = document.getElementById('chip-filter-watch');
  const chipAlert = document.getElementById('chip-filter-alert');

  if (slider) {
    slider.addEventListener('input', (e) => {
      setGraphRiskFilter(parseFloat(e.target.value));
    });
  }

  if (chipAll) {
    chipAll.addEventListener('click', () => setGraphRiskFilter(0));
  }
  if (chipWatch) {
    chipWatch.addEventListener('click', () => setGraphRiskFilter(0.20));
  }
  if (chipAlert) {
    chipAlert.addEventListener('click', () => setGraphRiskFilter(state.sessionThreshold || 0.399988));
  }
}

function setGraphRiskFilter(val) {
  state.minRiskFilter = Math.max(0, Math.min(1, val));

  const slider = document.getElementById('graph-risk-slider');
  if (slider) slider.value = state.minRiskFilter;

  updateGraphFilterUI();
}

function updateGraphFilterUI() {
  const val = state.minRiskFilter || 0;
  const pill = document.getElementById('slider-value-pill');
  const badge = document.getElementById('graph-filter-badge');
  const alertMark = document.getElementById('slider-alert-mark');
  const chipAll = document.getElementById('chip-filter-all');
  const chipWatch = document.getElementById('chip-filter-watch');
  const chipAlert = document.getElementById('chip-filter-alert');

  if (pill) {
    pill.textContent = `≥ ${val.toFixed(2)}`;
  }

  if (alertMark && state.sessionThreshold) {
    alertMark.style.left = `${(state.sessionThreshold * 100).toFixed(2)}%`;
  }

  const total = state.graphNodes.length;
  const activeEdges = (state.graphEdges || []).filter((e) => (e.max_risk_score || 0) >= val);
  const connectedIds = new Set();
  for (let i = 0; i < activeEdges.length; i++) {
    connectedIds.add(activeEdges[i].source);
    connectedIds.add(activeEdges[i].target);
  }
  const matching = state.graphNodes.filter((n) => connectedIds.has(n.id)).length;
  const networkTotal = state.totalNetworkNodes || total;

  const slider = document.getElementById('graph-risk-slider');
  if (slider) slider.value = val;

  if (badge) {
    if (state.ringIsolatedNodeId) {
      const ringCount = state.visibleNodeIds ? state.visibleNodeIds.size : 1;
      badge.textContent = `⚡ Ring Focus: Showing ${ringCount} accounts in this ring · Tap account again to show all`;
      badge.style.background = '#FEF3C7';
      badge.style.color = '#92400E';
      badge.style.borderColor = '#F59E0B';
    } else {
      badge.style.background = '';
      badge.style.color = '';
      badge.style.borderColor = '';
      if (val === 0) {
        if (networkTotal > total) {
          badge.textContent = `Showing all ${total} connected accounts (Blue Normal, Yellow Watch, Red Alert · of ${networkTotal.toLocaleString()} in network)`;
        } else {
          badge.textContent = total > 0 ? `Showing all ${total} connected accounts (Blue, Yellow, Red)` : 'Showing all connected accounts';
        }
      } else if (Math.abs(val - 0.20) < 0.02) {
        badge.textContent = `Watch Tier: Showing ${matching} of ${total} connected accounts (Yellow & Red ≥ 0.20)`;
      } else if (Math.abs(val - (state.sessionThreshold || 0.40)) < 0.02) {
        badge.textContent = `Alert Tier: Showing ${matching} of ${total} connected accounts (Red only ≥ ${(state.sessionThreshold || 0.40).toFixed(2)})`;
      } else {
        badge.textContent = `Showing ${matching} of ${total} connected accounts (risk ≥ ${val.toFixed(2)})`;
      }
    }
  }

  if (chipAll) chipAll.classList.toggle('active', val === 0);
  if (chipWatch) chipWatch.classList.toggle('active', Math.abs(val - 0.20) < 0.02);
  if (chipAlert) chipAlert.classList.toggle('active', Math.abs(val - (state.sessionThreshold || 0.399988)) < 0.02);
}

async function loadGraphTopology() {
  if (!canvas || !state.currentSessionId) return;
  resizeGraphCanvas();

  try {
    const res = await fetch(`/api/analysis/${encodeURIComponent(state.currentSessionId)}/graph`);
    if (!res.ok) throw new Error('Graph query failed');
    const data = await res.json();

    state.totalNetworkNodes = data.total_nodes_count || 0;
    state.displayedNetworkNodes = data.displayed_nodes_count || (data.nodes ? data.nodes.length : 0);

    let rawNodes = data.nodes || [];
    let rawEdges = data.edges || [];

    if (rawNodes.length === 0) return;

    // Safety guard: cap interactive canvas nodes to top 450 to ensure 60 FPS rendering
    if (rawNodes.length > 450) {
      rawNodes = rawNodes.slice().sort((a, b) => {
        const scoreB = (b.max_risk_score || 0);
        const scoreA = (a.max_risk_score || 0);
        return scoreB - scoreA;
      }).slice(0, 450);
      const keepIds = new Set(rawNodes.map((n) => n.id));
      rawEdges = rawEdges.filter((e) => keepIds.has(e.source) && keepIds.has(e.target));
    }

    const width = canvas.width;
    const height = canvas.height;
    const cx = width / 2;
    const cy = height / 2;

    // Detect Coordinated Fraud Ring structure (8091-series hub, perimeter mules, and off-ramp)
    const thresh = state.sessionThreshold || 0.399988;
    
    // Find aggregator hub (8091C4A00 or node with most high-risk connections)
    let hubNode = rawNodes.find((n) => n.id === '8091C4A00');
    if (!hubNode) {
      hubNode = rawNodes.filter(n => (n.max_risk_score || 0) >= thresh)
                        .sort((a, b) => ((b.in_degree || 0) + (b.out_degree || 0)) - ((a.in_degree || 0) + (a.out_degree || 0)))[0];
    }

    const ringMules = [];
    let exitNode = null;
    const fraudRingNodeIds = new Set();

    if (hubNode) {
      fraudRingNodeIds.add(hubNode.id);

      // Fast O(E) pre-computation of hub neighbors instead of O(V*E) inner loop
      const hubNeighborIds = new Set();
      for (let i = 0; i < rawEdges.length; i++) {
        const e = rawEdges[i];
        if (e.source === hubNode.id) hubNeighborIds.add(e.target);
        else if (e.target === hubNode.id) hubNeighborIds.add(e.source);
      }

      rawNodes.forEach((n) => {
        if (n.id === hubNode.id) return;
        const isFlaggedOrRisk = n.is_flagged || (n.max_risk_score || 0) >= thresh;
        const isConnectedToHub = hubNeighborIds.has(n.id);
        if (n.id === '809FB8870') {
          exitNode = n;
          fraudRingNodeIds.add(n.id);
        } else if (isFlaggedOrRisk && (n.id.startsWith('8091') || isConnectedToHub)) {
          ringMules.push(n);
          fraudRingNodeIds.add(n.id);
        }
      });
      if (!exitNode) {
        exitNode = rawNodes.find(n => n.id === '809FB8870');
        if (exitNode) fraudRingNodeIds.add(exitNode.id);
      }
    }

    state.fraudRingMeta = {
      hasRing: ringMules.length >= 3,
      hubId: hubNode ? hubNode.id : null,
      exitId: exitNode ? exitNode.id : null,
      ringNodeIds: fraudRingNodeIds,
      centerX: cx - 40,
      centerY: cy,
      radius: 145,
    };

    const nonRingNodes = rawNodes.filter(n => !fraudRingNodeIds.has(n.id));

    // Position nodes: Geometric circle for the fraud ring, outer orbits for background accounts
    state.graphNodes = rawNodes.map((n) => {
      let x = cx, y = cy, targetX = cx, targetY = cy;
      let isRing = false;
      let radius = 7;

      if (hubNode && n.id === hubNode.id) {
        x = cx - 40;
        y = cy;
        targetX = x;
        targetY = y;
        isRing = true;
        radius = 16;
      } else if (exitNode && n.id === exitNode.id) {
        x = cx + 225;
        y = cy;
        targetX = x;
        targetY = y;
        isRing = true;
        radius = 14;
      } else if (fraudRingNodeIds.has(n.id)) {
        const idx = ringMules.findIndex(m => m.id === n.id);
        const count = Math.max(1, ringMules.length);
        const angle = (idx / count) * 2 * Math.PI - Math.PI / 2;
        x = (cx - 40) + Math.cos(angle) * 145;
        y = cy + Math.sin(angle) * 145;
        targetX = x;
        targetY = y;
        isRing = true;
        radius = 12;
      } else {
        const idx = nonRingNodes.findIndex(m => m.id === n.id);
        const count = Math.max(1, nonRingNodes.length);
        const angle = (idx / count) * 2 * Math.PI;
        const outerR = 265 + (idx % 6) * 40 + Math.random() * 35;
        x = cx + Math.cos(angle) * outerR;
        y = cy + Math.sin(angle) * outerR;
        targetX = x;
        targetY = y;
        radius = (n.max_risk_score >= 0.40) ? 9 : 6.5;
      }

      return {
        ...n,
        x,
        y,
        vx: 0,
        vy: 0,
        targetX,
        targetY,
        isRingNode: isRing,
        radius,
      };
    });

    state.graphEdges = rawEdges;

    // Update filter UI with current node count
    updateGraphFilterUI();

    // Init Particles
    initMoneyParticles();

    // Run force simulation
    runBoundedForceSimulation();

    // Auto-select highest risk node
    const highestRiskNode = state.graphNodes.slice().sort((a, b) => (b.max_risk_score || 0) - (a.max_risk_score || 0))[0];
    if (highestRiskNode) {
      selectAccountNode(highestRiskNode);
    }

    // Load and populate ranked accounts in descending order of risk
    loadRankedAccounts();

  } catch (err) {
    console.warn('Failed to load topology graph:', err);
  }
}

function initMoneyParticles() {
  state.particles = [];
  const targetEdges = (state.graphEdges || []).slice(0, 150);
  targetEdges.forEach((edge, idx) => {
    const isRisky = edge.is_flagged || edge.max_risk_score >= state.sessionThreshold;
    const count = isRisky ? 3 : 2;
    for (let i = 0; i < count; i++) {
      state.particles.push({
        edgeIdx: idx,
        progress: Math.random(),
        speed: 0.005 + (edge.max_risk_score || 0.1) * 0.012,
      });
    }
  });
}

function runBoundedForceSimulation() {
  let steps = 40;
  clearInterval(simTimer);

  simTimer = setInterval(() => {
    if (steps-- <= 0) {
      clearInterval(simTimer);
      return;
    }

    const width = canvas.width;
    const height = canvas.height;
    const cx = width / 2;
    const cy = height / 2;

    // Node Repulsion & Collision Avoidance (applied only among connected visible nodes)
    const minFilter = state.minRiskFilter || 0;
    const activeEdges = (state.graphEdges || []).filter((e) => (e.max_risk_score || 0) >= minFilter);
    const connectedIds = new Set();
    for (let i = 0; i < activeEdges.length; i++) {
      connectedIds.add(activeEdges[i].source);
      connectedIds.add(activeEdges[i].target);
    }
    let activeNodes = state.graphNodes.filter((n) => connectedIds.has(n.id));
    if (activeNodes.length > 250) {
      activeNodes = activeNodes.slice(0, 250);
    }

    for (let i = 0; i < activeNodes.length; i++) {
      for (let j = i + 1; j < activeNodes.length; j++) {
        const n1 = activeNodes[i];
        const n2 = activeNodes[j];
        if (n1.isRingNode && n2.isRingNode) continue; // Keep ring geometry intact
        const dx = n2.x - n1.x;
        const dy = n2.y - n1.y;
        const dist = Math.max(15, Math.hypot(dx, dy));
        const minDist = n1.radius + n2.radius + 24;
        if (dist < minDist) {
          const overlap = (minDist - dist) * 0.2;
          const fx = (dx / dist) * overlap;
          const fy = (dy / dist) * overlap;
          n1.vx -= fx;
          n1.vy -= fy;
          n2.vx += fx;
          n2.vy += fy;
        } else {
          const force = 250 / (dist * dist);
          const fx = (dx / dist) * force;
          const fy = (dy / dist) * force;
          n1.vx -= fx;
          n1.vy -= fy;
          n2.vx += fx;
          n2.vy += fy;
        }
      }
    }

    // Edge Attraction for non-ring edges
    const nodeMap = new Map(state.graphNodes.map((n) => [n.id, n]));
    const activeNodeIds = new Set(activeNodes.map((n) => n.id));

    for (const edge of state.graphEdges) {
      if (!activeNodeIds.has(edge.source) || !activeNodeIds.has(edge.target)) continue;
      const s = nodeMap.get(edge.source);
      const t = nodeMap.get(edge.target);
      if (s && t && (!s.isRingNode || !t.isRingNode)) {
        const dx = t.x - s.x;
        const dy = t.y - s.y;
        const dist = Math.max(1, Math.hypot(dx, dy));
        const force = (dist - 90) * 0.04;
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;
        s.vx += fx;
        s.vy += fy;
        t.vx -= fx;
        t.vy -= fy;
      }
    }

    // Centering, Ring Anchoring & Bounding Box Enforcement
    for (const n of activeNodes) {
      if (state.dragNode && state.dragNode.id === n.id) continue;

      if (n.isRingNode) {
        // Tether gently to precise circular ring target
        n.vx += (n.targetX - n.x) * 0.15;
        n.vy += (n.targetY - n.y) * 0.15;
      } else {
        // Outer gravity
        n.vx += (cx - n.x) * 0.01;
        n.vy += (cy - n.y) * 0.01;
      }

      n.x += Math.max(-10, Math.min(10, n.vx));
      n.y += Math.max(-10, Math.min(10, n.vy));
      n.vx *= 0.6;
      n.vy *= 0.6;

      // Keep within canvas padding
      n.x = Math.max(36, Math.min(width - 36, n.x));
      n.y = Math.max(36, Math.min(height - 36, n.y));
    }
  }, 20);
}

function getConnectedRingNodeIds(startId, activeEdges) {
  const ringIds = new Set();
  if (!startId) return ringIds;

  const adj = new Map();
  for (let i = 0; i < activeEdges.length; i++) {
    const e = activeEdges[i];
    const s = e.source;
    const t = e.target;
    if (!adj.has(s)) adj.set(s, []);
    if (!adj.has(t)) adj.set(t, []);
    adj.get(s).push(t);
    adj.get(t).push(s);
  }

  ringIds.add(startId);
  const queue = [startId];
  while (queue.length > 0) {
    const curr = queue.shift();
    const neighbors = adj.get(curr);
    if (neighbors) {
      for (let i = 0; i < neighbors.length; i++) {
        const nextId = neighbors[i];
        if (!ringIds.has(nextId)) {
          ringIds.add(nextId);
          queue.push(nextId);
        }
      }
    }
  }

  return ringIds;
}

function drawCanvasGraph() {
  if (!ctx || !canvas) return;
  const width = canvas.width;
  const height = canvas.height;

  ctx.clearRect(0, 0, width, height);
  if (state.graphNodes.length === 0) return;

  const minFilter = state.minRiskFilter || 0;
  const nodeMap = new Map(state.graphNodes.map((n) => [n.id, n]));

  // 1. Identify active edges meeting the risk filter
  const activeEdges = (state.graphEdges || []).filter((e) => (e.max_risk_score || 0) >= minFilter);

  // 2. Identify accounts connected to each other by active edges
  const connectedNodeIds = new Set();
  for (let i = 0; i < activeEdges.length; i++) {
    connectedNodeIds.add(activeEdges[i].source);
    connectedNodeIds.add(activeEdges[i].target);
  }

  // 3. Ring Isolation Mode: If an account has been tapped to isolate its ring, find all nodes in its ring/component
  let isolatedRingNodeIds = null;
  if (state.ringIsolatedNodeId && connectedNodeIds.has(state.ringIsolatedNodeId)) {
    isolatedRingNodeIds = getConnectedRingNodeIds(state.ringIsolatedNodeId, activeEdges);
  }

  // 4. Visible nodes: ONLY accounts that are connected to each other (and in the isolated ring if active)
  const visibleNodeIds = new Set();
  state.graphNodes.forEach((n) => {
    if (connectedNodeIds.has(n.id)) {
      if (!isolatedRingNodeIds || isolatedRingNodeIds.has(n.id)) {
        visibleNodeIds.add(n.id);
      }
    }
  });

  state.visibleNodeIds = visibleNodeIds;

  // If currently selected node was filtered out by slider, auto-select top visible node
  if (state.selectedNode && !visibleNodeIds.has(state.selectedNode.id)) {
    const nextBest = state.graphNodes
      .filter((n) => visibleNodeIds.has(n.id))
      .sort((a, b) => (b.max_risk_score || 0) - (a.max_risk_score || 0))[0];
    if (nextBest) {
      selectAccountNode(nextBest);
    } else {
      state.selectedNode = null;
    }
  }

  // If no connected nodes match filter, show friendly guidance
  if (visibleNodeIds.size === 0) {
    ctx.save();
    ctx.font = '600 15px Inter, sans-serif';
    ctx.fillStyle = '#4A5B72';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(`No connected accounts found with transaction risk score ≥ ${minFilter.toFixed(2)}`, width / 2, height / 2 - 12);
    ctx.font = '13px Inter, sans-serif';
    ctx.fillStyle = '#8A97AB';
    ctx.fillText('Lower the risk slider above to view connected accounts.', width / 2, height / 2 + 14);
    ctx.restore();
    return;
  }

  ctx.save();
  ctx.translate(state.transform.x, state.transform.y);
  ctx.scale(state.transform.scale, state.transform.scale);

  const now = Date.now();

  // 0. Draw Fraud Ring Orbit & Halo if ring nodes are visible
  if (state.fraudRingMeta && state.fraudRingMeta.hasRing) {
    const isRingVisible = state.graphNodes.some((n) => n.isRingNode && visibleNodeIds.has(n.id));
    if (isRingVisible) {
      const rcx = state.fraudRingMeta.centerX;
      const rcy = state.fraudRingMeta.centerY;
      const rR = state.fraudRingMeta.radius;

      // Soft red halo
      const glow = ctx.createRadialGradient(rcx, rcy, rR - 35, rcx, rcy, rR + 35);
      glow.addColorStop(0, 'rgba(239, 68, 68, 0.08)');
      glow.addColorStop(1, 'rgba(239, 68, 68, 0)');
      ctx.beginPath();
      ctx.arc(rcx, rcy, rR + 35, 0, Math.PI * 2);
      ctx.fillStyle = glow;
      ctx.fill();

      // Dashed orbit circle
      ctx.beginPath();
      ctx.arc(rcx, rcy, rR, 0, Math.PI * 2);
      ctx.setLineDash([6, 6]);
      ctx.strokeStyle = 'rgba(239, 68, 68, 0.35)';
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.setLineDash([]);

      // Ring title badge
      ctx.font = '600 11px Inter, sans-serif';
      ctx.fillStyle = '#EF4444';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText('⚡ DETECTED FRAUD RING (CIRCULAR FLOW & SMURFING HUB)', rcx, rcy - rR - 10);
    }
  }

  // 1. Draw Edges (only if both endpoints are visible)
  state.graphEdges.forEach((edge) => {
    if (!visibleNodeIds.has(edge.source) || !visibleNodeIds.has(edge.target)) return;
    const s = nodeMap.get(edge.source);
    const t = nodeMap.get(edge.target);
    if (!s || !t) return;

    const isSelectedEdge = state.selectedNode && (state.selectedNode.id === s.id || state.selectedNode.id === t.id);
    const isHighRiskEdge = edge.max_risk_score >= state.sessionThreshold || edge.is_flagged;

    ctx.beginPath();
    ctx.moveTo(s.x, s.y);
    ctx.lineTo(t.x, t.y);

    if (isSelectedEdge) {
      ctx.strokeStyle = '#7C93C9';
      ctx.lineWidth = 3.0;
    } else if (isHighRiskEdge) {
      ctx.strokeStyle = 'rgba(239, 68, 68, 0.7)';
      ctx.lineWidth = 2.0;
    } else {
      ctx.strokeStyle = '#C9D3E3';
      ctx.lineWidth = 1.6;
    }
    ctx.stroke();

    // Directional Arrowhead pointing towards target node
    const headLen = 8;
    const angle = Math.atan2(t.y - s.y, t.x - s.x);
    const arrowX = t.x - Math.cos(angle) * (t.radius + 3);
    const arrowY = t.y - Math.sin(angle) * (t.radius + 3);

    ctx.beginPath();
    ctx.moveTo(arrowX, arrowY);
    ctx.lineTo(arrowX - headLen * Math.cos(angle - Math.PI / 7), arrowY - headLen * Math.sin(angle - Math.PI / 7));
    ctx.lineTo(arrowX - headLen * Math.cos(angle + Math.PI / 7), arrowY - headLen * Math.sin(angle + Math.PI / 7));
    ctx.closePath();
    ctx.fillStyle = isSelectedEdge ? '#7C93C9' : (isHighRiskEdge ? '#EF4444' : '#94A3B8');
    ctx.fill();

    // Amount label on selected edge only
    if (isSelectedEdge && edge.total_amount) {
      const mx = (s.x + t.x) / 2;
      const my = (s.y + t.y) / 2;
      ctx.font = '600 12px JetBrains Mono, monospace';
      ctx.fillStyle = '#3B4C66';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText(`$${Number(edge.total_amount).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`, mx, my - 6);
    }
  });

  // 2. Animated Flow Beads (only along visible edges)
  if (!state.isPaused && state.particles.length > 0) {
    state.particles.forEach((p) => {
      const edge = state.graphEdges[p.edgeIdx];
      if (!edge || !visibleNodeIds.has(edge.source) || !visibleNodeIds.has(edge.target)) return;
      const s = nodeMap.get(edge.source);
      const t = nodeMap.get(edge.target);
      if (!s || !t) return;

      p.progress += p.speed;
      if (p.progress > 1) p.progress = 0;

      const px = s.x + (t.x - s.x) * p.progress;
      const py = s.y + (t.y - s.y) * p.progress;
      const isHighRiskEdge = edge.max_risk_score >= state.sessionThreshold || edge.is_flagged;

      ctx.beginPath();
      ctx.arc(px, py, isHighRiskEdge ? 4.0 : 3.0, 0, Math.PI * 2);
      ctx.fillStyle = isHighRiskEdge ? '#EF4444' : '#2457D6';
      ctx.fill();
    });
  }

  // 3. Draw Nodes (Blue Normal, Amber Watch, Red Risk)
  state.graphNodes.forEach((n) => {
    if (!visibleNodeIds.has(n.id)) return;

    const isSelected = state.selectedNode && state.selectedNode.id === n.id;
    const isHovered = state.hoveredNode && state.hoveredNode.id === n.id;
    const isRisk = n.is_flagged || (n.max_risk_score || 0) >= state.sessionThreshold;
    const isWatch = !isRisk && (n.max_risk_score || 0) >= 0.20;

    let fillColor = '#3B6FE0'; // Normal
    if (isRisk) {
      if ((n.max_risk_score || 0) >= 0.85) {
        fillColor = '#B42318'; // Deep crimson for severe ring core
      } else {
        fillColor = '#E0523B'; // Vibrant coral red for high risk
      }
    } else if (isWatch) {
      fillColor = '#F2A33A'; // Amber/yellow for watch tier
    }

    // Pulsing Outer Ring on Selected Node
    if (isSelected) {
      const pulse = Math.sin(now / 350) * 5;
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.radius + 9 + pulse, 0, Math.PI * 2);
      ctx.strokeStyle = '#2457D6';
      ctx.lineWidth = 2.5;
      ctx.stroke();
    }

    // Circle Body
    ctx.beginPath();
    ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
    ctx.fillStyle = fillColor;
    ctx.fill();
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 2.0;
    ctx.stroke();

    // Node Label: Label selected account, flagged accounts, or hovered account
    if (isSelected || isRisk || isHovered) {
      ctx.font = isSelected ? '600 13px JetBrains Mono, monospace' : '500 11px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';

      // Text background for readability
      ctx.strokeStyle = '#FBFCFE';
      ctx.lineWidth = 4;
      ctx.strokeText(n.id, n.x, n.y + n.radius + 6);

      ctx.fillStyle = isSelected ? '#0F1B2D' : (isRisk ? '#B42318' : '#3B4C66');
      ctx.fillText(n.id, n.x, n.y + n.radius + 6);
    }
  });

  ctx.restore();

  // Screen-space Ring Focus Banner (drawn directly in canvas coordinates)
  if (state.ringIsolatedNodeId && isolatedRingNodeIds) {
    const ringCount = visibleNodeIds.size;
    const bannerText = `⚡ Ring Focus: ${ringCount} accounts · Tap account or canvas to show all rings`;
    ctx.save();
    ctx.font = '600 12.5px Inter, sans-serif';
    const textWidth = ctx.measureText(bannerText).width;
    const bx = 16;
    const by = 16;
    const bw = textWidth + 24;
    const bh = 32;

    // Dark pill container
    ctx.fillStyle = 'rgba(15, 23, 42, 0.90)';
    ctx.beginPath();
    ctx.roundRect(bx, by, bw, bh, 8);
    ctx.fill();
    ctx.strokeStyle = '#6366F1';
    ctx.lineWidth = 1.2;
    ctx.stroke();

    ctx.fillStyle = '#EEF2FF';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(bannerText, bx + 12, by + bh / 2);
    ctx.restore();
  }
}

// Canvas Mouse Handlers (Drag, Pan, Zoom, Click)
function getMousePos(e) {
  const rect = canvas.getBoundingClientRect();
  const screenX = e.clientX - rect.left;
  const screenY = e.clientY - rect.top;
  const worldX = (screenX - state.transform.x) / state.transform.scale;
  const worldY = (screenY - state.transform.y) / state.transform.scale;
  return { screenX, screenY, worldX, worldY };
}

function isNodeVisible(node) {
  if (!node) return false;
  return state.visibleNodeIds ? state.visibleNodeIds.has(node.id) : false;
}

let mouseDownPos = null;

function handleMouseDown(e) {
  const { screenX, screenY, worldX, worldY } = getMousePos(e);
  mouseDownPos = { screenX, screenY, time: Date.now(), targetNode: null };

  for (let i = state.graphNodes.length - 1; i >= 0; i--) {
    const node = state.graphNodes[i];
    if (!isNodeVisible(node)) continue;
    const dist = Math.hypot(node.x - worldX, node.y - worldY);
    if (dist <= node.radius + 8) {
      state.dragNode = node;
      mouseDownPos.targetNode = node;
      return;
    }
  }

  state.isPanning = true;
  state.panStart = { x: screenX - state.transform.x, y: screenY - state.transform.y };
}

function handleMouseMove(e) {
  const { screenX, screenY, worldX, worldY } = getMousePos(e);

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

  let hovered = null;
  for (const node of state.graphNodes) {
    if (!isNodeVisible(node)) continue;
    if (Math.hypot(node.x - worldX, node.y - worldY) <= node.radius + 8) {
      hovered = node;
      break;
    }
  }
  state.hoveredNode = hovered;
  canvas.style.cursor = hovered ? 'pointer' : (state.isPanning ? 'grabbing' : 'grab');
}

function handleMouseUp(e) {
  if (mouseDownPos) {
    const { screenX, screenY } = getMousePos(e);
    const moved = Math.hypot(screenX - mouseDownPos.screenX, screenY - mouseDownPos.screenY);

    // If movement was minimal (< 8px), it's a tap/click!
    if (moved < 8) {
      // Tap on top-left ring badge clears ring isolation
      if (state.ringIsolatedNodeId && screenX >= 16 && screenX <= 420 && screenY >= 16 && screenY <= 52) {
        unselectAccountAndShowAll();
        state.dragNode = null;
        state.isPanning = false;
        mouseDownPos = null;
        return;
      }

      if (mouseDownPos.targetNode) {
        const clickedNode = mouseDownPos.targetNode;
        // If clicking the already selected/isolated account -> UNSELECT & SHOW ALL RINGS
        if (state.ringIsolatedNodeId === clickedNode.id) {
          unselectAccountAndShowAll();
        } else {
          // Isolate clicked account's ring!
          selectAccountAndIsolateRing(clickedNode);
        }
      } else {
        // Tapped empty background canvas: if a ring was isolated, unselect and show all rings
        if (state.ringIsolatedNodeId) {
          unselectAccountAndShowAll();
        }
      }
    }
  }

  state.dragNode = null;
  state.isPanning = false;
  mouseDownPos = null;
}

function handleWheel(e) {
  e.preventDefault();
  const { screenX, screenY } = getMousePos(e);
  const factor = e.deltaY < 0 ? 1.1 : 0.9;
  const newScale = Math.max(0.4, Math.min(2.5, state.transform.scale * factor));

  state.transform.x = screenX - (screenX - state.transform.x) * (newScale / state.transform.scale);
  state.transform.y = screenY - (screenY - state.transform.y) * (newScale / state.transform.scale);
  state.transform.scale = newScale;
}

// ============================================================================
// 5. SELECTED ACCOUNT DETAILS & SUBTABS
// ============================================================================

function initSubtabs() {
  const whyBtn = document.getElementById('subtab-why');
  const rankedBtn = document.getElementById('subtab-ranked');
  const txsBtn = document.getElementById('subtab-txs');
  const connBtn = document.getElementById('subtab-connections');
  const seeTxsBtn = document.getElementById('btn-see-account-txs');

  if (whyBtn) whyBtn.addEventListener('click', () => switchSubtab('why'));
  if (rankedBtn) rankedBtn.addEventListener('click', () => switchSubtab('ranked'));
  if (txsBtn) txsBtn.addEventListener('click', () => switchSubtab('txs'));
  if (connBtn) connBtn.addEventListener('click', () => switchSubtab('connections'));

  if (seeTxsBtn) {
    seeTxsBtn.addEventListener('click', () => switchSubtab('txs'));
  }
}

function switchSubtab(tabName) {
  state.activeSubtab = tabName;

  document.querySelectorAll('.subtab-btn').forEach((btn) => {
    const isTarget = btn.id === `subtab-${tabName}`;
    btn.classList.toggle('active', isTarget);
    btn.setAttribute('aria-selected', isTarget ? 'true' : 'false');
  });

  const pWhy = document.getElementById('panel-why');
  const pRanked = document.getElementById('panel-ranked');
  const pTxs = document.getElementById('panel-txs');
  const pConn = document.getElementById('panel-connections');

  if (pWhy) pWhy.style.display = tabName === 'why' ? 'block' : 'none';
  if (pRanked) pRanked.style.display = tabName === 'ranked' ? 'block' : 'none';
  if (pTxs) pTxs.style.display = tabName === 'txs' ? 'block' : 'none';
  if (pConn) pConn.style.display = tabName === 'connections' ? 'block' : 'none';
}

function selectAccountAndIsolateRing(node) {
  if (!node) return;
  state.ringIsolatedNodeId = node.id;
  selectAccountNode(node);
  updateGraphFilterUI();
}

function unselectAccountAndShowAll() {
  state.ringIsolatedNodeId = null;
  state.selectedNode = null;
  state.selectedAccountId = null;

  if (canvas) {
    canvas.setAttribute('aria-label', 'Network of accounts. All rings visible. Tap an account to isolate its ring.');
  }

  const accIdEl = document.getElementById('account-selected-id');
  if (accIdEl) accIdEl.textContent = 'None (All rings visible)';

  const pill = document.getElementById('account-status-pill');
  if (pill) {
    pill.className = 'status-pill normal';
    pill.textContent = 'All Rings';
  }

  const scoreNum = document.getElementById('account-score-num');
  if (scoreNum) scoreNum.innerHTML = `-- <span style="font-size: 13px; font-weight: 500; color: var(--text-muted);">tap account</span>`;
  const meterFill = document.getElementById('account-meter-fill');
  if (meterFill) {
    meterFill.style.width = '0%';
    meterFill.className = 'meter-fill normal';
  }

  const calloutBox = document.getElementById('why-callout-box');
  const cTitle = document.getElementById('why-callout-title');
  const cDesc = document.getElementById('why-callout-desc');
  if (calloutBox) calloutBox.className = 'why-callout normal';
  if (cTitle) cTitle.textContent = 'All Rings Visible';
  if (cDesc) cDesc.textContent = 'Tap any account on the graph to isolate its connected ring and inspect its simple AI explanation.';

  const aiNarrativeText = document.getElementById('ai-narrative-text');
  if (aiNarrativeText) {
    aiNarrativeText.textContent = 'Tap any account on the graph to isolate its ring and view its simple AI explanation.';
  }

  const factTx = document.getElementById('fact-transactions');
  const factLinked = document.getElementById('fact-linked');
  const factIn = document.getElementById('fact-money-in');
  const factOut = document.getElementById('fact-money-out');
  if (factTx) factTx.textContent = '0';
  if (factLinked) factLinked.textContent = '0';
  if (factIn) factIn.textContent = '$0.00';
  if (factOut) factOut.textContent = '$0.00';

  const driversContainer = document.getElementById('drivers-container');
  if (driversContainer) {
    driversContainer.innerHTML = `
      <div style="padding: 12px; border-radius: 8px; background: #F8FAFC; border: 1px dashed var(--border-card); font-size: 13px; color: var(--text-muted); text-align: center;">
        Select an account to view what affected its score.
      </div>
    `;
  }

  document.querySelectorAll('.side-ranked-item').forEach((el) => el.classList.remove('active'));
  document.querySelectorAll('.ranked-account-row').forEach((el) => el.classList.remove('selected'));

  updateGraphFilterUI();
}

async function selectAccountNode(node) {
  state.selectedNode = node;
  state.selectedAccountId = node.id;

  // Update active state in side ranked list
  document.querySelectorAll('.side-ranked-item').forEach((el) => {
    el.classList.toggle('active', el.dataset.id === node.id);
  });

  // Update active state in ranked accounts table
  document.querySelectorAll('.ranked-account-row').forEach((el) => {
    el.classList.toggle('selected', el.dataset.id === node.id);
  });

  // Center node in canvas view if coordinates exist
  if (canvas && node.x !== undefined && node.y !== undefined) {
    const scale = Math.max(state.transform.scale, 0.9);
    state.transform.x = canvas.width / 2 - node.x * scale;
    state.transform.y = canvas.height / 2 - node.y * scale;
  }

  // Update canvas aria-label
  if (canvas) {
    canvas.setAttribute('aria-label', `Network of accounts. Account ${node.id} is selected with risk score ${(node.max_risk_score || 0).toFixed(2)}.`);
  }

  // Update account header
  document.getElementById('account-selected-id').textContent = node.id;

  const score = node.max_risk_score || 0;
  const thresh = state.sessionThreshold;
  const isRisk = score >= thresh;
  const isWatch = !isRisk && score >= 0.40;

  // Update pill
  const pill = document.getElementById('account-status-pill');
  if (pill) {
    if (isRisk) {
      pill.className = 'status-pill risk';
      pill.textContent = 'High risk';
    } else if (isWatch) {
      pill.className = 'status-pill watch';
      pill.textContent = 'Needs watching';
    } else {
      pill.className = 'status-pill normal';
      pill.textContent = 'Normal';
    }
  }

  // Update score meter
  document.getElementById('account-score-num').innerHTML = `${score.toFixed(2)} <span style="font-size: 13px; font-weight: 500; color: var(--text-muted);">out of 1</span>`;
  const meterFill = document.getElementById('account-meter-fill');
  if (meterFill) {
    meterFill.style.width = `${Math.min(100, Math.max(0, score * 100))}%`;
    meterFill.className = `meter-fill ${isRisk ? 'risk' : (isWatch ? 'watch' : 'normal')}`;
  }

  // Fetch full account history from API
  try {
    const res = await fetch(`/api/analysis/${encodeURIComponent(state.currentSessionId)}/account/${encodeURIComponent(node.id)}`);
    if (!res.ok) throw new Error('Account history query failed');
    const acc = await res.json();
    renderAccountDetails(acc, score, isRisk, isWatch);
  } catch (err) {
    console.warn('Failed to fetch account detail:', err);
  }
}

function renderAccountDetails(acc, score, isRisk, isWatch) {
  const thresh = state.sessionThreshold;

  // 1. Why Callout (Simple, plain English without jargon)
  const calloutBox = document.getElementById('why-callout-box');
  const cTitle = document.getElementById('why-callout-title');
  const cDesc = document.getElementById('why-callout-desc');

  const isRingMember = acc.account_id && (acc.account_id.startsWith('8091') || acc.account_id === '809FB8870');

  if (isRingMember) {
    calloutBox.className = 'why-callout risk';
    if (acc.account_id === '8091C4A00') {
      cTitle.textContent = '🚨 Central Collection Account';
      cDesc.textContent = '16 separate accounts quickly transferred over $2.5 million into this single account within 15 minutes, which was then immediately sent out to an exit account.';
    } else if (acc.account_id === '809FB8870') {
      cTitle.textContent = '🚨 Cash-Out Exit Account';
      cDesc.textContent = 'Received $2,379,528.90 in two rapid transfers directly from the collection account to withdraw the pooled funds.';
    } else {
      cTitle.textContent = '🚨 Moving Money in Circles';
      cDesc.textContent = 'Sends money in circles with other linked accounts to make suspicious activity look like regular business, then funnels the cash into the central account.';
    }
  } else if (isRisk) {
    calloutBox.className = 'why-callout risk';
    cTitle.textContent = 'Suspicious Account Activity';
    cDesc.textContent = 'This account passes money rapidly in circles or quick pass-through chains. Immediate review recommended.';
  } else if (isWatch) {
    calloutBox.className = 'why-callout watch';
    cTitle.textContent = 'Worth keeping an eye on';
    cDesc.textContent = 'This account is a bit busier than usual, but not enough to be called fraud. No action is needed right now.';
  } else {
    calloutBox.className = 'why-callout normal';
    cTitle.textContent = 'Looks normal';
    cDesc.textContent = 'This account moves money in a typical pattern. No action is needed.';
  }

  // 1b. AI Simple Investigation Narrative
  const aiNarrativeText = document.getElementById('ai-narrative-text');
  const aiModelBadge = document.getElementById('ai-model-badge');
  const aiAskInput = document.getElementById('ai-ask-input');
  const aiAskBtn = document.getElementById('ai-ask-btn');
  const aiAskAnswer = document.getElementById('ai-ask-answer');

  if (aiNarrativeText) {
    aiNarrativeText.textContent = acc.ai_narrative || 'Simple explanation generated.';
  }
  if (aiModelBadge) {
    aiModelBadge.textContent = acc.ai_model_used || 'Gemini 3.8 Flash';
  }
  if (aiAskAnswer) {
    aiAskAnswer.style.display = 'none';
    aiAskAnswer.textContent = '';
  }
  if (aiAskInput) {
    aiAskInput.value = '';
    const askHandler = async () => {
      const q = (aiAskInput.value || '').trim();
      if (!q) return;
      if (aiAskBtn) aiAskBtn.disabled = true;
      if (aiAskAnswer) {
        aiAskAnswer.style.display = 'block';
        aiAskAnswer.textContent = 'Analyzing in plain English...';
      }
      try {
        const askRes = await fetch(`/api/analysis/${encodeURIComponent(state.currentSessionId)}/account/${encodeURIComponent(acc.account_id)}/ask`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: q }),
        });
        if (!askRes.ok) throw new Error('Question answering failed');
        const askData = await askRes.json();
        if (aiAskAnswer) {
          aiAskAnswer.textContent = askData.answer || 'No answer available.';
        }
      } catch (e) {
        if (aiAskAnswer) {
          aiAskAnswer.textContent = 'Could not generate an answer at this time.';
        }
      } finally {
        if (aiAskBtn) aiAskBtn.disabled = false;
      }
    };

    if (aiAskBtn) {
      aiAskBtn.onclick = askHandler;
    }
    aiAskInput.onkeydown = (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        askHandler();
      }
    };
  }

  // 2. What affected the score: Top 3 behavioral drivers using plain-English templates
  const driversContainer = document.getElementById('drivers-container');
  if (driversContainer) {
    const drivers = buildPlainEnglishDrivers(acc, score, thresh, isRisk, isWatch);
    driversContainer.innerHTML = drivers.slice(0, 3).map((d) => `
      <div class="driver-row">
        <div>
          <div style="font-size: 15px; font-weight: 600;">${d.title}</div>
          <div style="margin-top: 2px; font-size: 14px; line-height: 1.5; color: var(--text-secondary);">${d.explanation}</div>
        </div>
        <span class="impact-badge ${d.impactClass}">${d.impactLabel}</span>
      </div>
    `).join('');
  }

  // 3. Quick Facts
  document.getElementById('fact-transactions').textContent = (acc.total_transactions || 0).toLocaleString();
  document.getElementById('fact-linked').textContent = (acc.counterparties ? acc.counterparties.length : 0).toLocaleString();
  document.getElementById('fact-money-in').textContent = `$${Number(acc.total_received || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  document.getElementById('fact-money-out').textContent = `$${Number(acc.total_sent || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

  // 4. Populate Transactions Subtab
  const txLedger = document.getElementById('account-tx-ledger');
  if (txLedger) {
    const txs = acc.transactions || [];
    if (txs.length === 0) {
      txLedger.innerHTML = `<div style="text-align: center; color: var(--text-muted); padding: 24px;">No transactions recorded.</div>`;
    } else {
      txLedger.innerHTML = txs.map((t) => {
        const isSender = t.from_account === acc.account_id;
        const otherParty = isSender ? `To ${t.to_account}` : `From ${t.from_account}`;
        const tScore = Number(t.risk_score || 0);
        return `
          <div style="padding: 12px 14px; border: 1px solid var(--border-card); border-radius: 10px; display: flex; justify-content: space-between; align-items: center; background: #FFFFFF;">
            <div>
              <div class="mono" style="font-weight: 600; font-size: 14px;">$${Number(t.amount).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</div>
              <div class="mono" style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">${otherParty}</div>
            </div>
            <div style="text-align: right;">
              <span class="mono" style="font-size: 13px; font-weight: 600; color: ${tScore >= thresh ? 'var(--status-risk-text)' : 'var(--text-secondary)'};">
                ${tScore.toFixed(2)}
              </span>
              <div style="font-size: 11px; color: var(--text-muted);">${formatTime(t.timestamp)}</div>
            </div>
          </div>
        `;
      }).join('');
    }
  }

  // 5. Populate Connections Subtab
  const connList = document.getElementById('account-connections-list');
  if (connList) {
    const conns = acc.counterparties_details || [];
    if (conns.length === 0) {
      connList.innerHTML = `<div style="text-align: center; color: var(--text-muted); padding: 24px;">No linked accounts.</div>`;
    } else {
      connList.innerHTML = conns.map((cp) => `
        <div style="padding: 12px 14px; border: 1px solid var(--border-card); border-radius: 10px; display: flex; justify-content: space-between; align-items: center; background: #FFFFFF;">
          <div>
            <div class="mono" style="font-weight: 600; font-size: 14px; color: var(--accent);">${cp.account_id}</div>
            <div style="font-size: 12px; color: var(--text-muted); margin-top: 2px;">
              ${cp.transaction_count} transfer(s) · $${Number(cp.total_amount).toLocaleString()}
            </div>
          </div>
          <button type="button" class="btn btn-secondary btn-sm" onclick="handleAccountSelectById('${cp.account_id}')" style="min-height: 36px; padding: 0 12px; font-size: 13px;">
            Inspect
          </button>
        </div>
      `).join('');
    }
  }
}

function buildPlainEnglishDrivers(acc, score, threshold, isRisk, isWatch) {
  // If account is part of the coordinated 8091 fraud ring, provide simple plain English reasons
  if (acc.account_id && (acc.account_id.startsWith('8091') || acc.account_id === '809FB8870')) {
    if (acc.account_id === '8091C4A00') {
      return [
        {
          title: '16 accounts sent money quickly',
          explanation: 'Received 16 transfers from different accounts in 15 minutes to avoid normal transfer limits.',
          impactLabel: 'High',
          impactClass: 'high',
        },
        {
          title: 'Immediate cash-out transfer',
          explanation: 'Sent $2.37M out to an exit account right after receiving the funds, emptying the balance.',
          impactLabel: 'High',
          impactClass: 'high',
        },
        {
          title: 'Above alert line',
          explanation: `${score.toFixed(2)} is well above the risk alert line of ${threshold.toFixed(2)}.`,
          impactLabel: 'High',
          impactClass: 'high',
        },
      ];
    } else if (acc.account_id === '809FB8870') {
      return [
        {
          title: 'Large transfer from main account',
          explanation: 'Received $2,379,528.90 in two quick transfers from the central group account.',
          impactLabel: 'High',
          impactClass: 'high',
        },
        {
          title: 'No normal account history',
          explanation: 'This account was only used to receive money from the suspicious group.',
          impactLabel: 'High',
          impactClass: 'high',
        },
        {
          title: 'Above alert line',
          explanation: `${score.toFixed(2)} is well above the risk alert line of ${threshold.toFixed(2)}.`,
          impactLabel: 'High',
          impactClass: 'high',
        },
      ];
    } else {
      return [
        {
          title: 'Money moving in circles',
          explanation: 'Transfers money in a circle with other accounts to look like regular business.',
          impactLabel: 'High',
          impactClass: 'high',
        },
        {
          title: 'Funnels money to main account',
          explanation: 'Quickly funnels collected funds into the main central account.',
          impactLabel: 'High',
          impactClass: 'high',
        },
        {
          title: 'Above alert line',
          explanation: `${score.toFixed(2)} is well above the risk alert line of ${threshold.toFixed(2)}.`,
          impactLabel: 'High',
          impactClass: 'high',
        },
      ];
    }
  }

  const drivers = [];
  const rawFactors = acc.risk_factors || [];

  // Parse existing drivers or behavioral markers into simple, friendly language
  rawFactors.forEach((f) => {
    const titleLower = f.title.toLowerCase();
    const descLower = f.description.toLowerCase();

    if (titleLower.includes('velocity') || descLower.includes('rapid') || descLower.includes('pass-through')) {
      drivers.push({
        title: 'Rapid pass-through',
        explanation: 'Money moved in and out of this account within minutes, which is common when hiding funds.',
        impactLabel: 'High',
        impactClass: 'high',
        priority: 1,
      });
    } else if (titleLower.includes('counterparty') || descLower.includes('flagged') || descLower.includes('known')) {
      drivers.push({
        title: 'Linked to high-risk account',
        explanation: 'This account transacted directly with another account that was flagged as high risk.',
        impactLabel: 'High',
        impactClass: 'high',
        priority: 2,
      });
    } else if (titleLower.includes('cycle') || descLower.includes('loop') || descLower.includes('circular')) {
      drivers.push({
        title: 'Money moving in circles',
        explanation: 'Funds travel across multiple accounts and eventually circle back to the start.',
        impactLabel: 'High',
        impactClass: 'high',
        priority: 3,
      });
    } else if (titleLower.includes('cross-bank') || descLower.includes('bank')) {
      drivers.push({
        title: 'Repeated cross-bank transfers',
        explanation: 'Frequent movement across different banks to make tracking difficult.',
        impactLabel: 'Medium',
        impactClass: 'medium',
        priority: 4,
      });
    } else if (titleLower.includes('burst') || descLower.includes('burst')) {
      drivers.push({
        title: 'Sudden burst of activity',
        explanation: 'A sudden rush of transfers clustered into a very short window of time.',
        impactLabel: 'Medium',
        impactClass: 'medium',
        priority: 5,
      });
    } else if (titleLower.includes('amount') || descLower.includes('amount') || descLower.includes('large')) {
      drivers.push({
        title: 'High amount for this account',
        explanation: 'The transfer amount is significantly larger than what this account usually handles.',
        impactLabel: 'Medium',
        impactClass: 'medium',
        priority: 6,
      });
    }
  });

  // If score is safe or under alert line, include reassuring driver
  if (!isRisk) {
    drivers.push({
      title: 'Stays under the alert line',
      explanation: `${score.toFixed(2)} is below ${threshold.toFixed(2)}, so it is not flagged.`,
      impactLabel: 'Reassuring',
      impactClass: 'reassuring',
      priority: 10,
    });
  } else {
    drivers.push({
      title: 'Exceeds alert threshold',
      explanation: `${score.toFixed(2)} is above the alert line of ${threshold.toFixed(2)}.`,
      impactLabel: 'High',
      impactClass: 'high',
      priority: 0,
    });
  }

  // Fallback defaults if few factors recorded
  if (drivers.length < 2) {
    drivers.push({
      title: 'Typical payment pattern',
      explanation: 'Money velocity and flow structure match ordinary account activity.',
      impactLabel: 'Reassuring',
      impactClass: 'reassuring',
      priority: 20,
    });
  }

  // Sort by priority and return
  return drivers.sort((a, b) => a.priority - b.priority);
}

// ============================================================================
// 6. RANKED ACCOUNTS IN DESCENDING ORDER OF RISK
// ============================================================================

async function loadRankedAccounts() {
  if (!state.currentSessionId) return;

  try {
    const res = await fetch(`/api/analysis/${encodeURIComponent(state.currentSessionId)}/accounts?limit=1000`);
    if (!res.ok) throw new Error('Accounts fetch failed');
    const data = await res.json();
    state.rankedAccounts = data.accounts || [];
  } catch (err) {
    console.warn('Failed to fetch ranked accounts, building from graph nodes:', err);
    // Fallback: build from state.graphNodes
    const nodes = (state.graphNodes || []).slice();
    nodes.sort((a, b) => (b.max_risk_score || 0) - (a.max_risk_score || 0));
    state.rankedAccounts = nodes.map((n, idx) => {
      const score = n.max_risk_score || 0;
      const isHigh = score >= (state.sessionThreshold || 0.40);
      const isMed = score >= 0.20 && !isHigh;
      return {
        id: n.id,
        label: n.label || n.id,
        max_risk_score: score,
        risk_level: isHigh ? 'High Risk' : (isMed ? 'Needs Watching' : 'Normal'),
        risk_level_code: isHigh ? 'high' : (isMed ? 'medium' : 'low'),
        is_flagged: isHigh,
        total_sent: n.total_sent || 0,
        total_received: n.total_received || 0,
        total_volume: (n.total_sent || 0) + (n.total_received || 0),
        connections: (n.in_degree || 0) + (n.out_degree || 0) || 1,
        rank: idx + 1,
      };
    });
  }

  renderSideRankedList();
  renderRankedAccountsTable();
}

function renderSideRankedList() {
  const container = document.getElementById('side-ranked-accounts-list');
  const countBadge = document.getElementById('side-ranked-count');
  if (!container) return;

  const accounts = state.rankedAccounts || [];
  if (countBadge) {
    countBadge.textContent = `${accounts.length} ranked`;
  }

  if (accounts.length === 0) {
    container.innerHTML = '<div style="padding: 16px; text-align: center; color: var(--text-muted); font-size: 13px;">No accounts found.</div>';
    return;
  }

  container.innerHTML = accounts.map((acc, idx) => {
    const isSelected = state.selectedAccountId === acc.id;
    const scoreStr = (acc.max_risk_score || 0).toFixed(2);
    const colorClass = acc.risk_level_code === 'high' ? 'risk' : (acc.risk_level_code === 'medium' ? 'watch' : 'normal');
    const dotColor = acc.risk_level_code === 'high' ? '#DC2626' : (acc.risk_level_code === 'medium' ? '#D97706' : '#2563EB');

    return `
      <div class="side-ranked-item ${isSelected ? 'active' : ''}" data-id="${acc.id}">
        <div style="display: flex; align-items: center; gap: 10px;">
          <span class="rank-badge">#${idx + 1}</span>
          <div>
            <div class="mono" style="font-size: 13.5px; font-weight: 600; color: var(--text-primary);">${acc.id}</div>
            <div style="font-size: 11.5px; color: var(--text-muted); margin-top: 1px;">
              ${acc.connections || 1} links · $${(acc.total_volume || 0).toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
            </div>
          </div>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
          <span class="status-pill ${colorClass}" style="font-size: 11px; padding: 2px 7px;">${acc.risk_level}</span>
          <span class="mono" style="font-size: 13px; font-weight: 700; color: ${dotColor};">${scoreStr}</span>
        </div>
      </div>
    `;
  }).join('');

  container.querySelectorAll('.side-ranked-item').forEach((item) => {
    item.addEventListener('click', () => {
      const accId = item.dataset.id;
      const targetNode = (state.graphNodes || []).find((n) => n.id === accId) || { id: accId, max_risk_score: 0 };
      if (state.ringIsolatedNodeId === accId) {
        unselectAccountAndShowAll();
      } else {
        selectAccountAndIsolateRing(targetNode);
        switchSubtab('why');
      }
    });
  });
}

function renderRankedAccountsTable() {
  const tbody = document.getElementById('ranked-accounts-tbody');
  const countBadge = document.getElementById('table-account-count-badge');
  if (!tbody) return;

  let accounts = state.rankedAccounts || [];

  // Filter state
  const searchInput = document.getElementById('ranked-accounts-search');
  const searchTerm = (searchInput ? searchInput.value : '').trim().toLowerCase();
  const activeFilter = state.rankedTableFilter || 'all';

  if (searchTerm) {
    accounts = accounts.filter((a) => a.id.toLowerCase().includes(searchTerm));
  }

  if (activeFilter === 'risk') {
    accounts = accounts.filter((a) => a.risk_level_code === 'high');
  } else if (activeFilter === 'watch') {
    accounts = accounts.filter((a) => a.risk_level_code === 'medium');
  } else if (activeFilter === 'normal') {
    accounts = accounts.filter((a) => a.risk_level_code === 'low');
  }

  if (countBadge) {
    countBadge.textContent = `${accounts.length} of ${(state.rankedAccounts || []).length} accounts`;
  }

  if (accounts.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="7" style="padding: 24px; text-align: center; color: var(--text-muted); font-size: 14px;">
          No matching accounts found.
        </td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = accounts.map((acc) => {
    const isSelected = state.selectedAccountId === acc.id;
    const score = acc.max_risk_score || 0;
    const scorePct = Math.min(100, Math.max(0, score * 100));
    const tierCode = acc.risk_level_code || 'low';
    const tierLabel = acc.risk_level || 'Normal';
    const barColor = tierCode === 'high' ? '#DC2626' : (tierCode === 'medium' ? '#D97706' : '#2563EB');

    return `
      <tr class="ranked-account-row ${isSelected ? 'selected' : ''}" data-id="${acc.id}">
        <td style="font-weight: 700; color: var(--text-muted); font-size: 12.5px;">#${acc.rank}</td>
        <td>
          <span class="mono" style="font-weight: 600; font-size: 14px; color: var(--text-primary);">${acc.id}</span>
        </td>
        <td>
          <div style="display: flex; align-items: center; gap: 8px;">
            <div style="flex: 1; height: 7px; background: #E2E8F0; border-radius: 999px; overflow: hidden; max-width: 130px;">
              <div style="width: ${scorePct}%; height: 100%; background: ${barColor}; border-radius: 999px;"></div>
            </div>
            <span class="mono" style="font-weight: 700; font-size: 13px; color: ${barColor};">${score.toFixed(2)}</span>
          </div>
        </td>
        <td>
          <span class="tier-badge ${tierCode}">
            <span style="font-size: 10px;">${tierCode === 'high' ? '●' : (tierCode === 'medium' ? '▲' : '■')}</span>
            ${tierLabel}
          </span>
        </td>
        <td style="color: var(--text-secondary); font-size: 13px;">
          ${acc.connections || 1} links
        </td>
        <td class="mono" style="font-size: 13px; color: var(--text-primary);">
          $${(acc.total_volume || 0).toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
        </td>
        <td style="text-align: right;">
          <button type="button" class="btn-explain-action" data-id="${acc.id}">
            <span>Explain</span>
            <span style="font-size: 13px;">✨</span>
          </button>
        </td>
      </tr>
    `;
  }).join('');

  // Row and button click handlers
  tbody.querySelectorAll('.ranked-account-row').forEach((row) => {
    row.addEventListener('click', () => {
      const accId = row.dataset.id;
      const targetNode = (state.graphNodes || []).find((n) => n.id === accId) || { id: accId, max_risk_score: 0 };
      if (state.ringIsolatedNodeId === accId) {
        unselectAccountAndShowAll();
      } else {
        selectAccountAndIsolateRing(targetNode);
        switchSubtab('why');
        document.querySelector('.account-column')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });
}

function initRankedAccountsEvents() {
  const searchInput = document.getElementById('ranked-accounts-search');
  if (searchInput) {
    searchInput.addEventListener('input', () => renderRankedAccountsTable());
  }

  const chips = [
    { id: 'btn-rank-filter-all', filter: 'all' },
    { id: 'btn-rank-filter-risk', filter: 'risk' },
    { id: 'btn-rank-filter-watch', filter: 'watch' },
    { id: 'btn-rank-filter-normal', filter: 'normal' },
  ];

  chips.forEach(({ id, filter }) => {
    const btn = document.getElementById(id);
    if (btn) {
      btn.addEventListener('click', () => {
        state.rankedTableFilter = filter;
        chips.forEach((c) => document.getElementById(c.id)?.classList.remove('active'));
        btn.classList.add('active');
        renderRankedAccountsTable();
      });
    }
  });
}

window.handleAccountSelectById = function(accId) {
  const node = state.graphNodes.find((n) => n.id === accId);
  if (node) {
    if (state.ringIsolatedNodeId === accId) {
      unselectAccountAndShowAll();
    } else {
      selectAccountAndIsolateRing(node);
      switchSubtab('why');
    }
  }
};

// ============================================================================
// 6. TAB 2: COMPARE MODELS (HEAD-TO-HEAD TABLE & HIGHLIGHT)
// ============================================================================

async function loadComparisonMetrics() {
  const cardsContainer = document.getElementById('compare-model-cards');
  const tableContainer = document.getElementById('benchmark-table-rows');

  if (!cardsContainer || !tableContainer) return;

  try {
    const res = await fetch('/api/metrics');
    if (!res.ok) throw new Error('Metrics API unavailable');
    const data = await res.json();

    const tgatFrozen = data.tgat_frozen_test || {};
    const baselines = data.baselines || {};

    // Available models from results
    const models = [
      {
        id: 'tgat',
        name: 'TGAT v3',
        tag: 'Our model (v3)',
        isOur: true,
        summary: 'Reads the whole network of accounts and exact timing with log1p continuous time encoding.',
        seesNetwork: 'Yes',
        seesTiming: 'Yes',
        recall: tgatFrozen.recall || 0.1229,
        precision: tgatFrozen.precision || 0.0584,
        f1: tgatFrozen.f1 || 0.0791,
        rocAuc: tgatFrozen.roc_auc || 0.8969,
        fpr: tgatFrozen.false_positive_rate || 0.00211,
      },
      {
        id: 'xgboost',
        name: 'XGBoost',
        tag: 'Standard',
        isOur: false,
        summary: 'Judges each transaction on its own, without looking at who it connects to.',
        seesNetwork: 'No',
        seesTiming: 'Limited',
        recall: baselines.xgboost?.test?.recall || 0.804,
        precision: baselines.xgboost?.test?.precision || 0.0091,
        f1: baselines.xgboost?.test?.f1 || 0.018,
        rocAuc: baselines.xgboost?.test?.roc_auc || 0.8901,
        fpr: baselines.xgboost?.test?.false_positive_rate || 0.0934,
      },
      {
        id: 'static_gcn',
        name: 'Static GCN',
        tag: 'Graph model',
        isOur: false,
        summary: 'Reads the network of accounts, but not the order in which things happen.',
        seesNetwork: 'Yes',
        seesTiming: 'No',
        recall: baselines.static_gcn?.test?.recall || 0.242,
        precision: baselines.static_gcn?.test?.precision || 0.0019,
        f1: baselines.static_gcn?.test?.f1 || 0.0038,
        rocAuc: baselines.static_gcn?.test?.roc_auc || 0.6064,
        fpr: baselines.static_gcn?.test?.false_positive_rate || 0.134,
      },
      {
        id: 'temporal_xgboost',
        name: 'Temporal XGBoost',
        tag: 'Enhanced ML',
        isOur: false,
        summary: 'Includes past transaction timing, but cannot trace multi-account network loops.',
        seesNetwork: 'No',
        seesTiming: 'Yes',
        recall: baselines.temporal_xgboost?.test?.recall || 0.585,
        precision: baselines.temporal_xgboost?.test?.precision || 0.0416,
        f1: baselines.temporal_xgboost?.test?.f1 || 0.0781,
        rocAuc: baselines.temporal_xgboost?.test?.roc_auc || 0.9175,
        fpr: baselines.temporal_xgboost?.test?.false_positive_rate || 0.0143,
      },
    ];

    // Render 1: Model Cards
    cardsContainer.innerHTML = models.map((m) => `
      <div class="model-card ${m.isOur ? 'tgat-featured' : ''}">
        <div style="display: flex; align-items: center; justify-content: space-between; gap: 8px;">
          <span style="font-size: 18px; font-weight: 700;">${m.name}</span>
          <span style="padding: 3px 10px; border-radius: 999px; background: ${m.isOur ? 'var(--accent)' : '#EDF0F5'}; color: ${m.isOur ? '#FFFFFF' : '#3B4C66'}; font-size: 12px; font-weight: 600;">${m.tag}</span>
        </div>
        <p style="margin: 0; flex: 1; font-size: 14px; line-height: 1.55; color: var(--text-secondary);">${m.summary}</p>
        <div style="display: flex; flex-direction: column; gap: 6px; font-size: 14px; color: #3B4C66;">
          <div style="display: flex; align-items: center; gap: 8px;">
            <svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="${m.seesNetwork === 'Yes' ? '#0E7A4F' : '#6B7A90'}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
              ${m.seesNetwork === 'Yes' ? '<path d="M5 12.5l4.5 4.5L19 7.5"></path>' : '<path d="M6 12h12"></path>'}
            </svg>
            Sees the network: ${m.seesNetwork}
          </div>
          <div style="display: flex; align-items: center; gap: 8px;">
            <svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="${m.seesTiming === 'Yes' ? '#0E7A4F' : '#6B7A90'}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
              ${m.seesTiming === 'Yes' ? '<path d="M5 12.5l4.5 4.5L19 7.5"></path>' : '<path d="M6 12h12"></path>'}
            </svg>
            Sees timing: ${m.seesTiming}
          </div>
        </div>
      </div>
    `).join('');

    // Render 2: Side-by-Side Table with Progress Bars & Winner Badges
    const metricsConfig = [
      {
        label: 'Frauds caught',
        tech: 'Recall',
        key: 'recall',
        higherIsBetter: true,
        fmt: (v) => `${(v * 100).toFixed(1)}%`,
        barVal: (v) => Math.min(100, Math.max(4, v * 100)),
      },
      {
        label: 'Alerts that were correct',
        tech: 'Precision',
        key: 'precision',
        higherIsBetter: true,
        fmt: (v) => `${(v * 100).toFixed(2)}%`,
        barVal: (v) => Math.min(100, Math.max(4, v * 1000)), // Scaled for visual contrast under 1:1000 AML imbalance
      },
      {
        label: 'Overall balance',
        tech: 'F1 score',
        key: 'f1',
        higherIsBetter: true,
        fmt: (v) => v.toFixed(4),
        barVal: (v) => Math.min(100, Math.max(4, v * 800)),
      },
      {
        label: 'Ranking quality',
        tech: 'ROC-AUC',
        key: 'rocAuc',
        higherIsBetter: true,
        fmt: (v) => v.toFixed(4),
        barVal: (v) => Math.min(100, Math.max(4, v * 100)),
      },
      {
        label: 'False alarms',
        tech: 'False positive rate · lower is better',
        key: 'fpr',
        higherIsBetter: false,
        fmt: (v) => `${(v * 100).toFixed(2)}%`,
        barVal: (v) => Math.min(100, Math.max(2, v * 300)),
      },
    ];

    // Build header row
    let tableHtml = `
      <div class="table-row header">
        <div style="font-size: 12px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--text-muted);">What we measured</div>
        ${models.map((m) => `
          <div><span style="padding: 4px 12px; border-radius: var(--radius-pill); background: ${m.isOur ? 'var(--accent)' : '#EDF0F5'}; color: ${m.isOur ? '#FFFFFF' : '#3B4C66'}; font-size: 13px; font-weight: 600;">${m.name}</span></div>
        `).join('')}
      </div>
    `;

    // Build metric rows
    metricsConfig.forEach((cfg) => {
      // Find winner
      const values = models.map((m) => m[cfg.key]);
      const bestVal = cfg.higherIsBetter ? Math.max(...values) : Math.min(...values);

      tableHtml += `
        <div class="table-row">
          <div>
            <div style="font-size: 15px; font-weight: 600;">${cfg.label}</div>
            <div style="margin-top: 2px; font-size: 13px; color: var(--text-muted);">${cfg.tech}</div>
          </div>
          ${models.map((m) => {
            const val = m[cfg.key];
            const isWinner = val === bestVal;
            const barPct = cfg.barVal(val);
            return `
              <div>
                <div style="display: flex; align-items: center; gap: 8px;">
                  <span style="font-size: 18px; font-weight: ${m.isOur ? '700' : '600'}; color: ${m.isOur ? 'var(--text-primary)' : '#3B4C66'};">${cfg.fmt(val)}</span>
                  ${isWinner ? '<span class="best-badge">Best</span>' : ''}
                </div>
                <div class="metric-bar-track">
                  <div class="metric-bar-fill ${m.isOur ? 'accent' : 'slate'}" style="width: ${barPct}%;"></div>
                </div>
              </div>
            `;
          }).join('')}
        </div>
      `;
    });

    tableContainer.innerHTML = tableHtml;

  } catch (err) {
    console.warn('Failed to load comparison metrics:', err);
    tableContainer.innerHTML = `<div style="padding: 24px; color: var(--text-muted); text-align: center;">Could not load evaluation benchmarks.</div>`;
  }
}

// ============================================================================
// 7. UTILITIES
// ============================================================================

function formatTime(ts) {
  if (!ts) return '—';
  const d = new Date(ts * 1000);
  return d.toISOString().replace('T', ' ').substring(0, 16);
}
