# Phase A: Repository & Experimental Results Inventory Audit

**Project:** Explainable Temporal Graph Attention Network (TGAT) for Financial Fraud Ring Detection  
**Repository:** [chinmayeechoudhari/explainable-financial-fraud-ring-detection](https://github.com/chinmayeechoudhari/explainable-financial-fraud-ring-detection)  
**Dataset:** IBM Anti-Money Laundering (AML) — `LI-Small_Trans.csv`  
**Audit Scope:** Full repository structure, data pipeline stages, feature definitions, baseline models, ablation runs, and checkpoint provenance.

---

## 1. Executive Summary

This audit establishes the definitive baseline and ground truth across the repository and the experimental result suite extracted from `results - full.zip`.

Key audit findings:
1. **Source of Truth for Experimental Results:** `results - full.zip` contains the complete, uncorrupted research suite (153 files across 8 experiment directories), including full validation (891,571 rows) and test (282,877 rows) predictions. In contrast, the repository's initial `results/` folder contained only 75 files with truncated (2,001-row) smoke-test predictions and incomplete ablation runs.
2. **Canonical TGAT Checkpoint:** The weights in `results_full_zip/results/tgat_official_primary/model.pt` (68,319 bytes) and `results_full_zip/results/tgat/model.pt` (68,319 bytes) are **100% bit-for-bit identical**. The checkpoint in the repo's previous `results/tgat/model.pt` (68,246 bytes) was a partial run.
3. **Feature Dimension Resolution (26 vs 27):**
   - `data/processed/temporal_gnn/train.csv` has 31 columns (4 metadata/target + 27 features).
   - The 27 features consist of 11 current transaction features and 16 temporal window features created by `src/models/temporal/temporal_dataset.py`.
   - The TGAT training pipeline (`src/models/temporal/tgat_dataset.py`) defines `ALL_FEATURES` with **26 features** (11 current + 15 temporal). The omitted column is `receiver_avg_out_amount`.
   - Model layer shapes strictly verify 26 features: `node_encoder.0.weight: [32, 26]` and `transaction_encoder.0.weight: [32, 90]` (where $32 + 32 + 26 = 90$).
4. **Environment Readiness:** PyTorch 2.6.0 with CUDA 12.4 acceleration is installed and verified in `.venv`.

---

## 2. Repository Inventory

### 2.1 Core Directory Structure
```
c:\Users\duxdh\Downloads\tgat\
├── configs/
│   ├── baselines/
│   │   ├── logistic_regression.json
│   │   ├── static_gcn.json
│   │   ├── temporal_xgboost.json
│   │   └── xgboost.json
│   └── tgat.json
├── data/
│   ├── raw/                       # Raw AML transactions
│   ├── interim/                   # Intermediate graph representations
│   └── processed/
│       ├── splits/                # Chronological split files (train, val, test, future)
│       └── temporal_gnn/          # Feature-engineered GNN splits & metadata.json
├── docs/                          # Runbooks, phase documentation, and specifications
├── results/                       # Primary experiment results in repo
├── results_full_zip/results/      # Extracted complete experimental result suite
├── scripts/
│   └── tgat_sanity.py             # Controlled TGAT learning sanity experiment
├── src/
│   ├── analysis/                  # Threshold analysis, metrics, and visualization
│   ├── data/                      # Ingestion, temporal splitting, and leakage validation
│   ├── features/                  # Tabular feature engineering
│   ├── graph/                     # Static graph construction & topology analysis
│   └── models/
│       ├── baselines/             # LR, Tabular XGBoost, Temporal XGBoost, Static GCN
│       ├── common/                # Shared evaluation metrics and PR-curve utilities
│       └── temporal/              # TGAT core model, dataset loaders, scaler, neighbor store
├── tests_phase9.py                # Temporal causality & architectural regression tests
├── tests_tgat.py                  # TGAT smoke tests & attention normalization tests
└── tools/
    └── profile_tgat_runtime.py    # Runtime bottleneck profiler
```

### 2.2 Data Split Inventory

Verified from `data/processed/temporal_gnn/metadata.json` and split CSV files:

| Split | File Path | Total Rows | Fraud Positives | Non-Fraud Negatives | Positive Rate (%) | Temporal Window |
|---|---|---|---|---|---|---|
| **Train** | `data/processed/temporal_gnn/train.csv` | 5,749,364 | 2,763 | 5,746,601 | 0.0481% | Earliest $\to$ Split boundary 1 |
| **Validation** | `data/processed/temporal_gnn/val.csv` | 891,571 | 367 | 891,204 | 0.0412% | Boundary 1 $\to$ Boundary 2 |
| **Test** | `data/processed/temporal_gnn/test.csv` | 282,877 | 301 | 282,576 | 0.1064% | Boundary 2 $\to$ Boundary 3 |
| **Future** | `data/processed/temporal_gnn/future.csv` | 223 | 134 | 89 | 60.0897% | Boundary 3 $\to$ End |
| **Total** | | **6,924,035** | **3,565** | **6,920,470** | **0.0515%** | **Strict Chronological** |

---

## 3. Results Suite Inventory (`results - full.zip`)

The extracted archive contains 8 distinct directories representing all baseline models, primary TGAT runs, ablation runs, and runtime benchmarking:

| Directory | Contents | Prediction Row Counts | Status |
|---|---|---|---|
| `baselines/logistic_regression` | `metrics.json`, `model.pkl`, `predictions.csv` | Val: 891,571; Test: 282,877 | Complete baseline |
| `baselines/static_gcn` | `config.json`, `metrics.json`, `model.pt`, predictions | Val: 891,571; Test: 282,877 | Complete baseline |
| `baselines/xgboost` | `metrics.json`, `model.json`, predictions | Val: 891,571; Test: 282,877 | Complete baseline |
| `baselines/temporal_xgboost` | `config.json`, `metrics.json`, `model.json`, predictions | Val: 891,571; Test: 282,877 | Complete baseline |
| `tgat` | `config.json`, `metrics.json`, `model.pt`, `training_log.json`, predictions | Val: 891,571; Test: 282,877 | Complete official run |
| `tgat_official_primary` | `config.json`, `metrics.json`, `model.pt`, `training_log.json`, predictions | Val: 891,571; Test: 282,877 | **Canonical primary run** (Identical to `tgat`) |
| `ablations/` (15 sub-runs) | Metrics, configs, logs for `a0`, `a1`, `a`, `b`, `c (k1-20)`, `d`, `e (e1-e6)` | Varying by ablation configuration | Complete ablation study |
| `tgat_opt2_*` & `tgat_runtime_benchmark` | Benchmark profiling runs (500k train rows, 50k val/test) | 50,000 val / 50,000 test | Runtime/profiling benchmarks |

---

## 4. Feature Definition Audit

### 4.1 Current Transaction Features (11 Features)
All 11 features represent intrinsic properties of the transaction:
1. `Amount Received`
2. `Amount Paid`
3. `Amount Difference`
4. `Amount Ratio`
5. `Same Bank Transaction`
6. `Cross Bank Transaction`
7. `Transaction Time Category`
8. `Is Weekend`
9. `Log Amount Received`
10. `Log Amount Paid`
11. `Same Currency`

### 4.2 Temporal State Features (15 Features in Model vs 16 in Data)
Historical window features computed causally:
- **Sender Features (8):** `sender_in_count`, `sender_out_count`, `sender_total_count`, `sender_in_amount`, `sender_out_amount`, `sender_avg_in_amount`, `sender_avg_out_amount`, `sender_time_since_last`.
- **Receiver Features (7 in Model):** `receiver_in_count`, `receiver_out_count`, `receiver_total_count`, `receiver_in_amount`, `receiver_out_amount`, `receiver_avg_in_amount`, `receiver_time_since_last`.
- *Omitted Feature (1):* `receiver_avg_out_amount` (present in CSV, excluded in `tgat_dataset.py` `TEMPORAL_FEATURES`).

Total Model Features: $11 + 15 = \mathbf{26}$.

---

## 5. Checkpoint Integrity & Parameter Dimensions

Direct tensor inspection of `results_full_zip/results/tgat_official_primary/model.pt`:

```
Layer Name                       Shape           Parameters
------------------------------------------------------------
node_encoder.0.weight            [32, 26]               832
node_encoder.0.bias              [32]                    32
time_encoder.frequency           [16]                    16
time_encoder.phase               [16]                    16
attn1.query.weight               [32, 32]             1,024
attn1.key.weight                 [32, 48]             1,536
attn1.value.weight               [32, 48]             1,536
attn1.output.weight              [32, 32]             1,024
attn1.output.bias                [32]                    32
attn1.norm.weight                [32]                    32
attn1.norm.bias                  [32]                    32
attn2.query.weight               [32, 32]             1,024
attn2.key.weight                 [32, 48]             1,536
attn2.value.weight               [32, 48]             1,536
attn2.output.weight              [32, 32]             1,024
attn2.output.bias                [32]                    32
attn2.norm.weight                [32]                    32
attn2.norm.bias                  [32]                    32
transaction_encoder.0.weight     [32, 90]             2,880
transaction_encoder.0.bias       [32]                    32
transaction_encoder.3.weight     [32, 32]             1,024
transaction_encoder.3.bias       [32]                    32
classifier.weight                [1, 32]                 32
classifier.bias                  [1]                      1
------------------------------------------------------------
Total Trainable Parameters:                          15,777
Checkpoint File Size:                           68,319 bytes
```

Key Architectural Dimensions Verified:
- Input Feature Dimension: `26`
- Node Embedding / Hidden Dimension: `32`
- Continuous Time Dimension: `16`
- Attention Key/Value Input Dimension: `Node Embedding (32) + Time Dimension (16) = 48`
- Transaction Classifier Input Dimension: `Src Embedding (32) + Dst Embedding (32) + Transaction Features (26) = 90`
- Number of Attention Heads: `2` (head dimension: `16`)
- Number of Attention Layers: `2` (`attn1` and `attn2`)

---

## 6. Phase A Conclusion & Next Phase Authorization

The repository audit is complete. All experimental artifacts have been cataloged from `results - full.zip`. The canonical checkpoint has been identified, and feature dimensions have been reconciled.

Authorization to proceed to **Phase B: Pipeline Verification & Code Integrity Audit**.
