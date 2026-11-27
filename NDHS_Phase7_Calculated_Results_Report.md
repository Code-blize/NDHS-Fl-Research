# NDHS Federated Learning Research — Calculated Results Report

**Purpose:** Manuscript-supporting record of the calculated results available at the Phase 7 closeout, before moving to Phase 8.  
**Study:** Federated learning for classifying facility delivery versus home delivery using the 2018 Nigeria Demographic and Health Survey (NDHS) Individual Recode analytic sample.  
**Status:** Results report, not a submission-ready manuscript. Survey-weighted sensitivity analysis remains outstanding.

---

## 1. Scope and evidence status

This report consolidates the documented calculations from the research report draft and the confirmatory validation metrics. It distinguishes completed calculations from analyses that are not yet evidenced.

- Confirmatory design: 2 feature configurations × 2 algorithms × 10 seeds (42–51) = **40 runs**.
- Primary checkpoint: communication round 10.
- Algorithms: FedAvg and FedProx (μ = 0.01).
- Fixed clients: Nigeria’s six geopolitical zones, with full participation.
- Validation metrics file: 3,080 records = 40 runs × 11 checkpoints (rounds 0–10) × 7 scopes (pooled + six regions).
- Test evaluation: 280 records = 40 models × 7 scopes.
- The survey-weighted sensitivity analysis is **not complete**. The aggregate metrics do not contain person-level predictions needed to apply individual survey weights.
- The results do not include a centralized pooled-training or local-only benchmark.

The ten seeds quantify stochastic training-run variability conditional on this dataset and partition. They are not ten independent DHS samples and do not quantify survey-sampling uncertainty.

## 2. Analytic sample and partition

### 2.1 Outcome and sample

| Outcome | n | Percent |
|---|---:|---:|
| Home delivery (0) | 12,441 | 58.02% |
| Facility delivery (1) | 9,001 | 41.98% |
| **Total** | **21,442** | **100.00%** |

The analytic sample was derived from the 2018 NDHS Individual Recode file `NGIR7BFL.DTA`. Home-delivery codes were 11 and 12; facility-delivery codes were 21, 22, 23, 26, 31 and 36. The target is facility delivery versus home delivery, not a general maternal-health outcome.

### 2.2 Regional outcome distribution

| Zone | N | Home | Facility | Facility prevalence |
|---|---:|---:|---:|---:|
| NC | 3,838 | 1,841 | 1,997 | 52.03% |
| NE | 4,498 | 3,329 | 1,169 | 25.99% |
| NW | 6,307 | 5,318 | 989 | 15.68% |
| SE | 2,338 | 451 | 1,887 | 80.71% |
| SS | 2,050 | 1,052 | 998 | 48.68% |
| SW | 2,411 | 450 | 1,961 | 81.34% |

The prevalence range is large: 15.68% in NW versus 81.34% in SW (and 80.71% in SE), supporting the presence of substantial outcome-distribution differences across the fixed regional clients.

### 2.3 Partition

| Partition | n | Share | PSUs |
|---|---:|---:|---:|
| Training | 12,851 | 59.94% | 836 |
| Validation | 4,299 | 20.05% | 279 |
| Test | 4,292 | 20.02% | 274 |

The partition was PSU-grouped and stratified; no PSU crossed partitions. Both feature configurations used the same row assignments. The test partition was not used for training or model selection.

## 3. Geographic heterogeneity calculations

Outcome-distribution divergence was calculated using base-2 Jensen–Shannon divergence between each client and the pooled outcome distribution.

| Zone | Outcome JS divergence |
|---|---:|
| NC | 0.007330 |
| NE | 0.020695 |
| NW | 0.062523 |
| SE | 0.118117 |
| SS | 0.003274 |
| SW | 0.122553 |

Feature-distribution heterogeneity was assessed using RBF-kernel MMD², comparing each zone with the pooled other five zones, 1,000 observations per side, gamma = 0.02, biased V-statistic, deterministic seeds 2027–2032, and training data only.

| Zone | Baseline MMD² (15 inputs) | Expanded MMD² (51 inputs) |
|---|---:|---:|
| NC | 0.002327 | 0.011836 |
| NE | 0.010286 | 0.029523 |
| NW | 0.017257 | 0.066556 |
| SE | 0.029465 | 0.102690 |
| SS | 0.016978 | 0.051556 |
| SW | 0.034352 | 0.103236 |

The MMD² ranking was the same in both feature spaces: SW, SE, NW, SS, NE, NC. Absolute MMD² magnitudes should **not** be compared directly across 15- and 51-dimensional spaces because feature dimension changes RBF-kernel geometry. These calculations are descriptive and do not establish a causal mechanism.

## 4. Confirmatory experiment and implementation

| Design element | Frozen setting |
|---|---|
| Feature configurations | Baseline (15 processed inputs); expanded (51 processed inputs) |
| Algorithms | FedAvg (μ = 0); FedProx (μ = 0.01) |
| Seeds | 42–51 |
| Communication rounds | 10 |
| Local epochs | 5 |
| Clients | Six regional clients; full participation |
| Aggregation | Sample-size weighted |
| Primary checkpoint | Round 10 |

All 40 runs were accepted. The manifests record ten rounds, six clients per round, correct regional identity/sample counts, and 77 validation metric records per run. Matched initial parameters were verified for all 20 FedAvg/FedProx pairs. Seeds 42–43 overlap with pilot runs; seeds 44–51 form the prespecified sensitivity subset. This overlap is disclosed.

The implementation used Flower 1.43.0, PyTorch 2.15.0+cpu and Python 3.12.10; the official implementation branch is identified as Flower v1.2.1. The model was input → 16 → 8 → 1 with ReLU hidden activations and one output logit. Training used BCEWithLogitsLoss, Adam (learning rate 0.001), batch size 32 and threshold 0.5. A new Adam optimizer was instantiated for each client in each communication round, resetting optimizer moments between client-rounds.

## 5. Final-round pooled validation performance

Values are mean (SD) across the ten seeds, expressed as percentages except Brier score and binary cross-entropy. SD is across seeds, not a survey-design standard error.

| Metric | Baseline FedAvg | Baseline FedProx | Expanded FedAvg | Expanded FedProx |
|---|---:|---:|---:|---:|
| Accuracy | 75.93 (0.24) | 75.93 (0.21) | 80.46 (0.18) | 80.70 (0.18) |
| Precision | 72.50 (0.31) | 72.42 (0.40) | 76.86 (0.79) | 77.15 (0.72) |
| Sensitivity | 68.79 (1.06) | 68.97 (1.22) | 76.57 (1.60) | 76.84 (1.48) |
| Specificity | 81.10 (0.52) | 80.97 (0.67) | 83.28 (1.05) | 83.49 (0.96) |
| F1 | 70.59 (0.50) | 70.65 (0.52) | 76.69 (0.48) | 76.98 (0.45) |
| Macro-F1 | 75.11 (0.30) | 75.12 (0.28) | 79.93 (0.24) | 80.18 (0.23) |
| Balanced accuracy | 74.94 (0.33) | 74.97 (0.33) | 79.92 (0.33) | 80.17 (0.31) |
| AUROC | 84.03 (0.09) | 84.07 (0.07) | 88.27 (0.06) | 88.32 (0.05) |
| Average precision / AUPRC | 77.89 (0.22) | 77.96 (0.20) | 83.72 (0.11) | 83.35 (0.15) |
| Brier score | 0.16069 (0.00049) | 0.16056 (0.00041) | 0.13669 (0.00051) | 0.13617 (0.00033) |
| Binary cross-entropy | 0.48934 (0.00128) | 0.48888 (0.00113) | 0.42559 (0.00118) | 0.42433 (0.00065) |
| Predicted positive prevalence | 39.86 (0.72) | 40.01 (0.89) | 41.86 (1.27) | 41.85 (1.17) |

Observed validation prevalence was 42.01% in every run. In this table, the percentages and SDs for predicted positive prevalence are percentage points.

### 5.1 Paired pooled comparisons: FedProx minus FedAvg

Paired comparisons use the same seed within each feature configuration. Differences below are percentage points; confidence intervals are paired t-based 95% CIs across ten seed-level differences.

| Feature set | Metric | Mean difference | 95% CI | p-value |
|---|---|---:|---:|---:|
| Baseline | Macro-F1 | +0.014 | [−0.156, +0.183] | 0.860 |
| Expanded | Macro-F1 | +0.248 | [+0.118, +0.379] | 0.00198 |
| Expanded | AUROC | +0.048 | [+0.016, +0.079] | 0.00774 |
| Expanded | AUPRC / average precision | −0.370 | [−0.457, −0.283] | 0.00000488 |

The expanded-feature pooled macro-F1 and AUROC differences favour FedProx, while AUPRC is lower. These are distinct metrics; no algorithm dominates every pooled metric. The baseline macro-F1 difference is negligible.

### 5.2 Expanded features versus baseline

Mean pooled macro-F1 across seeds was 79.93% for expanded FedAvg versus 75.11% for baseline FedAvg: **+4.83 percentage points** (paired 95% CI +4.65 to +5.00). For FedProx, expanded macro-F1 was 80.18% versus 75.12% for baseline: **+5.06 points** (paired 95% CI +4.85 to +5.27). Both paired comparisons were reported as p < .001.

## 6. Regional robustness: validation

The primary regional-robustness endpoint was the population standard deviation (ddof = 0) of balanced accuracy across the six fixed regions within each run. Paired differences are FedProx minus FedAvg. Negative regional-SD differences favour lower regional dispersion for FedProx.

| Feature set | Measure | Mean paired difference | 95% CI |
|---|---|---:|---:|
| Baseline | Regional BA SD | −0.00005 | [−0.00154, +0.00144] |
| Baseline | Worst-region BA | −0.00391 | [−0.00722, −0.00059] |
| Baseline | Regional BA range | +0.00033 | [−0.00482, +0.00547] |
| Baseline | Mean regional macro-F1 | −0.00092 | [−0.00317, +0.00133] |
| Expanded | Regional BA SD | −0.01598 | [−0.01898, −0.01298] |
| Expanded | Worst-region BA | +0.04021 | [+0.02974, +0.05067] |
| Expanded | Regional BA range | −0.04111 | [−0.05051, −0.03170] |
| Expanded | Mean regional macro-F1 | +0.01868 | [+0.01652, +0.02084] |

For baseline features, the regional-dispersion interval includes zero; there is no clear dispersion advantage. FedProx’s mean worst-region balanced accuracy was modestly lower. For expanded features, the estimates favour FedProx on regional SD, worst-region BA, regional range and mean regional macro-F1.

### 6.1 Eight-seed sensitivity analysis (seeds 44–51)

| Feature set | Measure | Mean paired difference | 95% CI |
|---|---|---:|---:|
| Baseline | Regional BA SD | +0.00008 | [−0.00183, +0.00199] |
| Baseline | Worst-region BA | −0.00444 | [−0.00823, −0.00064] |
| Baseline | Regional BA range | +0.00056 | [−0.00617, +0.00729] |
| Expanded | Regional BA SD | −0.01642 | [−0.02028, −0.01256] |
| Expanded | Worst-region BA | +0.04376 | [+0.03185, +0.05566] |
| Expanded | Regional BA range | −0.04419 | [−0.05502, −0.03336] |
| Expanded | Mean regional macro-F1 | +0.01854 | [+0.01595, +0.02114] |

The eight-seed sensitivity analysis retains the main pattern: little difference in regional dispersion for baseline features and lower dispersion for FedProx with expanded features. It is a sensitivity analysis, not a replacement for the ten-seed primary analysis.

## 7. Held-out test evaluation

All 40 saved final-round models were evaluated on the held-out test partition; no retraining or model selection was performed using test outcomes. The audit recorded 40 unique runs, 40 pooled and 240 regional records, 4,292 observations per model, expected regional counts, confusion-count totals, accuracy arithmetic and output hashes.

### 7.1 Paired regional-robustness results

| Feature set | Measure | Validation difference (95% CI) | Test difference (95% CI) |
|---|---|---|---|
| Baseline | Regional BA SD | −0.00005 [−0.00154, +0.00144] | +0.00074 [−0.00039, +0.00188] |
| Baseline | Worst-region BA | −0.00391 [−0.00722, −0.00059] | −0.00153 [−0.00556, +0.00250] |
| Baseline | Regional BA range | +0.00033 [−0.00482, +0.00547] | +0.00191 [−0.00237, +0.00619] |
| Expanded | Regional BA SD | −0.01598 [−0.01898, −0.01298] | −0.00762 [−0.00973, −0.00552] |
| Expanded | Worst-region BA | +0.04021 [+0.02974, +0.05067] | +0.03717 [+0.02661, +0.04773] |
| Expanded | Regional BA range | −0.04111 [−0.05051, −0.03170] | −0.03163 [−0.03892, −0.02434] |

The expanded-feature FedProx regional-consistency pattern persisted on test data, although the regional-SD reduction was smaller than on validation. Baseline features showed no clear regional-dispersion advantage on either split.

### 7.2 Pooled test metrics

For expanded features, FedProx minus FedAvg was:

| Metric | Mean difference | 95% CI |
|---|---:|---:|
| Macro-F1 | +0.00224 | [+0.00039, +0.00409] |
| AUROC | +0.00145 | [+0.00095, +0.00194] |
| AUPRC / average precision | −0.00037 | [−0.00139, +0.00066] |

On test data, expanded FedProx had higher pooled macro-F1 and AUROC; the AUPRC difference was slightly negative and its interval included zero.

## 8. Interpretation supported by the calculations

1. The fixed regional clients have materially different facility-delivery prevalence and measurable feature-distribution differences.
2. Under baseline features, the evidence does not show a clear FedProx advantage in regional balanced-accuracy dispersion on validation or test.
3. Under expanded features, FedProx has lower regional balanced-accuracy dispersion, higher worst-region balanced accuracy and a narrower regional range; the pattern persists on held-out test data.
4. Pooled metric rankings are mixed. Regional robustness and pooled predictive performance are different dimensions.
5. The evidence supports a **feature-configuration-dependent regional-robustness finding**, not universal FedProx superiority.
6. No causal mechanism is established by the heterogeneity calculations. The study also cannot quantify a federated-versus-centralized gap because centralized and local-only benchmarks are absent.

## 9. Outstanding analysis and manuscript qualifications

- **Survey-weighted sensitivity analysis:** outstanding. It requires per-person validation predictions (or final checkpoints from which predictions can be generated in one batch), joined to survey design metadata. Aggregate confusion counts cannot be reweighted person-by-person.
- The appropriate survey-design variance method for nonlinear metrics such as macro-F1 must be specified using the survey strata/PSU design; seed-level intervals are not survey-sampling intervals.
- The analysis plan was approved on 16 November 2026, after confirmatory training had begun. It must not be described as prospectively approved.
- Some expanded predictors may be measured during or after pregnancy/delivery; describe the task as retrospective classification rather than necessarily prospective prediction at pregnancy onset.
- Religion and ethnicity may proxy structural or socioeconomic factors and require careful interpretation.
- The six geopolitical zones are fixed clients, not a random sample of regions.
- Results concern the defined 2018 NDHS analytic sample, not all Nigerian women.
- The approved proposal and supervisor requirements should be checked for any required centralized or local-only benchmark before claiming the research design is complete.
- Confirm citations, official NDHS documentation, proposal alignment, and institutional formatting before submission.

## 10. Manuscript-ready concise result statement

Across 40 confirmatory federated runs (two feature configurations, two algorithms and ten seeds), expanded features improved pooled macro-F1 relative to baseline for both FedAvg (+4.83 percentage points; paired 95% CI +4.65 to +5.00) and FedProx (+5.06 points; +4.85 to +5.27). FedProx’s regional-robustness results depended on feature configuration: with baseline features, there was no clear difference in regional balanced-accuracy dispersion on validation or test; with expanded features, FedProx reduced regional dispersion on validation (−0.01598; 95% CI −0.01898 to −0.01298) and test (−0.00762; −0.00973 to −0.00552), while improving worst-region balanced accuracy. Pooled metric differences were mixed, so the findings support a feature-configuration-dependent regional-robustness advantage rather than universal FedProx superiority. Survey-weighted sensitivity analysis remains outstanding.

---

**Source/evidence note:** This report consolidates the available research report draft, roadmap, confirmatory validation metrics and run-manifest audit. Where an analysis is marked outstanding, no result is imputed or presented as completed.
