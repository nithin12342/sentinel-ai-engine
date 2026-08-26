# D7 — sentinel-ai-engine · Detailed Completion Plan

*Source evidence: reports/sentinel-ai-engine.md (verified 2026-08-24) · nodes D7-1..D7-DEPLOY
from deployment_ready_minimum_diff_plan.md · intent from project_intent_analysis.md §D7.*

---

## 0. Identity

| Field | Value |
|---|---|
| Local path | `C:\Users\thela\Downloads\projects context\Magnificent-Seven\sentinel-ai-engine` |
| Remote | `github.com/nithin12342/sentinel-ai-engine` |
| Branch | `master` (synced 2026-08-25; test-suite commit `848f2b4` already pushed) |
| Intent | Machines degrade gradually; catch anomalies early via autoencoder trained on normal-only telemetry, served over HTTP so any monitoring stack can call it |
| Input → Output | sensor time-series windows → autoencoder reconstruction-error training + threshold calibration on normal data → `{is_anomaly, anomaly_score, threshold}` API responses varying with input severity |
| Deploy target | Render free tier serving REAL anomaly scores from a trained checkpoint |
| Verdict at study time | 🔴 Skeleton: 13 files/0.1MB; 3 trainers exist but nothing ever ran; server returns hard-coded mocks (`is_anomaly: False`, yield `8500`); zero datasets/checkpoints/Triton artifacts/tests-pre-existing |
| Estimated effort | ~1 day |

## 1. Verified defect inventory

| # | Defect | Exact location | Reproduced? |
|---|---|---|---|
| B1 | Broken layer def compiles but AttributeError at construction | `src/training/biometric_trainer.py:29` (`nn.MaxPool2d(2, nn.Conv2d2)`) | ✅ py_compile OK ≠ runnable |
| B2 | Biometric main() body fully commented out | `src/training/biometric_trainer.py:244-251` | ✅ read |
| B3 | Anomaly trainer saves into nonexistent dir; `__main__` only prints "configured" | `src/training/anomaly_detector.py` end (torch.save → models/) | ✅ no models/ dir exists |
| B4 | ALL inference endpoints return hard-coded dicts; startup model loading commented out; /metrics constant zeros | `src/inference/server.py:52-55,69,105,141,166-173` | ✅ read |
| B5 | Azure ML score.py NameError: `io.BytesIO` at :49 vs local `import io` at :107; model path dead-codes to None | `serving/score.py:49,107,125` | ✅ read |
| B6 | Monitoring script calls nonexistent SDK APIs / private attr chains | `monitoring/model_monitor.py` (`_ml_client.monitors...`) | ✅ inspection |
| B7 | Feast layout invalid; registry URI = fabricated storage account | `feature_store/features.yaml:6` | ✅ inspection |
| B8 | Terraform targets Enterprise AML SKU (cost trap); no compute cluster defined | `infrastructure/azureml/terraform/main.tf` | ✅ inspection |
| B9 | Zero Triton configs despite headline claim (`config.pbtxt` count=0) | repo tree | ✅ glob |

## 2. Node plan

### P0 RUN-FIX

**Node D7-1 — scope cut: anomaly only**
```
GOAL      : single use case remains in serving path; broken cases parked honestly.
LOCATION  : README · src/training/biometric_trainer.py:29 (fix Conv2d2→Conv2d opportunistically,
            1 token) · agricultural_predictor.py parked behind ROADMAP · DELETE their mock
            endpoints from src/inference/server.py
MIN-DIFF  : README truth-pass + endpoint deletions + 1-token fix
VERIFY    : grep -c "biometric\|agricultural" src/inference/server.py → 0 in serving path.
            Artifact: verification/d7-1_scope.txt
SIBLINGS  : none yet
```

**Node D7-2 — training produces an artifact**
```
GOAL      : committed script trains the autoencoder and leaves a real checkpoint.
LOCATION  : src/training/anomaly_detector.py (os.makedirs("models", exist_ok=True) before
            torch.save; wire __main__ to actually train on synthetic windows;
            MLFLOW_TRACKING_URI=file:./mlruns)
VERIFY    : python src/training/anomaly_detector.py
EXPECTED  : models/anomaly_detector.pt exists; MLflow run visible in ./mlruns.
            Artifact: verification/d7-2_train.txt + checkpoint hash
```

**Node D7-3 — synthetic data source**
```
GOAL      : reproducible data feeding the trainer.
LOCATION  : NEW scripts/make_synthetic_data.py (~50 lines: normal patterns + injected
            fault signatures; CSV schema temperature/vibration per timestamp)
VERIFY    : regenerate → file stable; detector trains on it. Artifact: verification/d7-3_data.txt +
            small sample CSV committed
TAG       : ckpt-runs   (sibling re-runs: D7-1 scope check)
```

### P1 VERIFY

**Node D7-4 — server serves truth**
```
GOAL      : anomaly score is input-dependent and derived from the loaded checkpoint;
            high-anomaly payload crosses threshold.
LOCATION  : src/inference/server.py:52-55 (load checkpoint at startup), :69/:105/:141
            (replace remaining mocks), :166-173 (/metrics from real counters)
VERIFY    : pytest TestClient: two different payloads → different scores; anomalous > threshold flag true.
            Artifact: verification/d7-4_serve.txt (curl/TestClient transcript)
SIBLINGS  : D7-1..3 gates re-run first
```

**Node D7-5 — tests exist**
```
LOCATION  : NEW tests/test_anomaly.py (~8 tests: forward shape, threshold behavior, API contract)
VERIFY    : pytest -q → ≥8 passing incl. CI. Artifact: verification/d7-5_pytest.txt
TAG       : ckpt-tested   (+ .github/workflows CI green with CPU torch)
```

### P3 DEPLOY

**Node D7-DEPLOY — public scorer**
```
LOCATION  : Dockerfile (cpu torch slim base) + Render free deploy wiring
VERIFY    : live URL POST /api/v1/anomaly/detect → score varies with input (anti-mock proof)
PROOF     : transcript + URL → verification/d7_deploy_proof.txt. Tag ckpt-deployed.
ROADMAP    : biometric/agri use cases (need real datasets) · Triton serving (needs real
            config.pbtxt + model repo) · ONNX export · Azure ML free-workspace tracking ·
            walk-forward threshold recalibration
```

## 3. Out of scope (ROADMAP)

Biometric + agricultural trainers until real datasets exist · Triton/Feast/Snowflake/AzureML-
Enterprise claims (deleted from README, moved to ROADMAP) · AKS endpoints.

## 4. Execution contract

POST-FIX scope check per node · sibling gates re-run before tags · evidence committed atomically ·
tags pushed same day · toolchain: Python ✓ + CPU torch (pip install when batch starts).

## 5. P4 — PRODUCTION READINESS DELTA (target: `prod-ready` tag, L4)

Current level after P3 ≈ L3.
| Cat | Gap | Node | VERIFY artifact |
|---|---|---|---|
| G1/G2 | Missing/corrupt checkpoint behavior; input caps undefined | D7-P4a: fail-fast on missing model (no silent 500s); payload size + reading-count caps validated | fail-fast transcript + 422 responses |
| G2+ | Container runs as root by default | D7-P4b: Dockerfile non-root user + trivy image scan clean of CRITICALs | scan log |
| G3 | Checkpoint unversioned — can't roll back a bad model | D7-P4c: model registry convention (hash-named artifacts + metadata json); rollback = env var switch, drilled once | registry listing + rollback drill |
| G5 | /metrics real counters exist post-D7-4 but nothing scrapes/alerts | D7-P4d: uptime monitor (free) + anomaly-rate alert rule defined | monitor screenshot |
| G6/G8 | Threshold recalibration procedure documented (drift = false alarms) | D7-P4e: RUNBOOK.md incl. recalibration job steps | doc review |
| G7 | Serving latency unknown | D7-P4f: p99 measured at 50 rps burst; saturation behavior stated | load transcript |

TRACK=product (anomaly use case only).

### P4 audit addendum (production-readiness pass)
| Cat | Gap found in audit | Node | VERIFY artifact |
|---|---|---|---|
| UB | Universal Baseline UB1-UB6 applies | D7-P4g: LICENSE, gitleaks job, pip-audit job (torch slim image rescan), README badge row — uptime/error-tracking already in P4c/P4d | per-UB artifacts |

