# Hybrid Behavioral Authentication

Keystroke dynamics + risk-based authentication (RBA), built from the IC3IT 2025 paper *"ML and DL Powered Behavioral Authentication Using Typing and Login Risk Data"* (`docs/`), its two notebooks (`notebooks/original/`), and its two datasets (`data/raw/`).

The original notebooks were exploratory (label-encoded high-cardinality columns, `User ID`/IP as features, a missing TensorFlow install, placeholder paths). This project rebuilds them as a reproducible pipeline with leakage-safe features, proper verification metrics (EER/FAR/FRR), a score-fusion decision engine, tests, and an API.

## Pipeline

| Stage | Data | Task | Method |
|---|---|---|---|
| Keystroke identification | CMU DSL (20,400 reps, 51 users) | 51-way user ID | LR, KNN, SVM, RF, LightGBM, MLP; 5-fold CV |
| Keystroke verification | same | genuine vs impostor per user | Scaled-Manhattan / Mahalanobis one-class detectors; train on first 200 reps, test on remaining 200 + other users' reps; EER |
| RBA | `rba_dataset_cleaned.csv` (9,991 logins after dedup) | attack-IP detection (primary), login-success (paper baseline) | LR, RF, LightGBM, MLP; 5-fold CV |
| Fusion | simulated pairing of both | allow / challenge (step-up) / deny | logistic regression on keystroke z-score + context-risk logit; thresholds from FAR/FRR targets |
| Service | trained artifacts | `POST /authenticate` | FastAPI |

## Results (`reports/`, reproduced by `make run`, ~6 min CPU)

**Keystroke identification (accuracy, 5-fold CV):** LightGBM 94.55% · RandomForest 93.26% · MLP 91.00% · SVM-RBF 90.09% · LR 85.38% · KNN 84.16%. Matches the paper's LightGBM claim (94.68%).

**Keystroke verification (mean per-user EER):** Scaled-Manhattan 14.6% (±10.2) · Mahalanobis 15.8% (±8.7).

**RBA, attack-IP target (ROC-AUC / PR-AUC):** MLP 0.880/0.452 · LightGBM 0.880/0.451 · RF 0.879/0.430 · LR 0.875/0.408. Base rate is 10.2%, so accuracy is not reported as the headline. LightGBM at threshold 0.5 has recall 0.88, precision 0.32.

**RBA, login-success target (paper's target):** best accuracy 73.6% (AUC 0.785). The paper reports 86.05% and the original notebook reached 84.6% (RF); that came from using `User ID`, raw IP and raw user-agent as features. Not reproducible without that leakage.

**Fusion (pooled EER, simulated attempts):**

| P(impostor presents attack-like context) | Keystroke only | Context only | Fusion |
|---|---|---|---|
| 0.00 | 18.8% | 50.0% | 18.9% |
| 0.25 | 18.8% | 43.3% | 18.4% |
| 0.50 | 18.8% | 35.2% | 17.0% |
| 0.75 | 18.8% | 25.4% | 14.7% |
| 1.00 | 18.8% | 18.0% | 10.7% |

Fusion helps only to the extent attackers present risky context. Against a context-stealthy attacker (q=0) it equals keystroke-only.

**Decision policy (q=0.5, allow at 1% FAR, deny below 1% genuine quantile):** genuine users: 22.4% allow / 76.5% challenge / 1.1% deny. Impostors: 0.8% allow / 64.8% challenge / 34.4% deny. A strict 1% FAR makes most genuine logins step-up (OTP).

## Discrepancies between paper and supplied data

1. Keystroke dataset has **20,400 samples from 51 users** (one fixed password, 400 reps each). The paper states 204k samples / 4,000+ users (Aalto) — not what was supplied.
2. RBA file has **10,000 rows, 5,431 distinct User IDs**, not 500 users; paper also mentions 52k sessions.
3. `Is Account Takeover` is constant `False`; unusable as a label. `Is Attack IP` (10.2% positive) is used instead.
4. `Login Timestamp` is truncated to `mm:ss.s`; the paper's hour/weekday sine-cosine features cannot be built.
5. The two datasets share no user identities, so the paper's "feature fusion" and "user unification" cannot be done on real pairs. Fusion here is **simulated**; its numbers demonstrate the mechanism, not field performance.
6. Paper preprocessing steps (jitter augmentation, CNN/LSTM/GRU) are not reproduced; TensorFlow was unavailable in the original notebook's environment. A scikit-learn MLP stands in as the neural baseline.

## Layout

```
data/raw/            both CSVs
docs/                paper PDF
notebooks/original/  the two source notebooks, untouched
src/hba/             keystroke.py  rba.py  fusion.py  metrics.py  api.py
scripts/run_pipeline.py
tests/test_core.py
reports/             CSVs, results.json, figures/
models/              hba_artifacts.joblib (generated)
```

## Usage

```bash
pip install -r requirements.txt
make run      # train, evaluate, write reports/ and models/
make test
make serve    # http://127.0.0.1:8000/docs
```

```bash
curl -X POST localhost:8000/authenticate -H 'content-type: application/json' -d '{
  "subject":"s002",
  "keystroke":{"H.period":0.149,"DD.period.t":0.398, "...":"all 31 DSL timing features"},
  "context":{"country":"NO","asn":41164,"browser":"Chrome","os":"Windows","device_type":"desktop","round_trip_ms":539}
}'
```
Response: `decision` (allow/challenge/deny), `fused_score`, `keystroke_z`, `context_risk`, thresholds.

## Limitations

Single typed password, lab-collected keystrokes; verification EER is high for a production gate. The RBA label marks attack IPs, which is partly an IP-reputation proxy. Hyperparameters were not tuned. API enrolls all 400 reps per user and has no template update or rate limiting.
