"""End-to-end: keystroke ID + verification, RBA, fusion. Writes reports/ and models/."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hba import fusion as F, keystroke as K, rba as R
from hba.config import MODELS, REPORTS, SEED
from hba.metrics import eer

REPORTS.mkdir(exist_ok=True); (REPORTS / "figures").mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
t0 = time.time()
log = lambda m: print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
results = {}

# ---- 1. Keystroke identification
ks_df = K.load_keystroke()
log(f"keystroke data {ks_df.shape}, subjects={ks_df.subject.nunique()}")
ident = K.evaluate_identification(ks_df)
ident.to_csv(REPORTS / "keystroke_identification.csv", index=False)
results["keystroke_identification"] = ident.round(4).to_dict("records")
log("identification:\n" + ident.round(4).to_string())

# ---- 2. Keystroke verification (authentication proper)
ver = {}
scores = {}
for det in K.DETECTORS:
    sc = K.verification_scores(ks_df, det)
    per_user, m, s = K.verification_summary(sc)
    ver[det] = dict(mean_eer=m, std_eer=s)
    scores[det] = sc
    per_user.to_csv(REPORTS / f"keystroke_verification_eer_{det}.csv", index=False)
results["keystroke_verification"] = ver
log(f"verification EER: {ver}")
best_det = min(ver, key=lambda d: ver[d]["mean_eer"])
results["keystroke_verification_best_detector"] = best_det

# ---- 3. RBA
rba_df = R.load_rba()
log(f"rba data {rba_df.shape}; attack-IP rate={rba_df['Is Attack IP'].mean():.3f}")
rba_attack = R.evaluate(rba_df, "attack_ip")
rba_login = R.evaluate(rba_df, "login_success")
rba_attack.to_csv(REPORTS / "rba_attack_ip.csv", index=False)
rba_login.to_csv(REPORTS / "rba_login_success.csv", index=False)
results["rba_attack_ip"] = rba_attack.round(4).to_dict("records")
results["rba_login_success"] = rba_login.round(4).to_dict("records")
log("RBA attack_ip:\n" + rba_attack.round(4).to_string())
log("RBA login_success:\n" + rba_login.round(4).to_string())

# ---- 4. Fusion (simulated pairing)
p_risk = R.oof_risk(rba_df, "attack_ip")
is_attack = rba_df["Is Attack IP"].values
fusion_rows = []
for q in (0.0, 0.25, 0.5, 0.75, 1.0):
    tr, te = F.simulate_attempts(scores[best_det], p_risk, is_attack, stealth_q=q)
    fu = F.Fusion().fit(tr).fit_policy(tr)
    ev = F.evaluate_fusion(tr, te, fu)
    fusion_rows.append(dict(attacker_risky_context_prob=q, **{k: ev[k]["eer"] for k in
                       ("keystroke_only", "context_only", "fusion")}))
    if q == 0.5:
        results["policy_q0.5"] = ev["policy"]
        final_fusion, final_tr, final_te = fu, tr, te
fus = pd.DataFrame(fusion_rows)
fus.to_csv(REPORTS / "fusion_eer_by_attacker_context.csv", index=False)
results["fusion_eer"] = fus.round(4).to_dict("records")
log("fusion EER:\n" + fus.round(4).to_string())

# ---- 5. Deployable artifacts for API (all 400 reps per user enrolled)
raw = ks_df.sort_values(["subject", "sessionIndex", "rep"])
cols = K.timing_columns(raw)
rng = np.random.default_rng(SEED)
users = {}
for u, g in raw.groupby("subject"):
    det, mu, sd = K.fit_user_detector(g[cols].values, best_det, rng)
    users[u] = dict(det=det, mu=mu, sd=sd)
Xr = R.build_features(rba_df)
rba_final = R.rba_models()["LightGBM"].fit(Xr, rba_df["Is Attack IP"].astype(int).values)
joblib.dump(dict(users=users, cols=cols, detector=best_det, rba=rba_final, fusion=final_fusion),
            MODELS / "hba_artifacts.joblib", compress=3)

# ---- figures
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].barh(ident["model"][::-1], ident["accuracy"][::-1], xerr=ident["accuracy_std"][::-1], color="#3b6ea5")
ax[0].set_xlim(0, 1); ax[0].set_title("Keystroke identification accuracy (51 users, 5-fold CV)")
r = rba_attack.sort_values("roc_auc")
ax[1].barh(r["model"], r["roc_auc"], color="#b5533c"); ax[1].set_xlim(0.5, 1)
ax[1].set_title("RBA attack-IP detection ROC-AUC (5-fold CV)")
fig.tight_layout(); fig.savefig(REPORTS / "figures" / "model_comparison.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 4))
for c, col in (("keystroke_only", "#3b6ea5"), ("context_only", "#999999"), ("fusion", "#b5533c")):
    ax.plot(fus["attacker_risky_context_prob"], fus[c] * 100, marker="o", label=c.replace("_", " "), color=col)
ax.set_xlabel("P(impostor presents attack-like context)"); ax.set_ylabel("EER (%)")
ax.set_title("Fusion vs single-modality EER (simulated pairing)"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(REPORTS / "figures" / "fusion_eer.png", dpi=150); plt.close(fig)

results["runtime_seconds"] = round(time.time() - t0, 1)
(REPORTS / "results.json").write_text(json.dumps(results, indent=2, default=float))
log("done")
