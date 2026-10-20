"""Score-level fusion of keystroke verification and RBA context risk, plus decision policy.

The two public datasets share no user identities, so attempts are SIMULATED by pairing a
keystroke attempt (genuine = claimed user's own held-out reps, impostor = another subject's reps)
with an out-of-fold RBA context risk score. `stealth_q` is the probability that an impostor's
login context is drawn from the attack-IP pool; (1-q) of impostors present a benign-looking context.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .config import SEED
from .metrics import eer, far_frr

Z_CLIP = (-5.0, 30.0)


def _logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def features(z, p_risk):
    """Columns: keystroke genuineness (-z, clipped), context safety logit."""
    return np.column_stack([-np.clip(z, *Z_CLIP), _logit(1 - np.asarray(p_risk))])


def simulate_attempts(ks: pd.DataFrame, p_risk, is_attack, stealth_q=0.5, imp_per_source=4,
                      half=100, seed=SEED):
    """Build fusion train/test attempt tables from keystroke scores `ks` and RBA contexts.
    Split by keystroke rep index (<half = train, >=half = test) and by disjoint context halves."""
    rng = np.random.default_rng(seed)
    p_risk = np.asarray(p_risk); is_attack = np.asarray(is_attack).astype(bool)
    idx = rng.permutation(len(p_risk))
    ctx_split = {"train": idx[: len(idx) // 2], "test": idx[len(idx) // 2:]}
    pools = {s: dict(benign=ix[~is_attack[ix]], attack=ix[is_attack[ix]]) for s, ix in ctx_split.items()}
    res = {}
    for split in ("train", "test"):
        sel = (ks["rep_idx"] < half) if split == "train" else (ks["rep_idx"] >= half)
        d = ks[sel]
        g = d[d.kind == "genuine"]
        i = d[d.kind == "impostor"].groupby(["subject", "source"], group_keys=False).sample(
            n=imp_per_source, random_state=seed)
        att = pd.concat([g.assign(y=1), i.assign(y=0)], ignore_index=True)
        ctx = np.empty(len(att), dtype=int)
        for k, (y, ) in enumerate(zip(att["y"])):
            if y == 1 or rng.random() >= stealth_q:
                ctx[k] = rng.choice(pools[split]["benign"])
            else:
                ctx[k] = rng.choice(pools[split]["attack"])
        att["p_risk"] = p_risk[ctx]
        res[split] = att
    return res["train"], res["test"]


class Fusion:
    def __init__(self):
        self.lr = LogisticRegression(class_weight="balanced", max_iter=1000)

    def fit(self, att):
        self.lr.fit(features(att["z"].values, att["p_risk"].values), att["y"].values)
        return self

    def score(self, z, p_risk):
        return self.lr.predict_proba(features(np.atleast_1d(z), np.atleast_1d(p_risk)))[:, 1]

    def fit_policy(self, att, far_allow=0.01, frr_deny=0.01):
        """allow if score >= t_allow (FAR<=far_allow on train); deny if < t_deny (<=frr_deny genuine below);
        otherwise step-up challenge (e.g. OTP)."""
        s = self.score(att["z"].values, att["p_risk"].values)
        y = att["y"].values
        self.t_allow = float(np.quantile(s[y == 0], 1 - far_allow))
        self.t_deny = float(np.quantile(s[y == 1], frr_deny))
        self.t_deny = min(self.t_deny, self.t_allow)
        return self

    def decide(self, score):
        score = np.atleast_1d(score)
        return np.where(score >= self.t_allow, "allow", np.where(score < self.t_deny, "deny", "challenge"))


def evaluate_fusion(train_att, test_att, fusion: Fusion):
    s_f = fusion.score(test_att["z"].values, test_att["p_risk"].values)
    y = test_att["y"].values
    out = {}
    for name, s in {"keystroke_only": -np.clip(test_att["z"].values, *Z_CLIP),
                    "context_only": 1 - test_att["p_risk"].values,
                    "fusion": s_f}.items():
        e, thr = eer(y, s)
        out[name] = dict(eer=e)
    dec = fusion.decide(s_f)
    out["policy"] = dict(
        allow_genuine=float(np.mean(dec[y == 1] == "allow")),
        challenge_genuine=float(np.mean(dec[y == 1] == "challenge")),
        deny_genuine=float(np.mean(dec[y == 1] == "deny")),
        allow_impostor=float(np.mean(dec[y == 0] == "allow")),
        challenge_impostor=float(np.mean(dec[y == 0] == "challenge")),
        deny_impostor=float(np.mean(dec[y == 0] == "deny")),
        t_allow=fusion.t_allow, t_deny=fusion.t_deny)
    return out
