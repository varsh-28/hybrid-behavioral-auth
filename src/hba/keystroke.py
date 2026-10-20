"""Keystroke dynamics (CMU DSL-StrongPasswordData): identification and verification."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import lightgbm as lgb

from .config import KEYSTROKE_CSV, SEED
from .metrics import eer

META = ["subject", "sessionIndex", "rep"]


def load_keystroke(path=KEYSTROKE_CSV) -> pd.DataFrame:
    return pd.read_csv(path)


def timing_columns(df):
    return [c for c in df.columns if c not in META]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw 31 timings + aggregate rhythm features (paper: session length, hold/latency stats)."""
    out = df.copy()
    H = [c for c in df.columns if c.startswith("H.")]
    DD = [c for c in df.columns if c.startswith("DD.")]
    UD = [c for c in df.columns if c.startswith("UD.")]
    out["agg_H_mean"], out["agg_H_std"] = df[H].mean(axis=1), df[H].std(axis=1)
    out["agg_DD_mean"], out["agg_DD_std"] = df[DD].mean(axis=1), df[DD].std(axis=1)
    out["agg_UD_mean"], out["agg_UD_std"] = df[UD].mean(axis=1), df[UD].std(axis=1)
    out["agg_total_time"] = df[DD].sum(axis=1) + df[H[-1]]
    out["agg_neg_UD_frac"] = (df[UD] < 0).mean(axis=1)
    return out


def feature_columns(df):
    return [c for c in df.columns if c not in META]


def identification_models():
    return {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "SVM-RBF": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale")),
        "RandomForest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED),
        "LightGBM": lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                       random_state=SEED, verbose=-1, n_jobs=-1),
        "MLP": make_pipeline(StandardScaler(), MLPClassifier((128, 64), max_iter=400, early_stopping=True,
                                                             random_state=SEED)),
    }


def evaluate_identification(df, n_splits=5, models=None):
    """5-fold stratified CV, 51-way user identification (as in the paper/notebook)."""
    d = add_features(df)
    cols = feature_columns(d)
    X, y = d[cols].values, d["subject"].astype("category").cat.codes.values
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=SEED)
    rows = []
    for name, model in (models or identification_models()).items():
        accs, f1s, ps, rs = [], [], [], []
        for tr, te in skf.split(X, y):
            m = clone(model).fit(X[tr], y[tr])
            p = m.predict(X[te])
            accs.append(accuracy_score(y[te], p))
            f1s.append(f1_score(y[te], p, average="macro"))
            ps.append(precision_score(y[te], p, average="macro", zero_division=0))
            rs.append(recall_score(y[te], p, average="macro", zero_division=0))
        rows.append(dict(model=name, accuracy=np.mean(accs), accuracy_std=np.std(accs),
                         precision=np.mean(ps), recall=np.mean(rs), f1=np.mean(f1s)))
    return pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- verification
class ScaledManhattan:
    """Killourhy & Maxion (2009) scaled Manhattan detector; trained on genuine data only."""

    def fit(self, X):
        self.mu = np.mean(X, 0)
        self.mad = np.mean(np.abs(X - self.mu), 0) + 1e-6
        return self

    def distance(self, X):
        return np.sum(np.abs(X - self.mu) / self.mad, axis=1)


class Mahalanobis:
    def fit(self, X):
        self.mu = X.mean(0)
        cov = np.cov(X, rowvar=False) + 1e-4 * np.eye(X.shape[1])
        self.inv = np.linalg.inv(cov)
        return self

    def distance(self, X):
        d = X - self.mu
        return np.sqrt(np.einsum("ij,jk,ik->i", d, self.inv, d))


DETECTORS = {"ScaledManhattan": ScaledManhattan, "Mahalanobis": Mahalanobis}


def fit_user_detector(train, detector="ScaledManhattan", rng=None):
    """Fit a one-class detector on genuine reps; calibrate distance -> z-score via 5-fold CV."""
    rng = rng or np.random.default_rng(SEED)
    det = DETECTORS[detector]().fit(train)
    folds = np.array_split(rng.permutation(len(train)), 5)
    cal = np.concatenate([DETECTORS[detector]().fit(np.delete(train, f, 0)).distance(train[f])
                          for f in folds])
    return det, float(cal.mean()), float(cal.std() + 1e-9)


def verification_scores(df, detector="ScaledManhattan", n_train=200, seed=SEED):
    """Per-user one-class detector. Train on first `n_train` genuine reps (sessions 1-4);
    score the remaining genuine reps and all other users' reps (impostors).
    Returns long DataFrame: subject, kind(genuine/impostor), source, rep_idx, z (anomaly z-score)."""
    raw = df.sort_values(["subject", "sessionIndex", "rep"]).reset_index(drop=True)
    cols = timing_columns(raw)
    subjects = sorted(raw["subject"].unique())
    by_subj = {s: raw.loc[raw.subject == s, cols].values for s in subjects}
    rng = np.random.default_rng(seed)
    out = []
    for u in subjects:
        G = by_subj[u]
        tr, te = G[:n_train], G[n_train:]
        det, mu, sd = fit_user_detector(tr, detector, rng)
        parts = [("genuine", u, np.arange(len(te)), det.distance(te))]
        for v in subjects:
            if v == u:
                continue
            V = by_subj[v][n_train:]
            parts.append(("impostor", v, np.arange(len(V)), det.distance(V)))
        for kind, src, idx, d in parts:
            out.append(pd.DataFrame({"subject": u, "kind": kind, "source": src, "rep_idx": idx,
                                     "z": (d - mu) / sd}))
    return pd.concat(out, ignore_index=True)


def verification_summary(scores: pd.DataFrame):
    """Per-user EER (genuine vs all impostors) with score = -z; returns (per_user, mean, std)."""
    res = []
    for u, g in scores.groupby("subject"):
        y = (g["kind"] == "genuine").astype(int).values
        e, _ = eer(y, -g["z"].values)
        res.append((u, e))
    arr = np.array([e for _, e in res])
    return pd.DataFrame(res, columns=["subject", "eer"]), float(arr.mean()), float(arr.std())
