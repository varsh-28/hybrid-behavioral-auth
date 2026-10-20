"""Risk-based authentication on login context (rba_dataset_cleaned.csv)."""
import re
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import lightgbm as lgb

from .config import RBA_CSV, SEED

CAT = ["device_type", "browser_family", "os_family", "country", "asn"]
NUM = ["rtt_log", "ip_private", "is_bot", "is_mobile"]
TARGETS = {"attack_ip": "Is Attack IP", "login_success": "Login Successful"}


def load_rba(path=RBA_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="ISO-8859-1")
    return df.drop(columns=["index"], errors="ignore").drop_duplicates().reset_index(drop=True)


def _family(s: str) -> str:
    return re.sub(r"[\d.]+", "", str(s)).strip() or "unknown"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Leakage-safe context features. Dropped: User ID, raw IP, raw user-agent, timestamp
    (timestamps in the file are truncated to mm:ss.s, so hour/weekday cyclical features are impossible)."""
    out = pd.DataFrame(index=df.index)
    out["rtt_log"] = np.log1p(df["Round-Trip Time [ms]"].astype(float))
    ip = df["IP Address"].astype(str)
    out["ip_private"] = (ip.str.startswith("10.") | ip.str.startswith("192.168.")).astype(int)
    out["device_type"] = df["Device Type"].astype(str)
    out["is_bot"] = (out["device_type"] == "bot").astype(int)
    out["is_mobile"] = (out["device_type"] == "mobile").astype(int)
    out["browser_family"] = df["Browser Name and Version"].map(_family)
    out["os_family"] = df["OS Name and Version"].map(_family)
    out["country"] = df["Country"].astype(str)
    out["asn"] = df["ASN"].astype(str)
    return out


def preprocessor(scale_num=False):
    ohe = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=15, sparse_output=False)
    num = StandardScaler() if scale_num else "passthrough"
    return ColumnTransformer([("cat", ohe, CAT), ("num", num, NUM)])


def rba_models():
    return {
        "LogisticRegression": Pipeline([("pre", preprocessor(True)),
                                        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "RandomForest": Pipeline([("pre", preprocessor()),
                                  ("clf", RandomForestClassifier(300, class_weight="balanced", n_jobs=-1,
                                                                 min_samples_leaf=2, random_state=SEED))]),
        "LightGBM": Pipeline([("pre", preprocessor()),
                              ("clf", lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                                                         class_weight="balanced", random_state=SEED,
                                                         verbose=-1, n_jobs=-1))]),
        "MLP": Pipeline([("pre", preprocessor(True)),
                         ("clf", MLPClassifier((64, 32), max_iter=300, early_stopping=True, random_state=SEED))]),
    }


def evaluate(df, target="attack_ip", n_splits=5):
    X = build_features(df)
    y = df[TARGETS[target]].astype(int).values
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=SEED)
    rows = []
    for name, model in rba_models().items():
        m = {k: [] for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]}
        for tr, te in skf.split(X, y):
            mdl = clone(model).fit(X.iloc[tr], y[tr])
            p = mdl.predict_proba(X.iloc[te])[:, 1]
            pred = (p >= 0.5).astype(int)
            m["accuracy"].append(accuracy_score(y[te], pred))
            m["precision"].append(precision_score(y[te], pred, zero_division=0))
            m["recall"].append(recall_score(y[te], pred))
            m["f1"].append(f1_score(y[te], pred))
            m["roc_auc"].append(roc_auc_score(y[te], p))
            m["pr_auc"].append(average_precision_score(y[te], p))
        rows.append(dict(model=name, target=target, **{k: float(np.mean(v)) for k, v in m.items()},
                         accuracy_std=float(np.std(m["accuracy"]))))
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)


def oof_risk(df, target="attack_ip", n_splits=5) -> np.ndarray:
    """Out-of-fold P(risky) from LightGBM, so simulated contexts never use in-sample predictions."""
    X = build_features(df)
    y = df[TARGETS[target]].astype(int).values
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=SEED)
    return cross_val_predict(rba_models()["LightGBM"], X, y, cv=skf, method="predict_proba")[:, 1]
