import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pytest
from fastapi.testclient import TestClient

from hba import keystroke as K, rba as R, fusion as F
from hba.config import MODELS
from hba.metrics import eer, far_frr


def test_eer_perfect_and_random():
    y = np.array([1] * 50 + [0] * 50)
    assert eer(y, np.r_[np.ones(50), np.zeros(50)])[0] == 0.0
    rng = np.random.default_rng(0)
    assert 0.35 < eer(y, rng.random(100))[0] < 0.65


def test_far_frr():
    far, frr = far_frr([1, 1, 0, 0], [0.9, 0.2, 0.8, 0.1], 0.5)
    assert far == 0.5 and frr == 0.5


@pytest.fixture(scope="module")
def ks():
    return K.load_keystroke()


def test_keystroke_shape(ks):
    assert ks.shape == (20400, 34) and ks.subject.nunique() == 51
    assert K.add_features(ks).shape[1] == 34 + 8


def test_verification_beats_chance(ks):
    sub = ks[ks.subject.isin(sorted(ks.subject.unique())[:6])]
    sc = K.verification_scores(sub)
    _, m, _ = K.verification_summary(sc)
    assert m < 0.3
    assert set(sc.kind) == {"genuine", "impostor"}


def test_rba_features_no_leakage_columns():
    df = R.load_rba()
    X = R.build_features(df)
    banned = {"User ID", "IP Address", "User Agent String", "Login Timestamp"}
    assert not banned & set(X.columns)
    assert set(X.columns) == set(R.CAT + R.NUM)


def test_fusion_policy_ordering():
    rng = np.random.default_rng(1)
    import pandas as pd
    n = 400
    att = pd.DataFrame({"y": np.r_[np.ones(n), np.zeros(n)],
                        "z": np.r_[rng.normal(0, 1, n), rng.normal(6, 2, n)],
                        "p_risk": np.r_[rng.beta(1, 8, n), rng.beta(4, 3, n)]})
    fu = F.Fusion().fit(att).fit_policy(att)
    assert fu.t_deny <= fu.t_allow
    s = fu.score(att.z.values, att.p_risk.values)
    assert s[att.y == 1].mean() > s[att.y == 0].mean()


@pytest.mark.skipif(not (MODELS / "hba_artifacts.joblib").exists(), reason="run pipeline first")
def test_api_roundtrip(ks):
    from hba.api import app
    c = TestClient(app)
    assert c.get("/health").json() == {"status": "ok"}
    cols = K.timing_columns(ks)
    ctx_ok = dict(round_trip_ms=539, ip_address="84.210.17.29", country="NO", asn=41164,
                  browser="Chrome", os="Windows", device_type="desktop")

    def mean_score(subj, rows):
        return np.mean([c.post("/authenticate", json=dict(
            subject=subj, keystroke={k: float(r[k]) for k in cols}, context=ctx_ok)).json()["fused_score"]
            for _, r in rows.iterrows()])

    genuine = mean_score("s002", ks[ks.subject == "s002"].iloc[:40])
    impostor = mean_score("s002", ks[ks.subject.isin(["s003", "s004", "s005", "s007"])].iloc[::40].head(40))
    assert genuine > impostor
    body = c.post("/authenticate", json=dict(subject="s002", keystroke={k: 0.1 for k in cols})).json()
    assert body["decision"] in {"allow", "challenge", "deny"}
    assert c.post("/authenticate", json=dict(subject="nope", keystroke={})).status_code == 404
