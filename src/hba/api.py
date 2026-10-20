"""FastAPI service: POST /authenticate -> allow | challenge | deny."""
from functools import lru_cache
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import rba as R
from .config import MODELS

app = FastAPI(title="Hybrid Behavioral Authentication", version="0.1.0")


@lru_cache(maxsize=1)
def artifacts():
    path = MODELS / "hba_artifacts.joblib"
    if not path.exists():
        raise HTTPException(503, "Artifacts missing. Run: python scripts/run_pipeline.py")
    return joblib.load(path)


class Context(BaseModel):
    round_trip_ms: float = 539.0
    ip_address: str = "0.0.0.0"
    country: str = "US"
    asn: int = 0
    browser: str = "Chrome"
    os: str = "Windows"
    device_type: str = "desktop"


class AuthRequest(BaseModel):
    subject: str
    keystroke: Dict[str, float]  # the 31 DSL timing features (H.*, DD.*, UD.*)
    context: Optional[Context] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/authenticate")
def authenticate(req: AuthRequest):
    a = artifacts()
    if req.subject not in a["users"]:
        raise HTTPException(404, f"subject {req.subject} not enrolled")
    missing = [c for c in a["cols"] if c not in req.keystroke]
    if missing:
        raise HTTPException(422, f"missing keystroke features: {missing[:5]}...")
    u = a["users"][req.subject]
    x = np.array([[req.keystroke[c] for c in a["cols"]]])
    z = float((u["det"].distance(x)[0] - u["mu"]) / u["sd"])
    c = req.context or Context()
    row = pd.DataFrame([{
        "Round-Trip Time [ms]": c.round_trip_ms, "IP Address": c.ip_address, "Device Type": c.device_type,
        "Browser Name and Version": c.browser, "OS Name and Version": c.os,
        "Country": c.country, "ASN": c.asn}])
    p_risk = float(a["rba"].predict_proba(R.build_features(row))[0, 1])
    fu = a["fusion"]
    score = float(fu.score(z, p_risk)[0])
    return {"decision": str(fu.decide(score)[0]), "fused_score": score,
            "keystroke_z": z, "context_risk": p_risk,
            "thresholds": {"allow": fu.t_allow, "deny": fu.t_deny}}
