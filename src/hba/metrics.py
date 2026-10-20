import numpy as np
from sklearn.metrics import roc_curve


def far_frr(y_true, score, thr):
    """y_true: 1=genuine, 0=impostor; score: higher = more genuine."""
    y = np.asarray(y_true); s = np.asarray(score)
    pred = s >= thr
    far = float(np.mean(pred[y == 0])) if (y == 0).any() else 0.0
    frr = float(np.mean(~pred[y == 1])) if (y == 1).any() else 0.0
    return far, frr


def eer(y_true, score):
    """Equal error rate and its threshold. y_true: 1=genuine, 0=impostor."""
    fpr, tpr, thr = roc_curve(y_true, score)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])
