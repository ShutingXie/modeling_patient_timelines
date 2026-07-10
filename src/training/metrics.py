"""Multi-label evaluation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_codes: list[str],
) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n_labels = y_true.shape[1]

    per_auroc = {}
    per_ap = {}
    valid_aurocs = []
    valid_aps = []

    for i, code in enumerate(target_codes):
        yt = y_true[:, i]
        yp = y_prob[:, i]
        if len(np.unique(yt)) < 2:
            per_auroc[code] = None
        else:
            auc = roc_auc_score(yt, yp)
            per_auroc[code] = float(auc)
            valid_aurocs.append(auc)
        if yt.sum() > 0:
            ap = average_precision_score(yt, yp)
            per_ap[code] = float(ap)
            valid_aps.append(ap)
        else:
            per_ap[code] = None

    macro_auroc = float(np.mean(valid_aurocs)) if valid_aurocs else None
    macro_ap = float(np.mean(valid_aps)) if valid_aps else None

    try:
        micro_auroc = float(roc_auc_score(y_true.ravel(), y_prob.ravel()))
    except ValueError:
        micro_auroc = None
    try:
        micro_ap = float(average_precision_score(y_true.ravel(), y_prob.ravel()))
    except ValueError:
        micro_ap = None

    return {
        "macro_auroc": macro_auroc,
        "macro_ap": macro_ap,
        "mAP": macro_ap,
        "micro_auroc": micro_auroc,
        "micro_ap": micro_ap,
        "per_condition_auroc": per_auroc,
        "per_condition_ap": per_ap,
        "n_labels": n_labels,
    }
