import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix
)

def calculate_quantitative_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    anomaly_scores: np.ndarray,
    pipeline_name: str = "Pipeline"
) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, anomaly_scores))
    except Exception:
        roc_auc = 0.5

    try:
        prec_curve, rec_curve, _ = precision_recall_curve(y_true, anomaly_scores)
        pr_auc = float(auc(rec_curve, prec_curve))
    except Exception:
        pr_auc = 0.5

    far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    dr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    return {
        "pipeline": pipeline_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "false_alarm_rate": far,
        "detection_rate": dr,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }
