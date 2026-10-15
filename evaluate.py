"""
Evaluation metrics for Gene-Disease Association existence classification
and association strength regression.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Computes Expected Calibration Error (ECE) and Brier Score to quantify model overconfidence.
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_score >= bin_boundaries[i]) & (
            y_score < bin_boundaries[i + 1] if i < n_bins - 1 else y_score <= bin_boundaries[i + 1]
        )
        prop = np.mean(in_bin)
        if prop > 0:
            acc = np.mean(y_true[in_bin])
            conf = np.mean(y_score[in_bin])
            ece += np.abs(conf - acc) * prop

    brier = float(np.mean((y_score - y_true) ** 2))
    return {
        "ece": float(ece),
        "brier_score": brier,
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Computes AUROC, AUPRC (Average Precision), F1, Accuracy, Precision, Recall, ECE, and Brier Score.
    """
    metrics = {}
    # Handle single class edge cases in small subsets
    if len(np.unique(y_true)) > 1:
        metrics["auroc"] = float(roc_auc_score(y_true, y_score))
        metrics["auprc"] = float(average_precision_score(y_true, y_score))
    else:
        metrics["auroc"] = 0.5
        metrics["auprc"] = float(np.mean(y_true))

    y_pred = (y_score >= threshold).astype(int)
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))

    cal_metrics = compute_calibration_metrics(y_true, y_score)
    metrics.update(cal_metrics)

    return metrics


def compute_regression_metrics(
    s_true: np.ndarray,
    s_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Computes MSE, RMSE, MAE, Pearson r, and Spearman rho on non-zero / positive pairs.
    """
    metrics = {}
    if len(s_true) == 0 or len(s_pred) == 0:
        return {
            "mse": 0.0,
            "rmse": 0.0,
            "mae": 0.0,
            "pearson_r": 0.0,
            "spearman_rho": 0.0,
        }

    mse = float(mean_squared_error(s_true, s_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(s_true, s_pred))

    metrics["mse"] = mse
    metrics["rmse"] = rmse
    metrics["mae"] = mae

    if len(s_true) > 2 and np.std(s_true) > 1e-8 and np.std(s_pred) > 1e-8:
        r, _ = pearsonr(s_true, s_pred)
        rho, _ = spearmanr(s_true, s_pred)
        metrics["pearson_r"] = float(r) if not np.isnan(r) else 0.0
        metrics["spearman_rho"] = float(rho) if not np.isnan(rho) else 0.0
    else:
        metrics["pearson_r"] = 0.0
        metrics["spearman_rho"] = 0.0

    return metrics


def evaluate_predictions(
    y_true: np.ndarray,
    y_score: np.ndarray,
    s_true: np.ndarray,
    s_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluates both classification and regression.
    Evaluates strength metrics strictly on true positive associations (y_true == 1).
    """
    clf_metrics = compute_classification_metrics(y_true, y_score)

    pos_mask = y_true > 0.5
    if pos_mask.sum() > 0:
        regr_metrics = compute_regression_metrics(s_true[pos_mask], s_pred[pos_mask])
    else:
        regr_metrics = compute_regression_metrics(s_true, s_pred)

    combined = {**clf_metrics, **regr_metrics}
    return combined


def format_metrics_table(metrics_dict: Dict[str, Dict[str, float]]) -> str:
    """
    Formats a dictionary of model -> metrics into a markdown comparison table.
    """
    header = "| Model | AUROC | AUPRC | F1 | ECE | Accuracy | RMSE | Pearson r | Spearman rho |"
    divider = "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    lines = [header, divider]

    for model_name, m in metrics_dict.items():
        line = (
            f"| {model_name:<20} "
            f"| {m.get('auroc', 0.0):.4f} "
            f"| {m.get('auprc', 0.0):.4f} "
            f"| {m.get('f1', 0.0):.4f} "
            f"| {m.get('ece', 0.0):.4f} "
            f"| {m.get('accuracy', 0.0):.4f} "
            f"| {m.get('rmse', 0.0):.4f} "
            f"| {m.get('pearson_r', 0.0):.4f} "
            f"| {m.get('spearman_rho', 0.0):.4f} |"
        )
        lines.append(line)

    return "\n".join(lines)
