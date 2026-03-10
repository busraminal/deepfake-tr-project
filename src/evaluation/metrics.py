"""
Değerlendirme metrikleri: accuracy, AUC, EER, confusion matrix.
Threshold: 0.5 sabit yerine ROC'tan Youden (argmax(tpr-fpr)) ile optimal eşik kullanılabilir.
"""
from __future__ import annotations

import numpy as np
from typing import Sequence


def optimal_threshold_youden(y_true: Sequence[float], y_score: Sequence[float]) -> float:
    """
    ROC eğrisinden Youden indeksi: threshold = argmax(tpr - fpr).
    Skor dağılımı 0.5 etrafında değilse sabit 0.5 yerine bu kullanılmalı.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return 0.5
    try:
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        # Youden: max(tpr - fpr); threshold'lar fpr/tpr ile aynı uzunlukta olmayabilir (sklearn bir ek nokta ekler)
        if len(thresholds) == 0:
            return 0.5
        j = tpr - fpr
        idx = np.argmax(j)
        return float(thresholds[idx])
    except Exception:
        return 0.5


def accuracy(y_true: Sequence[float], y_pred: Sequence[float], threshold: float = 0.5) -> float:
    """Binary accuracy: pred >= threshold -> 1."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pred_bin = (y_pred >= threshold).astype(np.float32)
    return float(np.mean(pred_bin == y_true))


def auc_roc(y_true: Sequence[float], y_score: Sequence[float]) -> float:
    """ROC AUC. y_score = model çıktı (0-1 veya logit)."""
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_score))
    except ImportError:
        # Basit trapezoidal AUC
        order = np.argsort(y_score)
        y_true = np.asarray(y_true)[order]
        y_score = np.asarray(y_score)[order]
        n = len(y_true)
        if n < 2:
            return 0.5
        tpr = np.cumsum(y_true) / (np.sum(y_true) + 1e-8)
        fpr = np.cumsum(1 - y_true) / (np.sum(1 - y_true) + 1e-8)
        return float(np.trapz(tpr, fpr))


def eer_from_scores(y_true: Sequence[float], y_score: Sequence[float], n_thresholds: int = 101) -> float:
    """
    Equal Error Rate: FAR = FRR olduğu threshold'daki hata oranı.
    y_true: 0=real, 1=fake. y_score: sahte olasılığı (yüksek = fake).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    thresholds = np.linspace(0, 1, n_thresholds)
    best_eer = 1.0
    n_pos = max(1, int(np.sum(y_true)))
    n_neg = max(1, int(np.sum(1 - y_true)))
    for t in thresholds:
        pred = (y_score >= t).astype(np.float32)
        frr = np.sum((y_true == 1) & (pred == 0)) / n_pos
        far = np.sum((y_true == 0) & (pred == 1)) / n_neg
        eer = (frr + far) / 2
        if abs(frr - far) < abs(2 * best_eer - 1):
            best_eer = eer
    return float(best_eer)


def confusion_matrix_binary(y_true: Sequence[float], y_pred: Sequence[float], threshold: float = 0.5) -> dict:
    """TP, TN, FP, FN ve precision, recall, F1."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_pred) >= threshold).astype(np.int32)
    y_true = y_true.astype(np.int32)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
    }


def compute_all(
    y_true: Sequence[float],
    y_score: Sequence[float],
    threshold: float | None = None,
) -> dict:
    """
    Tüm metrikleri tek çağrıda hesapla.
    threshold=None ise ROC Youden ile optimal eşik kullanılır (skor dağılımı 0.5'te değilse önerilir).
    """
    if threshold is None:
        threshold = optimal_threshold_youden(y_true, y_score)
    acc = accuracy(y_true, y_score, threshold)
    try:
        auc = auc_roc(y_true, y_score)
    except Exception:
        auc = 0.5
    eer = eer_from_scores(y_true, y_score)
    cm = confusion_matrix_binary(y_true, y_score, threshold)
    return {
        "accuracy": acc,
        "auc": auc,
        "eer": eer,
        "threshold": float(threshold),
        **cm,
    }
