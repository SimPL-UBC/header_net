import logging
from typing import Iterable, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_classification_metrics(
    labels: Iterable[int], preds: Iterable[int], probs: Iterable
) -> dict:
    labels = list(labels)
    preds = list(preds)
    metrics = {
        "val_acc": 0.0,
        "val_precision": 0.0,
        "val_recall": 0.0,
        "val_f1": 0.0,
        "val_auc": 0.0,
    }
    try:
        metrics["val_acc"] = float(accuracy_score(labels, preds))
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="weighted", zero_division=0
        )
        metrics["val_precision"] = float(precision)
        metrics["val_recall"] = float(recall)
        metrics["val_f1"] = float(f1)

        unique_labels = set(labels)
        if len(unique_labels) == 2:
            try:
                probs_arr = np.array(list(probs))
                if probs_arr.ndim == 2 and probs_arr.shape[1] >= 2:
                    pos_probs = probs_arr[:, 1]
                else:
                    pos_probs = probs_arr
                metrics["val_auc"] = float(roc_auc_score(labels, pos_probs))
            except Exception as exc:  # AUC failures shouldn't break other metrics
                logging.warning("Failed to compute AUC: %s", exc)
    except Exception as exc:
        logging.error("Failed to compute classification metrics: %s", exc)

    return metrics
