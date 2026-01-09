from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
import numpy as np

def compute_classification_metrics(labels, preds, probs):
    """
    Computes accuracy, precision, recall, F1, and AUC.
    labels: ground truth (N,)
    preds: predictions (N,)
    probs: probabilities (N, C)
    """
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )
    
    # AUC (for binary classification)
    # Check if we have binary labels and probs
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 2 and probs.shape[1] >= 2:
        try:
            # Assuming class 1 is positive
            auc = roc_auc_score(labels, probs[:, 1])
        except ValueError:
            auc = 0.0
    else:
        auc = 0.0
        
    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }
