import pandas as pd
from pathlib import Path
from typing import List, Dict

def save_predictions(predictions: List[Dict], path: Path):
    """
    Writes val_predictions.csv.
    predictions: List of dicts with keys: video_id, half, frame, path, label, prob_header, prob_non_header, pred_label
    """
    if not predictions:
        return
        
    df = pd.DataFrame(predictions)
    
    # Ensure column order
    # Columns: video_id, half, frame, path, label, prob_header, prob_non_header, pred_label
    desired_cols = ["video_id", "half", "frame", "path", "label", "prob_header", "prob_non_header", "pred_label"]
    
    # Filter to available columns and reorder
    cols = [c for c in desired_cols if c in df.columns]
    df = df[cols]
    
    try:
        df.to_csv(path, index=False)
    except IOError as e:
        print(f"Error saving predictions to {path}: {e}")
