import csv
import logging
from pathlib import Path
from typing import List, Dict


PREDICTION_COLUMNS = [
    "video_id",
    "half",
    "frame",
    "path",
    "label",
    "prob_header",
    "prob_non_header",
    "pred_label",
]


def save_predictions(predictions: List[Dict], path: Path) -> None:
    """Write validation predictions to CSV with the required schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=PREDICTION_COLUMNS)
            writer.writeheader()
            for row in predictions:
                try:
                    writer.writerow({col: row.get(col, "") for col in PREDICTION_COLUMNS})
                except Exception as row_exc:
                    logging.error(
                        "Failed to write prediction row to %s: %s | row=%s", path, row_exc, row
                    )
    except Exception as exc:
        logging.error("Failed to save predictions to %s: %s", path, exc)
        raise
