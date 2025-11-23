import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.eval.metrics import compute_classification_metrics


class Trainer:
    def __init__(self, config, device: torch.device):
        self.config = config
        self.device = device
        self.criterion = self._build_criterion()

    def _build_criterion(self) -> nn.Module:
        weight = None
        try:
            from configs.header_default import CLASS_WEIGHTS

            weight = torch.tensor(
                [CLASS_WEIGHTS[0], CLASS_WEIGHTS[1]], dtype=torch.float32, device=self.device
            )
        except Exception as exc:
            logging.warning("Using unweighted loss; could not load class weights: %s", exc)
        return nn.CrossEntropyLoss(weight=weight)

    def train_one_epoch(
        self, model: torch.nn.Module, train_loader: DataLoader, optimizer, epoch: int
    ) -> Dict[str, float]:
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for videos, labels, _ in train_loader:
            videos = videos.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(videos)
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size

        avg_loss = total_loss / total_samples if total_samples else 0.0
        avg_acc = total_correct / total_samples if total_samples else 0.0
        logging.info("Epoch %d - train_loss=%.4f train_acc=%.4f", epoch, avg_loss, avg_acc)
        return {"train_loss": avg_loss, "train_acc": avg_acc}

    def validate(
        self, model: torch.nn.Module, val_loader: DataLoader, epoch: int
    ) -> Tuple[Dict[str, float], List[Dict]]:
        model.eval()
        total_loss = 0.0
        total_samples = 0

        all_labels: List[int] = []
        all_preds: List[int] = []
        all_probs: List[List[float]] = []
        predictions: List[Dict] = []

        with torch.no_grad():
            for videos, labels, metas in val_loader:
                videos = videos.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs = model(videos)
                loss = self.criterion(outputs, labels)
                probs = torch.softmax(outputs, dim=1)
                preds = probs.argmax(dim=1)

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                labels_cpu = labels.cpu()
                preds_cpu = preds.cpu()
                probs_cpu = probs.cpu()

                all_labels.extend(labels_cpu.tolist())
                all_preds.extend(preds_cpu.tolist())
                all_probs.extend(probs_cpu.tolist())

                for i in range(batch_size):
                    if isinstance(metas, dict):
                        meta = {k: v[i] for k, v in metas.items()}
                    elif isinstance(metas, (list, tuple)):
                        meta = metas[i]
                    else:
                        meta = {}
                    prob_row = probs_cpu[i].tolist()
                    prob_non_header = float(prob_row[0]) if len(prob_row) >= 1 else 0.0
                    prob_header = float(prob_row[1]) if len(prob_row) >= 2 else 1.0 - prob_non_header
                    predictions.append(
                        {
                            "video_id": str(meta.get("video_id", "")),
                            "half": str(meta.get("half", "")),
                            "frame": int(meta.get("frame", -1)),
                            "path": str(meta.get("path", "")),
                            "label": int(labels_cpu[i].item()),
                            "prob_header": prob_header,
                            "prob_non_header": prob_non_header,
                            "pred_label": int(preds_cpu[i].item()),
                        }
                    )

        avg_loss = total_loss / total_samples if total_samples else 0.0
        metrics = compute_classification_metrics(all_labels, all_preds, all_probs)
        metrics["val_loss"] = avg_loss
        logging.info(
            "Epoch %d - val_loss=%.4f val_acc=%.4f val_f1=%.4f",
            epoch,
            metrics["val_loss"],
            metrics["val_acc"],
            metrics["val_f1"],
        )
        return metrics, predictions
