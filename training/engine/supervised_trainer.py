from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from tqdm import tqdm
from ..eval.metrics import compute_classification_metrics


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        ce_loss = F.nll_loss(log_probs, targets, reduction="none")
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_term = (1.0 - p_t) ** self.gamma
        loss = focal_term * ce_loss

        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                alpha_tensor = torch.tensor(
                    self.alpha, device=logits.device, dtype=logits.dtype
                )
                alpha_t = alpha_tensor.gather(0, targets)
            else:
                # Use scalars directly - PyTorch handles broadcasting without allocating new tensors
                alpha_pos = float(self.alpha)
                alpha_t = torch.where(targets == 1, alpha_pos, 1.0 - alpha_pos)
            loss = loss * alpha_t

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class Trainer:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.gradient_accumulation_steps = max(
            1, int(getattr(config, "gradient_accumulation_steps", 1))
        )
        self.amp_enabled = bool(getattr(config, "amp", False) and device.type == "cuda")
        self.scaler = None
        if device.type == "cuda":
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        if getattr(config, "loss_type", "focal") == "focal":
            self.criterion = FocalLoss(
                gamma=getattr(config, "focal_gamma", 2.0),
                alpha=getattr(config, "focal_alpha", 0.75),
            )
        else:
            self.criterion = nn.CrossEntropyLoss()

    @staticmethod
    def _binary_f1(tp: int, fp: int, fn: int) -> float:
        denominator = (2 * tp) + fp + fn
        if denominator <= 0:
            return 0.0
        return (2.0 * tp) / float(denominator)

    def _autocast_context(self):
        if not self.amp_enabled:
            return nullcontext()
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)

    def _per_sample_loss(self, logits, targets):
        if getattr(self.config, "loss_type", "focal") == "focal":
            return FocalLoss(
                gamma=getattr(self.config, "focal_gamma", 2.0),
                alpha=getattr(self.config, "focal_alpha", 0.75),
                reduction="none",
            )(logits, targets)
        return F.cross_entropy(logits, targets, reduction="none")

    def train_one_epoch(self, model, train_loader, optimizer, epoch):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        tp = 0
        fp = 0
        fn = 0
        tn = 0

        progress = tqdm(
            train_loader,
            desc=f"Train Epoch {epoch}",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
            disable=not bool(getattr(self.config, "is_main_process", True)),
        )

        # train_loader yields (inputs, targets, meta)
        optimizer.zero_grad(set_to_none=True)
        pending_microbatches = 0
        total_batches = len(train_loader) if hasattr(train_loader, "__len__") else None
        for batch_idx, (inputs, targets, _) in enumerate(progress, start=1):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            should_step = pending_microbatches + 1 == self.gradient_accumulation_steps
            is_last_batch = total_batches is not None and batch_idx == total_batches
            should_sync = should_step or is_last_batch or total_batches is None
            sync_context = nullcontext()
            if not should_sync and hasattr(model, "no_sync"):
                sync_context = model.no_sync()

            with sync_context:
                with self._autocast_context():
                    outputs = model(inputs)
                    loss = self.criterion(outputs, targets)
                loss_for_backward = loss / self.gradient_accumulation_steps

                if self.scaler is not None and self.amp_enabled:
                    self.scaler.scale(loss_for_backward).backward()
                else:
                    loss_for_backward.backward()

            pending_microbatches += 1
            if pending_microbatches == self.gradient_accumulation_steps:
                if self.scaler is not None and self.amp_enabled:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending_microbatches = 0

            loss_sum += loss.item() * inputs.size(0)
            outputs_for_metrics = outputs.detach().float()
            _, predicted = torch.max(outputs_for_metrics, 1)
            sample_count += targets.size(0)
            predicted_positive = predicted.eq(1)
            target_positive = targets.eq(1)
            tp += int((predicted_positive & target_positive).sum().item())
            fp += int((predicted_positive & ~target_positive).sum().item())
            fn += int((~predicted_positive & target_positive).sum().item())
            tn += int((~predicted_positive & ~target_positive).sum().item())

            if sample_count > 0:
                correct = tp + tn
                progress.set_postfix(
                    loss=f"{loss_sum / sample_count:.4f}",
                    acc=f"{correct / sample_count:.4f}",
                    f1=f"{self._binary_f1(tp, fp, fn):.4f}",
                    refresh=False,
                )

        if pending_microbatches > 0:
            if self.scaler is not None and self.amp_enabled:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        correct = tp + tn
        epoch_loss = loss_sum / sample_count if sample_count > 0 else 0.0
        epoch_acc = correct / sample_count if sample_count > 0 else 0.0
        train_f1 = self._binary_f1(tp, fp, fn)

        return {
            "loss_sum": float(loss_sum),
            "sample_count": int(sample_count),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "train_loss": epoch_loss,
            "train_acc": epoch_acc,
            "train_f1": train_f1,
        }

    def validate(self, model, val_loader, epoch):
        model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        predictions_list = []
        total_batches = len(val_loader) if hasattr(val_loader, "__len__") else None
        progress_every = max(1, int(getattr(self.config, "val_progress_every", 1000)))
        start_time = time.time()
        
        with torch.inference_mode():
            for batch_idx, (inputs, targets, meta) in enumerate(val_loader, start=1):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                with self._autocast_context():
                    outputs = model(inputs)
                    sample_losses = self._per_sample_loss(outputs, targets)
                running_loss += sample_losses.sum().item()

                outputs_for_metrics = outputs.float()
                probs = F.softmax(outputs_for_metrics, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                # Collect per-sample predictions
                batch_size = inputs.size(0)
                for k in range(batch_size):
                    # meta is a dict of batches
                    pred_dict = {
                        "video_id": meta["video_id"][k],
                        "half": meta["half"][k],
                        "frame": meta["frame"][k].item(),
                        "row_idx": meta["row_idx"][k].item(),
                        "path": meta["path"][k],
                        "label": targets[k].item(),
                        "loss": float(sample_losses[k].item()),
                        # Assuming class 1 is header, class 0 is non-header
                        "prob_header": probs[k, 1].item() if probs.shape[1] > 1 else 0.0,
                        "prob_non_header": probs[k, 0].item(),
                        "pred_label": preds[k].item()
                    }
                    predictions_list.append(pred_dict)

                if batch_idx % progress_every == 0:
                    elapsed = max(time.time() - start_time, 1e-6)
                    rate = batch_idx / elapsed
                    if total_batches is not None:
                        print(
                            f"[VAL][Epoch {epoch}] "
                            f"{batch_idx}/{total_batches} batches "
                            f"({rate:.2f} batches/s)",
                            flush=True,
                        )
                    else:
                        print(
                            f"[VAL][Epoch {epoch}] "
                            f"{batch_idx} batches "
                            f"({rate:.2f} batches/s)",
                            flush=True,
                        )
                    
        total = len(all_labels)
        epoch_loss = running_loss / total if total > 0 else 0.0
        
        metrics = compute_classification_metrics(
            np.array(all_labels), 
            np.array(all_preds), 
            np.array(all_probs)
        )
        # Map metrics to expected keys if needed, but compute_classification_metrics returns standard keys
        metrics["val_loss"] = epoch_loss
        
        # Rename keys to match requirements: val_acc, val_precision, etc.
        # compute_classification_metrics returns acc, precision, recall, f1, auc
        # We should prefix them with val_
        final_metrics = {
            "val_loss": epoch_loss,
            "val_acc": metrics["acc"],
            "val_precision": metrics["precision"],
            "val_recall": metrics["recall"],
            "val_f1": metrics["f1"],
            "val_auc": metrics["auc"]
        }
        
        return final_metrics, predictions_list
