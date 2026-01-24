import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score
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
                alpha_pos = float(self.alpha)
                alpha_t = torch.where(
                    targets == 1,
                    torch.tensor(alpha_pos, device=logits.device, dtype=logits.dtype),
                    torch.tensor(1.0 - alpha_pos, device=logits.device, dtype=logits.dtype),
                )
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
        if getattr(config, "loss_type", "focal") == "focal":
            self.criterion = FocalLoss(
                gamma=getattr(config, "focal_gamma", 2.0),
                alpha=getattr(config, "focal_alpha", 0.75),
            )
        else:
            self.criterion = nn.CrossEntropyLoss()

    def train_one_epoch(self, model, train_loader, optimizer, epoch):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        # train_loader yields (inputs, targets, meta)
        for i, (inputs, targets, _) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            all_preds.extend(predicted.detach().cpu().numpy())
            all_labels.extend(targets.detach().cpu().numpy())
            
        epoch_loss = running_loss / total if total > 0 else 0.0
        epoch_acc = correct / total if total > 0 else 0.0
        train_f1 = f1_score(all_labels, all_preds, zero_division=0) if total > 0 else 0.0
        
        return {"train_loss": epoch_loss, "train_acc": epoch_acc, "train_f1": train_f1}

    def validate(self, model, val_loader, epoch):
        model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        predictions_list = []
        
        with torch.no_grad():
            for inputs, targets, meta in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item() * inputs.size(0)
                
                probs = F.softmax(outputs, dim=1)
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
                        "path": meta["path"][k],
                        "label": targets[k].item(),
                        # Assuming class 1 is header, class 0 is non-header
                        "prob_header": probs[k, 1].item() if probs.shape[1] > 1 else 0.0,
                        "prob_non_header": probs[k, 0].item(),
                        "pred_label": preds[k].item()
                    }
                    predictions_list.append(pred_dict)
                    
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
