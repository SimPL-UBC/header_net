import argparse
import torch
import pandas as pd
import json
import os
from pathlib import Path

# Use relative imports if running as a module (python -m training.cli_train_header)
from .config import Config, merge_cli_args
from .run_utils import set_seed, create_run_dir, save_config
from .data.header_dataset import build_dataloaders
from .models.factory import build_model
from .engine.supervised_trainer import Trainer
from .eval.predictions import save_predictions

def parse_args():
    parser = argparse.ArgumentParser(description="Header Net Training Phase 1")
    parser.add_argument("--train_csv", required=True, help="Path to training CSV")
    parser.add_argument("--val_csv", required=True, help="Path to validation CSV")
    parser.add_argument("--backbone", default="csn", help="Backbone model")
    parser.add_argument("--finetune_mode", default="full", help="Finetune mode")
    parser.add_argument("--run_name", required=True, help="Run name")
    parser.add_argument("--output_root", default="report/header_experiments", help="Output root directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers")
    parser.add_argument("--lr_backbone", type=float, default=0.001, help="Learning rate for backbone")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpus", type=int, nargs="+", help="GPU IDs")
    return parser.parse_args()

def main():
    args = parse_args()
    config = merge_cli_args(args)
    
    # Initialize run
    set_seed(config.seed)
    run_dir = create_run_dir(config.output_root, config.run_name)
    save_config(config, run_dir)
    
    print(f"Starting run: {config.run_name}")
    print(f"Output directory: {run_dir}")
    
    # Device
    if config.gpus and torch.cuda.is_available():
        device = torch.device(f"cuda:{config.gpus[0]}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")
    
    # Data
    print("Building dataloaders...")
    train_loader, val_loader = build_dataloaders(config)
    
    # Model
    print("Building model...")
    model, param_groups = build_model(config)
    model = model.to(device)
    
    if config.gpus and len(config.gpus) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpus)
        
    # Optimizer
    if config.optimizer_type == "sgd":
        optimizer = torch.optim.SGD(
            param_groups, 
            momentum=0.9, 
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer_type}")
        
    # Trainer
    trainer = Trainer(config, device)
    
    # Metrics tracking
    metrics_path = run_dir / "metrics_epoch.csv"
    best_metrics_path = run_dir / "best_metrics.json"
    predictions_path = run_dir / "val_predictions.csv"
    
    best_f1 = -1.0
    
    # Initialize metrics file
    with open(metrics_path, "w") as f:
        f.write("epoch,train_loss,val_loss,val_acc,val_precision,val_recall,val_f1,val_auc,lr_backbone\n")
        
    for epoch in range(1, config.epochs + 1):
        print(f"Epoch {epoch}/{config.epochs}")
        
        # Train
        train_metrics = trainer.train_one_epoch(model, train_loader, optimizer, epoch)
        print(f"Train Loss: {train_metrics['train_loss']:.4f} Acc: {train_metrics['train_acc']:.4f}")
        
        # Validate
        val_metrics, val_preds = trainer.validate(model, val_loader, epoch)
        print(f"Val Loss: {val_metrics['val_loss']:.4f} F1: {val_metrics['val_f1']:.4f}")
        
        # Save metrics
        current_lr = optimizer.param_groups[0]['lr']
        
        with open(metrics_path, "a") as f:
            f.write(f"{epoch},{train_metrics['train_loss']:.6f},{val_metrics['val_loss']:.6f},"
                    f"{val_metrics['val_acc']:.6f},{val_metrics['val_precision']:.6f},"
                    f"{val_metrics['val_recall']:.6f},{val_metrics['val_f1']:.6f},"
                    f"{val_metrics['val_auc']:.6f},{current_lr:.6f}\n")
                    
        # Check best
        if val_metrics['val_f1'] > best_f1:
            best_f1 = val_metrics['val_f1']
            print(f"New best F1: {best_f1:.4f}")
            
            # Save best metrics
            best_data = val_metrics.copy()
            best_data['epoch'] = epoch
            checkpoint_name = f"best_epoch_{epoch}.pt"
            best_data['checkpoint'] = f"checkpoints/{checkpoint_name}"
            
            with open(best_metrics_path, "w") as f:
                json.dump(best_data, f, indent=4)
                
            # Save predictions
            save_predictions(val_preds, predictions_path)
            
            # Save checkpoint
            checkpoint_path = run_dir / "checkpoints" / checkpoint_name
            
            # Handle DataParallel for state_dict
            if isinstance(model, torch.nn.DataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
                
            torch.save({
                'epoch': epoch,
                'state_dict': state_dict,
                'optimizer_state': optimizer.state_dict(),
                'config': vars(args)
            }, checkpoint_path)
            
if __name__ == "__main__":
    main()
