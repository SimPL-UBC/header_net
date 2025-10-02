#!/usr/bin/env python3
"""
Complete pipeline runner for soccer header detection using adapted NFL solution
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description="", check=True):
    """Run a shell command with error handling"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"CMD: {cmd}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print("STDOUT:")
        print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    if check and result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        sys.exit(1)
    
    return result

def setup_directories():
    """Create necessary directories"""
    dirs = [
        "header_net/cache",
        "header_net/outputs", 
        "header_net/logs",
        "header_net/checkpoints"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

def check_prerequisites():
    """Check if required files and dependencies exist"""
    required_files = [
        "../DeepImpact",
        "../1st_place_kaggle_player_contact_detection"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("Missing required files/directories:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\nPlease ensure DeepImpact and NFL solution directories are available.")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Run complete header detection pipeline')
    parser.add_argument('--step', type=str, choices=['all', 'ball_det', 'cache', 'train', 'eval'], 
                        default='all', help='Pipeline step to run')
    parser.add_argument('--deepimpact_path', type=str, default='../DeepImpact',
                        help='Path to DeepImpact directory')
    parser.add_argument('--yolo_dir', type=str, default='../DeepImpact/yolo_detections',
                        help='Directory containing YOLO detection results')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--gpus', nargs='+', type=int, default=[0], help='GPU IDs to use')
    
    args = parser.parse_args()
    
    print("Soccer Header Detection Pipeline")
    print("Adapting NFL winning solution to soccer header detection")
    print(f"Running step: {args.step}")
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Setup directories
    setup_directories()
    
    # Step 1: Build ball detection dictionary
    if args.step in ['all', 'ball_det']:
        print("\n" + "="*80)
        print("STEP 1: Building Ball Detection Dictionary")
        print("="*80)
        
        cmd = f"cd header_net/cache && python build_ball_det_dict.py"
        run_command(cmd, "Building ball detection dictionary from YOLO results")
    
    # Step 2: Create cache files
    if args.step in ['all', 'cache']:
        print("\n" + "="*80)
        print("STEP 2: Creating Cache Files")
        print("="*80)
        
        cmd = f"cd header_net/cache && python create_cache_header.py"
        run_command(cmd, "Creating cached training data with temporal windows")
    
    # Step 3: Train model
    if args.step in ['all', 'train']:
        print("\n" + "="*80)
        print("STEP 3: Training Header Detection Model")
        print("="*80)
        
        # Check if cache files exist
        cache_dir = "header_net/cache/cache_header"
        train_csv = os.path.join(cache_dir, "train_cache_header.csv")
        
        if not os.path.exists(train_csv):
            print(f"Cache file not found: {train_csv}")
            print("Please run cache creation step first")
            sys.exit(1)
        
        # Create train/val splits from cache CSV
        # This is a simplified version - you might want to implement proper cross-validation
        import pandas as pd
        
        df = pd.read_csv(train_csv)
        
        # Simple 80/20 split by video_id to avoid data leakage
        unique_videos = df['video_id'].unique()
        np.random.shuffle(unique_videos)
        
        split_idx = int(0.8 * len(unique_videos))
        train_videos = unique_videos[:split_idx]
        val_videos = unique_videos[split_idx:]
        
        train_df = df[df['video_id'].isin(train_videos)]
        val_df = df[df['video_id'].isin(val_videos)]
        
        # Create list files in format: path num_frames class_index
        train_list_path = "header_net/train_list.txt"
        val_list_path = "header_net/val_list.txt"
        
        with open(train_list_path, 'w') as f:
            for _, row in train_df.iterrows():
                # Assuming 11 temporal segments (from window_size config)
                f.write(f"{row['path']} 11 {row['label']}\n")
        
        with open(val_list_path, 'w') as f:
            for _, row in val_df.iterrows():
                f.write(f"{row['path']} 11 {row['label']}\n")
        
        print(f"Created train list: {len(train_df)} samples")
        print(f"Created val list: {len(val_df)} samples")
        
        # Train the model
        gpu_str = ','.join(map(str, args.gpus))
        cmd = (f"cd header_net && python train_header.py "
               f"--train_list train_list.txt "
               f"--val_list val_list.txt "
               f"--arch resnet50 "
               f"--epochs {args.epochs} "
               f"--batch_size {args.batch_size} "
               f"--lr 0.001 "
               f"--lr_type step "
               f"--lr_steps 20 40 "
               f"--gpus {gpu_str} "
               f"--store_name header_detection_model "
               f"--root_model checkpoints "
               f"--root_log logs")
        
        run_command(cmd, "Training 3D CNN model for header detection")
    
    # Step 4: Evaluation
    if args.step in ['all', 'eval']:
        print("\n" + "="*80)
        print("STEP 4: Model Evaluation")
        print("="*80)
        
        # Find best model checkpoint
        checkpoint_dir = "header_net/checkpoints"
        best_checkpoint = None
        
        for file in os.listdir(checkpoint_dir):
            if file.endswith('_best.pth.tar'):
                best_checkpoint = os.path.join(checkpoint_dir, file)
                break
        
        if not best_checkpoint:
            print("No best checkpoint found. Please train the model first.")
            sys.exit(1)
        
        # Run evaluation
        gpu_str = ','.join(map(str, args.gpus))
        cmd = (f"cd header_net && python train_header.py "
               f"--val_list val_list.txt "
               f"--arch resnet50 "
               f"--batch_size {args.batch_size} "
               f"--gpus {gpu_str} "
               f"--resume {best_checkpoint} "
               f"--evaluate")
        
        run_command(cmd, "Evaluating trained model on validation set")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    
    print("\nNext steps:")
    print("1. Review training logs in header_net/logs/")
    print("2. Check model checkpoints in header_net/checkpoints/")
    print("3. Implement temporal post-processing for final event detection")
    print("4. Test on held-out test videos")
    
    print("\nKey adaptations from NFL solution:")
    print("- Replaced dual-view (Endzone/Sideline) with single soccer broadcast view")
    print("- Replaced player/helmet tracking with ball tracking")
    print("- Adapted contact detection to header event detection")
    print("- Used temporal windowing and masking around ball position")
    print("- Applied class weighting for imbalanced header vs non-header data")

if __name__ == "__main__":
    main()
