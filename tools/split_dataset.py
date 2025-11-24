import argparse
import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser(description="Split dataset CSV into train and val")
    parser.add_argument("--input_csv", required=True, help="Path to input CSV")
    parser.add_argument("--output_dir", required=True, help="Output directory for train.csv and val.csv")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--root_dir", help="Root directory to make paths relative to")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Reading {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    
    # Fix paths
    if args.root_dir:
        print(f"Adjusting paths relative to {args.root_dir}...")
        # We want the path in the CSV to be relative to the project root when loaded by the dataset class
        # OR we can just make them absolute paths that are correct for this machine.
        # The user asked for "correct relative path".
        # Let's assume the training script runs from the project root.
        # If the data is in scratch_output/..., we want the CSV to contain "scratch_output/..."
        
        # However, the input CSV has absolute paths like /scratch/st-lyndiacw-1/gyan/...
        # And the files are actually at /home/aerial/antigravity/header_net/scratch_output/...
        # The common part might be the filename.
        
        # Let's look at an example line from the CSV:
        # /scratch/st-lyndiacw-1/gyan/dataset_generation/2015-04-11-16-30BayernMunich3-0EintrachtFrankfurt_half1_013451_1
        
        # And the file on disk:
        # /home/aerial/antigravity/header_net/scratch_output/generate_dataset_test/16_frames_ver/dataset_generation/2015-04-11-16-30BayernMunich3-0EintrachtFrankfurt_half1_013451_1_s.npy
        
        # So we should replace the directory part.
        
        root_path = Path(args.root_dir).resolve()
        
        def fix_path(p):
            # Extract filename from the path in CSV
            p_str = str(p)
            filename = os.path.basename(p_str)
            
            # Construct new path relative to current working directory (project root)
            # if we are running this script from project root.
            # But to be safe, let's just use the absolute path to the file on this machine.
            # Wait, user asked for "correct relative path".
            # If we run training from project root, relative path is best.
            
            # The file is at args.root_dir / filename
            # We need to return the path relative to os.getcwd()
            
            full_path = root_path / filename
            try:
                rel_path = full_path.relative_to(os.getcwd())
                return str(rel_path)
            except ValueError:
                # If not relative, return absolute
                return str(full_path)

        df['path'] = df['path'].apply(fix_path)
        
    # Split
    print(f"Splitting with seed {args.seed}, val_split {args.val_split}...")
    train_df, val_df = train_test_split(df, test_size=args.val_split, random_state=args.seed, shuffle=True)
    
    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_path = os.path.join(args.output_dir, "train.csv")
    val_path = os.path.join(args.output_dir, "val.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    print(f"Saved train.csv ({len(train_df)} samples) to {train_path}")
    print(f"Saved val.csv ({len(val_df)} samples) to {val_path}")

if __name__ == "__main__":
    main()
