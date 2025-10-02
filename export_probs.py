#!/usr/bin/env python3
"""
Export CNN probabilities per proposal for post-processing
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import pickle

# Add paths to import from NFL codebase and our modules
sys.path.append('../1st_place_kaggle_player_contact_detection/cnn/models')
sys.path.append('../1st_place_kaggle_player_contact_detection/cnn')
sys.path.append('./dataset')
sys.path.append('./configs')

from resnet3d_csn import ResNet3dCSN
from dataset_header_single import HeaderDataset, get_header_transforms
from header_default import *

def load_model(checkpoint_path, arch='resnet50', num_classes=2, device='cuda'):
    """Load trained CNN model"""
    print(f"Loading model from {checkpoint_path}")
    
    if arch == 'resnet50':
        model = ResNet3dCSN(
            pretrained2d=False,
            pretrained=None,
            depth=50,
            with_pool2=False,
            bottleneck_mode='ir',
            norm_eval=False,
            zero_init_residual=False,
            bn_frozen=False,
            num_classes=num_classes
        )
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present (from DataParallel)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully")
    return model

def load_proposals(proposals_path):
    """Load proposals from CSV or pickle file"""
    if proposals_path.endswith('.csv'):
        df = pd.read_csv(proposals_path)
        proposals = {}
        for _, row in df.iterrows():
            video_id = row['video_id']
            if video_id not in proposals:
                proposals[video_id] = []
            proposals[video_id].append({
                'frame_id': row['frame_id'],
                'probability': row.get('probability', 0.0)  # Pre-XGB probability
            })
    elif proposals_path.endswith('.pkl'):
        with open(proposals_path, 'rb') as f:
            proposals = pickle.load(f)
    else:
        raise ValueError("Proposals file must be .csv or .pkl")
    
    return proposals

def create_proposal_dataset(proposals, dataset_path, num_segments=11, modality='RGB'):
    """Create dataset from proposals for inference"""
    # Create list of samples in format expected by HeaderDataset
    samples = []
    
    for video_id, video_proposals in proposals.items():
        for prop in video_proposals:
            frame_id = prop['frame_id']
            pre_xgb_prob = prop.get('probability', 0.0)
            
            # Add to samples list (path will be constructed by dataset)
            samples.append({
                'video_id': video_id,
                'frame_id': frame_id,
                'pre_xgb_prob': pre_xgb_prob,
                'label': 0  # Dummy label for inference
            })
    
    print(f"Created proposal dataset with {len(samples)} samples")
    return samples

def export_probabilities(model, proposals, dataset_path, output_path, batch_size=16, 
                        num_segments=11, modality='RGB', device='cuda'):
    """Export CNN probabilities for all proposals"""
    
    # Create proposal samples
    proposal_samples = create_proposal_dataset(proposals, dataset_path, num_segments, modality)
    
    if len(proposal_samples) == 0:
        print("No proposal samples found!")
        return
    
    # Create temporary CSV file for HeaderDataset
    temp_csv_path = 'temp_proposals.csv'
    temp_df = pd.DataFrame(proposal_samples)
    temp_df.to_csv(temp_csv_path, index=False)
    
    # Create dataset
    transform = get_header_transforms(input_size=224, is_training=False)
    
    dataset = HeaderDataset(
        temp_csv_path,
        num_segments=num_segments,
        modality=modality,
        transform=transform,
        dense_sample=False,
        test_mode=True
    )
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    # Export probabilities
    all_probs = []
    all_metadata = []
    
    print(f"Exporting probabilities for {len(dataset)} proposals...")
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(tqdm(dataloader)):
            inputs = inputs.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            
            # Store results
            batch_probs = probs.cpu().numpy()
            batch_size_actual = len(batch_probs)
            
            for j in range(batch_size_actual):
                sample_idx = i * batch_size + j
                if sample_idx < len(proposal_samples):
                    sample = proposal_samples[sample_idx]
                    
                    all_probs.append({
                        'video_id': sample['video_id'],
                        'frame_id': sample['frame_id'],
                        'pre_xgb_prob': sample['pre_xgb_prob'],
                        'cnn_prob_0': batch_probs[j, 0],  # Non-header probability
                        'cnn_prob_1': batch_probs[j, 1],  # Header probability
                        'cnn_prob': batch_probs[j, 1]     # Header probability (main)
                    })
    
    # Clean up temporary file
    os.remove(temp_csv_path)
    
    # Convert to DataFrame and save
    probs_df = pd.DataFrame(all_probs)
    
    # Add ensemble probability (weighted combination of pre-XGB and CNN)
    # Similar to NFL solution: prob = 0.2*pre_xgb_prob + 0.8*cnn_prob
    probs_df['ensemble_prob'] = 0.2 * probs_df['pre_xgb_prob'] + 0.8 * probs_df['cnn_prob']
    
    # Save to CSV
    probs_df.to_csv(output_path, index=False)
    
    print(f"Exported {len(probs_df)} probabilities to {output_path}")
    
    # Print summary statistics
    print(f"\nSummary statistics:")
    print(f"Pre-XGB prob: mean={probs_df['pre_xgb_prob'].mean():.4f}, std={probs_df['pre_xgb_prob'].std():.4f}")
    print(f"CNN prob: mean={probs_df['cnn_prob'].mean():.4f}, std={probs_df['cnn_prob'].std():.4f}")
    print(f"Ensemble prob: mean={probs_df['ensemble_prob'].mean():.4f}, std={probs_df['ensemble_prob'].std():.4f}")
    
    return probs_df

def export_probabilities_all_frames(model, ball_det_dict, dataset_path, output_path, 
                                   batch_size=16, num_segments=11, modality='RGB', device='cuda'):
    """Export CNN probabilities for all frames with ball detections (no pre-filtering)"""
    
    # Create samples for all frames with ball detections
    all_samples = []
    
    for video_id, frame_detections in ball_det_dict.items():
        for frame_id in frame_detections.keys():
            if frame_detections[frame_id]:  # If ball detection exists
                all_samples.append({
                    'video_id': video_id,
                    'frame_id': frame_id,
                    'pre_xgb_prob': 0.0,  # No pre-filtering
                    'label': 0  # Dummy label
                })
    
    print(f"Created dataset with {len(all_samples)} frames (all ball detections)")
    
    if len(all_samples) == 0:
        print("No samples found!")
        return
    
    # Create temporary CSV file
    temp_csv_path = 'temp_all_frames.csv'
    temp_df = pd.DataFrame(all_samples)
    temp_df.to_csv(temp_csv_path, index=False)
    
    # Create dataset
    transform = get_header_transforms(input_size=224, is_training=False)
    
    dataset = HeaderDataset(
        temp_csv_path,
        num_segments=num_segments,
        modality=modality,
        transform=transform,
        dense_sample=False,
        test_mode=True
    )
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    # Export probabilities
    all_probs = []
    
    print(f"Exporting probabilities for {len(dataset)} frames...")
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(tqdm(dataloader)):
            inputs = inputs.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            
            # Store results
            batch_probs = probs.cpu().numpy()
            batch_size_actual = len(batch_probs)
            
            for j in range(batch_size_actual):
                sample_idx = i * batch_size + j
                if sample_idx < len(all_samples):
                    sample = all_samples[sample_idx]
                    
                    all_probs.append({
                        'video_id': sample['video_id'],
                        'frame_id': sample['frame_id'],
                        'pre_xgb_prob': 0.0,
                        'cnn_prob_0': batch_probs[j, 0],
                        'cnn_prob_1': batch_probs[j, 1],
                        'cnn_prob': batch_probs[j, 1],
                        'ensemble_prob': batch_probs[j, 1]  # Only CNN prob since no pre-XGB
                    })
    
    # Clean up temporary file
    os.remove(temp_csv_path)
    
    # Convert to DataFrame and save
    probs_df = pd.DataFrame(all_probs)
    probs_df.to_csv(output_path, index=False)
    
    print(f"Exported {len(probs_df)} probabilities to {output_path}")
    
    return probs_df

def main():
    parser = argparse.ArgumentParser(description='Export CNN probabilities for header detection')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained CNN checkpoint')
    parser.add_argument('--proposals', type=str, default='tree/pre_xgb/proposals.csv',
                        help='Path to proposals file (.csv or .pkl)')
    parser.add_argument('--ball_det_dict', type=str, default='cache/ball_det_dict.npy',
                        help='Path to ball detection dictionary')
    parser.add_argument('--dataset_path', type=str, default='../DeepImpact',
                        help='Path to dataset')
    parser.add_argument('--output', type=str, default='probs_export.csv',
                        help='Output CSV file for probabilities')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for inference')
    parser.add_argument('--arch', type=str, default='resnet50',
                        help='Model architecture')
    parser.add_argument('--num_segments', type=int, default=11,
                        help='Number of temporal segments')
    parser.add_argument('--modality', type=str, default='RGB',
                        help='Input modality')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--export_all', action='store_true',
                        help='Export probabilities for all frames (not just proposals)')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, args.arch, num_classes=2, device=device)
    
    if args.export_all:
        # Export for all frames with ball detections
        print("Loading ball detection dictionary...")
        ball_det_dict = np.load(args.ball_det_dict, allow_pickle=True).item()
        print(f"Loaded detections for {len(ball_det_dict)} videos")
        
        probs_df = export_probabilities_all_frames(
            model, ball_det_dict, args.dataset_path, args.output,
            args.batch_size, args.num_segments, args.modality, device
        )
    else:
        # Export for proposals only
        print(f"Loading proposals from {args.proposals}")
        proposals = load_proposals(args.proposals)
        print(f"Loaded proposals for {len(proposals)} videos")
        
        probs_df = export_probabilities(
            model, proposals, args.dataset_path, args.output,
            args.batch_size, args.num_segments, args.modality, device
        )
    
    print(f"Probability export complete! Results saved to {args.output}")

if __name__ == "__main__":
    main()
