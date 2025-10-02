#!/usr/bin/env python3
"""
Temporal NMS and event merging for header detection
"""

import os
import sys
import numpy as np
import pandas as pd
import argparse
from scipy.signal import find_peaks
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# Add paths
sys.path.append('../configs')
from header_default import *

def load_predictions(predictions_path):
    """Load final predictions from post-processing"""
    print(f"Loading predictions from {predictions_path}")
    df = pd.read_csv(predictions_path)
    
    required_cols = ['video_id', 'frame_id', 'final_prob']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"Loaded {len(df)} predictions")
    return df

def temporal_nms(video_predictions, threshold=0.5, nms_window=10):
    """
    Apply temporal Non-Maximum Suppression
    
    Args:
        video_predictions: DataFrame with frame_id and final_prob for one video
        threshold: Probability threshold for detection
        nms_window: Window size for NMS (±frames)
    
    Returns:
        List of detected events with frame_id and confidence
    """
    # Sort by frame_id
    video_predictions = video_predictions.sort_values('frame_id').reset_index(drop=True)
    
    # Get candidates above threshold
    candidates = video_predictions[video_predictions['final_prob'] >= threshold].copy()
    
    if len(candidates) == 0:
        return []
    
    # Apply NMS
    detections = []
    candidates = candidates.sort_values('final_prob', ascending=False).reset_index(drop=True)
    
    suppressed = set()
    
    for idx, row in candidates.iterrows():
        if idx in suppressed:
            continue
            
        frame_id = row['frame_id']
        prob = row['final_prob']
        
        # Add to detections
        detections.append({
            'frame_id': frame_id,
            'confidence': prob
        })
        
        # Suppress nearby detections
        for idx2, row2 in candidates.iterrows():
            if idx2 != idx and idx2 not in suppressed:
                frame_id2 = row2['frame_id']
                if abs(frame_id - frame_id2) <= nms_window:
                    suppressed.add(idx2)
    
    return detections

def peak_based_detection(video_predictions, threshold=0.5, min_peak_height=0.3, 
                        min_peak_distance=10, peak_prominence=0.1):
    """
    Alternative detection using peak finding
    
    Args:
        video_predictions: DataFrame with frame_id and final_prob for one video
        threshold: Minimum probability threshold
        min_peak_height: Minimum peak height
        min_peak_distance: Minimum distance between peaks (frames)
        peak_prominence: Minimum peak prominence
    
    Returns:
        List of detected events
    """
    # Sort by frame_id
    video_predictions = video_predictions.sort_values('frame_id').reset_index(drop=True)
    
    # Create continuous probability signal
    probs = video_predictions['final_prob'].values
    frame_ids = video_predictions['frame_id'].values
    
    # Find peaks
    peaks, properties = find_peaks(
        probs,
        height=min_peak_height,
        distance=min_peak_distance,
        prominence=peak_prominence
    )
    
    # Filter by threshold
    detections = []
    for peak_idx in peaks:
        prob = probs[peak_idx]
        if prob >= threshold:
            detections.append({
                'frame_id': frame_ids[peak_idx],
                'confidence': prob
            })
    
    return detections

def merge_nearby_detections(detections, merge_window=20):
    """
    Merge detections that are very close in time
    
    Args:
        detections: List of detection dictionaries
        merge_window: Merge detections within this window (frames)
    
    Returns:
        List of merged detections
    """
    if len(detections) <= 1:
        return detections
    
    # Sort by frame_id
    detections = sorted(detections, key=lambda x: x['frame_id'])
    
    merged = []
    current_group = [detections[0]]
    
    for i in range(1, len(detections)):
        current_det = detections[i]
        last_in_group = current_group[-1]
        
        if current_det['frame_id'] - last_in_group['frame_id'] <= merge_window:
            # Add to current group
            current_group.append(current_det)
        else:
            # Merge current group and start new group
            if len(current_group) == 1:
                merged.append(current_group[0])
            else:
                # Merge group by taking highest confidence detection
                best_det = max(current_group, key=lambda x: x['confidence'])
                merged.append(best_det)
            
            current_group = [current_det]
    
    # Handle last group
    if len(current_group) == 1:
        merged.append(current_group[0])
    else:
        best_det = max(current_group, key=lambda x: x['confidence'])
        merged.append(best_det)
    
    return merged

def apply_temporal_postprocessing(predictions_df, threshold=0.5, nms_window=10, 
                                merge_window=20, method='nms'):
    """
    Apply temporal post-processing to all videos
    
    Args:
        predictions_df: DataFrame with video_id, frame_id, final_prob
        threshold: Detection threshold
        nms_window: NMS window size
        merge_window: Merge window size
        method: 'nms' or 'peaks'
    
    Returns:
        DataFrame with final detections
    """
    print(f"Applying temporal post-processing with method='{method}', threshold={threshold}")
    
    all_detections = []
    
    for video_id, video_preds in predictions_df.groupby('video_id'):
        if method == 'nms':
            detections = temporal_nms(video_preds, threshold, nms_window)
        elif method == 'peaks':
            detections = peak_based_detection(video_preds, threshold)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Merge nearby detections
        detections = merge_nearby_detections(detections, merge_window)
        
        # Add video_id to detections
        for det in detections:
            det['video_id'] = video_id
            all_detections.append(det)
        
        print(f"Video {video_id}: {len(detections)} detections")
    
    detections_df = pd.DataFrame(all_detections)
    print(f"Total detections: {len(all_detections)}")
    
    return detections_df

def load_ground_truth_labels(dataset_path):
    """Load ground truth header labels for evaluation"""
    labels = {}
    
    header_dataset_path = os.path.join(dataset_path, "header_dataset")
    
    # Load from SoccerNetV2 annotations
    soccernet_path = os.path.join(header_dataset_path, "SoccerNetV2")
    if os.path.exists(soccernet_path):
        for league_dir in os.listdir(soccernet_path):
            league_path = os.path.join(soccernet_path, league_dir)
            if not os.path.isdir(league_path):
                continue
                
            for season_dir in os.listdir(league_path):
                season_path = os.path.join(league_path, season_dir)
                if not os.path.isdir(season_path):
                    continue
                    
                for match_dir in os.listdir(season_path):
                    match_path = os.path.join(season_path, match_dir)
                    if not os.path.isdir(match_path):
                        continue
                    
                    # Look for header annotation files
                    for half in [1, 2]:
                        for file_pattern in [f"{half}_framed.xlsx", f"{half}_HQ_framed.xlsx", f"{half}_framed.ods"]:
                            file_path = os.path.join(match_path, file_pattern)
                            if os.path.exists(file_path):
                                try:
                                    if file_path.endswith('.xlsx'):
                                        df = pd.read_excel(file_path)
                                    elif file_path.endswith('.ods'):
                                        df = pd.read_excel(file_path, engine='odf')
                                    
                                    # Extract frame numbers
                                    if len(df.columns) > 0:
                                        frame_numbers = df.iloc[:, 0].dropna().astype(int).tolist()
                                        
                                        video_id = f"{match_dir}_{half}"
                                        if video_id not in labels:
                                            labels[video_id] = []
                                        labels[video_id].extend(frame_numbers)
                                        
                                        print(f"Loaded {len(frame_numbers)} header frames from {file_path}")
                                
                                except Exception as e:
                                    print(f"Error loading {file_path}: {e}")
    
    return labels

def evaluate_detections(detections_df, ground_truth_labels, tolerance=25):
    """
    Evaluate detections against ground truth
    
    Args:
        detections_df: DataFrame with video_id, frame_id, confidence
        ground_truth_labels: Dict of video_id -> list of frame_ids
        tolerance: Tolerance for matching detections (±frames)
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"Evaluating detections with tolerance=±{tolerance} frames")
    
    total_gt = 0
    total_det = len(detections_df)
    true_positives = 0
    false_positives = 0
    
    # Count ground truth events
    for video_id, gt_frames in ground_truth_labels.items():
        total_gt += len(gt_frames)
    
    print(f"Ground truth events: {total_gt}")
    print(f"Detected events: {total_det}")
    
    # Evaluate each detection
    matched_gt = set()  # Track which GT events have been matched
    
    for _, det_row in detections_df.iterrows():
        video_id = det_row['video_id']
        det_frame = det_row['frame_id']
        
        is_true_positive = False
        
        if video_id in ground_truth_labels:
            gt_frames = ground_truth_labels[video_id]
            
            # Check if detection matches any GT event within tolerance
            for gt_frame in gt_frames:
                if abs(det_frame - gt_frame) <= tolerance:
                    # Match found
                    match_key = (video_id, gt_frame)
                    if match_key not in matched_gt:
                        # First match for this GT event
                        matched_gt.add(match_key)
                        is_true_positive = True
                        break
        
        if is_true_positive:
            true_positives += 1
        else:
            false_positives += 1
    
    false_negatives = total_gt - true_positives
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'total_gt': total_gt,
        'total_detected': total_det,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metrics

def print_evaluation_results(metrics):
    """Print evaluation results in a formatted way"""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"Ground Truth Events:    {metrics['total_gt']}")
    print(f"Detected Events:        {metrics['total_detected']}")
    print(f"True Positives:         {metrics['true_positives']}")
    print(f"False Positives:        {metrics['false_positives']}")
    print(f"False Negatives:        {metrics['false_negatives']}")
    
    print(f"\nPrecision:              {metrics['precision']:.4f}")
    print(f"Recall:                 {metrics['recall']:.4f}")
    print(f"F1 Score:               {metrics['f1']:.4f}")
    
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Apply temporal NMS and evaluate header detection')
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to predictions CSV file from post-processing')
    parser.add_argument('--dataset_path', type=str, default='../../DeepImpact',
                        help='Path to dataset for ground truth labels')
    parser.add_argument('--output_dir', type=str, default='eval_results',
                        help='Output directory for results')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Detection threshold')
    parser.add_argument('--nms_window', type=int, default=10,
                        help='NMS window size (frames)')
    parser.add_argument('--merge_window', type=int, default=20,
                        help='Merge window size (frames)')
    parser.add_argument('--method', type=str, choices=['nms', 'peaks'], default='nms',
                        help='Detection method')
    parser.add_argument('--tolerance', type=int, default=25,
                        help='Evaluation tolerance (frames)')
    parser.add_argument('--skip_eval', action='store_true',
                        help='Skip evaluation (only generate detections)')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load predictions
    predictions_df = load_predictions(args.predictions)
    
    # Apply temporal post-processing
    detections_df = apply_temporal_postprocessing(
        predictions_df, 
        threshold=args.threshold,
        nms_window=args.nms_window,
        merge_window=args.merge_window,
        method=args.method
    )
    
    # Save detections
    detections_path = os.path.join(args.output_dir, 'final_detections.csv')
    detections_df.to_csv(detections_path, index=False)
    print(f"Saved detections to {detections_path}")
    
    if not args.skip_eval:
        # Load ground truth for evaluation
        print("Loading ground truth labels...")
        ground_truth_labels = load_ground_truth_labels(args.dataset_path)
        
        if ground_truth_labels:
            # Evaluate detections
            metrics = evaluate_detections(detections_df, ground_truth_labels, args.tolerance)
            
            # Print results
            print_evaluation_results(metrics)
            
            # Save metrics
            metrics_path = os.path.join(args.output_dir, 'evaluation_metrics.csv')
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(metrics_path, index=False)
            print(f"Saved metrics to {metrics_path}")
            
            # Save detailed results
            results_summary = {
                'method': args.method,
                'threshold': args.threshold,
                'nms_window': args.nms_window,
                'merge_window': args.merge_window,
                'tolerance': args.tolerance,
                **metrics
            }
            
            summary_path = os.path.join(args.output_dir, 'results_summary.csv')
            summary_df = pd.DataFrame([results_summary])
            summary_df.to_csv(summary_path, index=False)
            
        else:
            print("No ground truth labels found. Skipping evaluation.")
    
    print(f"Post-processing complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
