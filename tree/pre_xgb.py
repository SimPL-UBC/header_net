#!/usr/bin/env python3
"""
Weak XGB prefilter for header detection
Computes ball kinematics features and trains XGB to prune easy negatives
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
import pickle
import argparse
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import xgboost as xgb

# Add paths
sys.path.append('../configs')
sys.path.append('../../DeepImpact/data-prep')

from header_default import *

def compute_kinematics_features(ball_positions, fps=25):
    """
    Compute kinematic features from ball trajectory
    
    Args:
        ball_positions: List of (frame_id, x, y, w, h, conf) tuples
        fps: Video frame rate
    
    Returns:
        dict: Features per frame
    """
    if len(ball_positions) < 3:
        return {}
    
    # Sort by frame_id
    ball_positions = sorted(ball_positions, key=lambda x: x[0])
    
    features = {}
    dt = 1.0 / fps  # Time step
    
    for i in range(1, len(ball_positions)-1):
        frame_id = ball_positions[i][0]
        
        # Current position
        x_curr, y_curr = ball_positions[i][1], ball_positions[i][2]
        conf_curr = ball_positions[i][5]
        
        # Previous and next positions
        x_prev, y_prev = ball_positions[i-1][1], ball_positions[i-1][2]
        x_next, y_next = ball_positions[i+1][1], ball_positions[i+1][2]
        
        # Velocity (pixels/second)
        vx = (x_next - x_prev) / (2 * dt)
        vy = (y_next - y_prev) / (2 * dt)
        speed = np.sqrt(vx**2 + vy**2)
        
        # Acceleration (pixels/second^2)
        ax = (x_next - 2*x_curr + x_prev) / (dt**2)
        ay = (y_next - 2*y_curr + y_prev) / (dt**2)
        accel_mag = np.sqrt(ax**2 + ay**2)
        
        # Direction change (radians)
        v_prev = np.array([x_curr - x_prev, y_curr - y_prev])
        v_next = np.array([x_next - x_curr, y_next - y_curr])
        
        angle_change = 0.0
        if np.linalg.norm(v_prev) > 1e-6 and np.linalg.norm(v_next) > 1e-6:
            cos_angle = np.dot(v_prev, v_next) / (np.linalg.norm(v_prev) * np.linalg.norm(v_next))
            cos_angle = np.clip(cos_angle, -1, 1)
            angle_change = np.arccos(cos_angle)
        
        # Speed change
        speed_prev = np.linalg.norm(v_prev) / dt
        speed_next = np.linalg.norm(v_next) / dt
        speed_change = abs(speed_next - speed_prev)
        speed_drop_ratio = max(0, (speed_prev - speed_next) / (speed_prev + 1e-6))
        
        # Curvature approximation
        curvature = 0.0
        if speed > 1e-6:
            curvature = angle_change / (speed * dt + 1e-6)
        
        # Jerk (change in acceleration)
        jerk = 0.0
        if i >= 2 and i < len(ball_positions) - 2:
            # Compute acceleration at previous and next time steps
            x_prev2 = ball_positions[i-2][1]
            y_prev2 = ball_positions[i-2][2]
            x_next2 = ball_positions[i+2][1]
            y_next2 = ball_positions[i+2][2]
            
            ax_prev = (x_curr - 2*x_prev + x_prev2) / (dt**2)
            ay_prev = (y_curr - 2*y_prev + y_prev2) / (dt**2)
            ax_next = (x_next2 - 2*x_next + x_curr) / (dt**2)
            ay_next = (y_next2 - 2*y_next + y_curr) / (dt**2)
            
            jerk_x = (ax_next - ax_prev) / (2 * dt)
            jerk_y = (ay_next - ay_prev) / (2 * dt)
            jerk = np.sqrt(jerk_x**2 + jerk_y**2)
        
        features[frame_id] = {
            'x': x_curr,
            'y': y_curr,
            'vx': vx,
            'vy': vy,
            'speed': speed,
            'ax': ax,
            'ay': ay,
            'accel_mag': accel_mag,
            'angle_change': angle_change,
            'speed_change': speed_change,
            'speed_drop_ratio': speed_drop_ratio,
            'curvature': curvature,
            'jerk': jerk,
            'confidence': conf_curr,
            'ball_size': ball_positions[i][3] * ball_positions[i][4]  # w * h
        }
    
    return features

def add_temporal_features(features, window=5):
    """Add temporal statistics over a sliding window"""
    frame_ids = sorted(features.keys())
    
    for i, frame_id in enumerate(frame_ids):
        # Get window around current frame
        start_idx = max(0, i - window//2)
        end_idx = min(len(frame_ids), i + window//2 + 1)
        window_frames = frame_ids[start_idx:end_idx]
        
        # Compute statistics over window
        speeds = [features[fid]['speed'] for fid in window_frames]
        accels = [features[fid]['accel_mag'] for fid in window_frames]
        angles = [features[fid]['angle_change'] for fid in window_frames]
        
        features[frame_id].update({
            'speed_mean_w': np.mean(speeds),
            'speed_std_w': np.std(speeds),
            'speed_max_w': np.max(speeds),
            'accel_mean_w': np.mean(accels),
            'accel_std_w': np.std(accels),
            'accel_max_w': np.max(accels),
            'angle_mean_w': np.mean(angles),
            'angle_std_w': np.std(angles),
            'angle_max_w': np.max(angles)
        })
    
    return features

def extract_features_from_video(video_id, ball_det_dict, fps=25):
    """Extract kinematic features for all frames in a video"""
    if video_id not in ball_det_dict:
        return {}
    
    # Collect all ball positions for this video
    ball_positions = []
    for frame_id, detections in ball_det_dict[video_id].items():
        if detections:  # If ball detections exist for this frame
            # Take the highest confidence detection
            best_det = max(detections.values(), key=lambda x: x.get('confidence', 0))
            box = best_det['box']
            conf = best_det.get('confidence', 0)
            
            ball_positions.append((frame_id, box[0], box[1], box[2], box[3], conf))
    
    if len(ball_positions) < 3:
        return {}
    
    # Compute kinematic features
    features = compute_kinematics_features(ball_positions, fps)
    
    # Add temporal features
    features = add_temporal_features(features, window=5)
    
    return features

def load_header_labels(dataset_path):
    """Load header labels from Excel/ODS files"""
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
                                    
                                    # Extract frame numbers (assuming first column contains frame numbers)
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

def create_training_data(ball_det_dict, header_labels, neg_sampling_ratio=3.0):
    """Create training dataset with features and labels"""
    feature_names = [
        'x', 'y', 'vx', 'vy', 'speed', 'ax', 'ay', 'accel_mag',
        'angle_change', 'speed_change', 'speed_drop_ratio', 'curvature', 'jerk',
        'confidence', 'ball_size',
        'speed_mean_w', 'speed_std_w', 'speed_max_w',
        'accel_mean_w', 'accel_std_w', 'accel_max_w',
        'angle_mean_w', 'angle_std_w', 'angle_max_w'
    ]
    
    X = []
    y = []
    metadata = []
    
    print("Extracting features from videos...")
    for video_id in tqdm(ball_det_dict.keys()):
        # Extract kinematic features
        features = extract_features_from_video(video_id, ball_det_dict)
        
        if not features:
            continue
        
        # Get header labels for this video
        header_frames = set(header_labels.get(video_id, []))
        
        # Collect positive samples
        positive_samples = []
        for frame_id, feat_dict in features.items():
            if frame_id in header_frames:
                feat_vector = [feat_dict.get(fn, 0.0) for fn in feature_names]
                X.append(feat_vector)
                y.append(1)  # Header
                metadata.append({'video_id': video_id, 'frame_id': frame_id, 'label': 'header'})
                positive_samples.append(frame_id)
        
        # Sample negative samples
        negative_frames = [fid for fid in features.keys() if fid not in header_frames]
        if negative_frames and positive_samples:
            # Sample negatives based on ratio
            n_negatives = min(len(negative_frames), int(len(positive_samples) * neg_sampling_ratio))
            sampled_negatives = np.random.choice(negative_frames, n_negatives, replace=False)
            
            for frame_id in sampled_negatives:
                feat_dict = features[frame_id]
                feat_vector = [feat_dict.get(fn, 0.0) for fn in feature_names]
                X.append(feat_vector)
                y.append(0)  # Non-header
                metadata.append({'video_id': video_id, 'frame_id': frame_id, 'label': 'background'})
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Created training data: {len(y)} samples")
    print(f"Positive samples: {np.sum(y)} ({np.mean(y)*100:.1f}%)")
    print(f"Negative samples: {len(y) - np.sum(y)} ({(1-np.mean(y))*100:.1f}%)")
    
    return X, y, metadata, feature_names

def train_pre_xgb(X, y, feature_names, output_dir, n_folds=5):
    """Train XGB prefilter with cross-validation"""
    os.makedirs(output_dir, exist_ok=True)
    
    # XGB parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_estimators': 100
    }
    
    # Cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_scores = []
    models = []
    
    print(f"Training XGB with {n_folds}-fold cross-validation...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold + 1}/{n_folds}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        # Validate
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred_proba)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, y_pred_proba > 0.5, average='binary'
        )
        
        print(f"Fold {fold + 1} - AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        cv_scores.append(auc)
        models.append(model)
        
        # Save fold model
        model_path = os.path.join(output_dir, f'pre_xgb_fold_{fold}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    
    print(f"\nCross-validation AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    # Train final model on all data
    print("Training final model on all data...")
    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X, y)
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'pre_xgb_final.pkl')
    with open(final_model_path, 'wb') as f:
        pickle.dump(final_model, f)
    
    # Save feature names
    feature_path = os.path.join(output_dir, 'feature_names.pkl')
    with open(feature_path, 'wb') as f:
        pickle.dump(feature_names, f)
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    importance_path = os.path.join(output_dir, 'feature_importance.csv')
    importance_df.to_csv(importance_path, index=False)
    
    print(f"\nTop 10 most important features:")
    print(importance_df.head(10))
    
    return final_model, models, cv_scores

def generate_proposals(ball_det_dict, model, feature_names, threshold=0.5, max_proposals_per_minute=5):
    """Generate proposals using trained XGB model"""
    proposals = {}
    
    print(f"Generating proposals with threshold={threshold}, max_per_min={max_proposals_per_minute}...")
    
    for video_id in tqdm(ball_det_dict.keys()):
        # Extract features
        features = extract_features_from_video(video_id, ball_det_dict)
        
        if not features:
            continue
        
        # Predict probabilities
        frame_probs = []
        for frame_id, feat_dict in features.items():
            feat_vector = [feat_dict.get(fn, 0.0) for fn in feature_names]
            prob = model.predict_proba([feat_vector])[0, 1]
            frame_probs.append((frame_id, prob))
        
        # Sort by probability
        frame_probs.sort(key=lambda x: x[1], reverse=True)
        
        # Apply threshold and max proposals constraint
        video_proposals = []
        for frame_id, prob in frame_probs:
            if prob >= threshold:
                video_proposals.append({
                    'frame_id': frame_id,
                    'probability': prob
                })
        
        # Limit proposals per video (assuming ~30 fps, 45 min half = ~81000 frames)
        # max_proposals_per_minute * 45 min = total budget
        max_proposals = max_proposals_per_minute * 45
        if len(video_proposals) > max_proposals:
            video_proposals = video_proposals[:max_proposals]
        
        proposals[video_id] = video_proposals
        
        print(f"Video {video_id}: {len(video_proposals)} proposals")
    
    return proposals

def main():
    parser = argparse.ArgumentParser(description='Train weak XGB prefilter for header detection')
    parser.add_argument('--dataset_path', type=str, default='../../DeepImpact', 
                        help='Path to dataset')
    parser.add_argument('--ball_det_dict_path', type=str, default='../cache/ball_det_dict.npy',
                        help='Path to ball detection dictionary')
    parser.add_argument('--output_dir', type=str, default='pre_xgb',
                        help='Output directory for models and results')
    parser.add_argument('--neg_ratio', type=float, default=3.0,
                        help='Negative to positive sampling ratio')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Probability threshold for proposals')
    parser.add_argument('--max_proposals_per_min', type=int, default=5,
                        help='Maximum proposals per minute of video')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of CV folds')
    
    args = parser.parse_args()
    
    # Load ball detection dictionary
    print(f"Loading ball detection dictionary from {args.ball_det_dict_path}")
    ball_det_dict = np.load(args.ball_det_dict_path, allow_pickle=True).item()
    print(f"Loaded detections for {len(ball_det_dict)} videos")
    
    # Load header labels
    print(f"Loading header labels from {args.dataset_path}")
    header_labels = load_header_labels(args.dataset_path)
    print(f"Loaded header labels for {len(header_labels)} videos")
    
    # Create training data
    X, y, metadata, feature_names = create_training_data(
        ball_det_dict, header_labels, args.neg_ratio
    )
    
    if len(X) == 0:
        print("No training data found!")
        return
    
    # Train model
    final_model, fold_models, cv_scores = train_pre_xgb(
        X, y, feature_names, args.output_dir, args.n_folds
    )
    
    # Generate proposals
    proposals = generate_proposals(
        ball_det_dict, final_model, feature_names, 
        args.threshold, args.max_proposals_per_min
    )
    
    # Save proposals
    proposals_path = os.path.join(args.output_dir, 'proposals.pkl')
    with open(proposals_path, 'wb') as f:
        pickle.dump(proposals, f)
    
    # Convert to CSV format for cache generation
    proposals_csv_data = []
    for video_id, video_proposals in proposals.items():
        for prop in video_proposals:
            proposals_csv_data.append({
                'video_id': video_id,
                'frame_id': prop['frame_id'],
                'probability': prop['probability'],
                'label_type': 'proposal'
            })
    
    proposals_df = pd.DataFrame(proposals_csv_data)
    proposals_csv_path = os.path.join(args.output_dir, 'proposals.csv')
    proposals_df.to_csv(proposals_csv_path, index=False)
    
    print(f"\nGenerated {len(proposals_csv_data)} total proposals")
    print(f"Average proposals per video: {len(proposals_csv_data) / len(proposals):.1f}")
    print(f"Saved proposals to {proposals_csv_path}")
    
    # Save training metadata
    metadata_df = pd.DataFrame(metadata)
    metadata_path = os.path.join(args.output_dir, 'training_metadata.csv')
    metadata_df.to_csv(metadata_path, index=False)
    
    print(f"Training complete! Models and results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
