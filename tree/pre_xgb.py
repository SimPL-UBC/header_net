#!/usr/bin/env python3
"""
Weak XGB prefilter for header detection
Computes ball kinematics features and trains XGB to prune easy negatives

Features include:
- Ball kinematics: position, velocity, acceleration, jerk, curvature
- Player spatial context: distance to nearest player/head, player density
"""

import sys
import json
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

HEADER_NET_ROOT = Path(__file__).resolve().parents[1]
if str(HEADER_NET_ROOT) not in sys.path:
    sys.path.append(str(HEADER_NET_ROOT))

from configs import header_default as cfg
from utils.detections import load_ball_det_dict, make_video_key
from utils.labels import load_header_labels, build_half_frame_lookup
from utils.player_features import (
    compute_player_features,
    extract_player_features_from_frame_data,
    PLAYER_FEATURE_NAMES
)

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


def extract_features_from_metadata_json(json_path, fps=25, use_player_features=True):
    """
    Extract kinematic AND player features from a metadata JSON file.

    Args:
        json_path: Path to *_meta.json file
        fps: Video frame rate
        use_player_features: Whether to include player-based features

    Returns:
        Tuple of (features_dict, sample_info) where:
        - features_dict: Dict mapping frame_id -> feature dict
        - sample_info: Dict with video_id, half, center_frame, label
    """
    with open(json_path, 'r') as f:
        meta = json.load(f)

    video_id = meta.get('video_id', '')
    half = meta.get('half', 1)
    center_frame = meta.get('frame', 0)
    label = meta.get('label', 0)
    frames_data = meta.get('frames', [])

    if not frames_data:
        return {}, {}

    sample_info = {
        'video_id': f"{video_id}_half{half}",
        'half': half,
        'center_frame': center_frame,
        'label': label
    }

    # Sort frames by offset to ensure temporal order
    frames_data = sorted(frames_data, key=lambda x: x.get('offset', 0))

    # First pass: collect ball positions for kinematic feature computation
    ball_positions = []
    frame_to_data = {}

    for frame_entry in frames_data:
        offset = frame_entry.get('offset', 0)
        frame_id = frame_entry.get('frame', center_frame + offset)
        ball_data = frame_entry.get('ball', {})

        if ball_data and 'box' in ball_data:
            box = ball_data['box']
            conf = ball_data.get('confidence', 0)
            ball_positions.append((frame_id, box[0], box[1], box[2], box[3], conf))

        frame_to_data[frame_id] = frame_entry

    if len(ball_positions) < 3:
        return {}, sample_info

    # Compute kinematic features
    features = compute_kinematics_features(ball_positions, fps)

    # Add temporal features
    features = add_temporal_features(features, window=5)

    # Second pass: add player features if enabled
    if use_player_features:
        frame_ids = sorted(frame_to_data.keys())
        for i, frame_id in enumerate(frame_ids):
            if frame_id not in features:
                continue

            frame_entry = frame_to_data[frame_id]

            # Get previous frame for velocity estimation
            prev_frame_data = None
            if i > 0:
                prev_frame_id = frame_ids[i - 1]
                prev_frame_data = frame_to_data.get(prev_frame_id)

            # Extract player features
            player_feats = extract_player_features_from_frame_data(
                frame_entry, prev_frame_data
            )

            # Merge player features into kinematic features
            features[frame_id].update(player_feats)

    return features, sample_info


def load_metadata_from_directory(metadata_dir, use_player_features=True):
    """
    Load all metadata JSON files from a directory.

    Args:
        metadata_dir: Directory containing *_meta.json files
        use_player_features: Whether to include player-based features

    Returns:
        Tuple of (all_features, all_samples) where:
        - all_features: Dict mapping (video_id, frame_id) -> feature dict
        - all_samples: List of sample info dicts with labels
    """
    metadata_dir = Path(metadata_dir)
    json_files = list(metadata_dir.glob('*_meta.json'))

    print(f"Found {len(json_files)} metadata JSON files in {metadata_dir}")

    all_features = {}
    all_samples = []

    for json_path in tqdm(json_files, desc="Loading metadata"):
        features, sample_info = extract_features_from_metadata_json(
            json_path, use_player_features=use_player_features
        )

        if not features or not sample_info:
            continue

        video_id = sample_info['video_id']
        center_frame = sample_info['center_frame']
        label = sample_info['label']

        # Store features keyed by (video_id, frame_id)
        for frame_id, feat_dict in features.items():
            all_features[(video_id, frame_id)] = feat_dict

        # Store sample info
        all_samples.append({
            'video_id': video_id,
            'frame_id': center_frame,
            'label': label,
            'json_path': str(json_path)
        })

    print(f"Loaded {len(all_samples)} samples with {len(all_features)} frame features")

    return all_features, all_samples


# Feature names including player features
KINEMATIC_FEATURE_NAMES = [
    'x', 'y', 'vx', 'vy', 'speed', 'ax', 'ay', 'accel_mag',
    'angle_change', 'speed_change', 'speed_drop_ratio', 'curvature', 'jerk',
    'confidence', 'ball_size',
    'speed_mean_w', 'speed_std_w', 'speed_max_w',
    'accel_mean_w', 'accel_std_w', 'accel_max_w',
    'angle_mean_w', 'angle_std_w', 'angle_max_w'
]

FULL_FEATURE_NAMES = KINEMATIC_FEATURE_NAMES + PLAYER_FEATURE_NAMES


def create_training_data_from_metadata(
    all_features, all_samples, use_player_features=True, neg_sampling_ratio=3.0
):
    """
    Create training dataset from metadata-extracted features.

    Args:
        all_features: Dict mapping (video_id, frame_id) -> feature dict
        all_samples: List of sample info dicts
        use_player_features: Whether to include player features
        neg_sampling_ratio: Ratio of negative to positive samples

    Returns:
        Tuple of (X, y, metadata, feature_names)
    """
    feature_names = FULL_FEATURE_NAMES if use_player_features else KINEMATIC_FEATURE_NAMES

    X = []
    y = []
    metadata = []

    # Separate positive and negative samples
    positive_samples = [s for s in all_samples if s['label'] == 1]
    negative_samples = [s for s in all_samples if s['label'] == 0]

    print(f"Total samples: {len(all_samples)}")
    print(f"Positive samples: {len(positive_samples)}")
    print(f"Negative samples: {len(negative_samples)}")

    # Process positive samples
    for sample in positive_samples:
        video_id = sample['video_id']
        frame_id = sample['frame_id']
        key = (video_id, frame_id)

        if key in all_features:
            feat_dict = all_features[key]
            feat_vector = [feat_dict.get(fn, 0.0) for fn in feature_names]
            X.append(feat_vector)
            y.append(1)
            metadata.append({
                'video_id': video_id,
                'frame_id': frame_id,
                'label': 'header'
            })

    # Sample negatives
    n_negatives = min(len(negative_samples), int(len(positive_samples) * neg_sampling_ratio))
    rng = np.random.default_rng(2024)
    sampled_negatives = rng.choice(len(negative_samples), n_negatives, replace=False)

    for idx in sampled_negatives:
        sample = negative_samples[idx]
        video_id = sample['video_id']
        frame_id = sample['frame_id']
        key = (video_id, frame_id)

        if key in all_features:
            feat_dict = all_features[key]
            feat_vector = [feat_dict.get(fn, 0.0) for fn in feature_names]
            X.append(feat_vector)
            y.append(0)
            metadata.append({
                'video_id': video_id,
                'frame_id': frame_id,
                'label': 'background'
            })

    X = np.array(X)
    y = np.array(y)

    print(f"Created training data: {len(y)} samples")
    if len(y) > 0:
        print(f"Positive samples: {np.sum(y)} ({np.mean(y)*100:.1f}%)")
        print(f"Negative samples: {len(y) - np.sum(y)} ({(1-np.mean(y))*100:.1f}%)")

    return X, y, metadata, feature_names


def create_training_data(ball_det_dict, labels_df, neg_sampling_ratio=3.0):
    """Create training dataset with features and labels (kinematic features only)"""
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

    label_lookup = build_half_frame_lookup(labels_df)
    rng = np.random.default_rng(2024)
    
    print("Extracting features from videos...")
    for video_id in tqdm(ball_det_dict.keys()):
        features = extract_features_from_video(video_id, ball_det_dict)
        if not features:
            continue

        header_frames = set(label_lookup.get(video_id, []))
        if not header_frames:
            continue

        positive_samples = []
        for frame_id, feat_dict in features.items():
            if frame_id in header_frames:
                feat_vector = [feat_dict.get(fn, 0.0) for fn in feature_names]
                X.append(feat_vector)
                y.append(1)
                metadata.append({'video_id': video_id, 'frame_id': frame_id, 'label': 'header'})
                positive_samples.append(frame_id)

        negative_frames = [fid for fid in features.keys() if fid not in header_frames]
        if negative_frames and positive_samples:
            n_negatives = min(len(negative_frames), int(len(positive_samples) * neg_sampling_ratio))
            sampled_negatives = rng.choice(negative_frames, n_negatives, replace=False)
            for frame_id in sampled_negatives:
                feat_dict = features[frame_id]
                feat_vector = [feat_dict.get(fn, 0.0) for fn in feature_names]
                X.append(feat_vector)
                y.append(0)
                metadata.append({'video_id': video_id, 'frame_id': frame_id, 'label': 'background'})

    X = np.array(X)
    y = np.array(y)

    print(f"Created training data: {len(y)} samples")
    if len(y) > 0:
        print(f"Positive samples: {np.sum(y)} ({np.mean(y)*100:.1f}%)")
        print(f"Negative samples: {len(y) - np.sum(y)} ({(1-np.mean(y))*100:.1f}%)")

    return X, y, metadata, feature_names

def train_pre_xgb(X, y, feature_names, output_dir, n_folds=5, n_estimators=100, n_jobs=1):
    """Train XGB prefilter with cross-validation"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_estimators': n_estimators,
        'n_jobs': n_jobs
    }

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_scores = []
    models = []

    print(f"Training XGB with {n_folds}-fold cross-validation...")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold + 1}/{n_folds}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        print(f'Training fold {fold + 1} with {len(train_idx)} samples', flush=True)
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred_proba)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, y_pred_proba > 0.5, average='binary'
        )

        print(f"Fold {fold + 1} - AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        cv_scores.append(auc)
        models.append(model)

        model_path = output_dir / f'pre_xgb_fold_{fold}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

    print(f"\nCross-validation AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    print("Training final model on all data...")
    print('Training final model', flush=True)
    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X, y)

    final_model_path = output_dir / 'pre_xgb_final.pkl'
    with open(final_model_path, 'wb') as f:
        pickle.dump(final_model, f)

    feature_path = output_dir / 'feature_names.pkl'
    with open(feature_path, 'wb') as f:
        pickle.dump(feature_names, f)

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)

    importance_path = output_dir / 'feature_importance.csv'
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

    # Data source arguments (mutually exclusive modes)
    parser.add_argument('--metadata_dir', type=str, default=None,
                        help='Directory containing *_meta.json files (enables player features)')
    parser.add_argument('--ball_det_dict_path', type=str, default=None,
                        help='Path to ball detection dictionary (kinematic features only)')

    # Dataset paths
    parser.add_argument('--dataset_path', type=str, default=str(cfg.DATASET_PATH),
                        help='Path to dataset root')
    parser.add_argument('--header_dataset', type=str, default=str(cfg.DATASET_PATH / 'header_dataset'),
                        help='Path to header annotation dataset')

    # Feature configuration
    parser.add_argument('--use_player_features', action='store_true', default=True,
                        help='Include player-based spatial features (requires --metadata_dir)')
    parser.add_argument('--no_player_features', action='store_true', default=False,
                        help='Disable player features even when using metadata')

    # Training arguments
    parser.add_argument('--output_dir', type=str, default=str(cfg.CACHE_PATH / 'pre_xgb'),
                        help='Output directory for models and results')
    parser.add_argument('--neg_ratio', type=float, default=3.0,
                        help='Negative to positive sampling ratio')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Probability threshold for proposals')
    parser.add_argument('--max_proposals_per_min', type=int, default=5,
                        help='Maximum proposals per minute of video')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of CV folds')
    parser.add_argument('--n_estimators', type=int, default=100,
                        help='Number of boosting rounds (trees)')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Number of parallel threads for XGBoost')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    use_player_features = args.use_player_features and not args.no_player_features

    # Determine which mode to use
    if args.metadata_dir:
        # Mode 1: Load from metadata JSON files (with player features)
        print("=" * 60)
        print("Running in METADATA MODE (with player features)")
        print("=" * 60)

        metadata_dir = Path(args.metadata_dir)
        if not metadata_dir.exists():
            print(f"Error: Metadata directory not found: {metadata_dir}")
            return

        # Load features from metadata JSON files
        all_features, all_samples = load_metadata_from_directory(
            metadata_dir, use_player_features=use_player_features
        )

        if not all_samples:
            print("No samples found in metadata directory!")
            return

        # Create training data
        X, y, metadata, feature_names = create_training_data_from_metadata(
            all_features, all_samples,
            use_player_features=use_player_features,
            neg_sampling_ratio=args.neg_ratio
        )

        # Note: Proposal generation in metadata mode uses the loaded samples
        ball_det_dict = None

    else:
        # Mode 2: Load from ball detection dictionary (kinematic features only)
        print("=" * 60)
        print("Running in BALL_DET_DICT MODE (kinematic features only)")
        print("=" * 60)

        ball_det_path = Path(args.ball_det_dict_path or cfg.BALL_DET_DICT_PATH)
        header_dataset_path = Path(args.header_dataset)

        print(f"Loading ball detection dictionary from {ball_det_path}")
        ball_det_dict = load_ball_det_dict(ball_det_path)
        print(f"Loaded detections for {len(ball_det_dict)} videos")

        print(f"Loading header labels from {header_dataset_path}")
        labels_df = load_header_labels(header_dataset_path)
        print(f"Loaded {len(labels_df)} header events across {labels_df['video_id'].nunique()} videos")

        X, y, metadata, feature_names = create_training_data(
            ball_det_dict, labels_df, args.neg_ratio
        )

    if len(X) == 0:
        print("No training data found!")
        return

    print(f"\nFeature set: {len(feature_names)} features")
    print(f"Features: {feature_names[:10]}... (showing first 10)")

    # Train model
    final_model, fold_models, cv_scores = train_pre_xgb(
        X,
        y,
        feature_names,
        output_dir,
        args.n_folds,
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
    )

    # Generate proposals (only in ball_det_dict mode for now)
    if ball_det_dict:
        proposals = generate_proposals(
            ball_det_dict, final_model, feature_names,
            args.threshold, args.max_proposals_per_min
        )

        # Save proposals
        proposals_path = output_dir / 'proposals.pkl'
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
        proposals_csv_path = output_dir / 'proposals.csv'
        proposals_df.to_csv(proposals_csv_path, index=False)

        print(f"\nGenerated {len(proposals_csv_data)} total proposals")
        if proposals:
            print(f"Average proposals per video: {len(proposals_csv_data) / len(proposals):.1f}")
        print(f"Saved proposals to {proposals_csv_path}")
    else:
        print("\nNote: Proposal generation skipped in metadata mode.")
        print("Use the trained model with --inference_only for generating proposals.")

    # Save training metadata
    metadata_df = pd.DataFrame(metadata)
    metadata_path = output_dir / 'training_metadata.csv'
    metadata_df.to_csv(metadata_path, index=False)

    # Save configuration info
    config_info = {
        'mode': 'metadata' if args.metadata_dir else 'ball_det_dict',
        'use_player_features': use_player_features,
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'neg_ratio': args.neg_ratio,
        'n_folds': args.n_folds,
        'cv_auc_mean': float(np.mean(cv_scores)),
        'cv_auc_std': float(np.std(cv_scores))
    }
    config_path = output_dir / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump(config_info, f, indent=2)

    print(f"\nTraining complete! Models and results saved to {output_dir}")

if __name__ == "__main__":
    main()
