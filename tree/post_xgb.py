#!/usr/bin/env python3
"""
XGB post-processor for header detection
Uses temporal neighborhood probabilities for final scoring
"""

import os
import sys
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import pickle
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import xgboost as xgb

# Add paths
sys.path.append('../configs')
from header_default import *

def load_probabilities(probs_path):
    """Load CNN probabilities from CSV"""
    print(f"Loading probabilities from {probs_path}")
    df = pd.read_csv(probs_path)
    
    # Ensure required columns exist
    required_cols = ['video_id', 'frame_id', 'cnn_prob', 'ensemble_prob']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"Loaded {len(df)} probability records")
    return df

def build_temporal_features(probs_df, window_size=15):
    """
    Build temporal features from CNN probabilities
    Following NFL solution: use neighboring probabilities as features
    
    Args:
        probs_df: DataFrame with columns [video_id, frame_id, cnn_prob, ensemble_prob, ...]
        window_size: Size of temporal window (total window = 2*window_size + 1)
    
    Returns:
        DataFrame with temporal features
    """
    print(f"Building temporal features with window size {window_size}...")
    
    temporal_features = []
    
    # Group by video for temporal processing
    for video_id, video_df in tqdm(probs_df.groupby('video_id')):
        # Sort by frame_id
        video_df = video_df.sort_values('frame_id').reset_index(drop=True)
        
        # Create frame lookup for fast access
        frame_to_idx = {row['frame_id']: idx for idx, row in video_df.iterrows()}
        
        for idx, row in video_df.iterrows():
            frame_id = row['frame_id']
            
            # Initialize feature dictionary
            features = {
                'video_id': video_id,
                'frame_id': frame_id,
                'center_pre_xgb_prob': row.get('pre_xgb_prob', 0.0),
                'center_cnn_prob': row['cnn_prob'],
                'center_ensemble_prob': row['ensemble_prob']
            }
            
            # Extract temporal window features
            cnn_probs = []
            ensemble_probs = []
            pre_xgb_probs = []
            
            for offset in range(-window_size, window_size + 1):
                target_idx = idx + offset
                
                if 0 <= target_idx < len(video_df):
                    # Valid frame within video
                    target_row = video_df.iloc[target_idx]
                    cnn_probs.append(target_row['cnn_prob'])
                    ensemble_probs.append(target_row['ensemble_prob'])
                    pre_xgb_probs.append(target_row.get('pre_xgb_prob', 0.0))
                else:
                    # Pad with zeros for frames outside video bounds
                    cnn_probs.append(0.0)
                    ensemble_probs.append(0.0)
                    pre_xgb_probs.append(0.0)
            
            # Add individual probability features (similar to NFL solution)
            for i, prob in enumerate(cnn_probs):
                features[f'cnn_prob_{i-window_size}'] = prob
            
            for i, prob in enumerate(ensemble_probs):
                features[f'ensemble_prob_{i-window_size}'] = prob
            
            # Add statistical features over the window
            features.update({
                'cnn_prob_mean': np.mean(cnn_probs),
                'cnn_prob_std': np.std(cnn_probs),
                'cnn_prob_max': np.max(cnn_probs),
                'cnn_prob_min': np.min(cnn_probs),
                'cnn_prob_median': np.median(cnn_probs),
                
                'ensemble_prob_mean': np.mean(ensemble_probs),
                'ensemble_prob_std': np.std(ensemble_probs),
                'ensemble_prob_max': np.max(ensemble_probs),
                'ensemble_prob_min': np.min(ensemble_probs),
                'ensemble_prob_median': np.median(ensemble_probs),
                
                'pre_xgb_prob_mean': np.mean(pre_xgb_probs),
                'pre_xgb_prob_std': np.std(pre_xgb_probs),
                'pre_xgb_prob_max': np.max(pre_xgb_probs)
            })
            
            # Add trend features
            if len(cnn_probs) >= 3:
                # Linear trend (slope)
                x = np.arange(len(cnn_probs))
                cnn_slope = np.polyfit(x, cnn_probs, 1)[0]
                ensemble_slope = np.polyfit(x, ensemble_probs, 1)[0]
                
                features.update({
                    'cnn_prob_slope': cnn_slope,
                    'ensemble_prob_slope': ensemble_slope
                })
            else:
                features.update({
                    'cnn_prob_slope': 0.0,
                    'ensemble_prob_slope': 0.0
                })
            
            # Add local maxima indicators
            center_idx = window_size
            is_local_max_cnn = (center_idx > 0 and center_idx < len(cnn_probs) - 1 and
                               cnn_probs[center_idx] >= cnn_probs[center_idx-1] and
                               cnn_probs[center_idx] >= cnn_probs[center_idx+1])
            
            is_local_max_ensemble = (center_idx > 0 and center_idx < len(ensemble_probs) - 1 and
                                   ensemble_probs[center_idx] >= ensemble_probs[center_idx-1] and
                                   ensemble_probs[center_idx] >= ensemble_probs[center_idx+1])
            
            features.update({
                'is_local_max_cnn': float(is_local_max_cnn),
                'is_local_max_ensemble': float(is_local_max_ensemble)
            })
            
            temporal_features.append(features)
    
    temporal_df = pd.DataFrame(temporal_features)
    print(f"Built temporal features for {len(temporal_df)} samples")
    
    return temporal_df

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
                                            labels[video_id] = set()
                                        labels[video_id].update(frame_numbers)
                                        
                                        print(f"Loaded {len(frame_numbers)} header frames from {file_path}")
                                
                                except Exception as e:
                                    print(f"Error loading {file_path}: {e}")
    
    return labels

def create_training_data_for_postprocessing(temporal_df, ground_truth_labels):
    """Create training data for post-processing XGB"""
    
    # Add ground truth labels
    temporal_df = temporal_df.copy()
    temporal_df['label'] = 0  # Default to non-header
    
    for idx, row in temporal_df.iterrows():
        video_id = row['video_id']
        frame_id = row['frame_id']
        
        if video_id in ground_truth_labels and frame_id in ground_truth_labels[video_id]:
            temporal_df.at[idx, 'label'] = 1  # Header
    
    print(f"Training data: {len(temporal_df)} samples")
    print(f"Positive samples: {temporal_df['label'].sum()} ({temporal_df['label'].mean()*100:.1f}%)")
    
    return temporal_df

def train_post_xgb(temporal_df, output_dir, window_size=15, n_folds=5):
    """Train post-processing XGB model"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare features
    exclude_cols = ['video_id', 'frame_id', 'label']
    feature_cols = [col for col in temporal_df.columns if col not in exclude_cols]
    
    X = temporal_df[feature_cols].values
    y = temporal_df['label'].values
    
    print(f"Training post-XGB with {len(feature_cols)} features on {len(X)} samples")
    
    # XGB parameters (similar to NFL solution)
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 4,  # Shallower for post-processing
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_estimators': 200
    }
    
    # Cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_scores = []
    models = []
    
    print(f"Training with {n_folds}-fold cross-validation...")
    
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
        model_path = os.path.join(output_dir, f'post_xgb_fold_{fold}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    
    print(f"\nCross-validation AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    # Train final model on all data
    print("Training final model on all data...")
    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X, y)
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'post_xgb_final.pkl')
    with open(final_model_path, 'wb') as f:
        pickle.dump(final_model, f)
    
    # Save feature names
    feature_path = os.path.join(output_dir, 'feature_names.pkl')
    with open(feature_path, 'wb') as f:
        pickle.dump(feature_cols, f)
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    importance_path = os.path.join(output_dir, 'feature_importance.csv')
    importance_df.to_csv(importance_path, index=False)
    
    print(f"\nTop 15 most important features:")
    print(importance_df.head(15))
    
    return final_model, models, cv_scores, feature_cols

def apply_post_processing(temporal_df, model, feature_cols, output_path):
    """Apply trained post-processing model to get final scores"""
    
    # Prepare features
    X = temporal_df[feature_cols].values
    
    # Predict final probabilities
    final_probs = model.predict_proba(X)[:, 1]
    
    # Add to dataframe
    result_df = temporal_df[['video_id', 'frame_id', 'center_cnn_prob', 'center_ensemble_prob']].copy()
    result_df['final_prob'] = final_probs
    
    # Save results
    result_df.to_csv(output_path, index=False)
    
    print(f"Applied post-processing to {len(result_df)} samples")
    print(f"Final prob stats: mean={np.mean(final_probs):.4f}, std={np.std(final_probs):.4f}")
    print(f"Results saved to {output_path}")
    
    return result_df

def main():
    parser = argparse.ArgumentParser(description='Train XGB post-processor for header detection')
    parser.add_argument('--probs_csv', type=str, required=True,
                        help='Path to CNN probabilities CSV file')
    parser.add_argument('--dataset_path', type=str, default='../../DeepImpact',
                        help='Path to dataset for ground truth labels')
    parser.add_argument('--output_dir', type=str, default='post_xgb',
                        help='Output directory for models and results')
    parser.add_argument('--window_size', type=int, default=15,
                        help='Temporal window size for features')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of CV folds')
    parser.add_argument('--inference_only', action='store_true',
                        help='Only run inference with pre-trained model')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to pre-trained model for inference')
    
    args = parser.parse_args()
    
    # Load probabilities
    probs_df = load_probabilities(args.probs_csv)
    
    # Build temporal features
    temporal_df = build_temporal_features(probs_df, args.window_size)
    
    if args.inference_only:
        # Inference only mode
        if not args.model_path or not os.path.exists(args.model_path):
            raise ValueError("Model path required for inference mode")
        
        print(f"Loading pre-trained model from {args.model_path}")
        with open(args.model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load feature names
        feature_path = os.path.join(os.path.dirname(args.model_path), 'feature_names.pkl')
        with open(feature_path, 'rb') as f:
            feature_cols = pickle.load(f)
        
        # Apply post-processing
        output_path = os.path.join(args.output_dir, 'final_predictions.csv')
        result_df = apply_post_processing(temporal_df, model, feature_cols, output_path)
        
    else:
        # Training mode
        print("Loading ground truth labels...")
        ground_truth_labels = load_ground_truth_labels(args.dataset_path)
        print(f"Loaded ground truth for {len(ground_truth_labels)} videos")
        
        # Create training data
        temporal_df = create_training_data_for_postprocessing(temporal_df, ground_truth_labels)
        
        if temporal_df['label'].sum() == 0:
            print("No positive samples found! Check ground truth labels.")
            return
        
        # Train model
        final_model, fold_models, cv_scores, feature_cols = train_post_xgb(
            temporal_df, args.output_dir, args.window_size, args.n_folds
        )
        
        # Apply to all data for final predictions
        output_path = os.path.join(args.output_dir, 'final_predictions.csv')
        result_df = apply_post_processing(temporal_df, final_model, feature_cols, output_path)
        
        print(f"Post-processing training complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
