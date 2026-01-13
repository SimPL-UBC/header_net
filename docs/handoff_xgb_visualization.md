# Handoff: XGBoost Visualization Implementation

## Overview

This document provides a complete handoff for implementing XGBoost visualization tools for the header detection pipeline. These visualizations will aid in debugging, interpreting results, and improving the pre-filter and post-filter XGBoost models.

---

## 1. Visualizations to Implement

### 1.1 Feature Importance Plots

**Purpose**: Understand which features contribute most to predictions

| Visualization | Description | Priority |
|--------------|-------------|----------|
| Bar chart importance | Horizontal bar chart of top N features | High |
| Importance comparison | Side-by-side pre-filter vs post-filter | Medium |
| Feature category breakdown | Grouped by kinematic/player/temporal | High |

### 1.2 Tree Structure Visualizations

**Purpose**: Debug individual tree decisions

| Visualization | Description | Priority |
|--------------|-------------|----------|
| Single tree plot | Graphviz/matplotlib tree diagram | Medium |
| Tree depth distribution | Histogram of tree depths | Low |
| Decision path visualization | Highlight path for specific sample | High |

### 1.3 Model Performance Plots

**Purpose**: Evaluate model quality

| Visualization | Description | Priority |
|--------------|-------------|----------|
| ROC curve | ROC with AUC for XGBoost models | High |
| Precision-Recall curve | PR curve with AP score | High |
| Calibration curve | Reliability diagram | Medium |
| Learning curves | Training vs validation over boosting rounds | High |
| Cross-validation scores | Box plot of CV fold scores | Medium |

### 1.4 Prediction Analysis

**Purpose**: Understand prediction patterns

| Visualization | Description | Priority |
|--------------|-------------|----------|
| Probability distribution | Histogram of predicted probabilities | High |
| Confusion matrix | 2x2 heatmap | High |
| Threshold analysis | Metrics vs decision threshold | Medium |
| SHAP summary plot | Global feature impact | High |
| SHAP dependence plots | Feature interactions | Medium |

### 1.5 Pipeline-Specific Visualizations

**Purpose**: Debug the full pipeline

| Visualization | Description | Priority |
|--------------|-------------|----------|
| Pre-filter → CNN flow | Sankey diagram of sample filtering | Medium |
| Temporal probability plot | CNN probs over 31-frame window | High |
| Post-filter smoothing | Before/after comparison | High |
| Error analysis | Characteristics of misclassified samples | High |

---

## 2. File Structure

Follow the existing pattern in `visualizations/`:

```
visualizations/
├── xgb/                                  # NEW: XGBoost visualization module
│   ├── __init__.py
│   ├── plot_feature_importance.py        # Feature importance charts
│   ├── plot_tree_structure.py            # Tree diagrams
│   ├── plot_xgb_performance.py           # ROC, PR, learning curves
│   ├── plot_shap_analysis.py             # SHAP visualizations
│   ├── plot_temporal_probs.py            # Post-filter temporal analysis
│   ├── plot_pipeline_flow.py             # Pipeline debugging
│   └── run_xgb_viz.py                    # CLI entry point
```

---

## 3. Implementation Details

### 3.1 Feature Importance (`plot_feature_importance.py`)

```python
"""Feature importance visualizations for XGBoost models."""

from pathlib import Path
from typing import Optional, List, Dict
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Import shared utilities
import sys
sys.path.append(str(Path(__file__).parents[2]))
from visualizations.utils import apply_global_style, COLORS


def plot_feature_importance(
    model_path: Path,
    feature_names: Optional[List[str]] = None,
    top_n: int = 20,
    importance_type: str = "weight",  # "weight", "gain", "cover"
    save_path: Optional[Path] = None,
    title: str = "Feature Importance"
) -> None:
    """
    Plot horizontal bar chart of feature importances.

    Args:
        model_path: Path to pickled XGBoost model
        feature_names: List of feature names (loaded from model dir if None)
        top_n: Number of top features to show
        importance_type: Type of importance metric
        save_path: Path to save PNG
        title: Plot title
    """
    apply_global_style()

    # Load model and feature names
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    if feature_names is None:
        feature_path = model_path.parent / 'feature_names.pkl'
        with open(feature_path, 'rb') as f:
            feature_names = pickle.load(f)

    # Get importances
    importances = model.feature_importances_

    # Sort and take top N
    indices = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))

    y_pos = np.arange(len(indices))
    ax.barh(y_pos, importances[indices], color=COLORS.get('train', '#1f77b4'))
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel(f'Importance ({importance_type})')
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')


def plot_feature_importance_by_category(
    model_path: Path,
    save_path: Optional[Path] = None
) -> None:
    """
    Plot feature importance grouped by category.

    Categories:
    - Kinematic: x, y, vx, vy, speed, ax, ay, etc.
    - Player: dist_to_nearest_*, num_players_*, etc.
    - Temporal: *_mean_w, *_std_w, *_max_w
    """
    # Define category mappings
    CATEGORIES = {
        'Position': ['x', 'y'],
        'Velocity': ['vx', 'vy', 'speed'],
        'Acceleration': ['ax', 'ay', 'accel_mag'],
        'Trajectory': ['angle_change', 'speed_change', 'speed_drop_ratio', 'curvature', 'jerk'],
        'Detection': ['confidence', 'ball_size'],
        'Temporal Stats': ['speed_mean_w', 'speed_std_w', 'speed_max_w',
                          'accel_mean_w', 'accel_std_w', 'accel_max_w',
                          'angle_mean_w', 'angle_std_w', 'angle_max_w'],
        'Player Distance': ['dist_to_nearest_player', 'dist_to_nearest_head', 'avg_player_density'],
        'Player Count': ['num_players_50px', 'num_players_100px', 'num_players_200px', 'player_count'],
        'Player Context': ['nearest_player_rel_vx', 'nearest_player_rel_vy',
                          'nearest_head_y_offset', 'ball_above_nearest_head', 'goalkeeper_nearby'],
    }

    # Implementation: Load model, group importances by category, create grouped bar chart
    # ...
```

### 3.2 SHAP Analysis (`plot_shap_analysis.py`)

```python
"""SHAP-based interpretability for XGBoost models."""

import shap
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional
import pickle


def plot_shap_summary(
    model_path: Path,
    X: np.ndarray,
    feature_names: list,
    save_path: Optional[Path] = None,
    max_display: int = 20
) -> None:
    """
    Create SHAP summary plot showing feature impact distribution.

    Args:
        model_path: Path to XGBoost model
        X: Feature matrix (n_samples, n_features)
        feature_names: List of feature names
        save_path: Output path for PNG
        max_display: Max features to display
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Create SHAP explainer (TreeExplainer is fast for XGBoost)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Summary plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X,
        feature_names=feature_names,
        max_display=max_display,
        show=False
    )

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')


def plot_shap_dependence(
    model_path: Path,
    X: np.ndarray,
    feature_name: str,
    feature_names: list,
    interaction_feature: Optional[str] = None,
    save_path: Optional[Path] = None
) -> None:
    """
    Create SHAP dependence plot for a specific feature.

    Shows how the feature value affects the prediction,
    optionally colored by interaction with another feature.
    """
    # Implementation...


def plot_shap_waterfall(
    model_path: Path,
    X_sample: np.ndarray,
    feature_names: list,
    save_path: Optional[Path] = None
) -> None:
    """
    Create SHAP waterfall plot for a single prediction.

    Shows how each feature contributes to pushing the prediction
    from the base value to the final output.
    """
    # Implementation...
```

### 3.3 Temporal Probability Analysis (`plot_temporal_probs.py`)

```python
"""Temporal probability visualizations for post-filter analysis."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# Import shared utilities
import sys
sys.path.append(str(Path(__file__).parents[2]))
from visualizations.utils import apply_global_style, COLORS


def plot_temporal_probability_window(
    video_id: str,
    frame_id: int,
    probs_df: pd.DataFrame,
    window_size: int = 15,
    save_path: Optional[Path] = None
) -> None:
    """
    Plot CNN probabilities over 31-frame temporal window.

    Shows:
    - Raw CNN probabilities (blue line)
    - Pre-XGB probabilities (orange dashed)
    - Ensemble probabilities (green)
    - Post-filter output (red, if available)
    - Ground truth marker (vertical line)
    """
    apply_global_style()

    # Filter to video and frame range
    video_probs = probs_df[probs_df['video_id'] == video_id].sort_values('frame_id')

    center_idx = video_probs[video_probs['frame_id'] == frame_id].index[0]
    start_idx = max(0, center_idx - window_size)
    end_idx = min(len(video_probs), center_idx + window_size + 1)

    window_data = video_probs.iloc[start_idx:end_idx]

    fig, ax = plt.subplots(figsize=(12, 5))

    x = range(len(window_data))
    ax.plot(x, window_data['cnn_prob'], 'b-', label='CNN prob', linewidth=2)
    ax.plot(x, window_data['ensemble_prob'], 'g--', label='Ensemble', linewidth=1.5)

    if 'pre_xgb_prob' in window_data.columns:
        ax.plot(x, window_data['pre_xgb_prob'], 'orange', linestyle=':', label='Pre-XGB')

    if 'final_prob' in window_data.columns:
        ax.plot(x, window_data['final_prob'], 'r-', label='Post-XGB', linewidth=2)

    # Mark center frame
    center_x = window_size if start_idx == center_idx - window_size else center_idx - start_idx
    ax.axvline(x=center_x, color='gray', linestyle='--', alpha=0.5, label='Center frame')

    ax.set_xlabel('Frame offset')
    ax.set_ylabel('Probability')
    ax.set_title(f'Temporal Probability Window: {video_id} @ frame {frame_id}')
    ax.legend()
    ax.set_ylim(0, 1)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')


def plot_smoothing_comparison(
    probs_df: pd.DataFrame,
    n_samples: int = 5,
    save_path: Optional[Path] = None
) -> None:
    """
    Compare before/after post-filter smoothing for multiple samples.

    Creates a grid showing how post-filter changes predictions.
    """
    # Implementation: Select samples, create subplot grid, show CNN vs final probs


def plot_error_temporal_patterns(
    probs_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    save_path: Optional[Path] = None
) -> None:
    """
    Analyze temporal patterns in false positives and false negatives.

    Helps debug why certain predictions fail.
    """
    # Implementation...
```

### 3.4 CLI Entry Point (`run_xgb_viz.py`)

```python
"""CLI for XGBoost visualizations."""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from plot_feature_importance import (
    plot_feature_importance,
    plot_feature_importance_by_category
)
from plot_xgb_performance import (
    plot_xgb_roc_curve,
    plot_xgb_pr_curve,
    plot_learning_curve,
    plot_cv_scores
)
from plot_shap_analysis import (
    plot_shap_summary,
    plot_shap_dependence
)
from plot_temporal_probs import (
    plot_temporal_probability_window,
    plot_smoothing_comparison
)


def parse_args():
    parser = argparse.ArgumentParser(description='XGBoost Visualization Tools')

    # Required arguments
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Directory containing XGBoost model and artifacts')
    parser.add_argument('--save-dir', type=str, required=True,
                        help='Output directory for visualizations')

    # Model type
    parser.add_argument('--model-type', type=str, choices=['pre', 'post'], default='pre',
                        help='Type of XGBoost model (pre-filter or post-filter)')

    # Optional data paths
    parser.add_argument('--probs-csv', type=str, default=None,
                        help='CNN probabilities CSV (for temporal analysis)')
    parser.add_argument('--training-data', type=str, default=None,
                        help='Training data path (for SHAP analysis)')

    # Visualization selection
    parser.add_argument('--all', action='store_true',
                        help='Generate all visualizations')
    parser.add_argument('--importance', action='store_true',
                        help='Generate feature importance plots')
    parser.add_argument('--performance', action='store_true',
                        help='Generate performance plots (ROC, PR, etc.)')
    parser.add_argument('--shap', action='store_true',
                        help='Generate SHAP analysis plots')
    parser.add_argument('--temporal', action='store_true',
                        help='Generate temporal probability plots')

    # SHAP options
    parser.add_argument('--shap-samples', type=int, default=500,
                        help='Number of samples for SHAP analysis')

    # Display options
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plots interactively')

    return parser.parse_args()


def main():
    args = parse_args()

    model_dir = Path(args.model_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Determine model path based on type
    if args.model_type == 'pre':
        model_path = model_dir / 'pre_xgb_final.pkl'
    else:
        model_path = model_dir / 'post_xgb_final.pkl'

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    # Feature importance (always quick to generate)
    if args.all or args.importance:
        print("Generating feature importance plots...")
        plot_feature_importance(
            model_path,
            save_path=save_dir / 'feature_importance.png',
            title=f'{"Pre" if args.model_type == "pre" else "Post"}-filter Feature Importance'
        )
        plot_feature_importance_by_category(
            model_path,
            save_path=save_dir / 'feature_importance_by_category.png'
        )
        print(f"  Saved: feature_importance.png, feature_importance_by_category.png")

    # Performance plots
    if args.all or args.performance:
        print("Generating performance plots...")
        # Load predictions and generate ROC, PR, confusion matrix
        # Implementation needed

    # SHAP analysis (slower, requires data)
    if args.shap:
        if args.training_data is None:
            print("Warning: --training-data required for SHAP analysis, skipping...")
        else:
            print("Generating SHAP analysis...")
            # Load training data, run SHAP
            # Implementation needed

    # Temporal analysis (requires probs CSV)
    if args.temporal:
        if args.probs_csv is None:
            print("Warning: --probs-csv required for temporal analysis, skipping...")
        else:
            print("Generating temporal probability plots...")
            # Implementation needed

    if not args.no_show:
        plt.show()

    print(f"\nVisualizations saved to {save_dir}")


if __name__ == '__main__':
    main()
```

---

## 4. Usage Examples

### Basic Feature Importance

```bash
conda activate deep_impact_env

# Pre-filter XGBoost
python -m visualizations.xgb.run_xgb_viz \
    --model-dir cache/pre_xgb_full \
    --save-dir cache/pre_xgb_full/visualizations \
    --model-type pre \
    --importance \
    --no-show

# Post-filter XGBoost
python -m visualizations.xgb.run_xgb_viz \
    --model-dir cache/post_xgb \
    --save-dir cache/post_xgb/visualizations \
    --model-type post \
    --importance \
    --no-show
```

### Full Analysis with SHAP

```bash
python -m visualizations.xgb.run_xgb_viz \
    --model-dir cache/pre_xgb_full \
    --save-dir cache/pre_xgb_full/visualizations \
    --model-type pre \
    --all \
    --shap \
    --training-data cache/pre_xgb_full/training_metadata.csv \
    --shap-samples 1000 \
    --no-show
```

### Temporal Analysis for Post-filter

```bash
python -m visualizations.xgb.run_xgb_viz \
    --model-dir cache/post_xgb \
    --save-dir cache/post_xgb/visualizations \
    --model-type post \
    --temporal \
    --probs-csv cache/cnn_probabilities.csv \
    --no-show
```

---

## 5. Dependencies to Add

Add to `requirements.txt`:

```
shap>=0.42.0
graphviz>=0.20  # For tree visualization (optional)
```

---

## 6. Integration with Existing Pipeline

### Option A: Standalone Module (Recommended)
Keep XGBoost visualizations separate in `visualizations/xgb/` with its own CLI.

### Option B: Integrate into run_all.py
Add flags to existing `run_all.py`:

```python
# In run_all.py
parser.add_argument('--enable-xgb-viz', action='store_true',
                    help='Generate XGBoost visualizations')
parser.add_argument('--pre-xgb-model', type=str, default=None,
                    help='Path to pre-filter XGBoost model')
parser.add_argument('--post-xgb-model', type=str, default=None,
                    help='Path to post-filter XGBoost model')
```

---

## 7. Key Insights for Implementation

**Design Principles:**
1. **Follow existing patterns**: Use `apply_global_style()` from `utils.py`, same color scheme
2. **SHAP is powerful but slow**: Use `TreeExplainer` (fast for XGBoost), limit samples
3. **Temporal plots are unique**: Post-filter's 31-frame window is a key debugging tool
4. **Feature categories matter**: Group kinematic vs player features for interpretability

**Most Valuable Visualizations for Debugging:**
1. **SHAP summary plot** - Shows which features drive predictions globally
2. **Temporal probability window** - Debug why post-filter changes predictions
3. **Feature importance by category** - Quickly see if player features are useful
4. **Error analysis** - Understand failure modes

---

## 8. Testing Checklist

- [ ] Feature importance plots render correctly
- [ ] SHAP analysis runs without memory issues (sample limiting)
- [ ] Temporal probability plots show all probability types
- [ ] CLI handles missing optional arguments gracefully
- [ ] PNG output at 300 DPI
- [ ] Works with `--no-show` for HPC environments
- [ ] Documentation added to `docs/visualizations/`

---

## 9. Files to Create

| File | Purpose |
|------|---------|
| `visualizations/xgb/__init__.py` | Package initializer |
| `visualizations/xgb/plot_feature_importance.py` | Feature importance plots |
| `visualizations/xgb/plot_xgb_performance.py` | ROC, PR, learning curves |
| `visualizations/xgb/plot_shap_analysis.py` | SHAP visualizations |
| `visualizations/xgb/plot_temporal_probs.py` | Post-filter temporal analysis |
| `visualizations/xgb/run_xgb_viz.py` | CLI entry point |
| `docs/visualizations/xgb_visualization_guide.md` | User documentation |

---

## 10. Reference Files

These existing files should be referenced for patterns:

| File | What to Reference |
|------|-------------------|
| `visualizations/utils.py` | Color scheme (`COLORS`), styling (`apply_global_style`), data loading |
| `visualizations/run_all.py` | CLI argument pattern, orchestration logic |
| `visualizations/plot_roc_curve.py` | Performance plot pattern |
| `visualizations/plot_confusion_matrix.py` | Heatmap pattern |
| `tree/pre_xgb.py` | Feature names (`KINEMATIC_FEATURE_NAMES`, `FULL_FEATURE_NAMES`) |
| `tree/post_xgb.py` | Temporal feature structure, post-filter logic |
| `utils/player_features.py` | `PLAYER_FEATURE_NAMES` for category grouping |

---

## 11. Priority Order for Implementation

1. **`plot_feature_importance.py`** - Quick win, immediately useful
2. **`run_xgb_viz.py`** - CLI framework for all visualizations
3. **`plot_temporal_probs.py`** - Critical for debugging post-filter
4. **`plot_shap_analysis.py`** - Powerful interpretability
5. **`plot_xgb_performance.py`** - Standard ML evaluation
6. **`plot_tree_structure.py`** - Lower priority, nice to have
