import os

# Dataset configuration
DATASET_PATH = "~/repository/heimdall_net/DeepImpact/"  # Update this path
CACHE_PATH = "header_net/cache"
OUTPUT_PATH = "header_net/outputs"

# Video processing
WINDOW_SIZE = [
    -24,
    -18,
    -12,
    -6,
    -3,
    0,
    3,
    6,
    12,
    18,
    24,
]  # Frame offsets around target frame
CROP_SCALE_FACTOR = 4.5  # Scale factor for cropping around ball (similar to NFL)
OUTPUT_SIZE = 128  # Final crop size after resizing
TEMPORAL_CHANNELS = 3  # Previous, current, next frame stacking

# Ball detection
BALL_CONFIDENCE_THRESHOLD = 0.3
KALMAN_ENABLED = True

# Training configuration
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 50
NUM_FOLDS = 5

# Class weights for imbalanced data (header vs non-header)
CLASS_WEIGHTS = {0: 1.0, 1: 10.0}  # Similar to DeepImpact weighting

# Paths
BALL_DET_DICT_PATH = os.path.join(DATASET_PATH, "ball_det_dict.npy")
HEADER_LABELS_PATH = os.path.join(DATASET_PATH, "header_labels.csv")
BACKGROUND_LABELS_PATH = os.path.join(DATASET_PATH, "background_labels.csv")

# Temporal post-processing
TEMPORAL_NMS_WINDOW = 10  # Frames to merge nearby detections
CONFIDENCE_THRESHOLD = 0.5

# Debug and visualization
DEBUG_MODE = True
SAVE_DEBUG_IMAGES = False
DEBUG_OUTPUT_PATH = os.path.join(OUTPUT_PATH, "debug")
