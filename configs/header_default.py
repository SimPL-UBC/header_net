from pathlib import Path

# Directory layout -----------------------------------------------------------
HEADER_NET_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = HEADER_NET_ROOT.parent

DATASET_PATH = (REPO_ROOT / "DeepImpact").resolve()
SOCCERNET_PATH = (DATASET_PATH / "SoccerNet").resolve()  # league/year/match/{video,json}
CACHE_PATH = (HEADER_NET_ROOT / "cache").resolve()
OUTPUT_PATH = (HEADER_NET_ROOT / "outputs").resolve()
HEADER_DATASET_PATH = (DATASET_PATH / "header_dataset").resolve()  # contains SoccerNetV2 spreadsheets
YOLO_DETECTIONS_PATH = (DATASET_PATH / "yolo_detections").resolve()

# Video processing -----------------------------------------------------------
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
OUTPUT_SIZE = 256  # Final crop size after resizing for high-res inputs
LOW_RES_OUTPUT_SIZE = 64  # Output size when working with low-resolution footage
LOW_RES_MAX_DIM = 512  # Images with max dimension <= this are considered low-res
TEMPORAL_CHANNELS = 3  # Previous, current, next frame stacking

# Ball detection -------------------------------------------------------------
BALL_CONFIDENCE_THRESHOLD = 0.3
KALMAN_ENABLED = True
FALLBACK_CONTEXT_FRAMES = 30  # Frames to include on either side when synthesising detections
RF_DETR_WEIGHTS = HEADER_NET_ROOT / "checkpoints" / "rf-detr-medium.pth"
SOCCERNET_RFDETR_WEIGHTS = HEADER_NET_ROOT / "RFDETR-Soccernet" / "weights" / "checkpoint_best_regular.pth"
RF_DETR_DEVICE = None
RF_DETR_BATCH_SIZE = 4
RF_DETR_SCORE_THRESHOLD = BALL_CONFIDENCE_THRESHOLD
RF_DETR_FRAME_STRIDE = 1
RF_DETR_TOPK = 5


# Training configuration -----------------------------------------------------
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 50
NUM_FOLDS = 5

# Class weights for imbalanced data (header vs non-header)
CLASS_WEIGHTS = {0: 1.0, 1: 10.0}

# Derived paths --------------------------------------------------------------
BALL_DET_DICT_PATH = CACHE_PATH / "ball_det_dict.npy"
BALL_PLAYER_DET_DICT_PATH = CACHE_PATH / "ball_player_det_dict.npy"
HEADER_LABELS_PATH = DATASET_PATH / "header_labels.csv"
BACKGROUND_LABELS_PATH = DATASET_PATH / "background_labels.csv"

# Temporal post-processing ---------------------------------------------------
TEMPORAL_NMS_WINDOW = 10  # Frames to merge nearby detections
CONFIDENCE_THRESHOLD = 0.5

# Debug and visualization ----------------------------------------------------
DEBUG_MODE = True
SAVE_DEBUG_IMAGES = False
DEBUG_OUTPUT_PATH = OUTPUT_PATH / "debug"
