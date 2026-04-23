"""
Configuration for the drone/helicopter/background acoustic classifier.
"""

from pathlib import Path


# Paths: use Colab's Drive mount when available, otherwise the project folder
# DRIVE_ROOT = Path("/content/drive/MyDrive/drone_detection")
DRIVE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = DRIVE_ROOT / "data"
TRAIN_ROOT = DRIVE_ROOT / "train"
CKPT_ROOT = DRIVE_ROOT / "checkpoints"
MODEL_ROOT = DRIVE_ROOT / "models"

# Public dataset sources
DATASETS = {
    "svanstrom": "https://zenodo.org/record/5500576",
    "al_emadi": "https://github.com/saraalemadi/DroneAudioDataset",
    "esc50": "https://github.com/karolpiczak/ESC-50",
}


# Classes
CLASSES = ["drone", "helicopter", "background"]
NUM_CLASSES = 3
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


# Audio / spectrogram
SAMPLE_RATE = 32000
WINDOW_SECONDS = 2.0
HOP_SECONDS = 1.0
N_FFT = 1024
HOP_LENGTH = 320
N_MELS = 128
FMIN = 50
FMAX = 14000


# Data splits (done at the recording level to prevent leakaging)
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
SEED = 42
HOLDOUT_SOURCE = "al_emadi"  # reserved for cross-source evaluation


# SVM baseline
SVM_FEATURES = [
    "mfcc",
    "spectral_centroid",
    "spectral_rolloff",
    "spectral_bandwidth",
    "zcr",
    "rms",
]
SVM_N_MFCC = 13
SVM_KERNEL = "linear"
SVM_C = 1.0
SVM_CLASS_WEIGHT = "balanced"


# CNN
CNN_CHANNELS = [32, 64, 128, 256, 512]
CNN_KERNEL_SIZE = 3
CNN_POOL_SIZE = 2
CNN_DROPOUT = 0.4
CNN_FC_DIM = 256
CNN_WEIGHT_DECAY = 1e-4
CNN_LR = 1e-3
CNN_BATCH_SIZE = 32
CNN_MAX_EPOCHS = 60
CNN_EARLY_STOPPING_PATIENCE = 10


# PANNs CNN10 fine-tuning (three-stage progressive unfreezing)
PANN_CKPT_URL = "https://zenodo.org/record/3987831/files/Cnn10_mAP%3D0.380.pth"
PANN_EMBEDDING_DIM = 512
PANN_BATCH_SIZE = 32
PANN_WEIGHT_DECAY = 1e-4
PANN_DROPOUT = 0.3
PANN_EARLY_STOPPING_PATIENCE = 8

# Each stage: (name, epochs, lr_head, lr_backbone, unfreeze_last_n_blocks)
PANN_STAGES = [
    ("stage1_head_only", 10, 1e-3, 0.0, 0),
    ("stage2_last_block", 15, 1e-3, 1e-5, 1),
    ("stage3_last_two_blocks", 20, 1e-3, 1e-5, 2),
]


# Augmentation
USE_SPECAUGMENT = True
TIME_MASK_PARAM = 30
FREQ_MASK_PARAM = 20
N_TIME_MASKS = 2
N_FREQ_MASKS = 2

USE_MIXUP = True
MIXUP_ALPHA = 0.2


# Evaluation
PRIMARY_METRIC = "macro_f1"
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_CI = 0.95
SNR_LEVELS_DB = ["clean", 10, 5, 0, -5]
NOISE_SOURCE = "esc50"


# Runtime
NUM_WORKERS = 2
PIN_MEMORY = True
CHECKPOINT_EVERY_N_EPOCHS = 2
LOG_EVERY_N_STEPS = 20


def ensure_dirs():
    for p in (DATA_ROOT, CACHE_ROOT, CKPT_ROOT, MANIFEST_ROOT):
        p.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print("Drone detection config")
    print(f"Classes: {CLASSES}")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Window: {WINDOW_SECONDS}s, hop {HOP_SECONDS}s")
    print(f"Mel bands: {N_MELS}")
    print(f"PANN stages: {[s[0] for s in PANN_STAGES]}")
    print(f"SNR sweep: {SNR_LEVELS_DB}")
