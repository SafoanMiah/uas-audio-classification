from pathlib import Path


# Paths (flip the comment when moving to Colab)
# ROOT = Path("/content/drive/MyDrive/drone_detection")
ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
CKPT_ROOT = ROOT / "checkpoints"


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


# Splits at the recording level to prevent leakage
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
SEED = 42


# SVM baseline
SVM_N_MFCC = 13
SVM_C = 1.0


# CNN (scratch)
CNN_CHANNELS = [32, 64, 128, 256, 512]
CNN_DROPOUT = 0.3
CNN_FC_DIM = 256
CNN_WEIGHT_DECAY = 1e-4
CNN_LR = 1e-3
CNN_BATCH_SIZE = 32
CNN_MAX_EPOCHS = 60
CNN_PATIENCE = 15


# PANNs CNN10 fine-tuning
PANN_CKPT_URL = "https://zenodo.org/record/3987831/files/Cnn10_mAP%3D0.380.pth"
PANN_EMBEDDING_DIM = 512
PANN_BATCH_SIZE = 32
PANN_WEIGHT_DECAY = 1e-4
PANN_DROPOUT = 0.3
PANN_PATIENCE = 8

# (name, epochs, lr_head, lr_backbone, unfreeze_last_n_blocks)
PANN_STAGES = [
    ("head_only", 10, 1e-3, 0.0, 0),
    ("last_block", 15, 1e-3, 1e-5, 1),
    ("last_two_blocks", 20, 1e-3, 1e-5, 2),
]


# Augmentation
TIME_MASK_PARAM = 30
FREQ_MASK_PARAM = 20
N_TIME_MASKS = 2
N_FREQ_MASKS = 2
USE_MIXUP = True
MIXUP_ALPHA = 0.2


# Evaluation
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_CI = 0.95
SNR_LEVELS_DB = ["clean", 10, 5, 0, -5]


# Runtime
NUM_WORKERS = 2
PIN_MEMORY = True


def ensure_dirs():
    for p in (DATA_ROOT, CKPT_ROOT):
        p.mkdir(parents=True, exist_ok=True)
