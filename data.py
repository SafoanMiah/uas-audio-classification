"""Data pipeline: manifests, splits, spectrograms, augmentation."""

import os
import random
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset, DataLoader

import config as cfg


# Reproducibility
torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)
random.seed(cfg.SEED)


# Svanstrom
def build_svanstrom_manifest():
    audio_dir = cfg.DATA_ROOT / "svanstrom" / "audio"
    rows = []
    for fname in sorted(os.listdir(audio_dir)):
        if not fname.lower().endswith(".wav"):
            continue
        prefix = fname.split("_")[0].lower()
        if prefix in cfg.CLASSES:
            rows.append(
                {
                    "path": str(audio_dir / fname),
                    "label": prefix,
                    "source": "svanstrom",
                    "recording_id": fname.replace(".wav", ""),
                }
            )
    return pd.DataFrame(rows)


# Al-Emadi
def build_al_emadi_manifest():
    base = cfg.DATA_ROOT / "al_emadi" / "Binary_Drone_Audio"
    rows = []
    for sub, label in [("yes_drone", "drone"), ("unknown", "background")]:
        folder = base / sub
        if not folder.exists():
            continue
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith(".wav"):
                rows.append(
                    {
                        "path": str(folder / fname),
                        "label": label,
                        "source": "al_emadi",
                        "recording_id": f"al_emadi_{fname.replace('.wav', '')}",
                    }
                )
    return pd.DataFrame(rows)


# ESC-50 noise subset for the SNR sweep
def build_esc50_manifest(noise_categories=None):
    if noise_categories is None:
        noise_categories = [
            "wind",
            "rain",
            "crackling_fire",
            "crickets",
            "thunderstorm",
            "footsteps",
            "car_horn",
            "engine",
        ]
    meta = pd.read_csv(cfg.DATA_ROOT / "esc50" / "meta" / "esc50.csv")
    meta = meta[meta["category"].isin(noise_categories)].copy()
    meta["path"] = meta["filename"].apply(
        lambda f: str(cfg.DATA_ROOT / "esc50" / "audio" / f)
    )
    return meta[["path", "category"]].reset_index(drop=True)


# Split by recording_id so overlapping windows never cross splits
def recording_level_split(df, seed=cfg.SEED):
    rng = np.random.RandomState(seed)
    train_ids, val_ids, test_ids = [], [], []

    for label in df["label"].unique():
        recs = rng.permutation(df[df["label"] == label]["recording_id"].unique())
        n = len(recs)
        n_train = int(cfg.TRAIN_FRAC * n)
        n_val = int(cfg.VAL_FRAC * n)
        train_ids += list(recs[:n_train])
        val_ids += list(recs[n_train : n_train + n_val])
        test_ids += list(recs[n_train + n_val :])

    df = df.copy()
    df["split"] = "unassigned"
    df.loc[df["recording_id"].isin(train_ids), "split"] = "train"
    df.loc[df["recording_id"].isin(val_ids), "split"] = "val"
    df.loc[df["recording_id"].isin(test_ids), "split"] = "test"
    return df


def load_audio(path, sr=cfg.SAMPLE_RATE):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


# Cut a clip into 2s windows with 50% overlap
def segment_audio(y, sr=cfg.SAMPLE_RATE):
    win = int(cfg.WINDOW_SECONDS * sr)
    hop = int(cfg.HOP_SECONDS * sr)
    if len(y) < win:
        return [np.pad(y, (0, win - len(y)))]
    return [y[i : i + win] for i in range(0, len(y) - win + 1, hop)]


def log_mel_spectrogram(y, sr=cfg.SAMPLE_RATE):
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        n_mels=cfg.N_MELS,
        fmin=cfg.FMIN,
        fmax=cfg.FMAX,
    )
    return librosa.power_to_db(S, ref=np.max).astype(np.float32)


# Hand-engineered features for the SVM baseline
def extract_svm_features(y, sr=cfg.SAMPLE_RATE):
    feats = []

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=cfg.SVM_N_MFCC)
    feats += list(mfcc.mean(axis=1)) + list(mfcc.std(axis=1))

    for fn in [
        librosa.feature.spectral_centroid,
        librosa.feature.spectral_rolloff,
        librosa.feature.spectral_bandwidth,
    ]:
        v = fn(y=y, sr=sr)
        feats += [v.mean(), v.std()]

    zcr = librosa.feature.zero_crossing_rate(y=y)
    feats += [zcr.mean(), zcr.std()]

    rms = librosa.feature.rms(y=y)
    feats += [rms.mean(), rms.std()]

    return np.array(feats, dtype=np.float32)


# SpecAugment: random time + frequency masks
def spec_augment(spec):
    spec = spec.copy()
    n_mels, n_frames = spec.shape
    fill = spec.min()

    for _ in range(cfg.N_TIME_MASKS):
        t = np.random.randint(0, cfg.TIME_MASK_PARAM)
        t0 = np.random.randint(0, max(1, n_frames - t))
        spec[:, t0 : t0 + t] = fill

    for _ in range(cfg.N_FREQ_MASKS):
        f = np.random.randint(0, cfg.FREQ_MASK_PARAM)
        f0 = np.random.randint(0, max(1, n_mels - f))
        spec[f0 : f0 + f, :] = fill

    return spec


# MixUp on a batch (y should be one-hot or soft labels)
def mixup_batch(x, y, alpha=cfg.MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0))
    return lam * x + (1 - lam) * x[idx], lam * y + (1 - lam) * y[idx]


# Add noise to signal at target SNR (dB)
def mix_at_snr(signal, noise, snr_db):
    if len(noise) < len(signal):
        noise = np.tile(noise, int(np.ceil(len(signal) / len(noise))))
    noise = noise[: len(signal)]

    sig_power = np.mean(signal**2) + 1e-10
    noise_power = np.mean(noise**2) + 1e-10
    scale = np.sqrt((sig_power / (10 ** (snr_db / 10))) / noise_power)
    return signal + noise * scale


class AudioDataset(Dataset):
    def __init__(self, manifest, augment=False, snr_db=None, noise_manifest=None):
        super(AudioDataset, self).__init__()
        self.augment = augment
        self.snr_db = snr_db
        self.noise_paths = (
            noise_manifest["path"].tolist() if noise_manifest is not None else None
        )

        # For SNR runs we keep raw audio (noise is added per sample).
        # Or we precompute log-mels once so epochs don't redo the STFT.
        self.labels = []
        self.audios = []  # populated only when snr_db is set
        self.specs = []  # populated only when snr_db is None

        for _, row in manifest.iterrows():
            y = load_audio(row["path"])
            for seg in segment_audio(y):
                self.labels.append(cfg.CLASS_TO_IDX[row["label"]])
                if snr_db is not None:
                    self.audios.append(seg)
                else:
                    self.specs.append(log_mel_spectrogram(seg))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]

        if self.snr_db is not None:
            y = self.audios[idx]
            if self.snr_db != "clean":
                noise = load_audio(random.choice(self.noise_paths))
                y = mix_at_snr(y, noise, self.snr_db)
            spec = log_mel_spectrogram(y)
        else:
            spec = self.specs[idx]

        if self.augment and cfg.USE_SPECAUGMENT:
            spec = spec_augment(spec)

        return torch.from_numpy(spec).unsqueeze(0), label


def make_loader(dataset, batch_size, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )


