import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

import config as cfg
import data
import models


def compute_class_weights(manifest, device):
    counts = manifest["label"].value_counts().to_dict()
    w = torch.tensor([1.0 / counts.get(c, 1) for c in cfg.CLASSES], dtype=torch.float32)
    w = w / w.sum() * cfg.NUM_CLASSES
    return w.to(device)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for inputs, labels in tqdm(loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        if cfg.USE_MIXUP:
            labels_oh = F.one_hot(labels, cfg.NUM_CLASSES).float()
            inputs_m, labels_m = data.mixup_batch(inputs, labels_oh)
            outputs = model(inputs_m)
            log_probs = F.log_softmax(outputs, dim=1)
            if criterion.weight is not None:
                log_probs = log_probs * criterion.weight
            loss = -(labels_m * log_probs).sum(dim=1).mean()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            running_loss += loss.item()
            all_preds.append(predicted.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return {
        "loss": running_loss / len(loader),
        "preds": np.concatenate(all_preds),
        "probs": np.concatenate(all_probs),
        "labels": np.concatenate(all_labels),
    }


# Restrict macro-F1 and AUC to the classes actually present so binary
# holdouts (e.g. Al-Emadi: drone + background only, SNR sweep:
# drone + helicopter only) aren't unfairly averaged against a class
# with zero support.
def compute_metrics(labels, preds, probs, classes=None):
    if classes is None:
        classes = cfg.CLASSES
    class_idx = [cfg.CLASS_TO_IDX[c] for c in classes]

    macro_f1 = f1_score(labels, preds, labels=class_idx, average="macro")

    auc_per_class = {}
    for c in classes:
        i = cfg.CLASS_TO_IDX[c]
        y_bin = (labels == i).astype(int)
        if 0 < y_bin.sum() < len(y_bin):
            auc_per_class[c] = roc_auc_score(y_bin, probs[:, i])
        else:
            auc_per_class[c] = float("nan")
    return macro_f1, auc_per_class


def bootstrap_ci(labels, preds, classes=None):
    if classes is None:
        classes = cfg.CLASSES
    class_idx = [cfg.CLASS_TO_IDX[c] for c in classes]

    rng = np.random.RandomState(cfg.SEED)
    scores = []
    n = len(labels)
    for _ in range(cfg.BOOTSTRAP_RESAMPLES):
        idx = rng.randint(0, n, size=n)
        scores.append(
            f1_score(labels[idx], preds[idx], labels=class_idx, average="macro")
        )

    ci = cfg.BOOTSTRAP_CI
    lo = np.percentile(scores, (1 - ci) / 2 * 100)
    hi = np.percentile(scores, (1 + ci) / 2 * 100)
    return np.mean(scores), lo, hi


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    patience,
    class_weights=None,
):
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_f1 = -1
    best_state = None
    epochs_since_best = 0
    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_out = evaluate(model, val_loader, criterion, device)
        val_f1, _ = compute_metrics(
            val_out["labels"], val_out["preds"], val_out["probs"]
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_out["loss"])
        history["val_f1"].append(val_f1)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_out['loss']:.4f}, Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= patience:
                print(f"Early stop at epoch {epoch + 1} (best val_f1 {best_f1:.4f})")
                break

    model.load_state_dict(best_state)
    return model, history


def train_pann_progressive(model, train_loader, val_loader, device, class_weights=None):
    all_history = {}

    for stage_name, epochs, lr_head, lr_bb, n_unfrozen in cfg.PANN_STAGES:
        print(f"\nPANN stage: {stage_name}")
        models.set_unfreeze_stage(model, n_unfrozen)
        total, trainable = models.count_params(model)
        print(f"Trainable: {trainable:,} / {total:,}")

        groups = models.param_groups(model, lr_head, lr_bb)
        optimizer = torch.optim.Adam(groups, weight_decay=cfg.PANN_WEIGHT_DECAY)

        model, hist = train_model(
            model,
            train_loader,
            val_loader,
            optimizer,
            device,
            num_epochs=epochs,
            patience=cfg.PANN_PATIENCE,
            class_weights=class_weights,
        )
        all_history[stage_name] = hist

    return model, all_history


def full_evaluate(model, loader, device, classes=None):
    criterion = nn.CrossEntropyLoss()
    out = evaluate(model, loader, criterion, device)
    macro_f1, auc_per_class = compute_metrics(
        out["labels"], out["preds"], out["probs"], classes=classes
    )
    cm = confusion_matrix(
        out["labels"], out["preds"], labels=list(range(cfg.NUM_CLASSES))
    )
    f1_mean, f1_lo, f1_hi = bootstrap_ci(out["labels"], out["preds"], classes=classes)

    return {
        "macro_f1": macro_f1,
        "auc_per_class": auc_per_class,
        "f1_ci": (f1_mean, f1_lo, f1_hi),
        "confusion_matrix": cm,
        "preds": out["preds"],
        "probs": out["probs"],
        "labels": out["labels"],
    }


def extract_svm_dataset(manifest):
    X, y = [], []
    for _, row in manifest.iterrows():
        wav = data.load_audio(row["path"])
        for seg in data.segment_audio(wav):
            X.append(data.extract_svm_features(seg))
            y.append(cfg.CLASS_TO_IDX[row["label"]])
    return np.array(X), np.array(y)


def train_svm(train_manifest):
    X, y = extract_svm_dataset(train_manifest)
    pipeline = models.build_svm()
    pipeline.fit(X, y)
    return pipeline


def evaluate_svm(pipeline, X, y, classes=None):
    preds = pipeline.predict(X)
    probs = pipeline.predict_proba(X)
    macro_f1, auc_per_class = compute_metrics(y, preds, probs, classes=classes)
    cm = confusion_matrix(y, preds, labels=list(range(cfg.NUM_CLASSES)))
    f1_mean, f1_lo, f1_hi = bootstrap_ci(y, preds, classes=classes)

    return {
        "macro_f1": macro_f1,
        "auc_per_class": auc_per_class,
        "f1_ci": (f1_mean, f1_lo, f1_hi),
        "confusion_matrix": cm,
        "preds": preds,
        "probs": probs,
        "labels": y,
    }


# Al-Emadi has drone + background only, so metrics are restricted to
# those two classes.
def cross_source_eval(model, holdout_manifest, device, is_torch=True, batch_size=32):
    classes = sorted(
        holdout_manifest["label"].unique(), key=lambda c: cfg.CLASS_TO_IDX[c]
    )

    if is_torch:
        ds = data.AudioDataset(holdout_manifest, augment=False)
        loader = data.make_loader(ds, batch_size=batch_size, shuffle=False)
        return full_evaluate(model, loader, device, classes=classes)

    X, y = extract_svm_dataset(holdout_manifest)
    return evaluate_svm(model, X, y, classes=classes)


# Background is excluded because it IS environmental audio — adding
# more noise to it has no SNR interpretation. Metrics are computed
# over drone + helicopter only.
def snr_degradation_curve(
    model,
    test_manifest,
    noise_manifest,
    device,
    is_torch=True,
    batch_size=32,
    classes=("drone", "helicopter"),
):
    test_manifest = test_manifest[test_manifest["label"].isin(classes)]
    results = []

    for snr in cfg.SNR_LEVELS_DB:
        if is_torch:
            ds = data.AudioDataset(
                test_manifest, augment=False, snr_db=snr, noise_manifest=noise_manifest
            )
            loader = data.make_loader(ds, batch_size=batch_size, shuffle=False)
            out = full_evaluate(model, loader, device, classes=list(classes))
        else:
            X, y = [], []
            for _, row in test_manifest.iterrows():
                wav = data.load_audio(row["path"])
                for seg in data.segment_audio(wav):
                    if snr != "clean":
                        noise = data.load_audio(
                            np.random.choice(noise_manifest["path"].values)
                        )
                        seg = data.mix_at_snr(seg, noise, snr)
                    X.append(data.extract_svm_features(seg))
                    y.append(cfg.CLASS_TO_IDX[row["label"]])
            out = evaluate_svm(model, np.array(X), np.array(y), classes=list(classes))

        results.append(
            {
                "snr_db": snr,
                "macro_f1": out["macro_f1"],
                "f1_lo": out["f1_ci"][1],
                "f1_hi": out["f1_ci"][2],
            }
        )
        print(
            f"SNR {snr}: macro_f1 = {out['macro_f1']:.4f} "
            f"[{out['f1_ci'][1]:.4f}, {out['f1_ci'][2]:.4f}]"
        )

    return pd.DataFrame(results)
