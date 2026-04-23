"""Training loops, evaluation, bootstrap CIs, cross-source and SNR runners."""

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
    weights = torch.tensor(
        [1.0 / counts.get(c, 1) for c in cfg.CLASSES],
        dtype=torch.float32,
    )
    weights = weights / weights.sum() * cfg.NUM_CLASSES
    return weights.to(device)


# One training pass. Returns (loss, accuracy).
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
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
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(train_loader), 100.0 * correct / total


# One evaluation pass. Returns loss, accuracy, preds, probs, labels.
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            running_loss += loss.item()
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.append(predicted.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return {
        "loss": running_loss / len(loader),
        "acc": 100.0 * correct / total,
        "preds": np.concatenate(all_preds),
        "probs": np.concatenate(all_probs),
        "labels": np.concatenate(all_labels),
    }


# Macro-F1 and per-class AUC-ROC (one-vs-rest)
def compute_metrics(labels, preds, probs):
    macro_f1 = f1_score(labels, preds, average="macro")
    auc_per_class = {}
    for i, cls in enumerate(cfg.CLASSES):
        y_bin = (labels == i).astype(int)
        if 0 < y_bin.sum() < len(y_bin):
            auc_per_class[cls] = roc_auc_score(y_bin, probs[:, i])
        else:
            auc_per_class[cls] = float("nan")
    return macro_f1, auc_per_class


# Drive the epoch loop with early stopping on validation macro-F1.
#
def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    patience,
    ckpt_path=None,
    class_weights=None,
):
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_f1 = -1
    best_state = None
    epochs_since_best = 0

    train_losses, train_accs = [], []
    val_losses, val_accs, val_f1s = [], [], []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_out = evaluate(model, val_loader, criterion, device)

        val_f1, _ = compute_metrics(
            val_out["labels"], val_out["preds"], val_out["probs"]
        )

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_out["loss"])
        val_accs.append(val_out["acc"])
        val_f1s.append(val_f1)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(
            f"Val Loss: {val_out['loss']:.4f}, Val Acc: {val_out['acc']:.2f}%, Val F1: {val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_best = 0
            if ckpt_path:
                torch.save(best_state, ckpt_path)
        else:
            epochs_since_best += 1
            if epochs_since_best >= patience:
                print(f"Early stop at epoch {epoch + 1} (best val_f1 {best_f1:.4f})")
                break

    model.load_state_dict(best_state)
    history = {
        "train_loss": train_losses,
        "train_acc": train_accs,
        "val_loss": val_losses,
        "val_acc": val_accs,
        "val_f1": val_f1s,
    }
    return model, history


# Three-stage PANN fine-tuning
def train_pann_progressive(
    model, train_loader, val_loader, device, save_ckpts=True, class_weights=None
):
    all_history = {}

    for stage_name, epochs, lr_head, lr_bb, n_unfrozen in cfg.PANN_STAGES:
        print(f"PANN {stage_name}")

        models.set_unfreeze_stage(model, n_unfrozen)
        total, trainable = models.count_params(model)
        print(f"Trainable: {trainable:,} / {total:,}")

        groups = models.param_groups(model, lr_head, lr_bb)
        optimizer = torch.optim.Adam(groups, weight_decay=cfg.PANN_WEIGHT_DECAY)

        ckpt = str(cfg.CKPT_ROOT / f"pann_{stage_name}.pt") if save_ckpts else None
        model, hist = train_model(
            model,
            train_loader,
            val_loader,
            optimizer,
            device,
            num_epochs=epochs,
            patience=cfg.PANN_EARLY_STOPPING_PATIENCE,
            ckpt_path=ckpt,
            class_weights=class_weights,
        )
        all_history[stage_name] = hist

    return model, all_history


# 95% bootstrap CI over a sample-level metric
def bootstrap_ci(
    labels,
    preds,
    probs=None,
    metric="macro_f1",
    n_resamples=cfg.BOOTSTRAP_RESAMPLES,
    ci=cfg.BOOTSTRAP_CI,
):
    rng = np.random.RandomState(cfg.SEED)
    scores = []
    n = len(labels)

    for _ in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        if metric == "macro_f1":
            scores.append(f1_score(labels[idx], preds[idx], average="macro"))
        elif metric == "macro_auc":
            aucs = []
            for i in range(cfg.NUM_CLASSES):
                y_bin = (labels[idx] == i).astype(int)
                if 0 < y_bin.sum() < len(y_bin):
                    aucs.append(roc_auc_score(y_bin, probs[idx, i]))
            scores.append(np.mean(aucs) if aucs else np.nan)

    lo = np.percentile(scores, (1 - ci) / 2 * 100)
    hi = np.percentile(scores, (1 + ci) / 2 * 100)
    return np.mean(scores), lo, hi


# Full evaluation: metrics + confusion matrix + bootstrap CI
def full_evaluate(model, loader, device):
    criterion = nn.CrossEntropyLoss()
    out = evaluate(model, loader, criterion, device)
    macro_f1, auc_per_class = compute_metrics(out["labels"], out["preds"], out["probs"])
    cm = confusion_matrix(out["labels"], out["preds"])
    f1_mean, f1_lo, f1_hi = bootstrap_ci(out["labels"], out["preds"])

    return {
        "loss": out["loss"],
        "acc": out["acc"],
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


# SVM evaluation (sklearn interface)
def evaluate_svm(pipeline, X_test, y_test):
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)
    macro_f1, auc_per_class = compute_metrics(y_test, preds, probs)
    cm = confusion_matrix(y_test, preds)
    f1_mean, f1_lo, f1_hi = bootstrap_ci(y_test, preds)

    return {
        "macro_f1": macro_f1,
        "auc_per_class": auc_per_class,
        "f1_ci": (f1_mean, f1_lo, f1_hi),
        "confusion_matrix": cm,
        "preds": preds,
        "probs": probs,
        "labels": y_test,
    }


# Cross-source gap: train on Svanstrom, test on held-out Al-Emadi
def cross_source_eval(model, holdout_manifest, device, is_torch=True, batch_size=32):
    if is_torch:
        ds = data.AudioDataset(holdout_manifest, augment=False)
        loader = data.make_loader(ds, batch_size=batch_size, shuffle=False)
        return full_evaluate(model, loader, device)

    # SVM path
    X, y = extract_svm_dataset(holdout_manifest)
    return evaluate_svm(model, X, y)


# SNR degradation sweep
def snr_degradation_curve(
    model,
    test_manifest,
    noise_manifest,
    device,
    is_torch=True,
    batch_size=32,
    classes=("drone", "helicopter"),
):
    """SNR sweep. Background class is excluded because it is already
    environmental audio — adding noise to it has no SNR interpretation."""
    test_manifest = test_manifest[test_manifest["label"].isin(classes)]
    results = []

    for snr in cfg.SNR_LEVELS_DB:
        if is_torch:
            ds = data.AudioDataset(
                test_manifest, augment=False, snr_db=snr, noise_manifest=noise_manifest
            )
            loader = data.make_loader(ds, batch_size=batch_size, shuffle=False)
            out = full_evaluate(model, loader, device)
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
            out = evaluate_svm(model, np.array(X), np.array(y))

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


