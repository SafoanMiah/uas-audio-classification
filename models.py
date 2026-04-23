import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

import config as cfg


def build_svm():
    base = LinearSVC(C=cfg.SVM_C, class_weight="balanced", max_iter=5000)
    clf = CalibratedClassifierCV(base, cv=3)  # needed for predict_proba
    return Pipeline([("scaler", StandardScaler()), ("svm", clf)])


# 5-block CNN trained from scratch on log-mel spectrograms.
# Dropout is applied only after the FC layer, not per-block, to avoid
class SimpleCNN(nn.Module):
    def __init__(self, n_classes=cfg.NUM_CLASSES):
        super(SimpleCNN, self).__init__()

        in_ch = 1
        blocks = []
        for out_ch in cfg.CNN_CHANNELS:
            blocks.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            blocks.append(nn.BatchNorm2d(out_ch))
            blocks.append(nn.ReLU())
            blocks.append(nn.MaxPool2d(2))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(cfg.CNN_CHANNELS[-1], cfg.CNN_FC_DIM)
        self.dropout = nn.Dropout(cfg.CNN_DROPOUT)
        self.fc2 = nn.Linear(cfg.CNN_FC_DIM, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# PANNs CNN10 (Kong et al. 2020). Block names match the pretrained
# checkpoint so state_dict loading works.
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, kernel_size=2)


class PANNCnn10(nn.Module):
    def __init__(self, n_classes=cfg.NUM_CLASSES):
        super(PANNCnn10, self).__init__()
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)

        self.fc1 = nn.Linear(512, cfg.PANN_EMBEDDING_DIM)
        self.dropout = nn.Dropout(cfg.PANN_DROPOUT)
        self.fc_out = nn.Linear(cfg.PANN_EMBEDDING_DIM, n_classes)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        # Global pool: mean over time, max over mel
        x = torch.mean(x, dim=3)
        x, _ = torch.max(x, dim=2)

        x = F.relu_(self.fc1(x))
        x = self.dropout(x)
        return self.fc_out(x)

    def blocks(self):
        return [self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4]


def download_pann_checkpoint(dest_path):
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if not dest_path.exists():
        print(f"Downloading PANN CNN10 to {dest_path}")
        urllib.request.urlretrieve(cfg.PANN_CKPT_URL, dest_path)
    return dest_path


def load_pann_cnn10(checkpoint_path, n_classes=cfg.NUM_CLASSES):
    model = PANNCnn10(n_classes=n_classes)

    state = torch.load(checkpoint_path, map_location="cpu")
    if "model" in state:
        state = state["model"]

    # Load backbone weights only; skip the original AudioSet head
    own = model.state_dict()
    loaded = 0
    for k, v in state.items():
        if k in own and own[k].shape == v.shape and not k.startswith("fc_out"):
            own[k] = v
            loaded += 1
    model.load_state_dict(own)
    if loaded == 0:
        raise RuntimeError(f"No pretrained weights loaded from {checkpoint_path}.")
    print(f"Loaded {loaded} pretrained tensors from {checkpoint_path}")
    return model


# Progressive unfreezing: 0 = head only, 1 = + last block, 2 = + last two blocks
def set_unfreeze_stage(model, n_blocks_unfrozen):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc1.parameters():
        p.requires_grad = True
    for p in model.fc_out.parameters():
        p.requires_grad = True
    if n_blocks_unfrozen > 0:
        for block in model.blocks()[-n_blocks_unfrozen:]:
            for p in block.parameters():
                p.requires_grad = True


# Differential learning rates for head vs backbone
def param_groups(model, lr_head, lr_backbone):
    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith(("fc1", "fc_out")):
            head_params.append(p)
        else:
            backbone_params.append(p)

    groups = [{"params": head_params, "lr": lr_head}]
    if backbone_params and lr_backbone > 0:
        groups.append({"params": backbone_params, "lr": lr_backbone})
    return groups


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
