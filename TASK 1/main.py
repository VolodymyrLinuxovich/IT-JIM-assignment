#!/usr/bin/env python3
import os
import re
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.metrics import f1_score

# Set device (prefer GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =============================================================================
# 1. Custom Dataset Class for Image Loading
# =============================================================================
class ArtifactDataset(Dataset):
    """
    Custom dataset for loading images.
    Files should follow the pattern:
       image_<frame_index>_<class label>.png
    For test data (label missing), set is_test=True.
    """
    def __init__(self, root_dir, transform=None, is_test=False):
        self.root = root_dir
        self.tf = transform
        self.test = is_test
        
        # Get list of .png files and sort them for sequence order.
        self.fnames = sorted([f for f in os.listdir(root_dir) if f.endswith('.png')])
        
    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fn = self.fnames[idx]
        img_path = os.path.join(self.root, fn)
        img = Image.open(img_path).convert("RGB")

        lbl = None
        # For non-test data, extract label from filename using regex.
        if not self.test:
            match = re.search(r'_(\d+)\.png$', fn)
            if match:
                lbl = int(match.group(1))
            else:
                raise ValueError(f"Filename {fn} does not match pattern!")
        
        if self.tf:
            img = self.tf(img)
        
        # For test data, return image and filename.
        return (img, fn) if self.test else (img, lbl)


# =============================================================================
# 2. Transformation Functions for Train, Validation, and Test
# =============================================================================
def get_train_transforms(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        # ImageNet normalization
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def get_valid_transforms(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def get_test_transforms(img_size=224):
    return get_valid_transforms(img_size)


# =============================================================================
# 3. Model Initialization Functions (Transfer Learning)
# =============================================================================
def get_resnet18_model(n_cls=2):
    m = models.resnet18(pretrained=True)
    in_f = m.fc.in_features
    m.fc = nn.Linear(in_f, n_cls)
    return m

def get_efficientnet_b0_model(n_cls=2):
    m = models.efficientnet_b0(pretrained=True)
    in_f = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_f, n_cls)
    return m


# =============================================================================
# 4. Functions for Training, Validation & Micro F1 Metric Calculation
# =============================================================================
def train_one_epoch(model, loader, crit, opt, dev):
    model.train()
    tot_loss = 0.0
    preds_all = []
    labels_all = []
    
    for imgs, lbls in tqdm(loader, desc="Training", leave=False):
        imgs = imgs.to(dev)
        lbls = lbls.to(dev)
        
        opt.zero_grad()
        outputs = model(imgs)
        loss = crit(outputs, lbls)
        loss.backward()
        opt.step()
        
        tot_loss += loss.item() * imgs.size(0)
        p = torch.argmax(outputs, dim=1)
        preds_all.extend(p.cpu().numpy())
        labels_all.extend(lbls.cpu().numpy())
    
    epoch_loss = tot_loss / len(loader.dataset)
    micro_f1 = f1_score(labels_all, preds_all, average="micro")
    return epoch_loss, micro_f1

def validate_model(model, loader, crit, dev):
    model.eval()
    tot_loss = 0.0
    preds_all = []
    labels_all = []
    
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="Validation", leave=False):
            imgs = imgs.to(dev)
            lbls = lbls.to(dev)
            
            outputs = model(imgs)
            loss = crit(outputs, lbls)
            tot_loss += loss.item() * imgs.size(0)
            p = torch.argmax(outputs, dim=1)
            preds_all.extend(p.cpu().numpy())
            labels_all.extend(lbls.cpu().numpy())
    
    epoch_loss = tot_loss / len(loader.dataset)
    micro_f1 = f1_score(labels_all, preds_all, average="micro")
    return epoch_loss, micro_f1


# =============================================================================
# 5. Inference Function to Get Predictions
# =============================================================================
def inference(model, loader, dev):
    model.eval()
    preds = []
    fnames = []
    
    with torch.no_grad():
        for imgs, fns in tqdm(loader, desc="Inference", leave=False):
            imgs = imgs.to(dev)
            outs = model(imgs)
            probs = nn.functional.softmax(outs, dim=1)
            p = torch.argmax(probs, dim=1)
            preds.extend(p.cpu().numpy())
            fnames.extend(fns)
    
    return fnames, preds


# =============================================================================
# 6. Ensemble Function (Averaging Probabilities from Multiple Models)
# =============================================================================
def inference_ensemble(models, loader, dev):
    """
    Obtain ensemble predictions by averaging probabilities from each model.
    """
    # Set all models to evaluation mode.
    for m in models:
        m.eval()
        
    all_fns = []
    ens_preds = []
    
    with torch.no_grad():
        for imgs, fns in tqdm(loader, desc="Ensemble Inference", leave=False):
            imgs = imgs.to(dev)
            bp = None  # batch probabilities
            for m in models:
                out = m(imgs)
                prob = nn.functional.softmax(out, dim=1)
                bp = prob if bp is None else bp + prob
            bp /= len(models)
            p = torch.argmax(bp, dim=1)
            ens_preds.extend(p.cpu().numpy())
            all_fns.extend(fns)
            
    return all_fns, ens_preds


# =============================================================================
# Global Execution: Training, Validation, and Inference (No main() function)
# =============================================================================

# Hyperparameters
n_ep = 10
bs = 32
lr = 1e-4
img_sz = 224
n_cls = 2

# Create transforms
tr_tf = get_train_transforms(img_sz)
val_tf = get_valid_transforms(img_sz)
test_tf = get_test_transforms(img_sz)

# Load training dataset from 'data/train'
ds = ArtifactDataset(root_dir="data/train", transform=tr_tf, is_test=False)

# Split dataset into training (80%) and validation (20%)
train_sz = int(0.8 * len(ds))
val_sz = len(ds) - train_sz
train_ds, val_ds = random_split(ds, [train_sz, val_sz])

train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4)

# Initialize two models using transfer learning approaches.
print("Initializing models...")
m_res = get_resnet18_model(n_cls=n_cls).to(device)
m_eff = get_efficientnet_b0_model(n_cls=n_cls).to(device)

# Define loss function and optimizers for each model.
crit = nn.CrossEntropyLoss()
opt_res = optim.Adam(m_res.parameters(), lr=lr)
opt_eff = optim.Adam(m_eff.parameters(), lr=lr)

# Save best models based on validation micro F1 score.
best_f1_res = 0.0
best_res = copy.deepcopy(m_res.state_dict())

best_f1_eff = 0.0
best_eff = copy.deepcopy(m_eff.state_dict())

# Train ResNet18 model.
print("Training ResNet18 model...")
for ep in range(n_ep):
    print(f"Epoch {ep+1}/{n_ep}")
    tr_loss, tr_f1 = train_one_epoch(m_res, train_loader, crit, opt_res, device)
    val_loss, val_f1 = validate_model(m_res, val_loader, crit, device)
    print(f"ResNet - Train Loss: {tr_loss:.4f} Train F1: {tr_f1:.4f} | Val Loss: {val_loss:.4f} Val F1: {val_f1:.4f}")
    if val_f1 > best_f1_res:
        best_f1_res = val_f1
        best_res = copy.deepcopy(m_res.state_dict())

# Load best weights for ResNet.
m_res.load_state_dict(best_res)

# Train EfficientNet-B0 model.
print("Training EfficientNet-B0 model...")
for ep in range(n_ep):
    print(f"Epoch {ep+1}/{n_ep}")
    tr_loss, tr_f1 = train_one_epoch(m_eff, train_loader, crit, opt_eff, device)
    val_loss, val_f1 = validate_model(m_eff, val_loader, crit, device)
    print(f"EfficientNet - Train Loss: {tr_loss:.4f} Train F1: {tr_f1:.4f} | Val Loss: {val_loss:.4f} Val F1: {val_f1:.4f}")
    if val_f1 > best_f1_eff:
        best_f1_eff = val_f1
        best_eff = copy.deepcopy(m_eff.state_dict())

# Load best weights for EfficientNet.
m_eff.load_state_dict(best_eff)

# =============================================================================
# Inference on Test Set and Ensemble Predictions
# =============================================================================
# Load test dataset from 'data/test'
test_ds = ArtifactDataset(root_dir="data/test", transform=test_tf, is_test=True)
test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=4)

# Get predictions from each model.
fn_res, preds_res = inference(m_res, test_loader, device)
fn_eff, preds_eff = inference(m_eff, test_loader, device)

# Compute ensemble predictions by averaging probabilities.
fn_ens, preds_ens = inference_ensemble([m_res, m_eff], test_loader, device)

# Create a DataFrame with the results and save as CSV.
sub = pd.DataFrame({
    "filename": fn_ens,
    "label": preds_ens
})

sub.sort_values("filename", inplace=True)
sub.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")