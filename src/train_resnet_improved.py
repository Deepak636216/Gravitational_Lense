"""
Gravitational Lensing Classification - IMPROVED ResNet34 Training
==================================================================

This script implements research-backed improvements to address the poor baseline performance:
1. Focal Loss to handle class imbalance during training
2. Rotation-invariant data augmentation (lensing images are rotationally symmetric)
3. Improved learning rate strategy with warmup
4. Better normalization based on domain statistics

Key Improvements:
- Focal Loss: Prioritizes hard-to-classify examples (addresses Subhalo/Sphere under-prediction)
- Data Augmentation: Rotations, flips, and subtle transformations
- OneCycleLR: Better convergence with warmup phase
- Domain-specific preprocessing

Expected Improvements:
- Baseline: 34.6% accuracy (random chance)
- Target: 70-80% accuracy (research benchmarks show 85-92% is achievable)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
import seaborn as sns
from collections import Counter


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Training configuration with improvements"""
    # Paths
    DATA_DIR = Path('./dataset')
    TRAIN_DIR = DATA_DIR / 'train'
    VAL_DIR = DATA_DIR / 'val'
    CHECKPOINT_DIR = Path('./checkpoints')
    LOG_DIR = Path('./logs')

    # Class mapping
    CLASS_NAMES = ['no', 'sphere', 'vort']  # Directory names
    CLASS_LABELS = ['No Substructure', 'Subhalo/Sphere', 'Vortex']  # Display names
    NUM_CLASSES = 3

    # Image properties
    IMG_SIZE = 150
    IMG_CHANNELS = 1

    # Training hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 50  # Reduced from 100 (with better LR strategy, we converge faster)
    LEARNING_RATE = 1e-4  # REDUCED: Lower LR for more stable training
    WEIGHT_DECAY = 1e-4

    # Regularization
    DROPOUT = 0.4

    # Focal Loss parameters (NEW)
    FOCAL_LOSS_ALPHA = 1.0  # Weight for all classes (can adjust per-class)
    FOCAL_LOSS_GAMMA = 2.0  # Focus on hard examples

    # Learning rate scheduling (IMPROVED)
    LR_SCHEDULER = 'onecycle'  # OneCycleLR with warmup
    MAX_LR = 1e-3  # Maximum LR during cycle
    PCT_START = 0.3  # 30% of training for warmup

    # Data augmentation (NEW)
    USE_AUGMENTATION = True
    ROTATION_DEGREES = 180  # Full rotation (lensing is rotation-invariant)
    HORIZONTAL_FLIP_PROB = 0.5
    VERTICAL_FLIP_PROB = 0.5
    TRANSLATE = (0.1, 0.1)  # Slight translations

    # WeightedRandomSampler (NEW - ensures balanced batches)
    USE_WEIGHTED_SAMPLER = True

    # Early stopping
    EARLY_STOPPING_PATIENCE = 15

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Reproducibility
    RANDOM_SEED = 42

    # Preprocessing (IMPROVED)
    USE_SQRT_STRETCH = True

    def __init__(self):
        # Create directories
        self.CHECKPOINT_DIR.mkdir(exist_ok=True)
        self.LOG_DIR.mkdir(exist_ok=True)

        # Create experiment directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_name = f'resnet34_improved_{timestamp}'
        self.experiment_dir = self.LOG_DIR / self.experiment_name
        self.experiment_dir.mkdir(exist_ok=True)

        # Save config
        self.save_config()

    def save_config(self):
        """Save configuration to JSON"""
        config_dict = {k: str(v) if isinstance(v, Path) else v
                      for k, v in self.__dict__.items()
                      if not k.startswith('_')}
        config_dict['DEVICE'] = str(self.DEVICE)

        with open(self.experiment_dir / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=4)


# ============================================================================
# Focal Loss (NEW - Critical for addressing class imbalance)
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification

    Addresses class imbalance by down-weighting easy examples and focusing
    on hard examples. This is critical for the Subhalo/Sphere class which
    the baseline model almost never predicts.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weighting factor (default=1.0 for all classes)
        gamma: Focusing parameter (default=2.0, higher = more focus on hard examples)
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) logits from model
            targets: (N,) class indices
        """
        # Get log probabilities
        log_probs = F.log_softmax(inputs, dim=1)

        # Get probabilities
        probs = torch.exp(log_probs)

        # Gather the probabilities for the target classes
        targets = targets.view(-1, 1)
        log_probs = log_probs.gather(1, targets).view(-1)
        probs = probs.gather(1, targets).view(-1)

        # Calculate focal loss
        focal_weight = (1 - probs) ** self.gamma
        loss = -self.alpha * focal_weight * log_probs

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# ============================================================================
# Data Augmentation Transforms (NEW)
# ============================================================================

class RandomRotation90:
    """Random 90-degree rotation (faster than continuous rotation)"""
    def __call__(self, x):
        k = np.random.randint(0, 4)  # 0, 90, 180, or 270 degrees
        return torch.rot90(x, k, dims=[-2, -1])


class GaussianNoise:
    """Add Gaussian noise to simulate observational noise"""
    def __init__(self, mean=0., std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        if self.std > 0:
            noise = torch.randn_like(x) * self.std + self.mean
            return x + noise
        return x


# ============================================================================
# Dataset with Augmentation
# ============================================================================

class GravitationalLensingDataset(Dataset):
    """Custom Dataset with data augmentation support"""

    def __init__(self, data_dir, class_names, use_sqrt_stretch=True, augment=False, config=None):
        """
        Args:
            data_dir: Path to data directory (train or val)
            class_names: List of class directory names
            use_sqrt_stretch: Apply square-root stretch preprocessing
            augment: Apply data augmentation (training only)
            config: Configuration object for augmentation parameters
        """
        self.data_dir = Path(data_dir)
        self.class_names = class_names
        self.use_sqrt_stretch = use_sqrt_stretch
        self.augment = augment
        self.config = config

        # Load all file paths and labels
        self.samples = []
        self.labels = []

        print(f"\nLoading data from {data_dir}...")
        for class_idx, class_name in enumerate(class_names):
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                raise ValueError(f"Class directory not found: {class_dir}")

            # Get all .npy files
            class_files = sorted(class_dir.glob('*.npy'))
            print(f"  {class_name}: {len(class_files)} samples")

            for file_path in class_files:
                self.samples.append(file_path)
                self.labels.append(class_idx)

        print(f"Total samples: {len(self.samples)}")
        if augment:
            print("  Data augmentation: ENABLED")
        print()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Load image
        img_path = self.samples[idx]
        image = np.load(img_path)  # Shape: (1, 150, 150) or (150, 150)

        # Remove channel dimension if present
        if len(image.shape) == 3 and image.shape[0] == 1:
            image = image[0]  # Shape: (150, 150)

        # Preprocessing: Square-root stretch (astronomy standard)
        if self.use_sqrt_stretch:
            # Ensure non-negative values
            image = np.maximum(image, 0)
            # Apply square-root
            image = np.sqrt(image)
            # Normalize to [0, 1]
            max_val = image.max()
            if max_val > 0:
                image = image / max_val

        # Convert to tensor and add channel dimension
        image = torch.from_numpy(image).float().unsqueeze(0)  # Shape: (1, 150, 150)

        # Apply data augmentation (training only)
        if self.augment and self.config is not None:
            # Random 90-degree rotations (most efficient for symmetric data)
            if np.random.rand() < 0.75:  # 75% chance
                k = np.random.randint(0, 4)
                image = torch.rot90(image, k, dims=[-2, -1])

            # Random horizontal flip
            if np.random.rand() < self.config.HORIZONTAL_FLIP_PROB:
                image = torch.flip(image, dims=[-1])

            # Random vertical flip
            if np.random.rand() < self.config.VERTICAL_FLIP_PROB:
                image = torch.flip(image, dims=[-2])

            # Gaussian noise (simulate observational noise)
            if np.random.rand() < 0.3:  # 30% chance
                noise = torch.randn_like(image) * 0.01
                image = image + noise
                image = torch.clamp(image, 0, 1)  # Keep in valid range

        # Get label
        label = self.labels[idx]

        return image, label


# ============================================================================
# ResNet34 Architecture (Same as baseline)
# ============================================================================

class BasicBlock(nn.Module):
    """Basic ResNet building block with skip connections"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        # First conv layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Second conv layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Downsample for skip connection (if dimensions change)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet34(nn.Module):
    """ResNet34 for gravitational lensing classification"""

    def __init__(self, num_classes=3, dropout=0.4):
        super(ResNet34, self).__init__()

        self.in_channels = 64

        # Initial convolution (grayscale input)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers (ResNet34 configuration: [3, 4, 6, 3])
        self.layer1 = self._make_layer(BasicBlock, 64, 3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 4, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 6, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 3, stride=2)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(128, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """Create a ResNet layer with multiple blocks"""
        downsample = None

        # If dimensions change, create downsample layer for skip connection
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        # First block (may change dimensions)
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNet blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)

        return x


# ============================================================================
# Training and Evaluation Functions
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, scheduler, device):
    """Train for one epoch with per-batch LR updates"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update learning rate (OneCycleLR steps per batch)
        if scheduler is not None:
            scheduler.step()

        # Statistics
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Update progress bar
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{current_lr:.2e}'})

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc


def evaluate(model, val_loader, criterion, device):
    """Evaluate on validation set"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Get probabilities
            probs = torch.softmax(outputs, dim=1)

            # Statistics
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc, np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_metrics(y_true, y_pred, y_probs, class_labels):
    """Compute comprehensive evaluation metrics"""
    metrics = {}

    # Overall accuracy
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))

    # Per-class metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # Macro averages
    metrics['macro_f1'] = float(np.mean(f1))
    metrics['macro_precision'] = float(np.mean(precision))
    metrics['macro_recall'] = float(np.mean(recall))

    # Per-class details
    metrics['per_class'] = {}
    for i, class_name in enumerate(class_labels):
        metrics['per_class'][class_name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i])
        }

    # ROC-AUC
    try:
        y_true_onehot = np.eye(len(class_labels))[y_true]
        metrics['roc_auc'] = float(roc_auc_score(y_true_onehot, y_probs, average='macro'))
    except:
        metrics['roc_auc'] = 0.0

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()

    return metrics


def plot_training_history(history, save_path):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o', markersize=3)
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s', markersize=3)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot accuracy
    axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o', markersize=3)
    axes[1].plot(history['val_acc'], label='Val Accuracy', marker='s', markersize=3)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    """Main training function"""

    # Set random seeds for reproducibility
    torch.manual_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.RANDOM_SEED)

    # Initialize config
    config = Config()

    print("="*70)
    print("GRAVITATIONAL LENSING CLASSIFICATION - ResNet34 IMPROVED")
    print("="*70)
    print(f"Experiment: {config.experiment_name}")
    print(f"Device: {config.DEVICE}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Learning Rate: {config.LEARNING_RATE} -> {config.MAX_LR}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"\nIMPROVEMENTS:")
    print(f"  [NEW] Focal Loss: alpha={config.FOCAL_LOSS_ALPHA}, gamma={config.FOCAL_LOSS_GAMMA}")
    print(f"  [NEW] Data Augmentation: {config.USE_AUGMENTATION}")
    print(f"  [NEW] Weighted Sampler: {config.USE_WEIGHTED_SAMPLER}")
    print(f"  [NEW] OneCycleLR Scheduler with {config.PCT_START*100:.0f}% warmup")
    print("="*70)

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = GravitationalLensingDataset(
        config.TRAIN_DIR,
        config.CLASS_NAMES,
        use_sqrt_stretch=config.USE_SQRT_STRETCH,
        augment=config.USE_AUGMENTATION,
        config=config
    )
    val_dataset = GravitationalLensingDataset(
        config.VAL_DIR,
        config.CLASS_NAMES,
        use_sqrt_stretch=config.USE_SQRT_STRETCH,
        augment=False  # No augmentation for validation
    )

    # Create sampler for balanced batches (NEW)
    if config.USE_WEIGHTED_SAMPLER:
        print("\nCreating WeightedRandomSampler for balanced batches...")
        class_counts = Counter(train_dataset.labels)
        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        sample_weights = [class_weights[label] for label in train_dataset.labels]
        sample_weights = torch.DoubleTensor(sample_weights)

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False  # Sampler handles shuffling
        print("  Class weights:")
        for cls_idx, cls_name in enumerate(config.CLASS_LABELS):
            print(f"    {cls_name}: {class_weights[cls_idx]:.6f}")
    else:
        sampler = None
        shuffle = True

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"\nDataset Statistics:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Batches per epoch: {len(train_loader)}")

    # Create model
    print("\nInitializing ResNet34 model...")
    model = ResNet34(num_classes=config.NUM_CLASSES, dropout=config.DROPOUT)
    model = model.to(config.DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Loss function - FOCAL LOSS (NEW)
    criterion = FocalLoss(alpha=config.FOCAL_LOSS_ALPHA, gamma=config.FOCAL_LOSS_GAMMA)
    print(f"\nLoss Function: Focal Loss (alpha={config.FOCAL_LOSS_ALPHA}, gamma={config.FOCAL_LOSS_GAMMA})")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler - OneCycleLR (NEW)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * config.NUM_EPOCHS

    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.MAX_LR,
        total_steps=total_steps,
        pct_start=config.PCT_START,
        anneal_strategy='cos',
        div_factor=10.0,  # initial_lr = max_lr / div_factor
        final_div_factor=100.0  # min_lr = max_lr / final_div_factor
    )
    print(f"Scheduler: OneCycleLR (max_lr={config.MAX_LR:.2e}, warmup={config.PCT_START*100:.0f}%)")

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': []
    }

    # Early stopping
    best_val_acc = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = 0

    # Training loop
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        print("-" * 50)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.2e}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, config.DEVICE)

        # Validate
        val_loss, val_acc, y_true, y_pred, y_probs = evaluate(model, val_loader, criterion, config.DEVICE)

        # Update history
        history['train_loss'].append(float(train_loss))
        history['train_acc'].append(float(train_acc))
        history['val_loss'].append(float(val_loss))
        history['val_acc'].append(float(val_acc))
        history['learning_rate'].append(float(current_lr))

        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0

            # Save best model
            best_model_path = config.CHECKPOINT_DIR / f'{config.experiment_name}_best.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, best_model_path)

            print(f"  >> New best model! Val Acc: {val_acc:.4f}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epoch(s)")

        # Early stopping
        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
            break

    # ========================================================================
    # Final Evaluation
    # ========================================================================

    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)

    # Load best model
    print("\nLoading best model...")
    best_checkpoint = torch.load(config.CHECKPOINT_DIR / f'{config.experiment_name}_best.pth')
    model.load_state_dict(best_checkpoint['model_state_dict'])

    # Evaluate on validation set
    print("Evaluating on validation set...")
    val_loss, val_acc, y_true, y_pred, y_probs = evaluate(model, val_loader, criterion, config.DEVICE)

    # Compute comprehensive metrics
    metrics = compute_metrics(y_true, y_pred, y_probs, config.CLASS_LABELS)
    metrics['best_epoch'] = best_epoch
    metrics['total_epochs'] = epoch + 1

    # Print results
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  Macro F1:     {metrics['macro_f1']:.4f}")
    print(f"  ROC-AUC:      {metrics['roc_auc']:.4f}")

    print(f"\nPer-Class Metrics:")
    for class_name, class_metrics in metrics['per_class'].items():
        print(f"\n  {class_name}:")
        print(f"    Precision: {class_metrics['precision']:.4f}")
        print(f"    Recall:    {class_metrics['recall']:.4f}")
        print(f"    F1-Score:  {class_metrics['f1']:.4f}")

    # Classification report
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(y_true, y_pred, target_names=config.CLASS_LABELS, digits=4))

    # Confusion matrix
    print("\nConfusion Matrix:")
    print(np.array(metrics['confusion_matrix']))

    # Save results
    print("\nSaving results...")

    # Save plots
    plot_training_history(history, config.experiment_dir / 'training_history.png')
    print("  Training history plot saved")

    plot_confusion_matrix(np.array(metrics['confusion_matrix']), config.CLASS_LABELS,
                         config.experiment_dir / 'confusion_matrix.png')
    print("  Confusion matrix plot saved")

    # Save metrics
    with open(config.experiment_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print("  Metrics saved to JSON")

    # Save training history
    with open(config.experiment_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=4)
    print("  Training history saved")

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nExperiment directory: {config.experiment_dir}")
    print(f"Best model: {config.CHECKPOINT_DIR / f'{config.experiment_name}_best.pth'}")
    print(f"Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")

    # Compare with baseline
    baseline_acc = 0.34653333333333336
    improvement = ((best_val_acc - baseline_acc) / baseline_acc) * 100
    print(f"\nImprovement over baseline: {improvement:+.1f}%")
    print(f"  Baseline: {baseline_acc:.4f}")
    print(f"  Improved: {best_val_acc:.4f}")
    print("\n")


if __name__ == '__main__':
    main()
