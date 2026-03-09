"""
Gravitational Lensing Classification - ResNet34 Baseline Training
==================================================================

This script trains a ResNet34 model from scratch on the gravitational lensing dataset.
NO DATA AUGMENTATION is used in this baseline to establish a reference performance.

Based on 2025-2026 research best practices:
- Raw pixels as input (no manual feature engineering)
- Square-root stretch preprocessing (astronomy standard)
- ResNet34 architecture for deep feature extraction
- Strong regularization to prevent overfitting

Dataset:
- Training: 30,000 images (10k per class)
- Validation: 7,500 images (2.5k per class)
- Classes: No Substructure, Subhalo/Sphere, Vortex
- Image size: 150x150 grayscale
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
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


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Training configuration"""
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
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Regularization
    DROPOUT = 0.4

    # Learning rate scheduling
    LR_SCHEDULER = 'cosine'  # 'cosine' or 'plateau'
    LR_PATIENCE = 5  # For plateau scheduler
    LR_FACTOR = 0.5  # For plateau scheduler
    MIN_LR = 1e-7

    # Early stopping
    EARLY_STOPPING_PATIENCE = 15

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Reproducibility
    RANDOM_SEED = 42

    # Preprocessing (based on research recommendations)
    USE_SQRT_STRETCH = True  # Square-root stretch (astronomy standard)

    def __init__(self):
        # Create directories
        self.CHECKPOINT_DIR.mkdir(exist_ok=True)
        self.LOG_DIR.mkdir(exist_ok=True)

        # Create experiment directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_name = f'resnet34_baseline_{timestamp}'
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
# Dataset
# ============================================================================

class GravitationalLensingDataset(Dataset):
    """Custom Dataset for Gravitational Lensing Images"""

    def __init__(self, data_dir, class_names, use_sqrt_stretch=True):
        """
        Args:
            data_dir: Path to data directory (train or val)
            class_names: List of class directory names
            use_sqrt_stretch: Apply square-root stretch preprocessing
        """
        self.data_dir = Path(data_dir)
        self.class_names = class_names
        self.use_sqrt_stretch = use_sqrt_stretch

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

        print(f"Total samples: {len(self.samples)}\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Load image
        img_path = self.samples[idx]
        image = np.load(img_path)  # Shape: (1, 150, 150)

        # Remove channel dimension if present
        if image.shape[0] == 1:
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

        # Get label
        label = self.labels[idx]

        return image, label


# ============================================================================
# ResNet34 Architecture
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

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
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

        # Statistics
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

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


def compute_metrics(y_true, y_pred, y_probs, class_names):
    """Compute comprehensive evaluation metrics"""
    metrics = {}

    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # Store per-class metrics
    for i, class_name in enumerate(class_names):
        metrics[f'{class_name}_precision'] = precision[i]
        metrics[f'{class_name}_recall'] = recall[i]
        metrics[f'{class_name}_f1'] = f1[i]
        metrics[f'{class_name}_support'] = support[i]

    # Macro averages
    metrics['macro_precision'] = np.mean(precision)
    metrics['macro_recall'] = np.mean(recall)
    metrics['macro_f1'] = np.mean(f1)

    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

    # ROC-AUC (one-vs-rest)
    try:
        # Convert labels to one-hot
        y_true_onehot = np.eye(len(class_names))[y_true]
        metrics['roc_auc'] = roc_auc_score(y_true_onehot, y_probs, average='macro')

        # Per-class AUC
        for i, class_name in enumerate(class_names):
            metrics[f'{class_name}_auc'] = roc_auc_score(y_true_onehot[:, i], y_probs[:, i])
    except Exception as e:
        print(f"Warning: Could not compute ROC-AUC: {e}")
        metrics['roc_auc'] = 0.0

    return metrics


def plot_training_history(history, save_path):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot accuracy
    axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o')
    axes[1].plot(history['val_acc'], label='Val Accuracy', marker='s')
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
    print("GRAVITATIONAL LENSING CLASSIFICATION - ResNet34 Baseline")
    print("="*70)
    print(f"Experiment: {config.experiment_name}")
    print(f"Device: {config.DEVICE}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"Preprocessing: Square-root stretch = {config.USE_SQRT_STRETCH}")
    print(f"Data Augmentation: None (Baseline)")
    print("="*70)

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = GravitationalLensingDataset(
        config.TRAIN_DIR,
        config.CLASS_NAMES,
        use_sqrt_stretch=config.USE_SQRT_STRETCH
    )
    val_dataset = GravitationalLensingDataset(
        config.VAL_DIR,
        config.CLASS_NAMES,
        use_sqrt_stretch=config.USE_SQRT_STRETCH
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
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

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer (AdamW with weight decay)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler
    if config.LR_SCHEDULER == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.NUM_EPOCHS,
            eta_min=config.MIN_LR
        )
    else:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.LR_FACTOR,
            patience=config.LR_PATIENCE,
            min_lr=config.MIN_LR,
            verbose=True
        )

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
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)

        # Validate
        val_loss, val_acc, y_true, y_pred, y_probs = evaluate(model, val_loader, criterion, config.DEVICE)

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rate'].append(current_lr)

        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        # Learning rate scheduling
        if config.LR_SCHEDULER == 'cosine':
            scheduler.step()
        else:
            scheduler.step(val_loss)

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
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, best_model_path)

            print(f"  ✓ New best model saved! Val Acc: {val_acc:.4f}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epoch(s)")

        # Early stopping
        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
            print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
            break

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = config.CHECKPOINT_DIR / f'{config.experiment_name}_epoch{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_path)

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

    # Print results
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:        {metrics['accuracy']:.4f}")
    print(f"  Macro F1-Score:  {metrics['macro_f1']:.4f}")
    print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:    {metrics['macro_recall']:.4f}")
    print(f"  ROC-AUC:         {metrics['roc_auc']:.4f}")

    print(f"\nPer-Class Metrics:")
    for class_name in config.CLASS_LABELS:
        print(f"\n  {class_name}:")
        print(f"    Precision: {metrics[f'{class_name}_precision']:.4f}")
        print(f"    Recall:    {metrics[f'{class_name}_recall']:.4f}")
        print(f"    F1-Score:  {metrics[f'{class_name}_f1']:.4f}")
        if f'{class_name}_auc' in metrics:
            print(f"    ROC-AUC:   {metrics[f'{class_name}_auc']:.4f}")

    # Classification report
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(y_true, y_pred, target_names=config.CLASS_LABELS, digits=4))

    # Confusion matrix
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])

    # Save results
    print("\nSaving results...")

    # Save training history plot
    plot_training_history(history, config.experiment_dir / 'training_history.png')
    print(f"  ✓ Training history plot saved")

    # Save confusion matrix plot
    plot_confusion_matrix(metrics['confusion_matrix'], config.CLASS_LABELS,
                         config.experiment_dir / 'confusion_matrix.png')
    print(f"  ✓ Confusion matrix plot saved")

    # Save metrics to JSON
    metrics_to_save = {k: v.tolist() if isinstance(v, np.ndarray) else v
                      for k, v in metrics.items()}
    with open(config.experiment_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_to_save, f, indent=4)
    print(f"  ✓ Metrics saved to JSON")

    # Save training history
    with open(config.experiment_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=4)
    print(f"  ✓ Training history saved")

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nExperiment directory: {config.experiment_dir}")
    print(f"Best model: {config.CHECKPOINT_DIR / f'{config.experiment_name}_best.pth'}")
    print(f"Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    print("\n")


if __name__ == '__main__':
    main()
