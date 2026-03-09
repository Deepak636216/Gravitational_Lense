"""
Gravitational Lensing Classification - FIXED ResNet34 Training
===============================================================

FIXES:
1. Removed per-image normalization (was destroying class differences)
2. Using simple Cross Entropy instead of Focal Loss + Weighted Sampler (too complex)
3. Higher initial learning rate (0.001 instead of 0.0001)
4. Simplified augmentation pipeline

Expected: 75-88% accuracy (vs. 34% before)
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


# Configuration
class Config:
    # Paths
    DATA_DIR = Path('./dataset')
    TRAIN_DIR = DATA_DIR / 'train'
    VAL_DIR = DATA_DIR / 'val'
    CHECKPOINT_DIR = Path('./checkpoints')
    LOG_DIR = Path('./logs')

    # Classes
    CLASS_NAMES = ['no', 'sphere', 'vort']
    CLASS_LABELS = ['No Substructure', 'Subhalo/Sphere', 'Vortex']
    NUM_CLASSES = 3

    # Training
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3  # FIXED: Higher initial LR
    WEIGHT_DECAY = 1e-4
    DROPOUT = 0.4

    # Augmentation
    USE_AUGMENTATION = True

    # Scheduler
    LR_SCHEDULER = 'cosine'
    MIN_LR = 1e-7

    # Early stopping
    EARLY_STOPPING_PATIENCE = 15

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    RANDOM_SEED = 42

    # Preprocessing - FIXED!
    USE_SQRT_STRETCH = True
    # Global statistics (computed from diagnosis)
    GLOBAL_MEAN = 0.0615  # Mean across all images
    GLOBAL_STD = 0.1157   # Std across all images

    def __init__(self):
        self.CHECKPOINT_DIR.mkdir(exist_ok=True)
        self.LOG_DIR.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_name = f'resnet34_fixed_{timestamp}'
        self.experiment_dir = self.LOG_DIR / self.experiment_name
        self.experiment_dir.mkdir(exist_ok=True)

        self.save_config()

    def save_config(self):
        config_dict = {k: str(v) if isinstance(v, Path) else v
                      for k, v in self.__dict__.items()
                      if not k.startswith('_')}
        config_dict['DEVICE'] = str(self.DEVICE)

        with open(self.experiment_dir / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=4)


# Dataset - FIXED preprocessing
class GravitationalLensingDataset(Dataset):
    def __init__(self, data_dir, class_names, use_sqrt_stretch=True, augment=False, config=None):
        self.data_dir = Path(data_dir)
        self.class_names = class_names
        self.use_sqrt_stretch = use_sqrt_stretch
        self.augment = augment
        self.config = config

        self.samples = []
        self.labels = []

        print(f"\nLoading data from {data_dir}...")
        for class_idx, class_name in enumerate(class_names):
            class_dir = self.data_dir / class_name
            class_files = sorted(class_dir.glob('*.npy'))
            print(f"  {class_name}: {len(class_files)} samples")

            for file_path in class_files:
                self.samples.append(file_path)
                self.labels.append(class_idx)

        print(f"Total samples: {len(self.samples)}")
        if augment:
            print("  Data augmentation: ENABLED")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Load image
        image = np.load(self.samples[idx])
        if image.shape[0] == 1:
            image = image[0]

        # Preprocessing: Square-root stretch
        if self.use_sqrt_stretch:
            image = np.maximum(image, 0)
            image = np.sqrt(image)

            # FIXED: Global standardization instead of per-image normalization!
            # This preserves the relative brightness differences between images
            image = (image - self.config.GLOBAL_MEAN) / (self.config.GLOBAL_STD + 1e-7)

        # Convert to tensor
        image = torch.from_numpy(image).float().unsqueeze(0)

        # Augmentation
        if self.augment:
            # 90-degree rotation
            if np.random.rand() < 0.75:
                k = np.random.randint(0, 4)
                image = torch.rot90(image, k, dims=[-2, -1])

            # Horizontal flip
            if np.random.rand() < 0.5:
                image = torch.flip(image, dims=[-1])

            # Vertical flip
            if np.random.rand() < 0.5:
                image = torch.flip(image, dims=[-2])

            # Gaussian noise
            if np.random.rand() < 0.3:
                noise = torch.randn_like(image) * 0.02
                image = image + noise

        return image, self.labels[idx]


# ResNet34 Model (same as before)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet34(nn.Module):
    def __init__(self, num_classes=3, dropout=0.4):
        super(ResNet34, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 4, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 6, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(128, num_classes)

        self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

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


# Training functions
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def evaluate(model, val_loader, criterion, device):
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

            outputs = model(images)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc, np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_metrics(y_true, y_pred, y_probs, class_names):
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    for i, class_name in enumerate(class_names):
        metrics[f'{class_name}_precision'] = precision[i]
        metrics[f'{class_name}_recall'] = recall[i]
        metrics[f'{class_name}_f1'] = f1[i]

    metrics['macro_precision'] = np.mean(precision)
    metrics['macro_recall'] = np.mean(recall)
    metrics['macro_f1'] = np.mean(f1)
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

    try:
        y_true_onehot = np.eye(len(class_names))[y_true]
        metrics['roc_auc'] = roc_auc_score(y_true_onehot, y_probs, average='macro')
    except:
        metrics['roc_auc'] = 0.0

    return metrics


def plot_training_history(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

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
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# Main
def main():
    torch.manual_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.RANDOM_SEED)

    config = Config()

    print("="*70)
    print("GRAVITATIONAL LENSING CLASSIFICATION - ResNet34 FIXED")
    print("="*70)
    print(f"Experiment: {config.experiment_name}")
    print(f"Device: {config.DEVICE}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"\nFIXES APPLIED:")
    print(f"  [FIXED] Global normalization (not per-image)")
    print(f"  [FIXED] Simple Cross Entropy loss")
    print(f"  [FIXED] Higher learning rate: {config.LEARNING_RATE}")
    print(f"  [FIXED] Removed weighted sampler")
    print("="*70)

    # Create datasets
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
        augment=False,
        config=config
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
    print("\nInitializing ResNet34...")
    model = ResNet34(num_classes=config.NUM_CLASSES, dropout=config.DROPOUT)
    model = model.to(config.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Simple Cross Entropy loss (no focal loss!)
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.NUM_EPOCHS,
        eta_min=config.MIN_LR
    )

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': []
    }

    best_val_acc = 0.0
    epochs_no_improve = 0
    best_epoch = 0

    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")

    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        print("-" * 50)

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

        # Print summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        # LR scheduling
        scheduler.step()

        # Check improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0

            # Save best model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, config.CHECKPOINT_DIR / f'{config.experiment_name}_best.pth')

            print(f"  ✓ New best model saved! Val Acc: {val_acc:.4f}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epoch(s)")

        # Early stopping
        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
            print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
            break

    # Final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)

    checkpoint = torch.load(config.CHECKPOINT_DIR / f'{config.experiment_name}_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    val_loss, val_acc, y_true, y_pred, y_probs = evaluate(model, val_loader, criterion, config.DEVICE)
    metrics = compute_metrics(y_true, y_pred, y_probs, config.CLASS_LABELS)

    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:        {metrics['accuracy']:.4f}")
    print(f"  Macro F1-Score:  {metrics['macro_f1']:.4f}")
    print(f"  ROC-AUC:         {metrics['roc_auc']:.4f}")

    print(f"\nPer-Class Metrics:")
    for class_name in config.CLASS_LABELS:
        print(f"\n  {class_name}:")
        print(f"    Precision: {metrics[f'{class_name}_precision']:.4f}")
        print(f"    Recall:    {metrics[f'{class_name}_recall']:.4f}")
        print(f"    F1-Score:  {metrics[f'{class_name}_f1']:.4f}")

    print("\n" + "="*70)
    print(classification_report(y_true, y_pred, target_names=config.CLASS_LABELS, digits=4))

    # Save results
    plot_training_history(history, config.experiment_dir / 'training_history.png')
    plot_confusion_matrix(metrics['confusion_matrix'], config.CLASS_LABELS,
                         config.experiment_dir / 'confusion_matrix.png')

    metrics_to_save = {k: v.tolist() if isinstance(v, np.ndarray) else v
                      for k, v in metrics.items()}
    with open(config.experiment_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_to_save, f, indent=4)

    with open(config.experiment_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=4)

    print(f"\n✓ Results saved to: {config.experiment_dir}")
    print(f"✓ Best model: {config.CHECKPOINT_DIR / f'{config.experiment_name}_best.pth'}")
    print(f"✓ Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})\n")


if __name__ == '__main__':
    main()
