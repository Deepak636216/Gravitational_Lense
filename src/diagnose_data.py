"""
Data Pipeline Diagnostic Script
================================
Verifies that data loading is correct and identifies any issues with class distribution.
"""

import torch
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Import from the training script
sys.path.append(str(Path(__file__).parent))
from train_resnet_baseline import Config, GravitationalLensingDataset
from torch.utils.data import DataLoader


def diagnose_dataset():
    """Run comprehensive dataset diagnostics"""

    print("="*70)
    print("DATASET DIAGNOSTICS")
    print("="*70)

    config = Config()

    # Create datasets
    print("\n1. Loading datasets...")
    train_dataset = GravitationalLensingDataset(
        config.TRAIN_DIR,
        config.CLASS_NAMES,
        use_sqrt_stretch=config.USE_SQRT_STRETCH
    )

    # Check overall distribution
    print("\n2. Checking class distribution in dataset...")
    label_counter = Counter(train_dataset.labels)
    print("\nClass distribution:")
    for class_idx, class_name in enumerate(config.CLASS_LABELS):
        count = label_counter[class_idx]
        percentage = (count / len(train_dataset)) * 100
        print(f"  {class_name}: {count} samples ({percentage:.1f}%)")

    # Create dataloader
    print("\n3. Checking batch distribution...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0  # Single-threaded for debugging
    )

    # Sample batches
    batch_labels = []
    num_batches_to_check = 50

    print(f"\nSampling {num_batches_to_check} random batches...")
    for i, (images, labels) in enumerate(train_loader):
        if i >= num_batches_to_check:
            break
        batch_labels.extend(labels.tolist())

        # Show first batch details
        if i == 0:
            print(f"\nFirst batch details:")
            print(f"  Image shape: {images.shape}")
            print(f"  Image dtype: {images.dtype}")
            print(f"  Image range: [{images.min():.4f}, {images.max():.4f}]")
            print(f"  Labels: {labels.tolist()}")
            print(f"  Unique labels in batch: {labels.unique().tolist()}")

    # Check batch distribution
    batch_counter = Counter(batch_labels)
    print(f"\nClass distribution in {num_batches_to_check} batches:")
    for class_idx, class_name in enumerate(config.CLASS_LABELS):
        count = batch_counter[class_idx]
        percentage = (count / len(batch_labels)) * 100
        print(f"  {class_name}: {count} samples ({percentage:.1f}%)")

    # Analyze pixel statistics
    print("\n4. Analyzing pixel statistics...")
    pixel_values = []
    num_samples_to_check = 100

    for i in range(min(num_samples_to_check, len(train_dataset))):
        image, _ = train_dataset[i]
        pixel_values.append(image.numpy())

    all_pixels = np.concatenate([img.flatten() for img in pixel_values])

    print(f"\nPixel statistics (from {num_samples_to_check} samples):")
    print(f"  Mean: {all_pixels.mean():.6f}")
    print(f"  Std:  {all_pixels.std():.6f}")
    print(f"  Min:  {all_pixels.min():.6f}")
    print(f"  Max:  {all_pixels.max():.6f}")
    print(f"  Median: {np.median(all_pixels):.6f}")
    print(f"  25th percentile: {np.percentile(all_pixels, 25):.6f}")
    print(f"  75th percentile: {np.percentile(all_pixels, 75):.6f}")

    # Visualize sample images from each class
    print("\n5. Visualizing sample images from each class...")
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    fig.suptitle('Sample Images from Each Class (after preprocessing)', fontsize=16)

    samples_per_class = 5
    for class_idx, class_name in enumerate(config.CLASS_LABELS):
        # Find indices for this class
        class_indices = [i for i, label in enumerate(train_dataset.labels) if label == class_idx]

        # Sample images
        sampled_indices = np.random.choice(class_indices, samples_per_class, replace=False)

        for i, idx in enumerate(sampled_indices):
            image, label = train_dataset[idx]
            ax = axes[class_idx, i]
            ax.imshow(image.squeeze(), cmap='viridis')
            ax.axis('off')
            if i == 0:
                ax.set_title(f'{class_name}', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('./logs/data_diagnostics_samples.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Sample images saved to ./logs/data_diagnostics_samples.png")

    # Pixel distribution histogram
    print("\n6. Creating pixel distribution histogram...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(all_pixels, bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Pixel Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Pixel Value Distribution')
    axes[0].grid(True, alpha=0.3)

    # Log-scale histogram
    axes[1].hist(all_pixels, bins=50, alpha=0.7, edgecolor='black', log=True)
    axes[1].set_xlabel('Pixel Value')
    axes[1].set_ylabel('Frequency (log scale)')
    axes[1].set_title('Pixel Value Distribution (Log Scale)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./logs/data_diagnostics_pixels.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Pixel distribution saved to ./logs/data_diagnostics_pixels.png")

    print("\n" + "="*70)
    print("DIAGNOSTICS COMPLETE")
    print("="*70)
    print("\nKey Findings:")
    print(f"  - Dataset is balanced: {len(set([label_counter[i] for i in range(3)])) == 1}")
    print(f"  - Batch sampling appears random: {abs(batch_counter[0] - batch_counter[1]) < 50}")
    print(f"  - Pixel normalization: [0, 1] range = {all_pixels.min() >= 0 and all_pixels.max() <= 1}")
    print("\nRecommendations:")

    # Check if distribution is skewed
    if abs(batch_counter[0] - batch_counter[2]) > 100:
        print("  ⚠ WARNING: Batch distribution appears skewed!")
        print("    → Use WeightedRandomSampler to ensure balanced batches")
    else:
        print("  ✓ Batch distribution looks good")

    # Check pixel distribution
    if all_pixels.std() < 0.1:
        print("  ⚠ WARNING: Very low pixel variance")
        print("    → Consider adjusting preprocessing or normalization")
    else:
        print("  ✓ Pixel variance looks reasonable")

    print("\n")


if __name__ == '__main__':
    diagnose_dataset()
