"""
Data Exploration Script for Gravitational Lensing Dataset
Analyzes .npy files and generates visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from collections import Counter

# Dataset paths
TRAIN_DIR = Path("dataset/train")
VAL_DIR = Path("dataset/val")
VIZ_DIR = Path("visualizations")

# Create visualization directory
VIZ_DIR.mkdir(exist_ok=True)

# Class names
CLASSES = ["no", "sphere", "vort"]
CLASS_LABELS = {
    "no": "No Substructure",
    "sphere": "Subhalo/Sphere Substructure",
    "vort": "Vortex Substructure"
}


def count_files_per_class(data_dir):
    """Count number of files in each class"""
    counts = {}
    for class_name in CLASSES:
        class_dir = data_dir / class_name
        npy_files = list(class_dir.glob("*.npy"))
        counts[class_name] = len(npy_files)
    return counts


def load_sample_images(data_dir, num_samples=5):
    """Load random sample images from each class"""
    samples = {}
    for class_name in CLASSES:
        class_dir = data_dir / class_name
        npy_files = list(class_dir.glob("*.npy"))

        # Randomly select samples
        selected_files = np.random.choice(npy_files, size=min(num_samples, len(npy_files)), replace=False)

        images = []
        for file_path in selected_files:
            img = np.load(file_path)
            images.append(img)

        samples[class_name] = images

    return samples


def analyze_image_statistics(data_dir, num_samples=100):
    """Analyze statistical properties of images"""
    stats = {class_name: {
        "shapes": [],
        "mins": [],
        "maxs": [],
        "means": [],
        "stds": []
    } for class_name in CLASSES}

    for class_name in CLASSES:
        class_dir = data_dir / class_name
        npy_files = list(class_dir.glob("*.npy"))

        # Sample random files
        selected_files = np.random.choice(npy_files, size=min(num_samples, len(npy_files)), replace=False)

        for file_path in selected_files:
            img = np.load(file_path)
            stats[class_name]["shapes"].append(img.shape)
            stats[class_name]["mins"].append(img.min())
            stats[class_name]["maxs"].append(img.max())
            stats[class_name]["means"].append(img.mean())
            stats[class_name]["stds"].append(img.std())

    return stats


def visualize_samples(samples, title="Sample Images", save_path=None):
    """Visualize sample images from each class"""
    num_classes = len(CLASSES)
    num_samples = len(samples[CLASSES[0]])

    fig, axes = plt.subplots(num_classes, num_samples, figsize=(15, 9))

    for i, class_name in enumerate(CLASSES):
        for j, img in enumerate(samples[class_name]):
            ax = axes[i, j] if num_classes > 1 else axes[j]
            # Remove channel dimension if present (1, 150, 150) -> (150, 150)
            img_display = img.squeeze() if img.ndim == 3 else img
            ax.imshow(img_display, cmap='gray')
            ax.axis('off')

            if j == 0:
                ax.set_ylabel(CLASS_LABELS[class_name], fontsize=12, rotation=0,
                            ha='right', va='center')

    plt.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.show()


def plot_class_distribution(train_counts, val_counts, save_path=None):
    """Plot class distribution for train and val sets"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Training set
    classes = [CLASS_LABELS[c] for c in CLASSES]
    train_values = [train_counts[c] for c in CLASSES]
    ax1.bar(classes, train_values, color=['#2ecc71', '#3498db', '#e74c3c'])
    ax1.set_title('Training Set Distribution', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Images', fontsize=12)
    ax1.tick_params(axis='x', rotation=15)

    for i, v in enumerate(train_values):
        ax1.text(i, v + 200, str(v), ha='center', va='bottom', fontweight='bold')

    # Validation set
    val_values = [val_counts[c] for c in CLASSES]
    ax2.bar(classes, val_values, color=['#2ecc71', '#3498db', '#e74c3c'])
    ax2.set_title('Validation Set Distribution', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Images', fontsize=12)
    ax2.tick_params(axis='x', rotation=15)

    for i, v in enumerate(val_values):
        ax2.text(i, v + 50, str(v), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved distribution plot to {save_path}")

    plt.show()


def plot_pixel_distributions(data_dir, num_samples=200, save_path=None):
    """Plot pixel value distributions for each class"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, class_name in enumerate(CLASSES):
        class_dir = data_dir / class_name
        npy_files = list(class_dir.glob("*.npy"))

        # Sample files
        selected_files = np.random.choice(npy_files, size=min(num_samples, len(npy_files)), replace=False)

        # Collect pixel values
        all_pixels = []
        for file_path in selected_files:
            img = np.load(file_path)
            all_pixels.extend(img.flatten())

        # Plot histogram
        axes[i].hist(all_pixels, bins=50, color=['#2ecc71', '#3498db', '#e74c3c'][i], alpha=0.7)
        axes[i].set_title(CLASS_LABELS[class_name], fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Pixel Value', fontsize=10)
        axes[i].set_ylabel('Frequency', fontsize=10)
        axes[i].grid(alpha=0.3)

    plt.suptitle('Pixel Value Distributions', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved pixel distribution plot to {save_path}")

    plt.show()


def print_statistics_report(train_stats, val_stats, train_counts, val_counts):
    """Print detailed statistics report"""
    print("\n" + "="*80)
    print(" " * 25 + "DATASET EXPLORATION REPORT")
    print("="*80)

    # Dataset overview
    print("\n📊 DATASET OVERVIEW")
    print("-" * 80)
    print(f"Training samples:   {sum(train_counts.values()):,}")
    print(f"Validation samples: {sum(val_counts.values()):,}")
    print(f"Total samples:      {sum(train_counts.values()) + sum(val_counts.values()):,}")
    print(f"Number of classes:  {len(CLASSES)}")

    # Class distribution
    print("\n📈 CLASS DISTRIBUTION")
    print("-" * 80)
    print(f"{'Class':<30} {'Train':<12} {'Val':<12} {'Total':<12}")
    print("-" * 80)
    for class_name in CLASSES:
        total = train_counts[class_name] + val_counts[class_name]
        print(f"{CLASS_LABELS[class_name]:<30} {train_counts[class_name]:<12} {val_counts[class_name]:<12} {total:<12}")

    # Image statistics
    print("\n📐 IMAGE STATISTICS (Training Set)")
    print("-" * 80)

    for class_name in CLASSES:
        print(f"\n{CLASS_LABELS[class_name]}:")

        # Check if all shapes are the same
        shapes = train_stats[class_name]["shapes"]
        unique_shapes = list(set([str(s) for s in shapes]))
        print(f"  Image shape:     {shapes[0] if shapes else 'N/A'}")
        print(f"  All same shape:  {len(unique_shapes) == 1}")

        # Pixel value statistics
        print(f"  Pixel min:       {np.mean(train_stats[class_name]['mins']):.4f} ± {np.std(train_stats[class_name]['mins']):.4f}")
        print(f"  Pixel max:       {np.mean(train_stats[class_name]['maxs']):.4f} ± {np.std(train_stats[class_name]['maxs']):.4f}")
        print(f"  Pixel mean:      {np.mean(train_stats[class_name]['means']):.4f} ± {np.std(train_stats[class_name]['means']):.4f}")
        print(f"  Pixel std:       {np.mean(train_stats[class_name]['stds']):.4f} ± {np.std(train_stats[class_name]['stds']):.4f}")

    print("\n" + "="*80)


def main():
    """Main exploration pipeline"""
    print("\n🔍 Starting Data Exploration...")

    # Set random seed for reproducibility
    np.random.seed(42)

    # 1. Count files
    print("\n📁 Counting files...")
    train_counts = count_files_per_class(TRAIN_DIR)
    val_counts = count_files_per_class(VAL_DIR)

    # 2. Analyze statistics
    print("📊 Analyzing image statistics...")
    train_stats = analyze_image_statistics(TRAIN_DIR, num_samples=100)
    val_stats = analyze_image_statistics(VAL_DIR, num_samples=100)

    # 3. Load sample images
    print("🖼️  Loading sample images...")
    train_samples = load_sample_images(TRAIN_DIR, num_samples=5)
    val_samples = load_sample_images(VAL_DIR, num_samples=5)

    # 4. Print report
    print_statistics_report(train_stats, val_stats, train_counts, val_counts)

    # 5. Create visualizations
    print("\n📊 Creating visualizations...")

    # Sample images
    visualize_samples(
        train_samples,
        title="Training Set - Sample Images per Class",
        save_path=VIZ_DIR / "train_samples.png"
    )

    visualize_samples(
        val_samples,
        title="Validation Set - Sample Images per Class",
        save_path=VIZ_DIR / "val_samples.png"
    )

    # Class distribution
    plot_class_distribution(
        train_counts,
        val_counts,
        save_path=VIZ_DIR / "class_distribution.png"
    )

    # Pixel distributions
    plot_pixel_distributions(
        TRAIN_DIR,
        num_samples=200,
        save_path=VIZ_DIR / "pixel_distributions.png"
    )

    print("\n✅ Data exploration complete!")
    print(f"📁 Visualizations saved to: {VIZ_DIR.absolute()}")


if __name__ == "__main__":
    main()
