# Converting .npy to PNG Guide - Visualizing Your Data

## 🎯 Learning Objective
Create a utility script to convert .npy files to PNG images for visualization and data exploration.

---

## 📚 Background Concepts

### Why Convert to PNG?

**Current format: .npy (NumPy array)**
- Binary format
- Fast to load in Python
- Can't view directly in image viewer
- Can't share easily

**PNG format:**
- ✅ Universal image format
- ✅ Can view in any image viewer
- ✅ Easy to share
- ✅ Good for presentations
- ✅ Lossless compression

**When to convert:**
- Visualizing samples during data exploration
- Creating presentations or reports
- Debugging data loading
- Sharing examples with collaborators

**When NOT to convert:**
- For training (use .npy directly - faster)
- For large-scale processing (conversion is slow)

---

## 🔍 Understanding the Data

### Current .npy structure:
```python
data = np.load('dataset/train/no/1.npy')
# Shape: (1, 150, 150)
# Type: float64
# Range: [0.0, 1.0]
```

### What we need for PNG:
```python
# Shape: (150, 150) or (150, 150, 3) for RGB
# Type: uint8 (0-255 integers)
# Range: [0, 255]
```

### Conversion steps:
```
(1, 150, 150) float64 [0, 1]
        ↓ squeeze channel dimension
(150, 150) float64 [0, 1]
        ↓ scale to 0-255
(150, 150) float64 [0, 255]
        ↓ convert to uint8
(150, 150) uint8 [0, 255]
        ↓ save as PNG
image.png
```

---

## 🏗️ Building the Conversion Script - Step by Step

### Step 1: Import Required Libraries

```python
import os
import numpy as np
from PIL import Image  # Python Imaging Library
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
```

**Library purposes:**
- `numpy`: Load .npy files
- `PIL`: Save PNG files (efficient)
- `matplotlib`: Alternative visualization
- `tqdm`: Progress bar
- `argparse`: Command-line arguments

---

### Step 2: Basic Conversion Function

```python
def npy_to_png(npy_path, png_path, colormap=None):
    """
    Convert a single .npy file to PNG

    Args:
        npy_path: Path to input .npy file
        png_path: Path to output PNG file
        colormap: Optional colormap name (e.g., 'viridis', 'gray')

    Returns:
        Success status (True/False)
    """
    try:
        # Load .npy file
        data = np.load(npy_path)

        # TODO: Process data
        # 1. Remove channel dimension if exists
        # 2. Ensure 2D array
        # 3. Convert to 0-255 range
        # 4. Convert to uint8
        # 5. Save as PNG

        return True

    except Exception as e:
        print(f"Error converting {npy_path}: {e}")
        return False
```

**Your task:** Fill in the TODO section. Here's the approach:

```python
# Step 1: Remove channel dimension
if data.shape[0] == 1 and len(data.shape) == 3:
    data = data.squeeze(0)  # (1, 150, 150) → (150, 150)

# Step 2: Ensure 2D
assert len(data.shape) == 2, f"Expected 2D array, got shape {data.shape}"

# Step 3: Normalize to [0, 1] if not already
if data.max() > 1.0:
    data = (data - data.min()) / (data.max() - data.min())

# Step 4: Scale to [0, 255]
data_scaled = (data * 255).astype(np.uint8)

# Step 5: Save
if colormap is None:
    # Grayscale
    image = Image.fromarray(data_scaled, mode='L')
else:
    # Apply colormap using matplotlib
    cmap = plt.get_cmap(colormap)
    data_colored = cmap(data)  # (150, 150, 4) RGBA
    data_rgb = (data_colored[:, :, :3] * 255).astype(np.uint8)
    image = Image.fromarray(data_rgb, mode='RGB')

image.save(png_path)
```

---

### Step 3: Batch Conversion Function

```python
def convert_directory(input_dir, output_dir, colormap=None, max_files=None):
    """
    Convert all .npy files in a directory to PNG

    Args:
        input_dir: Directory containing .npy files
        output_dir: Directory to save PNG files
        colormap: Optional colormap
        max_files: Maximum number of files to convert (None = all)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find all .npy files
    npy_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]

    # Limit number of files if specified
    if max_files is not None:
        npy_files = npy_files[:max_files]

    print(f"Converting {len(npy_files)} files from {input_dir}")

    # Convert each file
    success_count = 0
    for npy_filename in tqdm(npy_files):
        npy_path = os.path.join(input_dir, npy_filename)
        png_filename = npy_filename.replace('.npy', '.png')
        png_path = os.path.join(output_dir, png_filename)

        if npy_to_png(npy_path, png_path, colormap):
            success_count += 1

    print(f"Successfully converted {success_count}/{len(npy_files)} files")
```

---

### Step 4: Convert Entire Dataset

```python
def convert_dataset(dataset_root, output_root, colormap=None, samples_per_class=10):
    """
    Convert samples from each class in train and val sets

    Args:
        dataset_root: Root directory of dataset (contains train/ and val/)
        output_root: Root directory for output
        colormap: Optional colormap
        samples_per_class: How many samples to convert per class
    """
    splits = ['train', 'val']
    classes = ['no', 'sphere', 'vort']

    for split in splits:
        print(f"\n{'='*60}")
        print(f"Converting {split} set...")
        print('='*60)

        for class_name in classes:
            input_dir = os.path.join(dataset_root, split, class_name)
            output_dir = os.path.join(output_root, split, class_name)

            convert_directory(input_dir, output_dir, colormap, max_files=samples_per_class)
```

---

### Step 5: Visualization Function

Sometimes you want to visualize multiple samples side-by-side:

```python
def visualize_samples(npy_paths, titles=None, colormap='gray', save_path=None):
    """
    Visualize multiple .npy files in a grid

    Args:
        npy_paths: List of paths to .npy files
        titles: Optional list of titles for each image
        colormap: Colormap to use
        save_path: Optional path to save the figure
    """
    n_samples = len(npy_paths)
    cols = min(5, n_samples)
    rows = (n_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    # Handle single row/col case
    if n_samples == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for idx, npy_path in enumerate(npy_paths):
        # Load data
        data = np.load(npy_path)

        # Remove channel dimension
        if data.shape[0] == 1:
            data = data.squeeze(0)

        # Plot
        axes[idx].imshow(data, cmap=colormap)
        axes[idx].axis('off')

        # Title
        if titles is not None:
            axes[idx].set_title(titles[idx], fontsize=10)
        else:
            axes[idx].set_title(os.path.basename(npy_path), fontsize=8)

    # Hide unused subplots
    for idx in range(n_samples, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")

    plt.show()
```

---

### Step 6: Command-Line Interface

Make the script usable from terminal:

```python
def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description='Convert .npy files to PNG images')

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory or file path'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory or file path'
    )

    parser.add_argument(
        '--colormap',
        type=str,
        default='gray',
        help='Colormap to use (gray, viridis, plasma, etc.)'
    )

    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='Maximum number of files to convert'
    )

    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Visualize samples instead of saving'
    )

    args = parser.parse_args()

    # Check if input is file or directory
    if os.path.isfile(args.input):
        # Single file conversion
        npy_to_png(args.input, args.output, args.colormap)
        print(f"Converted {args.input} → {args.output}")

    elif os.path.isdir(args.input):
        # Directory conversion
        convert_directory(args.input, args.output, args.colormap, args.max_files)

    else:
        print(f"Error: {args.input} is not a valid file or directory")


if __name__ == "__main__":
    main()
```

---

## 🎨 Choosing Colormaps

### Grayscale (Default)
```python
colormap='gray'
```
**Best for:** Scientific accuracy, seeing exact intensity values

### Viridis (Perceptually Uniform)
```python
colormap='viridis'
```
**Best for:** General visualization, colorblind-friendly

### Plasma
```python
colormap='plasma'
```
**Best for:** Highlighting features, presentations

### Jet (Controversial but Popular)
```python
colormap='jet'
```
**Best for:** High contrast, but not perceptually uniform
**Warning:** Can be misleading, not colorblind-friendly

### Custom Colormaps
```python
# For lensing images, you might want:
# - 'hot': Highlights bright regions
# - 'coolwarm': Shows positive/negative deviations
# - 'seismic': Red-blue diverging
```

**Exercise:** Try different colormaps and see which reveals features best.

---

## 🧪 Testing the Script

### Test 1: Convert Single File
```python
from src.utils.convert_npy_to_png import npy_to_png

# Convert one file
npy_to_png(
    'dataset/train/no/1.npy',
    'test_output/sample_1.png',
    colormap='gray'
)

# View the result
from PIL import Image
img = Image.open('test_output/sample_1.png')
img.show()
```

### Test 2: Convert Multiple Classes
```python
# Convert a few samples from each class
from src.utils.convert_npy_to_png import visualize_samples

npy_paths = [
    'dataset/train/no/1.npy',
    'dataset/train/sphere/1.npy',
    'dataset/train/vort/1.npy'
]

titles = ['No Substructure', 'Sphere Substructure', 'Vortex Substructure']

visualize_samples(npy_paths, titles, colormap='viridis', save_path='visualizations/samples.png')
```

### Test 3: Command-Line Usage
```bash
# Convert all files in a directory
python src/utils/convert_npy_to_png.py \
    --input dataset/train/no \
    --output png_samples/train/no \
    --colormap gray \
    --max-files 10

# Convert single file
python src/utils/convert_npy_to_png.py \
    --input dataset/train/no/1.npy \
    --output sample.png \
    --colormap viridis
```

---

## 📊 Creating a Data Exploration Notebook

Create a comparison visualization:

```python
def create_class_comparison(dataset_root, n_samples=5):
    """
    Create a grid showing samples from each class

    Args:
        dataset_root: Root directory of dataset
        n_samples: Number of samples per class
    """
    classes = ['no', 'sphere', 'vort']
    class_labels = ['No Substructure', 'Subhalo', 'Vortex']

    fig, axes = plt.subplots(len(classes), n_samples, figsize=(n_samples*3, len(classes)*3))

    for i, (class_name, label) in enumerate(zip(classes, class_labels)):
        class_dir = os.path.join(dataset_root, 'train', class_name)
        files = sorted([f for f in os.listdir(class_dir) if f.endswith('.npy')])[:n_samples]

        for j, filename in enumerate(files):
            filepath = os.path.join(class_dir, filename)
            data = np.load(filepath)

            if data.shape[0] == 1:
                data = data.squeeze(0)

            axes[i, j].imshow(data, cmap='gray')
            axes[i, j].axis('off')

            if j == 0:
                axes[i, j].set_ylabel(label, fontsize=12, fontweight='bold')

    plt.suptitle('Gravitational Lensing Image Samples by Class', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('visualizations/class_comparison.png', dpi=200, bbox_inches='tight')
    plt.show()
```

---

## 💡 Practical Use Cases

### Use Case 1: Data Exploration
```python
# Visualize random samples to understand data
import random

def explore_random_samples(dataset_root, n_samples=9):
    classes = ['no', 'sphere', 'vort']
    samples = []
    titles = []

    for _ in range(n_samples):
        cls = random.choice(classes)
        cls_dir = os.path.join(dataset_root, 'train', cls)
        files = [f for f in os.listdir(cls_dir) if f.endswith('.npy')]
        file = random.choice(files)
        samples.append(os.path.join(cls_dir, file))
        titles.append(f"{cls} - {file}")

    visualize_samples(samples, titles, save_path='visualizations/random_samples.png')
```

### Use Case 2: Debugging Data Loading
```python
# Verify data is loaded correctly
def verify_dataloader(dataloader):
    images, labels = next(iter(dataloader))

    # Save first batch as PNGs
    for i in range(min(8, len(images))):
        img = images[i].squeeze().numpy()
        img_scaled = (img * 255).astype(np.uint8)
        Image.fromarray(img_scaled, mode='L').save(f'debug/batch_sample_{i}_label_{labels[i]}.png')
```

### Use Case 3: Presentation/Report
```python
# Create high-quality figures for reports
def create_report_figure():
    """Create a publication-quality figure"""
    # Set matplotlib style
    plt.style.use('seaborn-v0_8-paper')

    # Your visualization code here
    # Use high DPI (300+) for publications
    plt.savefig('report_figure.png', dpi=300, bbox_inches='tight')
```

---

## ✅ Checklist - Conversion Working?

- [ ] Can load .npy files correctly
- [ ] Can convert to PNG (grayscale)
- [ ] Output images look correct (not inverted, rotated, etc.)
- [ ] Can apply different colormaps
- [ ] Batch conversion works
- [ ] Command-line interface works
- [ ] Can visualize multiple samples in grid
- [ ] Saved PNGs have correct dimensions (150x150)

---

## 🎓 Learning Exercises

### Exercise 1: Understand Data Range
Load several samples and check min/max values. Are they all [0, 1]?

### Exercise 2: Compare Colormaps
Convert the same image with 5 different colormaps. Which is best for lensing data?

### Exercise 3: Image Statistics
Calculate and visualize pixel intensity histograms for each class.

### Exercise 4: Data Quality Check
Create a script to detect corrupted or anomalous files.

---

## 🐛 Common Issues & Solutions

### Issue 1: Image looks inverted
**Cause:** Some colormaps invert black/white
**Solution:** Use `colormap='gray'` or invert with `img = 255 - img`

### Issue 2: Image looks wrong orientation
**Cause:** Array indexing vs. image coordinates
**Solution:** Use `np.rot90()` or `np.flip()` if needed

### Issue 3: Low contrast
**Cause:** Data range is small
**Solution:** Apply histogram equalization:
```python
from skimage import exposure
data_eq = exposure.equalize_hist(data)
```

### Issue 4: PNG file is huge
**Cause:** Saving as 24-bit RGB instead of 8-bit grayscale
**Solution:** Ensure `mode='L'` in `Image.fromarray()`

---

## 📝 Summary

**What you learned:**
1. How to convert NumPy arrays to PNG images
2. Data preprocessing (normalization, scaling)
3. Working with PIL and matplotlib
4. Creating visualization utilities
5. Command-line interface design
6. Colormap selection for scientific data

**When to use this:**
- Data exploration and understanding
- Debugging data pipelines
- Creating presentations
- Sharing samples with collaborators

---

## 💡 Pro Tips

1. **Keep .npy for training**: Only convert for visualization
2. **Use grayscale for accuracy**: Colormaps can be misleading
3. **Check data range**: Always verify min/max values
4. **Save high DPI**: Use dpi=300 for publications
5. **Batch processing**: Use tqdm for progress bars
6. **Document colormap choice**: Different colormaps for different purposes

---

## 🔗 Useful Resources

- [PIL Documentation](https://pillow.readthedocs.io/)
- [Matplotlib Colormaps](https://matplotlib.org/stable/tutorials/colors/colormaps.html)
- [NumPy Image Processing](https://numpy.org/doc/stable/user/quickstart.html)
- [Choosing Colormaps](https://www.kennethmoreland.com/color-advice/)

Now you can visualize your lensing images! 🖼️
