# DataLoader Guide - Understanding Data Loading in PyTorch

## 🎯 Learning Objective
Build a custom PyTorch Dataset and DataLoader to load gravitational lensing images from .npy files.

---

## 📚 Background Concepts

### What is a Dataset in PyTorch?
A **Dataset** is a class that defines:
1. How to access a single data sample
2. The total number of samples
3. How to transform/preprocess each sample

Think of it like a **catalog** - you ask for item #5, it gives you that specific item.

### What is a DataLoader?
A **DataLoader** wraps a Dataset and provides:
- **Batching**: Groups multiple samples together
- **Shuffling**: Randomizes order (important for training)
- **Parallel loading**: Uses multiple CPU workers for faster loading

Think of it like a **delivery service** - it packages items in batches and delivers them efficiently.

---

## 🔍 Understanding Our Data

### Current Structure:
```
dataset/
├── train/
│   ├── no/      → 10,000 .npy files (class 0)
│   ├── sphere/  → 10,000 .npy files (class 1)
│   └── vort/    → 10,000 .npy files (class 2)
└── val/
    ├── no/      → 2,500 .npy files
    ├── sphere/  → 2,500 .npy files
    └── vort/    → 2,500 .npy files
```

### Each .npy file contains:
- Shape: `(1, 150, 150)` → 1 channel, 150x150 pixels
- Data type: `float64`
- Value range: `[0.0, 1.0]` (already normalized)

---

## 🏗️ Building the Dataset Class - Step by Step

### Step 1: Import Required Libraries

```python
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
```

**Why each library?**
- `os`: Navigate directories and list files
- `numpy`: Load .npy files
- `torch`: Convert to PyTorch tensors
- `Dataset`: Base class we'll inherit from
- `DataLoader`: Will use this to batch our data

---

### Step 2: Create the Dataset Class Structure

```python
class LensingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory path (e.g., 'dataset/train')
            transform (callable, optional): Optional transform to apply
        """
        pass  # We'll fill this

    def __len__(self):
        """Return the total number of samples"""
        pass

    def __getitem__(self, idx):
        """Load and return sample #idx"""
        pass
```

**Three required methods:**
1. `__init__`: Initialize and collect all file paths
2. `__len__`: Return total count (PyTorch needs this)
3. `__getitem__`: Load one sample by index

---

### Step 3: Implement `__init__` - Collecting File Paths

**Goal**: Find all .npy files and create labels.

**Think about:**
- How to map folder names to numbers? (no=0, sphere=1, vort=2)
- Where to store file paths and their labels?

**Approach:**
```python
def __init__(self, root_dir, transform=None):
    self.root_dir = root_dir
    self.transform = transform

    # Define class mapping
    self.class_to_idx = {
        'no': 0,
        'sphere': 1,
        'vort': 2
    }

    # Store (filepath, label) pairs
    self.samples = []

    # TODO: Loop through each class folder
    # TODO: For each .npy file, append (filepath, label) to self.samples
```

**Exercise for you:**
1. How would you loop through the three class folders?
2. How would you get all .npy files in each folder?
3. How would you create the full file path?

**Hint structure:**
```python
for class_name, class_idx in self.class_to_idx.items():
    class_dir = os.path.join(root_dir, class_name)
    # List all files in class_dir
    # For each .npy file, append (full_path, class_idx) to self.samples
```

---

### Step 4: Implement `__len__` - Easy!

```python
def __len__(self):
    return len(self.samples)
```

**Why?** PyTorch needs to know how many samples exist for batching.

---

### Step 5: Implement `__getitem__` - The Core Logic

**Goal**: Given an index, load that specific image and return it with its label.

**Steps:**
1. Get the filepath and label for this index
2. Load the .npy file
3. Process the image (shape, type conversions)
4. Convert to PyTorch tensor
5. Return (image_tensor, label)

**Template:**
```python
def __getitem__(self, idx):
    # Step 1: Get file path and label
    filepath, label = self.samples[idx]

    # Step 2: Load .npy file
    image = np.load(filepath)  # Shape: (1, 150, 150)

    # Step 3: Process image
    # Current shape: (1, 150, 150)
    # We might want: (150, 150) for augmentations, then back to (1, 150, 150)

    # Step 4: Convert to tensor
    # TODO: Use torch.from_numpy()
    # TODO: Ensure shape is (1, 150, 150)
    # TODO: Convert to float32

    # Step 5: Return
    return image_tensor, label
```

**Questions to consider:**
- What's the difference between `float64` and `float32`? (Hint: memory)
- Should we squeeze the channel dimension? When?
- What if we want to apply augmentations?

---

### Step 6: Understanding Shape Transformations

```
Original .npy: (1, 150, 150)
                ↓
Option A: Keep as (1, 150, 150) → Good for direct tensor conversion
Option B: Squeeze to (150, 150) → Good for some augmentation libraries
                ↓
Apply transforms (optional)
                ↓
Final tensor: (1, 150, 150) → Required for CNN input
```

**Recommended approach:**
```python
# Remove channel dimension for processing
if image.shape[0] == 1:
    image = image.squeeze(0)  # (150, 150)

# Convert to float32 (more efficient than float64)
image = image.astype(np.float32)

# Apply transforms if any
if self.transform:
    image = self.transform(image)
else:
    # Convert to tensor and add channel back
    image = torch.from_numpy(image).unsqueeze(0)  # (1, 150, 150)
```

---

## 🔄 Data Augmentation (Optional but Recommended)

### Why Augmentation?
Even with balanced classes, augmentation helps the model generalize better.

### What augmentations make sense for lensing images?

**Good augmentations:**
- ✅ Rotations (lenses are rotationally symmetric)
- ✅ Horizontal/Vertical flips
- ✅ Small shifts and scales
- ✅ Gaussian noise (simulates observation noise)

**Bad augmentations:**
- ❌ Color changes (grayscale images)
- ❌ Large crops (might remove important features)

### Two popular libraries:

**Option 1: torchvision.transforms**
```python
from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.RandomRotation(45),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip()
])
```

**Option 2: albumentations (more powerful)**
```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transforms = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
    A.GaussNoise(var_limit=(0.0, 0.01), p=0.3),
    ToTensorV2()  # Converts to tensor at the end
])
```

**Exercise:** Research what each parameter means (p, shift_limit, etc.)

---

## 📦 Creating the DataLoader

### Basic DataLoader:
```python
# Create dataset
train_dataset = LensingDataset(root_dir='dataset/train')

# Wrap in DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,        # How many samples per batch?
    shuffle=True,         # Randomize order for training
    num_workers=4,        # Parallel loading
    pin_memory=True       # Faster GPU transfer
)
```

### Important Parameters:

**batch_size**: How many images to process at once
- Small batch (8-16): Less memory, noisier gradients
- Large batch (64-128): More memory, smoother gradients
- Start with 32, adjust based on GPU memory

**shuffle**:
- `True` for training (prevents learning order)
- `False` for validation (reproducible results)

**num_workers**:
- 0 = single process (slower, good for debugging)
- 4-8 = multiple processes (faster, but uses more CPU)

**pin_memory**:
- `True` if using GPU (faster data transfer)
- `False` if CPU only

---

## 🧪 Testing Your DataLoader

Always test before using in training!

```python
if __name__ == "__main__":
    # Create dataloader
    train_loader = DataLoader(
        LensingDataset('dataset/train'),
        batch_size=8,
        shuffle=True,
        num_workers=0  # 0 for debugging
    )

    # Test: Get one batch
    images, labels = next(iter(train_loader))

    # Print information
    print(f"Batch shape: {images.shape}")  # Should be (8, 1, 150, 150)
    print(f"Labels: {labels}")              # Should be tensor with values 0, 1, 2
    print(f"Data type: {images.dtype}")     # Should be torch.float32
    print(f"Value range: [{images.min():.3f}, {images.max():.3f}]")  # Should be [0, 1]
```

**Expected output:**
```
Batch shape: torch.Size([8, 1, 150, 150])
Labels: tensor([0, 2, 1, 0, 2, 1, 0, 1])
Data type: torch.float32
Value range: [0.000, 1.000]
```

---

## ✅ Checklist - Is Your DataLoader Working?

- [ ] Can load all 30,000 training samples
- [ ] Can load all 7,500 validation samples
- [ ] Batch shape is `(batch_size, 1, 150, 150)`
- [ ] Labels are integers: 0, 1, or 2
- [ ] Data type is `torch.float32`
- [ ] Values are in range [0, 1]
- [ ] Training data is shuffled
- [ ] Validation data is not shuffled
- [ ] Can iterate through entire dataset without errors

---

## 🎓 Learning Exercises

### Exercise 1: Basic Implementation
Implement the Dataset class without augmentations first. Get it working!

### Exercise 2: Add Augmentation
Add rotation and flip augmentations. Visualize before/after.

### Exercise 3: Verify Class Balance
Write code to count how many samples per class in train and val sets.

```python
from collections import Counter
labels = [label for _, label in train_dataset.samples]
print(Counter(labels))  # Should be {0: 10000, 1: 10000, 2: 10000}
```

### Exercise 4: Visualize Samples
Load a few samples and display them to verify they look correct.

---

## 🐛 Common Issues & Solutions

### Issue 1: "Can't convert np.ndarray to Tensor"
**Solution**: Use `torch.from_numpy()` not `torch.tensor()`

### Issue 2: Shape mismatch errors
**Solution**: Check shapes at each step. Use `.shape` liberally!

### Issue 3: DataLoader hangs with num_workers > 0 on Windows
**Solution**: Set `num_workers=0` or use `if __name__ == "__main__":`

### Issue 4: Out of memory
**Solution**: Reduce `batch_size`

---

## 📝 Summary

**What you learned:**
1. How PyTorch Dataset and DataLoader work
2. How to load custom .npy files
3. How to create class labels from folder structure
4. Shape transformations and tensor conversions
5. Data augmentation strategies
6. Testing and debugging data pipelines

**Next step:** Once your DataLoader is working, move to [02_model_guide.md](02_model_guide.md) to build the CNN!

---

## 💡 Pro Tips

1. **Always test with small batch first**: Use `batch_size=4, num_workers=0` for debugging
2. **Print shapes everywhere**: Confusion usually comes from shape mismatches
3. **Visualize your data**: Make sure augmentations look reasonable
4. **Start simple**: Get basic loading working before adding augmentations
5. **Use assertions**: Add checks like `assert image.shape == (1, 150, 150)`

---

## 🔗 Useful Resources

- [PyTorch Dataset Documentation](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)
- [DataLoader Documentation](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
- [Albumentations Documentation](https://albumentations.ai/docs/)

Happy coding! 🚀
