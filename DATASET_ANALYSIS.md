# Gravitational Lensing Classification Dataset - Comprehensive Analysis

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Dataset Overview](#dataset-overview)
3. [Physical Context](#physical-context)
4. [Statistical Analysis](#statistical-analysis)
5. [Class Characteristics](#class-characteristics)
6. [Model Architecture Recommendations](#model-architecture-recommendations)
7. [Hyperparameter Selection Guide](#hyperparameter-selection-guide)
8. [Training Strategy](#training-strategy)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Expected Challenges](#expected-challenges)
11. [Recommended Configurations](#recommended-configurations)

---

## Executive Summary

This dataset contains 37,500 grayscale images (150×150 pixels) of simulated gravitational lensing events, divided into three equally-balanced classes representing different dark matter substructure scenarios. The classification task is challenging due to subtle visual differences between classes (mean pixel intensity varies by < 0.002), requiring sophisticated deep learning architectures with careful regularization.

**Key Characteristics:**
- ✅ Perfectly balanced classes (no resampling needed)
- ✅ Pre-normalized pixel values [0, 1]
- ✅ Consistent image dimensions
- ⚠️ Very subtle inter-class differences
- ⚠️ Requires deep feature extraction

---

## Dataset Overview

### Dataset Composition

| Metric | Training Set | Validation Set | Total |
|--------|-------------|----------------|-------|
| **Total Samples** | 30,000 | 7,500 | 37,500 |
| **No Substructure** | 10,000 | 2,500 | 12,500 |
| **Subhalo/Sphere** | 10,000 | 2,500 | 12,500 |
| **Vortex Substructure** | 10,000 | 2,500 | 12,500 |
| **Class Distribution** | 33.33% each | 33.33% each | 33.33% each |

### Image Specifications

```
Format: NumPy array
Shape: (1, 150, 150)
Channels: 1 (Grayscale)
Data Type: float32
Pixel Range: [0.0, 1.0]
File Organization: Separate directories per class
```

### Directory Structure

```
data/
├── train/
│   ├── no_sub/           # No Substructure (10,000 images)
│   ├── sphere_sub/       # Subhalo/Sphere (10,000 images)
│   └── vortex_sub/       # Vortex (10,000 images)
└── val/
    ├── no_sub/           # No Substructure (2,500 images)
    ├── sphere_sub/       # Subhalo/Sphere (2,500 images)
    └── vortex_sub/       # Vortex (2,500 images)
```

---

## Physical Context

### What is Gravitational Lensing?

Gravitational lensing occurs when massive objects (galaxies, galaxy clusters) bend light from background sources due to spacetime curvature. This creates distorted, magnified images including:

- **Einstein Rings**: Perfect circular alignments
- **Arcs**: Partial rings from slight misalignments
- **Multiple Images**: Same source appearing multiple times

### The Three Classes Explained

#### 1. **No Substructure**
- **Physical Meaning**: Smooth dark matter distribution
- **Visual Appearance**: Clean, symmetric Einstein rings or arcs
- **Characteristics**: Continuous intensity, minimal perturbations
- **Use Case**: Baseline gravitational lens model

#### 2. **Subhalo/Sphere Substructure**
- **Physical Meaning**: Compact dark matter clumps (subhalos) in the lens
- **Visual Appearance**: Localized distortions, asymmetric arcs, brightness anomalies
- **Characteristics**: Point-like or spherical perturbations
- **Use Case**: Tests Cold Dark Matter predictions

#### 3. **Vortex Substructure**
- **Physical Meaning**: Rotating/angular momentum in dark matter
- **Visual Appearance**: Spiral or twisted patterns, tangential distortions
- **Characteristics**: Non-spherical, rotation-like features
- **Use Case**: Alternative dark matter models

### Scientific Importance

Distinguishing these patterns helps:
- Test dark matter theories (Cold vs. Warm Dark Matter)
- Understand dark matter distribution in galaxies
- Validate cosmological simulations
- Discover new physics beyond standard models

---

## Statistical Analysis

### Pixel Intensity Statistics (Training Set)

| Class | Min | Max | Mean | Std Dev |
|-------|-----|-----|------|---------|
| **No Substructure** | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0611 ± 0.0077 | 0.1157 ± 0.0152 |
| **Subhalo/Sphere** | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0626 ± 0.0081 | 0.1179 ± 0.0155 |
| **Vortex** | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0607 ± 0.0080 | 0.1134 ± 0.0146 |

### Key Observations

#### 1. **Normalization Status**
```
All pixels ∈ [0, 1]
```
- ✅ **Already normalized** - no rescaling needed
- Images are very dark (mean ≈ 0.06), typical for astrophysical data
- 94% of pixels are near-zero (background)

#### 2. **Inter-Class Similarity**
```
Maximum mean difference: 0.0626 - 0.0607 = 0.0019 (0.19%)
Maximum std difference: 0.1179 - 0.1134 = 0.0045 (3.8%)
```
- ⚠️ **Classes are statistically very similar**
- Simple thresholding or histogram-based methods will fail
- Requires learning **spatial patterns** rather than intensity distributions

#### 3. **Consistency**
- All images exactly 150×150 pixels
- All single-channel grayscale
- No corrupted or malformed images detected
- No missing data

---

## Class Characteristics

### Feature Complexity Analysis

| Feature Type | No Substructure | Subhalo/Sphere | Vortex |
|--------------|-----------------|----------------|--------|
| **Global Symmetry** | High | Medium | Low |
| **Local Perturbations** | None | Strong | Medium |
| **Spatial Frequency** | Low | Mixed | High |
| **Arc Continuity** | Continuous | Broken | Twisted |
| **Detection Difficulty** | Baseline | Hard | Hard |

### Visual Patterns to Learn

#### No Substructure
```
Expected Features:
- Smooth intensity gradients
- Radial symmetry (if Einstein ring)
- Predictable arc shapes
- Gaussian-like brightness profiles
```

**Model Focus**: Learn smooth, symmetric patterns

#### Subhalo/Sphere Substructure
```
Expected Features:
- Localized brightness spikes
- Arc discontinuities
- Small-scale asymmetries
- Flux ratio anomalies between multiple images
```

**Model Focus**: Detect local anomalies and perturbations

#### Vortex Substructure
```
Expected Features:
- Spiral/twisted arc morphology
- Angular momentum signatures
- Non-radial intensity patterns
- Curvilinear distortions
```

**Model Focus**: Capture global geometric transformations

---

## Model Architecture Recommendations

### Architecture Comparison

| Architecture | Pros | Cons | Recommendation |
|--------------|------|------|----------------|
| **ResNet (34/50)** | Skip connections preserve subtle features, proven on ImageNet | May overfit on small dataset | ⭐ **Highly Recommended** |
| **EfficientNet (B0-B3)** | Excellent parameter efficiency, compound scaling | Slower training | ⭐ **Recommended** |
| **Vision Transformer** | Global attention, good for spatial relationships | Requires more data, computationally expensive | ⚠️ Use with strong augmentation |
| **Custom CNN** | Full control, interpretable | Requires expertise to design | ✅ Good baseline |
| **DenseNet** | Feature reuse, compact | Memory intensive | ✅ Alternative to ResNet |
| **MobileNet** | Lightweight, fast | Lower accuracy ceiling | ❌ Too simple for task |

### Recommended Custom CNN Architecture

```python
# Example: Gravitational Lensing CNN (GLCNN)

Layer Structure:
┌─────────────────────────────────────┐
│ Input: (150, 150, 1)                │
├─────────────────────────────────────┤
│ Conv Block 1: 32 filters (3×3)      │
│ → BatchNorm → ReLU → MaxPool(2×2)   │
│ Output: (75, 75, 32)                │
├─────────────────────────────────────┤
│ Conv Block 2: 64 filters (3×3)      │
│ → BatchNorm → ReLU → MaxPool(2×2)   │
│ Output: (37, 37, 64)                │
├─────────────────────────────────────┤
│ Conv Block 3: 128 filters (3×3)     │
│ → BatchNorm → ReLU → MaxPool(2×2)   │
│ Output: (18, 18, 128)               │
├─────────────────────────────────────┤
│ Conv Block 4: 256 filters (3×3)     │
│ → BatchNorm → ReLU → MaxPool(2×2)   │
│ Output: (9, 9, 256)                 │
├─────────────────────────────────────┤
│ Conv Block 5: 512 filters (3×3)     │
│ → BatchNorm → ReLU → AdaptivePool   │
│ Output: (4, 4, 512)                 │
├─────────────────────────────────────┤
│ Flatten: 8,192 features             │
├─────────────────────────────────────┤
│ Dense: 256 → Dropout(0.5) → ReLU    │
├─────────────────────────────────────┤
│ Dense: 128 → Dropout(0.4) → ReLU    │
├─────────────────────────────────────┤
│ Output: 3 (Softmax)                 │
└─────────────────────────────────────┘

Total Parameters: ~2.3M
Receptive Field: ~127 pixels (covers most image)
```

### Transfer Learning Strategy

```python
# For ResNet/EfficientNet:

1. Load ImageNet pre-trained weights
2. Replace first Conv layer: RGB (3) → Grayscale (1)
   - Option A: Average RGB weights → single channel
   - Option B: Random initialization + freeze rest
3. Replace final FC layer: 1000 classes → 3 classes
4. Training stages:
   - Stage 1 (10-20 epochs): Train only final layers
   - Stage 2 (30-50 epochs): Unfreeze all, low LR
```

---

## Hyperparameter Selection Guide

### 1. Input Parameters

```python
# Fixed by dataset
INPUT_SHAPE = (150, 150, 1)
NUM_CLASSES = 3
PIXEL_RANGE = [0.0, 1.0]  # Already normalized
```

### 2. Preprocessing Options

#### Option A: No Additional Preprocessing
```python
# Use as-is (recommended baseline)
X = images  # Already in [0, 1]
```

#### Option B: Standardization
```python
# Zero-mean, unit variance per image
mean = 0.0615  # Global mean across all classes
std = 0.1157   # Global std

X_standardized = (images - mean) / std
# Range: approximately [-0.5, 8.1]
```

#### Option C: Per-Image Normalization
```python
# Normalize each image independently
X_normalized = (images - images.min()) / (images.max() - images.min())
# Enhances contrast in each individual image
```

**Recommendation**: Start with **Option A**, try **Option B** if training unstable.

### 3. Batch Size Selection

| Batch Size | Training Speed | Generalization | GPU Memory | Recommendation |
|------------|----------------|----------------|------------|----------------|
| 16 | Slow | Best | Low | Small GPU / High regularization |
| 32 | Medium | Good | Medium | ⭐ **Recommended** |
| 64 | Fast | Good | High | Large GPU available |
| 128 | Very Fast | Worse | Very High | Not recommended |

**Formula**:
```
Optimal Batch Size ≈ sqrt(Training Samples) / 10
                   = sqrt(30,000) / 10
                   ≈ 17-34
```

**Choice**: **32** balances speed and generalization

### 4. Learning Rate Strategy

#### Initial Learning Rate
```python
# Rule of thumb: Start at 1e-3 for Adam
INITIAL_LR = 1e-3

# If using SGD with momentum:
INITIAL_LR_SGD = 1e-2
```

#### Learning Rate Schedules

**A. Cosine Annealing** (Recommended)
```python
from tensorflow.keras.optimizers.schedules import CosineDecay

lr_schedule = CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=EPOCHS * STEPS_PER_EPOCH,
    alpha=1e-6  # Minimum LR
)

# Benefit: Smooth decay, helps escape local minima
```

**B. Reduce on Plateau**
```python
from tensorflow.keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,        # LR → LR/2
    patience=5,        # Wait 5 epochs
    min_lr=1e-7
)

# Benefit: Adaptive to training dynamics
```

**C. Step Decay**
```python
# Drop LR by 10x at specific epochs
def step_decay(epoch):
    if epoch < 30: return 1e-3
    if epoch < 60: return 1e-4
    return 1e-5

# Benefit: Simple, interpretable
```

**Recommendation**: Use **Cosine Annealing** with **ReduceLROnPlateau** as backup.

### 5. Optimizer Selection

| Optimizer | Pros | Cons | Use Case |
|-----------|------|------|----------|
| **Adam** | Fast convergence, adaptive LR per parameter | Can overshoot, higher memory | ⭐ Default choice |
| **AdamW** | Adam + weight decay decoupling | Slightly slower | ⭐ If overfitting |
| **SGD + Momentum** | Better generalization, simpler | Requires LR tuning | Long training runs |
| **RMSprop** | Good for RNNs | Outdated for CNNs | ❌ Not recommended |

**Recommended Configuration**:
```python
# Primary: AdamW with weight decay
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=1e-3,
    weight_decay=1e-4,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-7
)

# Alternative: Adam with L2 regularization in layers
optimizer = tf.keras.optimizers.Adam(
    learning_rate=1e-3,
    beta_1=0.9,
    beta_2=0.999
)
```

### 6. Regularization Parameters

#### Dropout
```python
# Network depth dependent:
DROPOUT_RATES = {
    'shallow (< 10 layers)': 0.2,
    'medium (10-30 layers)': 0.3,
    'deep (30-50 layers)': 0.4,
    'very deep (> 50 layers)': 0.5
}

# For gravitational lensing (medium-deep):
DROPOUT = 0.4  # After dense layers
SPATIAL_DROPOUT = 0.2  # After conv layers (optional)
```

#### L2 Regularization
```python
# Apply to convolutional and dense layers
L2_REG = 1e-4  # Standard value

# In Keras:
Conv2D(64, (3,3),
       kernel_regularizer=tf.keras.regularizers.l2(1e-4))
```

#### Batch Normalization
```python
# Use after EVERY Conv layer (before activation)
BATCH_NORM = True
BATCH_NORM_MOMENTUM = 0.99
BATCH_NORM_EPSILON = 1e-3

# Layer order:
# Conv → BatchNorm → Activation → Dropout (optional)
```

### 7. Data Augmentation

#### Astrophysical Symmetries

Gravitational lensing has specific physical invariances:

```python
augmentation_config = {
    # VALID AUGMENTATIONS (preserve physics):
    'rotation_range': 360,        # Space is isotropic
    'horizontal_flip': True,      # Mirror symmetry
    'vertical_flip': True,        # Mirror symmetry
    'zoom_range': 0.1,            # ±10% scale invariance
    'width_shift_range': 0.05,    # Small translations OK
    'height_shift_range': 0.05,

    # INVALID AUGMENTATIONS (break physics):
    'brightness_range': None,     # Already normalized
    'shear_range': None,          # Would distort lensing
    'channel_shift': None,        # Grayscale only
}
```

#### Implementation Example

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rotation_range=360,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.1,
    width_shift_range=0.05,
    height_shift_range=0.05,
    fill_mode='constant',
    cval=0.0  # Fill with black (space background)
)

# Validation: NO augmentation
val_datagen = ImageDataGenerator()
```

#### Advanced: Mixup/Cutout

```python
# Mixup: Blend two images (helps regularization)
def mixup(images, labels, alpha=0.2):
    lambda_ = np.random.beta(alpha, alpha)
    mixed_images = lambda_ * images + (1 - lambda_) * images[::-1]
    mixed_labels = lambda_ * labels + (1 - lambda_) * labels[::-1]
    return mixed_images, mixed_labels

# Cutout: Random masking (simulates occlusion)
# Use with caution - might remove critical features
```

### 8. Training Duration

```python
# Epochs
MAX_EPOCHS = 100-150
EARLY_STOPPING_PATIENCE = 15-20

# Steps per epoch (if using generators)
STEPS_PER_EPOCH = len(train_data) // BATCH_SIZE
VALIDATION_STEPS = len(val_data) // BATCH_SIZE

# Expected training time (RTX 3080):
# - Custom CNN: ~3-5 minutes/epoch → 5-12 hours total
# - ResNet50: ~5-8 minutes/epoch → 8-20 hours total
# - EfficientNetB3: ~8-12 minutes/epoch → 13-30 hours total
```

---

## Training Strategy

### Phase 1: Baseline Establishment (First 10-15 Epochs)

**Goal**: Verify data pipeline and get initial accuracy

```python
config_baseline = {
    'model': 'Simple CNN (3-4 layers)',
    'optimizer': 'Adam',
    'learning_rate': 1e-3,
    'batch_size': 32,
    'augmentation': 'None',
    'regularization': 'Light (dropout=0.2)',
    'epochs': 15
}

# Expected Results:
# - Training accuracy: 50-70%
# - Validation accuracy: 45-65%
# - Should clearly beat random (33%)
```

### Phase 2: Model Selection (Epochs 15-50)

**Goal**: Find best architecture

```python
# Test 3-5 architectures in parallel:
candidates = [
    'Custom CNN (medium depth)',
    'ResNet34',
    'EfficientNetB0',
    'DenseNet121',
    'Vision Transformer (small)'
]

# Compare:
# 1. Validation accuracy
# 2. Training speed
# 3. Overfitting (train-val gap)
# 4. Parameter count

# Expected Results:
# - Best model: 70-85% val accuracy
# - Clear winner should emerge
```

### Phase 3: Hyperparameter Tuning (Epochs 50-100)

**Goal**: Optimize the winning architecture

```python
# Grid search or Bayesian optimization:
search_space = {
    'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3],
    'batch_size': [16, 32, 64],
    'dropout': [0.3, 0.4, 0.5],
    'l2_reg': [1e-5, 1e-4, 1e-3],
    'augmentation_strength': ['light', 'medium', 'heavy']
}

# Expected Results:
# - Best config: 80-92% val accuracy
# - ~2-5% improvement over default
```

### Phase 4: Ensemble & Fine-tuning (Epochs 100-150)

**Goal**: Squeeze final performance

```python
# Strategies:
1. Ensemble top 3-5 models (different seeds)
2. Test-time augmentation (TTA)
3. Fine-tune with very low LR (1e-6)
4. Mix predictions with different augmentations

# Expected Results:
# - Ensemble: +1-3% over single model
# - Final accuracy: 85-95%
```

### Training Monitoring

#### Callbacks to Use

```python
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)

callbacks = [
    # Save best model
    ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),

    # Stop if no improvement
    EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),

    # Reduce LR on plateau
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),

    # Visualize training
    TensorBoard(
        log_dir='./logs',
        histogram_freq=1
    ),

    # Save metrics
    CSVLogger('training_log.csv')
]
```

#### What to Watch For

```python
# GOOD SIGNS:
✅ Val accuracy steadily increasing
✅ Train-val gap < 10%
✅ Loss decreasing smoothly
✅ Confusion matrix improving evenly

# WARNING SIGNS:
⚠️ Val accuracy plateaus early (< 70%)
   → Increase model capacity
⚠️ Train-val gap > 20%
   → Increase regularization/augmentation
⚠️ Loss oscillating wildly
   → Reduce learning rate
⚠️ One class always misclassified
   → Check class balance, add class weights
```

---

## Evaluation Metrics

### Primary Metrics

#### 1. Accuracy
```python
# Suitable because classes are balanced
Accuracy = (TP + TN) / (TP + TN + FP + FN)

Target: > 85% validation accuracy
Excellent: > 90%
```

#### 2. Per-Class Precision & Recall

```python
# Precision: How many predictions were correct?
Precision_i = TP_i / (TP_i + FP_i)

# Recall: How many actual cases were caught?
Recall_i = TP_i / (TP_i + FN_i)

# Goal: Balanced across all 3 classes
# Watch for: One class consistently worse → model bias
```

#### 3. F1-Score

```python
# Harmonic mean of precision & recall
F1_i = 2 * (Precision_i * Recall_i) / (Precision_i + Recall_i)

# Macro-averaged F1 (equal weight per class):
F1_macro = (F1_no_sub + F1_sphere + F1_vortex) / 3

Target: > 0.85 macro F1
```

### Secondary Metrics

#### 4. Confusion Matrix

```
Actual vs Predicted:
                  Predicted
                No_Sub  Sphere  Vortex
Actual  No_Sub    a       b       c
        Sphere    d       e       f
        Vortex    g       h       i

Ideal: High diagonal (a, e, i), low off-diagonal
```

**Interpretation**:
- High `b` (No_Sub → Sphere): Model confuses smooth with subhalo
- High `c` (No_Sub → Vortex): Seeing vortices where there are none
- etc.

#### 5. ROC-AUC (One-vs-Rest)

```python
# For each class i:
# AUC_i = Area Under ROC Curve (class i vs. others)

# Average:
AUC_macro = (AUC_no_sub + AUC_sphere + AUC_vortex) / 3

Target: > 0.92 macro AUC
```

#### 6. Top-2 Accuracy

```python
# Percentage where true class is in top 2 predictions
# Useful for understanding near-misses

Top2_Acc = P(true_class in top_2_predictions)

Expected: 95-98% (higher than top-1)
```

### Diagnostic Metrics

#### 7. Calibration (Expected Calibration Error)

```python
# Does predicted probability match actual accuracy?
# E.g., predictions with 80% confidence should be 80% correct

ECE = Σ |accuracy_in_bin - confidence_in_bin| * fraction_in_bin

Target: < 0.05 (well-calibrated)
```

#### 8. Per-Sample Confidence

```python
# Average confidence on correct predictions
Confidence_correct = mean(max(probabilities) | correct predictions)

# Should be high (> 0.85)

# Average confidence on incorrect predictions
Confidence_incorrect = mean(max(probabilities) | incorrect predictions)

# Should be lower (ideally < 0.60) → model "knows when it's wrong"
```

### Evaluation Code Template

```python
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, classification_report
)

def evaluate_model(model, val_data, val_labels):
    # Predictions
    y_pred_probs = model.predict(val_data)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(val_labels, axis=1)

    # 1. Accuracy
    acc = accuracy_score(y_true, y_pred_classes)

    # 2. Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred_classes, average=None
    )

    # 3. Macro F1
    f1_macro = np.mean(f1)

    # 4. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)

    # 5. ROC-AUC (one-vs-rest)
    auc = roc_auc_score(val_labels, y_pred_probs, average='macro')

    # 6. Classification Report
    report = classification_report(y_true, y_pred_classes,
                                   target_names=['No_Sub', 'Sphere', 'Vortex'])

    # Print results
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print("\nPer-Class Metrics:")
    for i, name in enumerate(['No_Sub', 'Sphere', 'Vortex']):
        print(f"  {name}: P={precision[i]:.3f}, R={recall[i]:.3f}, F1={f1[i]:.3f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nDetailed Report:")
    print(report)

    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'auc': auc,
        'confusion_matrix': cm
    }
```

---

## Expected Challenges

### Challenge 1: Subtle Inter-Class Differences

**Problem**: Mean pixel difference between classes < 0.002

**Impact**:
- Simple models will struggle
- High risk of overfitting to noise
- Requires learning spatial patterns, not intensity

**Solutions**:
1. Use deep architectures (ResNet, EfficientNet)
2. Strong augmentation (rotations, flips, zooms)
3. Regularization (dropout 0.4-0.5, L2 1e-4)
4. Ensemble multiple models
5. Focus on convolutional feature extraction

### Challenge 2: Overfitting Risk

**Problem**: 30,000 samples for complex patterns

**Indicators**:
- Train accuracy > 95%, Val accuracy < 75%
- Training loss → 0, Validation loss increasing
- Model memorizes training samples

**Solutions**:
```python
# Aggressive regularization:
- Dropout: 0.5 in dense layers
- L2 regularization: 1e-4
- Data augmentation: rotation, flip, zoom, shift
- Early stopping: patience=15-20
- Reduce model capacity if gap > 20%

# Augmentation multiplier:
# Effective training samples = 30,000 * augmentation_factor
# With rotation + flips: ~30,000 * 8 = 240,000 variations
```

### Challenge 3: Class Confusion Patterns

**Expected Confusions**:
1. **Vortex → No_Sub**: Subtle vortices look smooth
2. **Sphere → No_Sub**: Small subhalos hard to detect
3. **Sphere ↔ Vortex**: Both are perturbations

**Solutions**:
- Analyze confusion matrix after training
- Collect hard examples (correctly classified with < 60% confidence)
- Create focused augmentation for confused pairs
- Use focal loss to emphasize hard examples

```python
# Focal loss (emphasizes hard examples)
def focal_loss(y_true, y_pred, gamma=2.0):
    pt = tf.reduce_sum(y_true * y_pred, axis=-1)
    focal_weight = (1 - pt) ** gamma
    ce = -tf.math.log(pt + 1e-7)
    return focal_weight * ce
```

### Challenge 4: Computational Constraints

**Resource Requirements**:

| Model | Parameters | GPU Memory | Training Time (100 epochs) |
|-------|-----------|------------|---------------------------|
| Custom CNN | ~2M | 4 GB | 5-8 hours |
| ResNet50 | ~25M | 8 GB | 15-20 hours |
| EfficientNetB3 | ~12M | 6 GB | 20-25 hours |
| Vision Transformer | ~22M | 10 GB | 25-35 hours |

**Solutions**:
- Start with custom CNN baseline
- Use mixed precision training (`tf.keras.mixed_precision`)
- Gradient accumulation for small GPUs
- Use smaller batch sizes (16 instead of 32)

### Challenge 5: Hyperparameter Sensitivity

**Critical Parameters** (in order of importance):
1. **Learning Rate**: ±0.5x can swing accuracy by 10%
2. **Regularization**: Too low → overfit, too high → underfit
3. **Architecture Depth**: Too shallow → underfits, too deep → overfits
4. **Augmentation**: Too weak → overfit, too strong → unrealistic samples

**Solution**: Systematic search
```python
# Priority order:
1. Fix architecture (choose one: ResNet34)
2. Tune LR: [1e-4, 5e-4, 1e-3, 5e-3]
3. Tune regularization: dropout [0.3, 0.4, 0.5]
4. Tune augmentation: zoom [0.05, 0.1, 0.15]
5. Ensemble best 3 configs
```

---

## Recommended Configurations

### Configuration 1: Quick Baseline (Fastest)

**Use Case**: First experiment, limited GPU

```python
config_baseline = {
    # Model
    'architecture': 'Custom CNN',
    'conv_layers': 4,
    'filters': [32, 64, 128, 256],
    'dense_layers': [128],
    'dropout': 0.3,
    'l2_reg': 1e-4,

    # Training
    'optimizer': 'Adam',
    'learning_rate': 1e-3,
    'lr_schedule': 'ReduceLROnPlateau',
    'batch_size': 32,
    'epochs': 50,
    'early_stopping': 10,

    # Data
    'augmentation': {
        'rotation': 180,
        'horizontal_flip': True,
        'vertical_flip': True,
    },
    'preprocessing': 'none',  # Use [0,1] as-is

    # Metrics
    'loss': 'categorical_crossentropy',
    'metrics': ['accuracy', 'f1_score'],
}

# Expected Results:
# - Training time: 3-5 hours
# - Validation accuracy: 70-80%
# - Parameters: ~1.5M
```

### Configuration 2: High Performance (Recommended)

**Use Case**: Best accuracy, reasonable training time

```python
config_high_performance = {
    # Model
    'architecture': 'ResNet34',
    'pretrained': False,  # Train from scratch
    'dropout': 0.4,
    'l2_reg': 1e-4,
    'batch_norm': True,

    # Training
    'optimizer': 'AdamW',
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'lr_schedule': 'CosineAnnealing',
    'batch_size': 32,
    'epochs': 100,
    'early_stopping': 15,

    # Data
    'augmentation': {
        'rotation': 360,
        'horizontal_flip': True,
        'vertical_flip': True,
        'zoom': 0.1,
        'shift': 0.05,
    },
    'preprocessing': 'standardize',  # (x - 0.0615) / 0.1157

    # Metrics
    'loss': 'categorical_crossentropy',
    'metrics': ['accuracy', 'precision', 'recall', 'f1_score'],

    # Advanced
    'mixup_alpha': 0.2,  # Optional mixup
    'label_smoothing': 0.1,  # Regularization
}

# Expected Results:
# - Training time: 12-18 hours
# - Validation accuracy: 85-92%
# - Parameters: ~21M
```

### Configuration 3: Maximum Accuracy (Competition-Grade)

**Use Case**: Best possible performance, long training OK

```python
config_max_accuracy = {
    # Model (Ensemble of 5 models)
    'ensemble': [
        'ResNet50',
        'EfficientNetB3',
        'DenseNet121',
        'Custom CNN (deep)',
        'Vision Transformer (small)'
    ],
    'dropout': 0.5,
    'l2_reg': 1e-4,

    # Training
    'optimizer': 'AdamW',
    'learning_rate': 5e-4,  # Lower for stability
    'weight_decay': 1e-4,
    'lr_schedule': 'CosineAnnealing + Warmup',
    'warmup_epochs': 10,
    'batch_size': 16,  # Smaller for better generalization
    'epochs': 150,
    'early_stopping': 20,

    # Data
    'augmentation': {
        'rotation': 360,
        'horizontal_flip': True,
        'vertical_flip': True,
        'zoom': [0.9, 1.1],
        'shift': 0.1,
        'mixup_alpha': 0.2,
        'cutout_size': 30,  # 30×30 pixel cutouts
    },
    'preprocessing': 'standardize',

    # Metrics
    'loss': 'focal_loss',  # Better for hard examples
    'focal_gamma': 2.0,
    'label_smoothing': 0.1,
    'metrics': ['accuracy', 'precision', 'recall', 'f1_score', 'auc'],

    # Advanced
    'test_time_augmentation': True,  # Average over 10 augmentations
    'ensemble_method': 'weighted_average',
    'cross_validation': 5,  # 5-fold CV for robustness
}

# Expected Results:
# - Training time: 4-7 days (all models + CV)
# - Validation accuracy: 90-95%
# - Parameters: ~80M total (ensemble)
# - Inference time: ~500ms per image (TTA + ensemble)
```

### Configuration 4: Fast Inference (Production)

**Use Case**: Deploy in real-time system, need speed

```python
config_production = {
    # Model (Optimized for speed)
    'architecture': 'EfficientNetB0',  # Best speed/accuracy tradeoff
    'quantization': 'float16',  # Half precision
    'pruning': 0.3,  # Remove 30% of weights
    'dropout': 0.4,
    'l2_reg': 1e-4,

    # Training (same as Config 2)
    'optimizer': 'AdamW',
    'learning_rate': 1e-3,
    'batch_size': 64,  # Larger for throughput
    'epochs': 80,

    # Data
    'augmentation': {
        'rotation': 360,
        'horizontal_flip': True,
        'vertical_flip': True,
        'zoom': 0.1,
    },
    'preprocessing': 'standardize',

    # Optimization
    'tflite_conversion': True,
    'onnx_export': True,

    # Metrics
    'loss': 'categorical_crossentropy',
    'metrics': ['accuracy'],
}

# Expected Results:
# - Training time: 8-12 hours
# - Validation accuracy: 82-88%
# - Inference time: 5-10ms per image (GPU), 50-100ms (CPU)
# - Model size: 15 MB (vs. 45 MB uncompressed)
```

---

## Summary Checklist

### Before Training
- [ ] Verify data loading (all 37,500 images)
- [ ] Check class balance (10k/10k/10k train)
- [ ] Visualize sample images from each class
- [ ] Confirm pixel range [0, 1]
- [ ] Test augmentation pipeline

### During Training
- [ ] Monitor train vs. val accuracy (gap < 15%)
- [ ] Watch learning rate schedule
- [ ] Check confusion matrix every 10 epochs
- [ ] Save best model checkpoints
- [ ] Log metrics to TensorBoard

### After Training
- [ ] Evaluate on validation set (all metrics)
- [ ] Analyze confusion matrix for patterns
- [ ] Test on hard examples (low confidence)
- [ ] Visualize misclassified samples
- [ ] Compare with baseline/other models

### Optimization
- [ ] Try 2-3 different architectures
- [ ] Hyperparameter search (LR, dropout, L2)
- [ ] Test different augmentation strengths
- [ ] Consider ensemble if needed
- [ ] Document final configuration

---

## Additional Resources

### Recommended Reading

1. **Gravitational Lensing Physics**:
   - "Gravitational Lensing: Strong, Weak and Micro" (Schneider, Kochanek, Wambsganss)
   - ArXiv papers on substructure detection

2. **Deep Learning for Astronomy**:
   - "Deep Learning for Cosmology" review papers
   - "Morphological Classification of Galaxies with Deep Learning" (Huertas-Company et al.)

3. **CNN Architectures**:
   - ResNet paper: "Deep Residual Learning for Image Recognition"
   - EfficientNet paper: "EfficientNet: Rethinking Model Scaling for CNNs"

### Tools & Libraries

```python
# Essential
- TensorFlow / Keras: Model building
- NumPy: Data manipulation
- Matplotlib: Visualization
- scikit-learn: Metrics

# Advanced
- TensorBoard: Training monitoring
- Weights & Biases: Experiment tracking
- Optuna: Hyperparameter optimization
- SHAP/Grad-CAM: Model interpretability
```

### Next Steps

1. **Run baseline experiment** (Config 1)
2. **Analyze results** (identify failure modes)
3. **Iterate architecture** (try Config 2)
4. **Hyperparameter tuning** (systematic search)
5. **Ensemble best models** (Config 3 if needed)
6. **Document findings** (report performance)

---

## Contact & Contribution

This analysis is part of the **ML4Sci DeepLense** project for automated gravitational lensing classification.

For questions or improvements to this documentation:
- Open an issue in the project repository
- Refer to the main README.md for project overview
- Check visualization outputs in `visualizations/` directory

---

**Document Version**: 1.0
**Last Updated**: 2026-03-06
**Author**: Automated Dataset Analysis Pipeline
