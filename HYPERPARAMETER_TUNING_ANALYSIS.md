# Hyperparameter Tuning Results - Critical Analysis

## Results Summary

### Performance:
```
Baseline XGBoost:  45.89%
Optimized XGBoost: 46.23%
Improvement:       +0.33 percentage points
```

### Training Details:
- **Search time**: 657.1s (~11 minutes)
- **Best CV score**: 45.22%
- **Final validation**: 46.23%
- **Overfitting gap**: 28.38% (74.60% train → 46.23% val)

---

## 🔍 **Key Findings**

### **1. Hyperparameter Tuning BARELY Helped (+0.33%)**

**Why?**
- ✅ The hyperparameter search worked correctly
- ✅ Best params found: `max_depth=4, n_estimators=600, learning_rate=0.1`
- ❌ **Problem is NOT hyperparameters - it's the features themselves**

**Evidence:**
```
Cross-validation score: 45.22%
Validation score:       46.23%
```
These are very close → hyperparameters are well-tuned, but **ceiling is around 46%**.

---

### **2. The REAL Problem: Feature Quality**

Look at the **top features**:

```
1. radial_7              (radial profile bin 7)
2. radial_6              (radial profile bin 6)
3. radial_1              (radial profile bin 1)
4. radial_2              (radial profile bin 2)
5. radial_8              (radial profile bin 8)
6. radial_0              (radial profile bin 0)
7. radial_3              (radial profile bin 3)
8. radial_9              (radial profile bin 9)
```

**8 out of top 15 features are from radial profile!**

**What this tells us:**
- ✅ **Radial brightness pattern** is THE most important signal
- ⚠️ But we're using 50 radial bins → **too granular, creating noise**
- ⚠️ Individual bins (radial_7, radial_6) are not meaningful physics features
- ⚠️ Missing **derived features** from radial profile (slope, curvature, peaks)

---

### **3. Class-Specific Analysis**

```
Class               Precision  Recall   F1-Score
No Substructure     46.66%     50.80%   48.64%   ✓ (best)
Subhalo/Sphere      37.20%     32.72%   34.82%   ✗ (worst - 33% recall!)
Vortex              53.47%     55.16%   54.30%   ✓ (best)
```

**Critical Insight:**
- **Subhalo/Sphere class is being ignored** (only 33% recall)
- Even with `scale_pos_weight=2`, it's still the worst
- This class has the **subtlest differences** from data analysis

**Why class weights didn't help:**
```python
Class weight analysis:
  No Substructure     : 1.000
  Subhalo/Sphere      : 1.000
  Vortex              : 1.000
```
Classes are perfectly balanced (12,500 each) → weights = 1.0 for all!

**The `compute_class_weight('balanced')` did nothing because classes are already balanced.**

---

### **4. Overfitting Analysis**

```
Training:   74.60%
Validation: 46.23%
Gap:        28.38%
```

**This is concerning!**

Despite aggressive regularization:
```python
max_depth = 4           # Very shallow trees
reg_alpha = 1.0         # High L1 regularization
reg_lambda = 2.0        # High L2 regularization
min_child_weight = 7    # High minimum leaf weight
```

**Still 28% overfitting gap.**

**What this means:**
- Model is memorizing **noise** in the training data
- Features contain spurious correlations that don't generalize
- Need **better features**, not just more regularization

---

## 🧠 **What We Learned (Expert Insights)**

### **1. Radial Profile is Key, But We're Using It Wrong**

**Current approach:**
- 50 bins of radial profile
- Model picks individual bins (radial_7, radial_6, etc.)
- These are **noisy point estimates**

**Better approach:**
```python
# Instead of 50 individual bins, extract DERIVED features:

def radial_profile_features(image, center=(75, 75)):
    """
    Extract meaningful physics-based features from radial profile.
    """
    # Get radial profile (keep granular for analysis)
    profile = radial_profile(image, center, num_bins=50)

    # Derived features (more meaningful than individual bins)
    features = {
        # 1. Overall shape
        'radial_peak_value': profile.max(),
        'radial_peak_location': profile.argmax(),
        'radial_decay_rate': -np.polyfit(range(len(profile)), profile, 1)[0],

        # 2. Curvature (2nd derivative)
        'radial_curvature': np.mean(np.diff(profile, n=2)),

        # 3. Brightness distribution
        'radial_central_brightness': profile[:10].mean(),    # Central region
        'radial_middle_brightness': profile[10:30].mean(),   # Middle ring
        'radial_outer_brightness': profile[30:].mean(),      # Outer region
        'radial_contrast': profile[:10].mean() / (profile[30:].mean() + 1e-8),

        # 4. Einstein ring characteristics
        'ring_sharpness': profile.std(),
        'ring_completeness': (profile > 0.5 * profile.max()).sum() / len(profile),

        # 5. Symmetry (compare left/right halves)
        'radial_symmetry': np.corrcoef(profile[:25], profile[25:][::-1])[0, 1],
    }

    return np.array(list(features.values()))
```

**Why this is better:**
- ✅ **Physically meaningful** (decay rate, Einstein ring, curvature)
- ✅ **Less noisy** (aggregated statistics vs individual bins)
- ✅ **Generalizes better** (based on domain knowledge, not data mining)

---

### **2. Missing Critical Features**

Looking at the feature importance, we're missing:

#### **A. Texture Features (Haralick, LBP)**
```python
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

def texture_features(image):
    """
    Gravitational lensing creates texture patterns.
    Subhalo substructure creates local irregularities.
    """
    if image.ndim == 3:
        image = image.squeeze()

    # Convert to uint8 for GLCM
    image_uint8 = (image * 255).astype(np.uint8)

    # Gray-Level Co-occurrence Matrix
    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(image_uint8, distances, angles, levels=256, symmetric=True, normed=True)

    # Haralick features
    contrast = graycoprops(glcm, 'contrast').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    ASM = graycoprops(glcm, 'ASM').mean()

    # Local Binary Patterns
    lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)

    features = np.concatenate([
        [contrast, dissimilarity, homogeneity, energy, correlation, ASM],
        lbp_hist
    ])

    return features  # 16 features
```

**Why this matters:**
- Subhalo/Sphere creates **local texture irregularities**
- Vortex creates **spiral patterns** (different texture)
- Current features don't capture this

---

#### **B. Frequency Domain Features (Fourier Transform)**
```python
def frequency_features(image):
    """
    Different lensing types have different frequency signatures.

    - No substructure: Smooth, low frequency
    - Subhalo: High frequency components (irregularities)
    - Vortex: Specific frequency patterns (spiral)
    """
    if image.ndim == 3:
        image = image.squeeze()

    # 2D FFT
    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)

    # Power spectrum
    power_spectrum = magnitude ** 2

    # Divide into frequency bands
    center = (magnitude.shape[0] // 2, magnitude.shape[1] // 2)
    y, x = np.ogrid[:magnitude.shape[0], :magnitude.shape[1]]
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)

    # Low, mid, high frequency energy
    low_freq = power_spectrum[r < 20].sum()
    mid_freq = power_spectrum[(r >= 20) & (r < 50)].sum()
    high_freq = power_spectrum[r >= 50].sum()

    # Frequency ratios
    total_energy = low_freq + mid_freq + high_freq
    low_ratio = low_freq / total_energy
    mid_ratio = mid_freq / total_energy
    high_ratio = high_freq / total_energy

    # Peak frequency
    peak_freq = r[magnitude.argmax()]

    features = np.array([
        low_freq, mid_freq, high_freq,
        low_ratio, mid_ratio, high_ratio,
        peak_freq,
        magnitude.max(),
        magnitude.mean()
    ])

    return features  # 9 features
```

**Why this matters:**
- Subhalo creates **high-frequency noise** (small-scale perturbations)
- No substructure is **smooth** (low frequency)
- Vortex has **specific frequency patterns**

---

#### **C. Arc-Specific Features**
```python
def arc_features(image, threshold_percentile=95):
    """
    Gravitational lensing creates arc patterns.
    Extract arc geometry.
    """
    if image.ndim == 3:
        image = image.squeeze()

    # Threshold to get bright arc
    threshold = np.percentile(image, threshold_percentile)
    arc_mask = image > threshold

    # Connected components
    from scipy.ndimage import label
    labeled, num_features = label(arc_mask)

    if num_features == 0:
        return np.zeros(8)

    # Find largest component (main arc)
    sizes = [(labeled == i).sum() for i in range(1, num_features + 1)]
    largest_component = np.argmax(sizes) + 1
    main_arc = (labeled == largest_component)

    # Arc geometry
    y_coords, x_coords = np.where(main_arc)

    if len(y_coords) < 5:
        return np.zeros(8)

    # Fit ellipse to arc
    from skimage.measure import EllipseModel
    ellipse = EllipseModel()

    points = np.column_stack([x_coords, y_coords])
    success = ellipse.estimate(points)

    if success:
        xc, yc, a, b, theta = ellipse.params
        eccentricity = np.sqrt(1 - (b/a)**2) if a > 0 else 0
    else:
        xc, yc, a, b, theta, eccentricity = 0, 0, 0, 0, 0, 0

    # Arc statistics
    arc_length = main_arc.sum()
    arc_brightness_mean = image[main_arc].mean()
    arc_brightness_std = image[main_arc].std()

    features = np.array([
        arc_length,
        arc_brightness_mean,
        arc_brightness_std,
        a, b,  # Ellipse semi-axes
        eccentricity,
        theta,
        num_features  # Number of arc fragments
    ])

    return features  # 8 features
```

**Why this matters:**
- No substructure: **Complete, smooth arc**
- Subhalo: **Fragmented arc** (multiple components)
- Vortex: **Spiral/curved arc**

---

## 🎯 **Recommended Next Steps**

### **Option 1: Quick Fix (10 minutes) - Test New Features**

Add just **radial profile derived features** to see immediate improvement:

```python
# Add to feature extraction
def extract_all_features_v2(image):
    """Enhanced feature extraction with radial profile derivatives."""
    feature_list = []

    # Original features
    feature_list.append(radial_profile(image, num_bins=50))  # Keep for now
    feature_list.append(azimuthal_statistics(image, num_sectors=8))
    feature_list.append(quadrant_features(image))
    feature_list.append(edge_features(image))
    feature_list.append(hu_moment_features(image))
    feature_list.append(pixel_statistics(image))
    feature_list.append(spatial_variance_features(image))

    # NEW: Radial profile derived features
    feature_list.append(radial_profile_features(image))  # +11 features

    return np.concatenate(feature_list)

# Expected improvement: +2-3% accuracy
```

---

### **Option 2: Full Re-engineering (1 hour) - Professional Approach**

Replace pixel-level features with **physics-based domain features**:

```python
def extract_all_features_v3(image):
    """
    Physics-based gravitational lensing features.
    Based on domain knowledge, not data mining.
    """
    feature_list = []

    # 1. Radial analysis (11 derived features, not 50 bins)
    feature_list.append(radial_profile_features(image))

    # 2. Arc geometry (8 features)
    feature_list.append(arc_features(image))

    # 3. Texture (16 features - Haralick + LBP)
    feature_list.append(texture_features(image))

    # 4. Frequency domain (9 features)
    feature_list.append(frequency_features(image))

    # 5. Azimuthal symmetry (9 features)
    feature_list.append(azimuthal_statistics(image, num_sectors=8))

    # 6. Edge patterns (5 features)
    feature_list.append(edge_features(image))

    # 7. Shape moments (7 features - Hu moments)
    feature_list.append(hu_moment_features(image))

    # 8. Global statistics (10 features)
    feature_list.append(pixel_statistics(image))

    # Total: ~75 meaningful features (vs 95 noisy features)
    return np.concatenate(feature_list)

# Expected improvement: +5-8% accuracy → 51-54%
```

---

### **Option 3: Ensemble Different Feature Sets (30 minutes)**

Train separate models on different feature groups:

```python
# Model 1: Radial features only
model_radial = XGBClassifier(...)
model_radial.fit(X_train_radial_features, y_train)

# Model 2: Texture features only
model_texture = XGBClassifier(...)
model_texture.fit(X_train_texture_features, y_train)

# Model 3: Frequency features only
model_frequency = XGBClassifier(...)
model_frequency.fit(X_train_frequency_features, y_train)

# Model 4: Arc features only
model_arc = XGBClassifier(...)
model_arc.fit(X_train_arc_features, y_train)

# Ensemble: Weighted voting
ensemble = VotingClassifier([
    ('radial', model_radial),
    ('texture', model_texture),
    ('frequency', model_frequency),
    ('arc', model_arc)
], voting='soft')

# Expected: +3-5% accuracy → 49-51%
```

---

## 📈 **Expected Outcomes**

| Approach | Time | Expected Accuracy | Improvement |
|----------|------|------------------|-------------|
| **Current (baseline)** | - | 46.23% | - |
| **Option 1: Radial derivatives** | 10 min | 48-49% | +2-3% |
| **Option 2: Full re-engineering** | 1 hour | 51-54% | +5-8% |
| **Option 3: Feature ensemble** | 30 min | 49-51% | +3-5% |

---

## 🎓 **Key Takeaway for ML Expertise**

> **"When hyperparameter tuning gives <1% improvement, the problem is NOT the model - it's the features."**

You've proven:
1. ✅ Hyperparameters are well-tuned (11 min search, only +0.33%)
2. ✅ Model is appropriate (XGBoost works well)
3. ❌ **Features are the bottleneck** (too granular, missing physics)

**Next move for ML expert:**
- Don't waste more time on hyperparameters
- **Re-engineer features** based on domain knowledge
- Focus on **radial profile derivatives, texture, and frequency features**

---

## 💡 **My Recommendation**

Given limited time, do **Option 1 (Quick Fix)** NOW:

1. Add `radial_profile_features()` function (5 min)
2. Re-run feature extraction (3 min)
3. Re-run XGBoost with same hyperparams (2 min)
4. **Expected: 48-49% accuracy (+2-3%)**

This shows you:
- ✅ Identified the real problem (features, not hyperparams)
- ✅ Applied domain knowledge (radial derivatives)
- ✅ Got measurable improvement
- ✅ **Look like an ML expert who thinks critically**

Then, if you have more time, do **Option 2** for the full solution.

---

**Which option do you want to implement?**
