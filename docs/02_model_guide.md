# Model Architecture Guide - Building a CNN for Image Classification

## 🎯 Learning Objective
Design and implement a Convolutional Neural Network (CNN) to classify gravitational lensing images into 3 classes.

---

## 📚 Background Concepts

### What is a Convolutional Neural Network (CNN)?
A CNN is a deep learning architecture specifically designed for image data. It learns hierarchical features:
- **Early layers**: Detect edges, corners, simple patterns
- **Middle layers**: Detect shapes, textures
- **Deep layers**: Detect complex patterns (e.g., "vortex structure")

### Key Components of a CNN:

#### 1. Convolutional Layer (`nn.Conv2d`)
Applies filters to detect patterns.
```
Input: (batch, channels, height, width)
Output: (batch, num_filters, new_height, new_width)
```

**Parameters:**
- `in_channels`: Number of input channels (1 for grayscale)
- `out_channels`: Number of filters to learn
- `kernel_size`: Size of the filter (e.g., 3x3, 5x5)
- `stride`: How much to move the filter
- `padding`: Add zeros around borders to preserve size

#### 2. Activation Function (ReLU)
Introduces non-linearity: `ReLU(x) = max(0, x)`

#### 3. Pooling Layer (`nn.MaxPool2d`)
Reduces spatial dimensions (downsampling).
```
Input: (batch, channels, H, W)
Output: (batch, channels, H/2, W/2)  # if kernel_size=2
```

#### 4. Batch Normalization (`nn.BatchNorm2d`)
Normalizes activations → faster training, better generalization.

#### 5. Fully Connected Layer (`nn.Linear`)
Combines features for final classification.

#### 6. Dropout (`nn.Dropout`)
Randomly zeros some activations → prevents overfitting.

---

## 🏗️ Model Design Strategies

### Strategy 1: Build from Scratch (Good for Learning)
Design your own architecture to understand how CNNs work.

**Pros:**
- ✅ Full control
- ✅ Learn CNN principles
- ✅ Smaller model size

**Cons:**
- ❌ May require more tuning
- ❌ Might not match pre-trained performance

### Strategy 2: Use Pre-trained Model (Transfer Learning)
Use a model trained on ImageNet and fine-tune it.

**Pros:**
- ✅ Often better performance
- ✅ Faster convergence
- ✅ Less data needed

**Cons:**
- ❌ Larger model size
- ❌ ImageNet features may not perfectly match lensing images

**Popular choices:**
- ResNet (18, 34, 50)
- EfficientNet (B0-B7)
- MobileNet
- DenseNet

---

## 🎨 Designing a Custom CNN - Step by Step

### Step 1: Understand Input/Output Dimensions

**Input:**
- Shape: `(batch_size, 1, 150, 150)`
- 1 channel (grayscale)
- 150x150 pixels

**Output:**
- Shape: `(batch_size, 3)`
- 3 classes (no, sphere, vortex)

**Goal:** Transform (1, 150, 150) → (3,)

---

### Step 2: Design the Architecture

**General pattern:**
```
Input (1, 150, 150)
    ↓
[Conv → ReLU → Pool] × N times  # Feature extraction
    ↓
Flatten
    ↓
[Linear → ReLU → Dropout] × M times  # Classification
    ↓
Linear → Output (3 classes)
```

**Example Architecture:**
```
Layer                   Output Shape           Parameters
================================================================
Input                   (1, 150, 150)          -
Conv2d(1→32, 3x3)       (32, 150, 150)        → 320 params
BatchNorm2d             (32, 150, 150)        → 64 params
ReLU                    (32, 150, 150)        -
MaxPool2d(2x2)          (32, 75, 75)          -

Conv2d(32→64, 3x3)      (64, 75, 75)          → 18,496 params
BatchNorm2d             (64, 75, 75)          → 128 params
ReLU                    (64, 75, 75)          -
MaxPool2d(2x2)          (64, 37, 37)          -

Conv2d(64→128, 3x3)     (128, 37, 37)         → 73,856 params
BatchNorm2d             (128, 37, 37)         → 256 params
ReLU                    (128, 37, 37)         -
MaxPool2d(2x2)          (128, 18, 18)         -

Conv2d(128→256, 3x3)    (256, 18, 18)         → 295,168 params
BatchNorm2d             (256, 18, 18)         → 512 params
ReLU                    (256, 18, 18)         -
MaxPool2d(2x2)          (256, 9, 9)           -

Flatten                 (20736,)               -

Linear(20736→512)       (512,)                 → 10,617,344 params
ReLU                    (512,)                 -
Dropout(0.5)            (512,)                 -

Linear(512→128)         (128,)                 → 65,664 params
ReLU                    (128,)                 -
Dropout(0.3)            (128,)                 -

Linear(128→3)           (3,)                   → 387 params
================================================================
Total: ~11M parameters
```

---

### Step 3: Calculate Output Dimensions

**Formula for Conv2d:**
```
output_size = (input_size + 2*padding - kernel_size) / stride + 1
```

**Formula for MaxPool2d:**
```
output_size = input_size / kernel_size  (assuming stride=kernel_size)
```

**Example:**
```
150x150 → Conv(3x3, pad=1) → 150x150  (padding preserves size)
150x150 → Pool(2x2) → 75x75
75x75 → Conv(3x3, pad=1) → 75x75
75x75 → Pool(2x2) → 37x37
37x37 → Conv(3x3, pad=1) → 37x37
37x37 → Pool(2x2) → 18x18
18x18 → Conv(3x3, pad=1) → 18x18
18x18 → Pool(2x2) → 9x9

Final: 256 channels × 9 × 9 = 20,736 features
```

---

### Step 4: Implement the Model Class

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LensingCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(LensingCNN, self).__init__()

        # TODO: Define convolutional layers
        # Hint: Use nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d

        # TODO: Define fully connected layers
        # Hint: Use nn.Linear, nn.Dropout

    def forward(self, x):
        # TODO: Define forward pass
        # Apply layers in sequence
        # Don't forget activation functions!
        pass
```

**Questions to consider:**
1. How many convolutional blocks do you need?
2. How should filter sizes increase? (32→64→128?)
3. When should you apply BatchNorm and Dropout?
4. What's the final feature map size before flattening?

---

### Step 5: Implementing Forward Pass

**Template:**
```python
def forward(self, x):
    # Block 1
    x = self.conv1(x)
    x = self.bn1(x)
    x = F.relu(x)
    x = self.pool(x)

    # Block 2
    x = self.conv2(x)
    x = self.bn2(x)
    x = F.relu(x)
    x = self.pool(x)

    # ... more blocks ...

    # Flatten
    x = x.view(x.size(0), -1)  # or torch.flatten(x, 1)

    # Fully connected layers
    x = F.relu(self.fc1(x))
    x = self.dropout1(x)
    x = F.relu(self.fc2(x))
    x = self.dropout2(x)
    x = self.fc3(x)  # No activation here! (handled by loss function)

    return x
```

**Important:**
- Don't apply softmax in forward pass if using `nn.CrossEntropyLoss`
- CrossEntropyLoss expects raw logits (unnormalized scores)

---

## 🔄 Transfer Learning Approach

### Using a Pre-trained Model

```python
import torchvision.models as models

# Option 1: ResNet18
model = models.resnet18(pretrained=True)

# Modify first layer for grayscale input
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# Modify last layer for 3 classes
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 3)
```

**Key modifications:**
1. **First layer**: Change from 3 channels (RGB) to 1 (grayscale)
2. **Last layer**: Change from 1000 classes (ImageNet) to 3 classes

### Fine-tuning Strategy

**Strategy A: Fine-tune all layers**
```python
# All parameters are trainable
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

**Strategy B: Freeze early layers, train only later layers**
```python
# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last few layers
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True

# Only train unfrozen parameters
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
```

**When to use which?**
- Freeze if: Small dataset, similar to ImageNet
- Fine-tune all if: Large dataset, different from ImageNet (like ours!)

---

## 🧪 Testing Your Model

### Test 1: Forward Pass
```python
# Create model
model = LensingCNN(num_classes=3)

# Create dummy input
x = torch.randn(8, 1, 150, 150)  # batch_size=8

# Forward pass
output = model(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")  # Should be (8, 3)
print(f"Output values: {output[0]}")     # Should be 3 numbers (logits)
```

**Expected output:**
```
Input shape: torch.Size([8, 1, 150, 150])
Output shape: torch.Size([8, 3])
Output values: tensor([-0.2341,  0.5421, -0.1234], grad_fn=<SelectBackward>)
```

### Test 2: Count Parameters
```python
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total trainable parameters: {count_parameters(model):,}")
```

**Typical sizes:**
- Custom small CNN: 1-5M parameters
- ResNet18: ~11M parameters
- ResNet50: ~25M parameters

### Test 3: Check Output Range
```python
# Before softmax, logits can be any value
output = model(x)
print(f"Logit range: [{output.min():.2f}, {output.max():.2f}]")

# After softmax, should sum to 1
probs = F.softmax(output, dim=1)
print(f"Probabilities: {probs[0]}")
print(f"Sum: {probs[0].sum()}")  # Should be 1.0
```

---

## 🎓 Design Decisions & Trade-offs

### 1. Model Depth (Number of Layers)
- **Shallow (3-4 conv layers)**: Faster, fewer parameters, might underfit
- **Deep (5+ conv layers)**: Slower, more parameters, might overfit
- **Sweet spot**: Start with 4-5 layers, adjust based on results

### 2. Filter Sizes
- **Pattern**: Usually double: 32→64→128→256
- **Why?** As spatial size decreases, increase channels to capture complexity

### 3. Kernel Size
- **3x3**: Most common, efficient
- **5x5 or 7x7**: Larger receptive field, more parameters
- **1x1**: Reduce dimensions without spatial convolution

### 4. Padding
- **padding=1 with kernel=3**: Preserves spatial size
- **padding=0**: Reduces size (avoid for small images)

### 5. Dropout Rate
- **0.3-0.5**: Good starting point
- **Higher (0.6-0.7)**: If overfitting
- **Lower (0.2)**: If underfitting

---

## 🏆 Architecture Recommendations

### Recommendation 1: Start Simple
```python
# 3-4 conv blocks + 2 FC layers
# ~2-5M parameters
# Fast to train, good baseline
```

### Recommendation 2: If Baseline Works Well
```python
# Add more conv blocks
# Add residual connections (like ResNet)
# Increase model capacity
```

### Recommendation 3: If Time is Limited
```python
# Use pretrained ResNet18 or EfficientNet-B0
# Fine-tune on your data
# Often best performance with least effort
```

---

## 🔧 Advanced Techniques (Optional)

### 1. Residual Connections (ResNet-style)
```python
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # Skip connection
        out = F.relu(out)
        return out
```

### 2. Global Average Pooling
Instead of flattening, use average pooling:
```python
# Instead of:
x = x.view(x.size(0), -1)  # Flatten

# Use:
x = F.adaptive_avg_pool2d(x, (1, 1))  # (batch, channels, 1, 1)
x = x.view(x.size(0), -1)  # (batch, channels)
```
**Benefits:** Fewer parameters in FC layer, better generalization

### 3. Attention Mechanisms
Add channel attention or spatial attention to focus on important features.

---

## 📊 Model Comparison

| Model | Parameters | Speed | Accuracy (Expected) | Best For |
|-------|-----------|-------|---------------------|----------|
| Custom Small CNN | 2-5M | Fast | 85-90% | Learning, fast iteration |
| Custom Deep CNN | 10-15M | Medium | 90-93% | Custom control |
| ResNet18 (pretrained) | 11M | Medium | 92-95% | Good balance |
| ResNet50 (pretrained) | 25M | Slow | 93-96% | Max performance |
| EfficientNet-B0 | 5M | Medium | 92-95% | Efficiency |

---

## ✅ Checklist - Is Your Model Ready?

- [ ] Model accepts input shape `(batch, 1, 150, 150)`
- [ ] Model outputs shape `(batch, 3)`
- [ ] Forward pass works without errors
- [ ] Can print model summary with `print(model)`
- [ ] Parameter count is reasonable (< 50M for this task)
- [ ] No activation (softmax) on final layer if using CrossEntropyLoss
- [ ] Model can be moved to GPU: `model.to('cuda')`

---

## 🎓 Learning Exercises

### Exercise 1: Calculate Receptive Field
What's the receptive field after each conv+pool block? (How much of the original image does each pixel "see"?)

### Exercise 2: Visualize Architecture
Use `torchsummary` to visualize your model:
```python
from torchsummary import summary
summary(model, (1, 150, 150))
```

### Exercise 3: Compare Architectures
Build two models with different depths. Compare parameter counts.

### Exercise 4: Implement ResBlock
Add residual connections to your custom CNN.

---

## 🐛 Common Issues & Solutions

### Issue 1: Shape mismatch in forward pass
**Solution:** Print shapes after each layer to find where it breaks

### Issue 2: "RuntimeError: mat1 and mat2 shapes cannot be multiplied"
**Solution:** Wrong input size to first FC layer. Recalculate flattened size.

### Issue 3: Model too large for GPU
**Solution:** Reduce batch size or model depth

### Issue 4: Gradients exploding or vanishing
**Solution:** Use BatchNorm, check learning rate, use gradient clipping

---

## 📝 Summary

**What you learned:**
1. CNN architecture components and their purposes
2. How to design a custom CNN from scratch
3. How to use transfer learning with pre-trained models
4. How to calculate layer dimensions
5. Model testing and debugging strategies

**Next step:** Proceed to [03_train_guide.md](03_train_guide.md) to train your model!

---

## 💡 Pro Tips

1. **Start small**: Get a simple model working first, then increase complexity
2. **Print shapes**: Add `print(x.shape)` after each layer during development
3. **Use BatchNorm**: Almost always helps with convergence
4. **Don't over-engineer**: 4-5 conv layers is often enough
5. **Consider transfer learning**: ResNet18 is a safe, strong baseline

---

## 🔗 Useful Resources

- [PyTorch CNN Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [Torchvision Models](https://pytorch.org/vision/stable/models.html)
- [Understanding Convolutions](https://poloclub.github.io/cnn-explainer/)
- [Receptive Field Calculator](https://fomoro.com/research/article/receptive-field-calculator)

Happy model building! 🏗️
