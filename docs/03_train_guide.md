# Training Guide - Teaching Your Model to Classify Lensing Images

## 🎯 Learning Objective
Implement a complete training loop with validation, checkpointing, and monitoring to train your CNN classifier.

---

## 📚 Background Concepts

### What is Training?
Training is the process of adjusting model weights to minimize the difference between predictions and true labels.

**The Training Loop:**
```
1. Forward pass: Input → Model → Predictions
2. Calculate loss: Compare predictions to true labels
3. Backward pass: Calculate gradients
4. Update weights: Adjust parameters using optimizer
5. Repeat for all batches (1 epoch)
6. Repeat for multiple epochs
```

### Key Components:

#### 1. Loss Function
Measures how wrong the predictions are.
- **CrossEntropyLoss**: Standard for multi-class classification
- Combines LogSoftmax + NLLLoss
- Input: Raw logits (no softmax needed)

#### 2. Optimizer
Updates model weights based on gradients.
- **Adam**: Adaptive learning rate, good default choice
- **SGD**: Simple, requires manual learning rate tuning
- **AdamW**: Adam with weight decay (better generalization)

#### 3. Learning Rate
Controls how much to update weights.
- Too high: Training unstable, doesn't converge
- Too low: Training too slow, might get stuck
- Typical range: 1e-4 to 1e-2

#### 4. Learning Rate Scheduler
Adjusts learning rate during training.
- Start high, gradually decrease
- Helps convergence and avoids overfitting

---

## 🏗️ Building the Training Script - Step by Step

### Step 1: Import Required Libraries

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from tqdm import tqdm  # Progress bar
import matplotlib.pyplot as plt

# Your custom modules
from dataloader import get_dataloaders
from model import LensingCNN  # or your model name
```

---

### Step 2: Set Up Configuration

**Why separate config?** Easy to experiment with different hyperparameters.

```python
# Training configuration
config = {
    'batch_size': 32,
    'num_epochs': 30,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'num_workers': 4,

    # Paths
    'train_dir': 'dataset/train',
    'val_dir': 'dataset/val',
    'checkpoint_dir': 'checkpoints',
    'log_dir': 'logs',

    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    # Augmentation
    'use_augmentation': True,
}
```

**Exercise:** Research what each hyperparameter does.

---

### Step 3: Initialize Model, Loss, and Optimizer

```python
def setup_training(config):
    # Create model
    model = LensingCNN(num_classes=3)
    model = model.to(config['device'])

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )

    return model, criterion, optimizer, scheduler
```

**Questions:**
- Why move model to device?
- What is weight_decay? (Hint: regularization)
- What does ReduceLROnPlateau do?

---

### Step 4: Implement Training Function for One Epoch

```python
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch

    Returns:
        average_loss: Average loss over the epoch
        accuracy: Accuracy on training set
    """
    model.train()  # Set to training mode (enables dropout, batchnorm training)

    running_loss = 0.0
    correct = 0
    total = 0

    # Progress bar
    pbar = tqdm(dataloader, desc='Training')

    for images, labels in pbar:
        # Move to device
        images = images.to(device)
        labels = labels.to(device)

        # TODO: Implement training step
        # 1. Zero gradients
        # 2. Forward pass
        # 3. Calculate loss
        # 4. Backward pass
        # 5. Update weights

        # Update metrics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar
        pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc
```

**Your task:** Fill in the TODO section. Here's the structure:
```python
# 1. Zero gradients
optimizer.zero_grad()

# 2. Forward pass
outputs = model(images)

# 3. Calculate loss
loss = criterion(outputs, labels)

# 4. Backward pass
loss.backward()

# 5. Update weights
optimizer.step()
```

**Why each step?**
1. **zero_grad()**: Clears old gradients (PyTorch accumulates them by default)
2. **Forward pass**: Get predictions
3. **Calculate loss**: How wrong are we?
4. **backward()**: Calculate gradients (∂loss/∂weights)
5. **step()**: Update weights using gradients

---

### Step 5: Implement Validation Function

```python
def validate(model, dataloader, criterion, device):
    """
    Evaluate the model on validation set

    Returns:
        average_loss: Average loss
        accuracy: Accuracy
        all_labels: True labels (for ROC/AUC later)
        all_preds: Predicted probabilities (for ROC/AUC later)
    """
    model.eval()  # Set to evaluation mode (disables dropout, batchnorm uses running stats)

    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []

    # No gradient computation needed for validation
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Get predictions
            probs = torch.softmax(outputs, dim=1)  # Convert logits to probabilities
            _, predicted = torch.max(outputs, 1)

            # Update metrics
            running_loss += loss.item() * images.size(0)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Store for ROC/AUC calculation
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(probs.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc, np.array(all_labels), np.array(all_preds)
```

**Key differences from training:**
- `model.eval()` instead of `model.train()`
- `torch.no_grad()` context (saves memory, faster)
- No `optimizer.zero_grad()` or `loss.backward()`
- Store predictions for evaluation metrics

---

### Step 6: Implement Model Checkpointing

```python
def save_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Checkpoint loaded: {filepath}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Loss: {checkpoint['loss']:.4f}")
    print(f"  Accuracy: {checkpoint['accuracy']:.2f}%")

    return checkpoint['epoch']
```

**Why save checkpoints?**
- Resume training if interrupted
- Save best model based on validation accuracy
- Save models at regular intervals

---

### Step 7: Implement the Main Training Loop

```python
def train(config):
    """Main training function"""

    # Setup
    print("Setting up training...")
    model, criterion, optimizer, scheduler = setup_training(config)

    # Data loaders
    print("Loading data...")
    train_loader, val_loader = get_dataloaders(
        train_dir=config['train_dir'],
        val_dir=config['val_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        use_augmentation=config['use_augmentation']
    )

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }

    best_val_acc = 0.0

    # Training loop
    print(f"\nStarting training on {config['device']}...")
    print(f"Total epochs: {config['num_epochs']}")
    print("="*60)

    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 60)

        start_time = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, config['device']
        )

        # Validate
        val_loss, val_acc, _, _ = validate(
            model, val_loader, criterion, config['device']
        )

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print epoch summary
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc,
                f"{config['checkpoint_dir']}/best_model.pth"
            )
            print(f"  ✓ New best model! (Val Acc: {val_acc:.2f}%)")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc,
                f"{config['checkpoint_dir']}/checkpoint_epoch_{epoch+1}.pth"
            )

    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

    # Plot training curves
    plot_training_curves(history)

    return model, history
```

---

### Step 8: Visualize Training Progress

```python
def plot_training_curves(history):
    """Plot loss and accuracy curves"""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('logs/training_curves.png', dpi=150)
    plt.show()

    print("Training curves saved to logs/training_curves.png")
```

---

## 🎓 Understanding Training Dynamics

### Interpreting Training Curves

#### Scenario 1: Good Training
```
Train Loss: Decreasing steadily
Val Loss:   Decreasing steadily, close to train loss
Train Acc:  Increasing
Val Acc:    Increasing, close to train acc
```
✅ **Model is learning well!**

#### Scenario 2: Overfitting
```
Train Loss: Decreasing
Val Loss:   Increasing after some epoch
Train Acc:  High (95%+)
Val Acc:    Lower than train acc, plateauing
```
⚠️ **Model memorizing training data**
**Solutions:**
- Add more dropout
- Use data augmentation
- Reduce model complexity
- Add weight decay
- Use early stopping

#### Scenario 3: Underfitting
```
Train Loss: High, not decreasing much
Val Loss:   High, similar to train loss
Train Acc:  Low (< 80%)
Val Acc:    Low, similar to train acc
```
⚠️ **Model too simple or learning rate too low**
**Solutions:**
- Increase model capacity (more layers/filters)
- Increase learning rate
- Train for more epochs
- Check if data is loaded correctly

#### Scenario 4: High Variance (Unstable Training)
```
Train Loss: Jumps around erratically
Val Loss:   Jumps around erratically
```
⚠️ **Learning rate too high**
**Solutions:**
- Decrease learning rate
- Use gradient clipping
- Check for data issues

---

## ⚙️ Hyperparameter Tuning

### Learning Rate
**Most important hyperparameter!**

**Finding the right LR:**
1. Start with 1e-3 (0.001)
2. If loss doesn't decrease: Try 1e-4
3. If loss explodes: Try 1e-4 or 1e-5
4. Use learning rate finder (advanced)

**Learning Rate Schedules:**
```python
# Option 1: Reduce on plateau (recommended)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

# Option 2: Cosine annealing
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])

# Option 3: Step decay
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
```

### Batch Size
- **Small (8-16)**: More noise, better generalization, slower
- **Medium (32-64)**: Good balance
- **Large (128+)**: Faster, smoother gradients, might need higher LR

**Rule of thumb:** Largest batch size that fits in GPU memory.

### Number of Epochs
- Monitor validation loss
- Use early stopping if val loss increases for N epochs
- Typical range: 20-50 epochs

---

## 🛠️ Advanced Techniques (Optional)

### 1. Early Stopping
```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
```

### 2. Gradient Clipping
Prevents exploding gradients:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 3. Mixed Precision Training
Faster training, less memory:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop:
with autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 4. Class Weights (if imbalanced dataset)
```python
# If classes are imbalanced (ours are not)
class_weights = torch.tensor([0.8, 1.2, 1.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

---

## 🧪 Testing the Training Script

### Test 1: Overfitting on Small Batch
```python
# Test if model can overfit a single batch (should reach ~100% acc)
small_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
images, labels = next(iter(small_loader))

for i in range(100):
    outputs = model(images.to(device))
    loss = criterion(outputs, labels.to(device))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if i % 20 == 0:
        _, preds = torch.max(outputs, 1)
        acc = (preds == labels.to(device)).float().mean()
        print(f"Step {i}: Loss={loss.item():.4f}, Acc={acc.item():.2f}")
```

**Expected:** Loss → 0, Acc → 1.0
If this doesn't work, there's a bug in your model or training code.

---

## ✅ Checklist - Is Your Training Ready?

- [ ] DataLoader returns batches correctly
- [ ] Model accepts batches and produces correct output shape
- [ ] Loss function is appropriate (CrossEntropyLoss for 3 classes)
- [ ] Optimizer is configured
- [ ] Learning rate scheduler is set up
- [ ] Training loop includes: forward, loss, backward, step
- [ ] Validation loop has `torch.no_grad()` and `model.eval()`
- [ ] Checkpointing works (can save and load)
- [ ] Training curves are plotted
- [ ] GPU is being used if available

---

## 🎓 Learning Exercises

### Exercise 1: Implement Early Stopping
Add the EarlyStopping class to your training loop.

### Exercise 2: Experiment with Learning Rates
Train with LR: 1e-2, 1e-3, 1e-4. Which works best?

### Exercise 3: Compare Optimizers
Try Adam, SGD, and AdamW. Compare convergence speed.

### Exercise 4: Visualize Predictions
During validation, save some misclassified examples to understand errors.

---

## 🐛 Common Issues & Solutions

### Issue 1: Loss is NaN
**Causes:** Learning rate too high, gradient explosion
**Solutions:** Reduce LR, use gradient clipping, check data for NaNs

### Issue 2: Loss not decreasing
**Causes:** LR too low, model too simple, data issue
**Solutions:** Increase LR, check if data is loaded correctly, verify labels

### Issue 3: Training very slow
**Causes:** Large batch size, inefficient data loading
**Solutions:** Increase num_workers, check if GPU is used, reduce batch size

### Issue 4: Out of memory (OOM)
**Causes:** Batch size too large, model too large
**Solutions:** Reduce batch_size, use gradient accumulation, smaller model

### Issue 5: Validation accuracy plateaus early
**Causes:** Model converged, might be underfitting
**Solutions:** Increase model capacity, check learning rate, add regularization

---

## 📝 Summary

**What you learned:**
1. Complete training loop structure
2. Validation and metric tracking
3. Model checkpointing and saving
4. Hyperparameter tuning strategies
5. Interpreting training curves
6. Debugging training issues

**Next step:** Proceed to [04_evaluate_guide.md](04_evaluate_guide.md) to evaluate with ROC/AUC!

---

## 💡 Pro Tips

1. **Always validate**: Never trust training accuracy alone
2. **Save often**: Checkpoint every few epochs, save best model
3. **Monitor curves**: Training curves tell you what's wrong
4. **Start simple**: Get baseline working before adding complexity
5. **Use GPU**: Training on CPU is 10-50x slower
6. **Be patient**: Good models take time to train (20-30 epochs)
7. **Experiment**: Try different architectures, hyperparameters

---

## 🔗 Useful Resources

- [PyTorch Training Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [CS231n: Training Neural Networks](http://cs231n.github.io/neural-networks-3/)
- [Understanding Learning Rate](https://www.jeremyjordan.me/nn-learning-rate/)
- [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)

Happy training! 🚀
