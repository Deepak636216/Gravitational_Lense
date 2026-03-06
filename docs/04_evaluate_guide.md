# Evaluation Guide - ROC Curves and AUC Scores

## 🎯 Learning Objective
Evaluate your trained model using ROC curves and AUC scores - the required metrics for this task.

---

## 📚 Background Concepts

### What is ROC Curve?
**ROC = Receiver Operating Characteristic**

A ROC curve plots:
- **X-axis**: False Positive Rate (FPR)
- **Y-axis**: True Positive Rate (TPR)

It shows the trade-off between detecting positive cases and avoiding false alarms.

### What is AUC?
**AUC = Area Under the ROC Curve**

- **AUC = 1.0**: Perfect classifier
- **AUC = 0.9-1.0**: Excellent
- **AUC = 0.8-0.9**: Good
- **AUC = 0.7-0.8**: Fair
- **AUC = 0.5**: Random guessing (coin flip)
- **AUC < 0.5**: Worse than random (model is confused)

---

## 🤔 Multi-Class ROC: One-vs-Rest Strategy

**Problem:** ROC curves are designed for binary classification, but we have 3 classes!

**Solution:** One-vs-Rest (OvR) approach
- Treat each class vs. all others as a binary problem
- Create 3 separate ROC curves:
  1. "no" vs. "sphere + vortex"
  2. "sphere" vs. "no + vortex"
  3. "vortex" vs. "no + sphere"

**Example:**
```
True label: "sphere" (class 1)
Predicted probabilities: [0.1, 0.8, 0.1]  # [no, sphere, vortex]

For ROC of class 1 (sphere):
- This is a POSITIVE example (true label = 1)
- Confidence score = 0.8
- For other classes, this is a NEGATIVE example
```

---

## 📊 Key Metrics Definitions

### Confusion Matrix Terms:
```
                Predicted Positive    Predicted Negative
Actual Positive    True Positive (TP)    False Negative (FN)
Actual Negative    False Positive (FP)   True Negative (TN)
```

### True Positive Rate (TPR) / Recall / Sensitivity:
```
TPR = TP / (TP + FN)
```
"Of all actual positives, how many did we correctly identify?"

### False Positive Rate (FPR):
```
FPR = FP / (FP + TN)
```
"Of all actual negatives, how many did we incorrectly call positive?"

### Precision:
```
Precision = TP / (TP + FP)
```
"Of all predicted positives, how many were actually positive?"

### F1-Score:
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
Harmonic mean of precision and recall.

---

## 🏗️ Building the Evaluation Script - Step by Step

### Step 1: Import Required Libraries

```python
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
from tqdm import tqdm

# Your custom modules
from dataloader import get_dataloaders
from model import LensingCNN
```

**Why sklearn?**
- Industry-standard metrics library
- Well-tested implementations
- Easy to use

---

### Step 2: Load Trained Model

```python
def load_trained_model(checkpoint_path, model, device):
    """
    Load trained model from checkpoint

    Args:
        checkpoint_path: Path to saved checkpoint
        model: Model instance
        device: 'cuda' or 'cpu'

    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Validation Accuracy: {checkpoint['accuracy']:.2f}%")

    return model
```

---

### Step 3: Get Predictions on Validation Set

```python
def get_predictions(model, dataloader, device):
    """
    Get predictions for all samples in dataloader

    Returns:
        true_labels: Ground truth labels (numpy array)
        pred_probs: Predicted probabilities for each class (numpy array)
        pred_classes: Predicted class labels (numpy array)
    """
    model.eval()

    all_labels = []
    all_probs = []

    print("Getting predictions...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)

            # Forward pass
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)  # Convert logits to probabilities

            # Store results
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Convert to numpy arrays
    true_labels = np.array(all_labels)
    pred_probs = np.array(all_probs)  # Shape: (num_samples, 3)
    pred_classes = np.argmax(pred_probs, axis=1)

    return true_labels, pred_probs, pred_classes
```

**Key points:**
- `F.softmax()` converts logits to probabilities (sum to 1)
- `pred_probs` has shape `(num_samples, 3)` - probability for each class
- `pred_classes` is the class with highest probability

---

### Step 4: Calculate Per-Class Metrics

```python
def calculate_metrics(true_labels, pred_probs, pred_classes, class_names):
    """
    Calculate accuracy, precision, recall, F1 for each class

    Args:
        true_labels: Ground truth labels
        pred_probs: Predicted probabilities (N, 3)
        pred_classes: Predicted classes
        class_names: List of class names ['no', 'sphere', 'vortex']

    Returns:
        Dictionary with metrics
    """
    # Overall accuracy
    accuracy = np.mean(pred_classes == true_labels) * 100

    # Classification report (precision, recall, F1 per class)
    report = classification_report(
        true_labels,
        pred_classes,
        target_names=class_names,
        digits=4
    )

    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(report)

    print(f"\nOverall Accuracy: {accuracy:.2f}%")

    return {
        'accuracy': accuracy,
        'report': report
    }
```

---

### Step 5: Calculate ROC Curves and AUC Scores

This is the main requirement of your task!

```python
def calculate_roc_auc(true_labels, pred_probs, class_names):
    """
    Calculate ROC curves and AUC scores for each class (One-vs-Rest)

    Args:
        true_labels: Ground truth labels (N,)
        pred_probs: Predicted probabilities (N, 3)
        class_names: List of class names

    Returns:
        Dictionary with ROC data for each class
    """
    n_classes = len(class_names)
    roc_data = {}

    print("\n" + "="*60)
    print("ROC-AUC SCORES (One-vs-Rest)")
    print("="*60)

    for i in range(n_classes):
        # Binarize: class i vs. rest
        binary_labels = (true_labels == i).astype(int)
        class_probs = pred_probs[:, i]

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(binary_labels, class_probs)

        # Calculate AUC
        roc_auc = auc(fpr, tpr)

        # Store results
        roc_data[class_names[i]] = {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': roc_auc
        }

        print(f"{class_names[i]:10s} - AUC: {roc_auc:.4f}")

    # Micro-average (aggregate all classes)
    # Flatten labels and predictions
    true_binary = np.eye(n_classes)[true_labels].ravel()
    pred_binary = pred_probs.ravel()
    fpr_micro, tpr_micro, _ = roc_curve(true_binary, pred_binary)
    roc_auc_micro = auc(fpr_micro, tpr_micro)

    roc_data['micro'] = {
        'fpr': fpr_micro,
        'tpr': tpr_micro,
        'auc': roc_auc_micro
    }

    print(f"{'Micro-avg':10s} - AUC: {roc_auc_micro:.4f}")

    # Macro-average (mean of per-class AUCs)
    macro_auc = np.mean([roc_data[cls]['auc'] for cls in class_names])
    print(f"{'Macro-avg':10s} - AUC: {macro_auc:.4f}")

    return roc_data
```

**Understanding the code:**
1. For each class, treat it as positive, others as negative
2. Use probability of that class as confidence score
3. Calculate FPR and TPR at different thresholds
4. Calculate area under the curve

---

### Step 6: Plot ROC Curves

```python
def plot_roc_curves(roc_data, class_names, save_path='visualizations/roc_curves.png'):
    """
    Plot ROC curves for all classes

    Args:
        roc_data: Dictionary with ROC data from calculate_roc_auc()
        class_names: List of class names
        save_path: Where to save the plot
    """
    plt.figure(figsize=(10, 8))

    # Plot ROC curve for each class
    colors = ['blue', 'red', 'green']
    for i, class_name in enumerate(class_names):
        plt.plot(
            roc_data[class_name]['fpr'],
            roc_data[class_name]['tpr'],
            color=colors[i],
            lw=2,
            label=f'{class_name} (AUC = {roc_data[class_name]["auc"]:.4f})'
        )

    # Plot micro-average
    plt.plot(
        roc_data['micro']['fpr'],
        roc_data['micro']['tpr'],
        color='darkorange',
        linestyle='--',
        lw=2,
        label=f'Micro-average (AUC = {roc_data["micro"]["auc"]:.4f})'
    )

    # Plot diagonal (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier (AUC = 0.50)')

    # Formatting
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Multi-Class (One-vs-Rest)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nROC curves saved to {save_path}")
```

**What makes a good ROC curve?**
- Curves close to top-left corner (high TPR, low FPR)
- Large area under curve (high AUC)
- Far from diagonal line (better than random)

---

### Step 7: Plot Confusion Matrix

```python
def plot_confusion_matrix(true_labels, pred_classes, class_names,
                         save_path='visualizations/confusion_matrix.png'):
    """
    Plot confusion matrix

    Args:
        true_labels: Ground truth labels
        pred_classes: Predicted classes
        class_names: List of class names
        save_path: Where to save the plot
    """
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, pred_classes)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', ax=ax, values_format='d')

    plt.title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Confusion matrix saved to {save_path}")

    # Print per-class accuracy
    print("\nPer-class Accuracy:")
    for i, class_name in enumerate(class_names):
        class_acc = cm[i, i] / cm[i].sum() * 100
        print(f"  {class_name:10s}: {class_acc:.2f}%")
```

**Reading confusion matrix:**
```
              Predicted
              no  sph vrt
Actual  no   [A   B   C]
        sph  [D   E   F]
        vrt  [G   H   I]
```
- Diagonal (A, E, I): Correct predictions
- Off-diagonal: Misclassifications
- Goal: Large diagonal values, small off-diagonal

---

### Step 8: Main Evaluation Function

```python
def evaluate_model(config):
    """
    Complete evaluation pipeline

    Args:
        config: Dictionary with configuration
    """
    print("="*60)
    print("MODEL EVALUATION")
    print("="*60)

    device = config['device']
    class_names = ['no', 'sphere', 'vortex']

    # Load model
    model = LensingCNN(num_classes=3)
    model = load_trained_model(config['checkpoint_path'], model, device)

    # Load validation data
    _, val_loader = get_dataloaders(
        train_dir=config['train_dir'],
        val_dir=config['val_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        use_augmentation=False  # No augmentation for evaluation
    )

    # Get predictions
    true_labels, pred_probs, pred_classes = get_predictions(model, val_loader, device)

    # Calculate metrics
    metrics = calculate_metrics(true_labels, pred_probs, pred_classes, class_names)

    # Calculate ROC-AUC
    roc_data = calculate_roc_auc(true_labels, pred_probs, class_names)

    # Plot ROC curves
    plot_roc_curves(roc_data, class_names)

    # Plot confusion matrix
    plot_confusion_matrix(true_labels, pred_classes, class_names)

    # Save results
    results = {
        'accuracy': metrics['accuracy'],
        'roc_data': roc_data,
        'classification_report': metrics['report']
    }

    # Save to file
    import json
    with open('logs/evaluation_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_roc_data = {}
        for cls in roc_data:
            json_roc_data[cls] = {
                'auc': float(roc_data[cls]['auc']),
                'fpr': roc_data[cls]['fpr'].tolist(),
                'tpr': roc_data[cls]['tpr'].tolist()
            }
        json.dump({'accuracy': metrics['accuracy'], 'roc_auc': json_roc_data}, f, indent=2)

    print("\n" + "="*60)
    print("Evaluation completed!")
    print("Results saved to logs/evaluation_results.json")
    print("="*60)

    return results


if __name__ == "__main__":
    config = {
        'checkpoint_path': 'checkpoints/best_model.pth',
        'train_dir': 'dataset/train',
        'val_dir': 'dataset/val',
        'batch_size': 32,
        'num_workers': 4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    results = evaluate_model(config)
```

---

## 📊 Interpreting Results

### Good Results (Expected for this task):

**AUC Scores:**
```
no       - AUC: 0.95-0.99
sphere   - AUC: 0.95-0.99
vortex   - AUC: 0.95-0.99
Micro-avg - AUC: 0.95-0.99
```

**Overall Accuracy:** 90-95%+

**Confusion Matrix:**
- Most values on diagonal
- Few misclassifications

### What if results are poor?

**AUC < 0.8:**
- Model didn't train well
- Check training curves
- Try different architecture
- Train for more epochs

**One class has low AUC:**
- That class is harder to distinguish
- May need more training data
- May need better features

**All classes similar AUC but low:**
- Model too simple
- Learning rate issues
- Data loading problems

---

## 🎓 Advanced Analysis (Optional)

### 1. Per-Sample Analysis
```python
def analyze_misclassifications(model, dataloader, device, num_samples=10):
    """Find and visualize worst predictions"""
    model.eval()

    mistakes = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            # Find mistakes
            wrong_idx = (preds != labels.to(device))
            if wrong_idx.sum() > 0:
                for i in range(len(images)):
                    if wrong_idx[i]:
                        mistakes.append({
                            'image': images[i].cpu(),
                            'true': labels[i].item(),
                            'pred': preds[i].cpu().item(),
                            'confidence': probs[i].max().cpu().item()
                        })

    # Visualize top mistakes
    mistakes = sorted(mistakes, key=lambda x: x['confidence'], reverse=True)[:num_samples]

    # TODO: Plot these images with true vs predicted labels
```

### 2. Threshold Analysis
```python
def find_optimal_threshold(true_labels, pred_probs, class_idx):
    """Find threshold that maximizes F1 score"""
    binary_labels = (true_labels == class_idx).astype(int)
    class_probs = pred_probs[:, class_idx]

    fpr, tpr, thresholds = roc_curve(binary_labels, class_probs)

    # Calculate F1 for each threshold
    f1_scores = []
    for threshold in thresholds:
        preds = (class_probs >= threshold).astype(int)
        precision = np.sum((preds == 1) & (binary_labels == 1)) / np.sum(preds == 1)
        recall = np.sum((preds == 1) & (binary_labels == 1)) / np.sum(binary_labels == 1)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)

    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]
```

### 3. Calibration Plot
Check if predicted probabilities match actual frequencies:
```python
from sklearn.calibration import calibration_curve

def plot_calibration_curve(true_labels, pred_probs, class_names):
    """Plot calibration curve for each class"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, class_name in enumerate(class_names):
        binary_labels = (true_labels == i).astype(int)
        class_probs = pred_probs[:, i]

        fraction_of_positives, mean_predicted_value = calibration_curve(
            binary_labels, class_probs, n_bins=10
        )

        axes[i].plot(mean_predicted_value, fraction_of_positives, "s-")
        axes[i].plot([0, 1], [0, 1], "k--")
        axes[i].set_xlabel("Predicted Probability")
        axes[i].set_ylabel("True Frequency")
        axes[i].set_title(f"Class: {class_name}")

    plt.tight_layout()
    plt.show()
```

---

## ✅ Checklist - Evaluation Complete?

- [ ] Model loaded successfully from checkpoint
- [ ] Predictions obtained on validation set
- [ ] Overall accuracy calculated
- [ ] Per-class precision, recall, F1 calculated
- [ ] ROC curve calculated for each class (One-vs-Rest)
- [ ] AUC scores calculated for each class
- [ ] ROC curves plotted and saved
- [ ] Confusion matrix plotted and saved
- [ ] Results saved to file
- [ ] AUC scores are > 0.8 (ideally > 0.9)

---

## 🎓 Learning Exercises

### Exercise 1: Understand ROC
Plot ROC curves with different classifiers (random, perfect) to understand the curve.

### Exercise 2: Manual ROC Calculation
Implement ROC calculation from scratch (without sklearn) to understand the algorithm.

### Exercise 3: Cost-Sensitive Analysis
If misclassifying "vortex" as "no" is worse than the reverse, how would you adjust your threshold?

### Exercise 4: Bootstrap Confidence Intervals
Calculate confidence intervals for AUC scores using bootstrapping.

---

## 🐛 Common Issues & Solutions

### Issue 1: AUC = 0.5 (Random Performance)
**Causes:** Model not trained, wrong checkpoint loaded, labels incorrect
**Solutions:** Verify model is trained, check checkpoint path, verify label mapping

### Issue 2: Predictions all same class
**Causes:** Severe overfitting or underfitting
**Solutions:** Check training curves, retrain model

### Issue 3: "ValueError: Only one class present"
**Causes:** Batch contains only one class
**Solutions:** Use full validation set, check data loading

### Issue 4: Inconsistent results
**Causes:** Model in training mode, random augmentation applied
**Solutions:** Use `model.eval()`, disable augmentation for evaluation

---

## 📝 Summary

**What you learned:**
1. How ROC curves work and what they measure
2. AUC score interpretation
3. One-vs-Rest strategy for multi-class ROC
4. Complete evaluation pipeline
5. Visualizing and interpreting results
6. Advanced metrics and analysis

**Task Requirements Met:**
✅ ROC curves for each class
✅ AUC scores calculated
✅ Visualizations saved
✅ Results documented

---

## 💡 Pro Tips

1. **Always use validation set**: Never evaluate on training data
2. **One-vs-Rest is standard**: For multi-class ROC/AUC
3. **Visualize confusion matrix**: Quickly see which classes are confused
4. **Report all metrics**: Accuracy, precision, recall, F1, AUC
5. **Compare to baseline**: Random classifier (AUC=0.5) or simple model
6. **Save everything**: Plots, metrics, predictions for later analysis

---

## 🔗 Useful Resources

- [Scikit-learn ROC Tutorial](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html)
- [Understanding ROC Curves](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)
- [Interpreting Confusion Matrix](https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/)
- [Multi-class ROC](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#multiclass-settings)

Congratulations on completing the evaluation! 🎉
