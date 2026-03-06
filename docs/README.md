# Gravitational Lensing Multi-Class Classification - Learning Guide

## 📋 Project Overview

This is a **learning-oriented guide** for building a deep learning classifier to identify different types of gravitational lensing images. The project uses PyTorch to classify images into 3 categories:

1. **No substructure** - Regular strong gravitational lensing
2. **Subhalo/Sphere substructure** - Lensing with subhalo patterns
3. **Vortex substructure** - Lensing with vortex patterns

---

## 🎯 Task Requirements

**Objective:** Build a CNN classifier using PyTorch or Keras

**Dataset:**
- 30,000 training images (10,000 per class)
- 7,500 validation images (2,500 per class)
- Format: .npy files (150x150 grayscale, normalized to [0, 1])

**Evaluation Metrics:**
- ROC curves (Receiver Operating Characteristic)
- AUC scores (Area Under the ROC Curve)

---

## 📁 Project Structure

```
DeepLense_ml4sci/
├── dataset/                    # Your data
│   ├── train/
│   │   ├── no/                # 10,000 .npy files
│   │   ├── sphere/            # 10,000 .npy files
│   │   └── vort/              # 10,000 .npy files
│   └── val/
│       ├── no/                # 2,500 .npy files
│       ├── sphere/            # 2,500 .npy files
│       └── vort/              # 2,500 .npy files
│
├── src/                       # Your source code
│   ├── dataloader.py          # Data loading and augmentation
│   ├── model.py               # CNN architecture
│   ├── train.py               # Training loop
│   ├── evaluate.py            # ROC/AUC evaluation
│   └── utils/
│       └── convert_npy_to_png.py  # Visualization utility
│
├── docs/                      # Learning guides (START HERE!)
│   ├── README.md              # This file
│   ├── 01_dataloader_guide.md
│   ├── 02_model_guide.md
│   ├── 03_train_guide.md
│   ├── 04_evaluate_guide.md
│   └── 05_convert_npy_to_png_guide.md
│
├── checkpoints/               # Saved models
├── logs/                      # Training logs
└── visualizations/            # Plots and figures
```

---

## 🚀 Getting Started

### Step 1: Read the Learning Guides

Each guide is designed to teach you concepts step-by-step. Read them in order:

1. **[01_dataloader_guide.md](01_dataloader_guide.md)**
   - Understanding PyTorch Dataset and DataLoader
   - Loading .npy files
   - Data augmentation strategies
   - **Estimated time:** 1-2 hours

2. **[02_model_guide.md](02_model_guide.md)**
   - CNN architecture design
   - Building custom models vs. transfer learning
   - Layer dimensions and parameter calculations
   - **Estimated time:** 2-3 hours

3. **[03_train_guide.md](03_train_guide.md)**
   - Complete training loop
   - Loss functions and optimizers
   - Hyperparameter tuning
   - Interpreting training curves
   - **Estimated time:** 2-3 hours

4. **[04_evaluate_guide.md](04_evaluate_guide.md)**
   - ROC curves and AUC scores
   - Multi-class evaluation (One-vs-Rest)
   - Confusion matrices
   - **Estimated time:** 1-2 hours

5. **[05_convert_npy_to_png_guide.md](05_convert_npy_to_png_guide.md)**
   - Visualizing .npy data
   - Data exploration utilities
   - **Estimated time:** 30 minutes

---

## 🛠️ Implementation Workflow

### Phase 1: Data Exploration (Day 1)
```
1. Convert some .npy files to PNG (Guide 5)
2. Visualize samples from each class
3. Understand data characteristics
```

### Phase 2: Data Pipeline (Day 1-2)
```
1. Implement Dataset class (Guide 1)
2. Implement DataLoader
3. Test with small batch
4. Add data augmentation (optional)
```

### Phase 3: Model Building (Day 2-3)
```
1. Design CNN architecture (Guide 2)
2. Test forward pass with dummy data
3. Count parameters
4. OR: Use pretrained model (ResNet18)
```

### Phase 4: Training (Day 3-4)
```
1. Implement training loop (Guide 3)
2. Add validation
3. Implement checkpointing
4. Train for 20-30 epochs
5. Monitor training curves
```

### Phase 5: Evaluation (Day 4-5)
```
1. Load best model (Guide 4)
2. Calculate ROC curves and AUC
3. Create visualizations
4. Analyze results
```

---

## 📚 Learning Philosophy

### This is NOT a Copy-Paste Project

Each guide is designed to:
- ✅ Explain **WHY** not just **HOW**
- ✅ Provide **conceptual understanding**
- ✅ Include **exercises** for practice
- ✅ Show **common mistakes** and solutions
- ✅ Encourage **experimentation**

### How to Use These Guides

1. **Read carefully**: Don't skip the explanations
2. **Type code yourself**: Don't copy-paste
3. **Experiment**: Try different approaches
4. **Debug**: Understand errors when they happen
5. **Ask questions**: Research concepts you don't understand

### Expected Learning Outcomes

By the end of this project, you should understand:
- How PyTorch data pipelines work
- CNN architecture design principles
- Training loop mechanics
- Evaluation metrics for classification
- Debugging ML models
- Hyperparameter tuning strategies

---

## 📊 Expected Results

### Good Performance Targets:
- **Overall Accuracy:** 90-95%
- **AUC Scores:** 0.95-0.99 per class
- **Training Time:** 30-60 minutes (with GPU)

### If Results Are Poor:
- Check data loading (are images correct?)
- Verify model architecture (forward pass works?)
- Check training curves (overfitting? underfitting?)
- Adjust hyperparameters (learning rate, batch size)
- Train for more epochs

---

## 🔧 Technical Requirements

### Required Libraries:
```bash
pip install torch torchvision
pip install numpy pillow matplotlib
pip install scikit-learn
pip install tqdm
pip install albumentations  # For augmentation (optional)
```

### Hardware:
- **Minimum:** CPU only (slow, ~2-3 hours training)
- **Recommended:** GPU (CUDA) (fast, ~30-60 min training)
- **RAM:** 8GB minimum, 16GB recommended

---

## 💡 Tips for Success

### 1. Start Small
- Test with small batch (8-16 samples) first
- Use simple model initially
- Get everything working before optimizing

### 2. Debug Methodically
- Print shapes everywhere
- Test each component independently
- Use small dataset subset for debugging

### 3. Monitor Training
- Watch training curves
- Save checkpoints frequently
- Don't over-train

### 4. Experiment
- Try different architectures
- Test various hyperparameters
- Compare augmentation strategies

### 5. Document Your Work
- Keep notes on what works/doesn't work
- Save training logs
- Record final results

---

## 🎓 Learning Checkpoints

### After DataLoader (Guide 1):
- [ ] Can load .npy files correctly
- [ ] Can create batches of images
- [ ] Understand shape transformations
- [ ] Can apply augmentations

### After Model (Guide 2):
- [ ] Understand CNN components
- [ ] Can design custom architecture
- [ ] Can calculate layer dimensions
- [ ] Model accepts correct input shape

### After Training (Guide 3):
- [ ] Understand training loop mechanics
- [ ] Can interpret training curves
- [ ] Can tune hyperparameters
- [ ] Model trains without errors

### After Evaluation (Guide 4):
- [ ] Understand ROC/AUC metrics
- [ ] Can calculate multi-class ROC
- [ ] Can interpret confusion matrix
- [ ] Results meet requirements (AUC > 0.9)

---

## 🐛 Troubleshooting

### Common Issues:

**Problem:** Can't load .npy files
```python
Solution: Check file path, use os.path.join()
```

**Problem:** Shape mismatch errors
```python
Solution: Print shapes at each step, check transformations
```

**Problem:** Training too slow
```python
Solution: Use GPU, increase num_workers, reduce batch size
```

**Problem:** Loss not decreasing
```python
Solution: Check learning rate, verify data loading, check labels
```

**Problem:** Overfitting
```python
Solution: Add dropout, use augmentation, reduce model complexity
```

---

## 📈 Next Steps After Completion

### Improve Performance:
1. Try ensemble methods (combine multiple models)
2. Implement cross-validation
3. Use more advanced augmentations
4. Try different architectures (EfficientNet, DenseNet)

### Expand Capabilities:
1. Add class activation maps (CAM) for interpretability
2. Implement uncertainty estimation
3. Create web demo with Gradio/Streamlit
4. Deploy model as API

### Learn More:
1. Study attention mechanisms
2. Learn about Vision Transformers
3. Explore semi-supervised learning
4. Research domain adaptation

---

## 📚 Additional Resources

### PyTorch:
- [Official PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

### Deep Learning:
- [CS231n: CNNs for Visual Recognition](http://cs231n.stanford.edu/)
- [Deep Learning Book](https://www.deeplearningbook.org/)
- [Fast.ai Course](https://course.fast.ai/)

### Computer Vision:
- [PyImageSearch](https://www.pyimagesearch.com/)
- [Papers with Code](https://paperswithcode.com/)

### Evaluation Metrics:
- [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Understanding ROC Curves](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)

---

## 🤝 Contributing & Feedback

This is a learning project. If you:
- Find errors or unclear explanations
- Have suggestions for improvements
- Want to share your results
- Need help with specific issues

Feel free to reach out or create an issue!

---

## 📝 Summary

**This project teaches you:**
1. End-to-end deep learning pipeline
2. PyTorch fundamentals
3. CNN architecture design
4. Training and evaluation best practices
5. ROC/AUC metrics for classification

**Remember:**
- Learning takes time - don't rush
- Errors are part of the process
- Understanding > getting it working quickly
- Experiment and explore

**Good luck with your learning journey! 🚀**

---

## 🏁 Quick Start Checklist

Before you begin, ensure:
- [ ] Dataset downloaded and extracted
- [ ] Python 3.7+ installed
- [ ] PyTorch installed (with CUDA if available)
- [ ] Required libraries installed
- [ ] Read this README completely
- [ ] Ready to start with Guide 1

**Start here:** [01_dataloader_guide.md](01_dataloader_guide.md)
