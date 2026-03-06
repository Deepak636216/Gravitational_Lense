# Gravitational Lensing Multi-Class Classification

Deep learning project for classifying gravitational lensing images into three categories: no substructure, subhalo/sphere substructure, and vortex substructure.

## 📋 Project Overview

This project implements and compares multiple machine learning approaches for gravitational lensing image classification:

- **Traditional ML**: Logistic Regression, Random Forest, SVM, XGBoost
- **Deep Learning**: Custom CNN architecture
- **Transfer Learning**: Fine-tuned pretrained models (ResNet18, EfficientNet)

## 🎯 Task

Build a classifier to identify different types of gravitational lensing images:
1. **No substructure** - Regular strong gravitational lensing
2. **Subhalo/Sphere substructure** - Lensing with subhalo patterns
3. **Vortex substructure** - Lensing with vortex patterns

## 📊 Dataset

- **Training**: 30,000 images (10,000 per class)
- **Validation**: 7,500 images (2,500 per class)
- **Format**: `.npy` files (150×150 grayscale, normalized to [0, 1])
- **Balanced**: Equal samples per class

## 📁 Project Structure

```
DeepLense_ml4sci/
├── src/                       # Source code
│   ├── explore_data.py        # Data exploration and visualization
│   ├── dataloader.py          # Data loading pipeline
│   ├── models/                # Model architectures
│   ├── train.py               # Training scripts
│   └── evaluate.py            # Evaluation and metrics
├── docs/                      # Documentation and guides
├── visualizations/            # Generated plots and figures
├── checkpoints/               # Saved model weights
├── logs/                      # Training logs
└── requirements.txt           # Python dependencies
```

## 🚀 Getting Started

### Installation

```bash
# Clone repository
git clone https://github.com/Deepak636216/Gravitational_Lense.git
cd Gravitational_Lense

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Exploration

```bash
python src/explore_data.py
```

This will generate:
- Dataset statistics report
- Sample visualizations
- Class distribution plots
- Pixel value distributions

## 📈 Evaluation Metrics

- **ROC Curves** (Receiver Operating Characteristic)
- **AUC Scores** (Area Under the ROC Curve)
- **Confusion Matrix**
- **Classification Report**

**Target Performance**: AUC > 0.95 per class

## 🛠️ Tech Stack

- **Python 3.8+**
- **PyTorch** - Deep learning framework
- **scikit-learn** - ML models and metrics
- **NumPy** - Array operations
- **Matplotlib** - Visualizations

## 📝 Results

_Coming soon after model training and evaluation_

## 🤝 Contributing

This is a learning project for the ML4Sci DeepLense challenge. Contributions, suggestions, and feedback are welcome!

## 📄 License

MIT License

## 🙏 Acknowledgments

- ML4Sci DeepLense Project
- Dataset provided by the DeepLense collaboration
