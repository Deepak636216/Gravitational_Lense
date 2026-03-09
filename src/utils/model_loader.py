"""
Model Loader Utility
====================

Helper functions to load trained models and make predictions.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from train_resnet_baseline import ResNet34, Config


def load_best_model(checkpoint_path=None, device=None):
    """
    Load the best trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file. If None, finds latest best model.
        device: Device to load model on. If None, uses CUDA if available.

    Returns:
        model: Loaded ResNet34 model in evaluation mode
        checkpoint: Full checkpoint dict with metadata
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Find checkpoint if not provided
    if checkpoint_path is None:
        checkpoint_dir = Path('./checkpoints')
        checkpoint_files = list(checkpoint_dir.glob('resnet34_baseline_*_best.pth'))

        if not checkpoint_files:
            raise FileNotFoundError("No checkpoint files found in ./checkpoints/")

        # Get the most recent one
        checkpoint_path = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
        print(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model
    model = ResNet34(num_classes=3, dropout=0.4)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()  # Set to evaluation mode

    print(f"Model loaded successfully!")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Validation Accuracy: {checkpoint['val_acc']:.4f}")
    print(f"  Validation Loss: {checkpoint['val_loss']:.4f}")

    return model, checkpoint


def preprocess_image(image, use_sqrt_stretch=True):
    """
    Preprocess a single image for inference.

    Args:
        image: NumPy array of shape (150, 150) or (1, 150, 150)
        use_sqrt_stretch: Apply square-root stretch

    Returns:
        Preprocessed image tensor ready for model input
    """
    # Remove channel dimension if present
    if len(image.shape) == 3 and image.shape[0] == 1:
        image = image[0]

    # Apply preprocessing
    if use_sqrt_stretch:
        image = np.maximum(image, 0)
        image = np.sqrt(image)
        max_val = image.max()
        if max_val > 0:
            image = image / max_val

    # Convert to tensor and add batch + channel dimensions
    image_tensor = torch.from_numpy(image).float()
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 150, 150)

    return image_tensor


def predict_single(model, image, device=None, return_probabilities=True):
    """
    Make prediction on a single image.

    Args:
        model: Trained ResNet34 model
        image: NumPy array of shape (150, 150) or (1, 150, 150)
        device: Device to run inference on
        return_probabilities: If True, return class probabilities

    Returns:
        If return_probabilities=True:
            predicted_class (int), probabilities (numpy array)
        Else:
            predicted_class (int)
    """
    if device is None:
        device = next(model.parameters()).device

    # Preprocess
    image_tensor = preprocess_image(image)
    image_tensor = image_tensor.to(device)

    # Inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    if return_probabilities:
        return predicted_class, probabilities.cpu().numpy()[0]
    else:
        return predicted_class


def predict_batch(model, images, device=None, batch_size=32):
    """
    Make predictions on a batch of images.

    Args:
        model: Trained ResNet34 model
        images: NumPy array of shape (N, 150, 150) or (N, 1, 150, 150)
        device: Device to run inference on
        batch_size: Batch size for processing

    Returns:
        predicted_classes (numpy array), probabilities (numpy array)
    """
    if device is None:
        device = next(model.parameters()).device

    # Preprocess all images
    processed_images = []
    for img in images:
        img_tensor = preprocess_image(img)
        processed_images.append(img_tensor)

    # Stack into batch
    images_tensor = torch.cat(processed_images, dim=0).to(device)

    # Inference in batches
    all_predictions = []
    all_probabilities = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(images_tensor), batch_size):
            batch = images_tensor[i:i+batch_size]
            outputs = model(batch)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    return np.array(all_predictions), np.array(all_probabilities)


def get_class_name(class_idx):
    """Convert class index to class name."""
    class_names = ['No Substructure', 'Subhalo/Sphere', 'Vortex']
    return class_names[class_idx]


# Example usage
if __name__ == '__main__':
    """
    Example of how to use this module
    """
    print("="*70)
    print("MODEL LOADER - EXAMPLE USAGE")
    print("="*70)

    # Load model
    print("\n1. Loading trained model...")
    model, checkpoint = load_best_model()

    # Load a sample image
    print("\n2. Loading sample image...")
    sample_image_path = Path('./dataset/val/no/1.npy')

    if sample_image_path.exists():
        image = np.load(sample_image_path)
        print(f"   Image shape: {image.shape}")

        # Make prediction
        print("\n3. Making prediction...")
        predicted_class, probabilities = predict_single(model, image)

        print(f"   Predicted class: {predicted_class} ({get_class_name(predicted_class)})")
        print(f"   Probabilities:")
        print(f"     No Substructure: {probabilities[0]:.4f}")
        print(f"     Subhalo/Sphere:  {probabilities[1]:.4f}")
        print(f"     Vortex:          {probabilities[2]:.4f}")
    else:
        print(f"   Sample image not found: {sample_image_path}")

    print("\n" + "="*70)
    print("Example complete!")
    print("="*70)
