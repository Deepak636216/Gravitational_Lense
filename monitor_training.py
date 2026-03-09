"""
Training Monitor - Check progress of ongoing training
"""

import json
import time
from pathlib import Path
from datetime import datetime

def find_latest_experiment():
    """Find the most recent experiment directory"""
    log_dir = Path('./logs')
    if not log_dir.exists():
        return None

    experiments = [d for d in log_dir.iterdir() if d.is_dir() and d.name.startswith('resnet34_baseline')]
    if not experiments:
        return None

    # Sort by creation time
    latest = max(experiments, key=lambda x: x.stat().st_mtime)
    return latest

def load_history(exp_dir):
    """Load training history if available"""
    history_file = exp_dir / 'history.json'
    if history_file.exists():
        with open(history_file, 'r') as f:
            return json.load(f)
    return None

def display_progress(exp_dir):
    """Display training progress"""
    print("="*70)
    print("TRAINING PROGRESS MONITOR")
    print("="*70)
    print(f"\nExperiment: {exp_dir.name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load config
    config_file = exp_dir / 'config.json'
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"\nConfiguration:")
        print(f"  Batch Size: {config.get('BATCH_SIZE', 'N/A')}")
        print(f"  Learning Rate: {config.get('LEARNING_RATE', 'N/A')}")
        print(f"  Max Epochs: {config.get('NUM_EPOCHS', 'N/A')}")

    # Load history
    history = load_history(exp_dir)
    if history:
        completed_epochs = len(history['train_loss'])
        print(f"\nProgress:")
        print(f"  Completed Epochs: {completed_epochs}")

        if completed_epochs > 0:
            print(f"\nLatest Metrics (Epoch {completed_epochs}):")
            print(f"  Train Loss: {history['train_loss'][-1]:.4f}")
            print(f"  Train Accuracy: {history['train_acc'][-1]:.4f}")
            print(f"  Val Loss: {history['val_loss'][-1]:.4f}")
            print(f"  Val Accuracy: {history['val_acc'][-1]:.4f}")
            print(f"  Learning Rate: {history['learning_rate'][-1]:.6f}")

            # Best so far
            best_val_acc = max(history['val_acc'])
            best_epoch = history['val_acc'].index(best_val_acc) + 1
            print(f"\nBest So Far:")
            print(f"  Best Val Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")

            # Show improvement trend
            if completed_epochs >= 5:
                recent_avg = sum(history['val_acc'][-5:]) / 5
                print(f"  Recent 5-epoch average: {recent_avg:.4f}")
    else:
        print("\n⏳ Training just started, no history available yet...")

    # Check if metrics file exists (training complete)
    metrics_file = exp_dir / 'metrics.json'
    if metrics_file.exists():
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        print(f"\nFinal Results:")
        print(f"  Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
        print(f"  Macro F1: {metrics.get('macro_f1', 'N/A'):.4f}")
        print(f"  ROC-AUC: {metrics.get('roc_auc', 'N/A'):.4f}")

        print(f"\nPer-Class F1 Scores:")
        for class_name in ['No Substructure', 'Subhalo/Sphere', 'Vortex']:
            f1_key = f'{class_name}_f1'
            if f1_key in metrics:
                print(f"  {class_name}: {metrics[f1_key]:.4f}")

    print("\n" + "="*70)

def main():
    """Main monitoring function"""
    exp_dir = find_latest_experiment()

    if exp_dir is None:
        print("No training experiments found in ./logs/")
        return

    display_progress(exp_dir)

    print("\nFiles in experiment directory:")
    for file in sorted(exp_dir.iterdir()):
        print(f"  - {file.name}")

    print("\nTo monitor in real-time, run this script periodically.")
    print("Results will be saved to:", exp_dir)

if __name__ == '__main__':
    main()
