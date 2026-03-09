# 🚀 GPU Setup Guide - Enable NVIDIA GPU in Your Virtual Environment

## Current Situation

✅ **You have**: NVIDIA GeForce GTX 1650 (4GB VRAM)
✅ **CUDA Version**: 12.1 (installed and working)
❌ **Problem**: Your `venv` has CPU-only PyTorch (version 2.10.0+cpu)

**Goal**: Install PyTorch with CUDA support in your virtual environment for 10-15x faster training.

---

## Prerequisites Check

### 1. Verify GPU is Working

Open PowerShell and run:

```powershell
nvidia-smi
```

**Expected output:**
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 591.74                 Driver Version: 591.74         CUDA Version: 13.1     |
+-----------------------------------------+------------------------+----------------------+
|   0  NVIDIA GeForce GTX 1650      WDDM  |   00000000:01:00.0 Off |                  N/A |
```

✅ If you see this, your GPU is working!

---

## Step-by-Step Installation

### Step 1: Activate Your Virtual Environment

```powershell
cd "D:\Projects\Contributions\DeepLense_ml4sci"

# Activate venv
.\venv\Scripts\Activate.ps1
```

**You should see**: `(venv)` prefix in your terminal

---

### Step 2: Uninstall CPU-only PyTorch

```powershell
pip uninstall torch torchvision torchaudio
```

**When prompted**, type `y` and press Enter to confirm.

**Expected output:**
```
Successfully uninstalled torch-2.10.0+cpu
Successfully uninstalled torchvision-...
```

---

### Step 3: Install PyTorch with CUDA Support

Your GPU supports CUDA 12.1, so install the matching PyTorch version:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**This will download ~2.5 GB**, so it takes 5-10 minutes depending on your internet speed.

**Expected output:**
```
Downloading torch-2.5.1+cu121-cp312-cp312-win_amd64.whl (2.5 GB)
Successfully installed torch-2.5.1+cu121 torchvision-...
```

---

### Step 4: Verify GPU Installation

Run the check script:

```powershell
python check_gpu.py
```

**Expected output:**
```
======================================================================
GPU CHECK
======================================================================

PyTorch Version: 2.5.1+cu121
CUDA Available: True
CUDA Version: 12.1
GPU Count: 1
GPU Name: NVIDIA GeForce GTX 1650
GPU Memory: 4.00 GB

✅ GPU IS READY!
======================================================================
```

✅ If you see `CUDA Available: True`, you're all set!

---

### Step 5: Update requirements.txt (Optional)

If you want to save this configuration for future use:

```powershell
# Generate new requirements
pip freeze > requirements_gpu.txt
```

Or manually update `requirements.txt`:

```txt
numpy
matplotlib
scikit-learn
torch==2.5.1+cu121
torchvision==0.20.1+cu121
Pillow
tqdm
seaborn
```

---

## Running Training with GPU

Now you can train with GPU:

```powershell
# Make sure venv is activated (you should see (venv) prefix)
python src\train_resnet_improved.py
```

**You should now see:**
```
======================================================================
GRAVITATIONAL LENSING CLASSIFICATION - ResNet34 IMPROVED
======================================================================
Device: cuda  ← Should say "cuda" now!
GPU: NVIDIA GeForce GTX 1650
======================================================================
```

---

## Performance Comparison

| Metric | CPU (Before) | GPU (After) | Improvement |
|--------|--------------|-------------|-------------|
| **Time per epoch** | ~30 minutes | ~2-3 minutes | **10-15x faster** |
| **Time per batch** | ~1.8 seconds | ~0.15 seconds | **12x faster** |
| **Total training** | 20-50 hours | **2-4 hours** | **10-15x faster** |
| **Batches/second** | 0.5 | 6-7 | **12-14x faster** |

---

## Expected Training Output (with GPU)

```
Epoch 1/50
--------------------------------------------------
Learning Rate: 1.00e-04
Training: 100%|████████████| 938/938 [02:15<00:00, 6.92it/s, loss=0.8234]
Validation: 100%|██████████| 235/235 [00:28<00:00, 8.39it/s, loss=0.7892]

Epoch 1 Summary:
  Train Loss: 0.8234 | Train Acc: 0.6345
  Val Loss:   0.7892 | Val Acc:   0.6680
  ✓ New best model saved! Val Acc: 0.6680
```

**Key indicators that GPU is working:**
- ✅ `Device: cuda` in header
- ✅ Speed: ~7 batches/second (not 0.5)
- ✅ Epoch takes 2-3 minutes (not 30 minutes)

---

## Troubleshooting

### Problem 1: "pip install" is very slow

**Cause**: Downloading 2.5 GB package

**Solution**: Be patient, it takes 5-10 minutes. You can monitor progress.

---

### Problem 2: After installation, still shows "CUDA Available: False"

**Solution 1**: Restart your terminal and reactivate venv
```powershell
deactivate
.\venv\Scripts\Activate.ps1
python check_gpu.py
```

**Solution 2**: Check CUDA compatibility
```powershell
nvidia-smi
# Note the CUDA version shown (should be 12.1 or higher)
```

If your CUDA version is different, install matching PyTorch:
- CUDA 11.8: `--index-url https://download.pytorch.org/whl/cu118`
- CUDA 12.1: `--index-url https://download.pytorch.org/whl/cu121`

---

### Problem 3: "CUDA out of memory" error during training

**Cause**: GTX 1650 has 4GB VRAM, might be tight with batch size 32

**Solution**: Reduce batch size in the config

Edit `src/train_resnet_improved.py`, line ~69:
```python
BATCH_SIZE = 16  # Changed from 32
```

Or run with smaller batch:
```python
# In the Config class
BATCH_SIZE = 24  # Try 24 first, then 16 if still fails
```

---

### Problem 4: Training still slow after GPU installation

**Check 1**: Verify GPU is being used
```powershell
# While training is running, open another terminal:
nvidia-smi

# You should see:
# | Processes:                                          GPU Memory |
# |   0   python.exe              ...                    3500MiB |
```

**Check 2**: Verify device in training output
```
Device: cuda  ← Must say "cuda", not "cpu"
```

If it says "cpu", the script isn't detecting GPU. Run `python check_gpu.py` again.

---

## Complete Installation Commands (Copy-Paste)

Here's the complete sequence to run:

```powershell
# Step 1: Navigate to project
cd "D:\Projects\Contributions\DeepLense_ml4sci"

# Step 2: Activate venv
.\venv\Scripts\Activate.ps1

# Step 3: Uninstall CPU PyTorch
pip uninstall -y torch torchvision torchaudio

# Step 4: Install GPU PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Step 5: Verify installation
python check_gpu.py

# Step 6: Start training!
python src\train_resnet_improved.py
```

**Copy and paste these commands one by one in PowerShell.**

---

## Alternative: Use Anaconda Environment (Already Has GPU PyTorch)

If you have issues with venv, you can use Anaconda which already has GPU PyTorch:

```powershell
# Deactivate venv if active
deactivate

# Use Anaconda Python directly (no venv)
C:\Users\Asus\anaconda3\python.exe src\train_resnet_improved.py
```

This will use Anaconda's Python which already has `torch 2.5.1+cu121` with GPU support.

---

## Best Practices

### ✅ Do's:

1. **Always activate venv** before running training
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. **Verify GPU before long training**
   ```powershell
   python check_gpu.py
   ```

3. **Monitor GPU usage** during training
   ```powershell
   # In another terminal:
   nvidia-smi
   ```

4. **Save your model regularly**
   - The script auto-saves best model
   - Check `checkpoints/` folder

5. **Monitor training progress**
   ```powershell
   # In another terminal:
   python monitor_training.py
   ```

### ❌ Don'ts:

1. **Don't run without activating venv**
   - You'll use the wrong Python version

2. **Don't assume GPU is working**
   - Always check the "Device:" line in output

3. **Don't use too large batch size**
   - GTX 1650 has 4GB VRAM
   - Stick to batch size 16-32

4. **Don't close terminal during training**
   - Training will stop
   - Use `monitor_training.py` in another terminal instead

---

## Quick Reference

### Check if GPU is working:
```powershell
python check_gpu.py
```

### Install GPU PyTorch:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Start training:
```powershell
python src\train_resnet_improved.py
```

### Monitor training:
```powershell
python monitor_training.py
```

### Check GPU usage:
```powershell
nvidia-smi
```

---

## Expected Timeline (with GPU)

| Phase | Duration | What's Happening |
|-------|----------|------------------|
| **Setup** | 10-15 min | Install GPU PyTorch |
| **Epoch 1** | 2-3 min | Initial learning |
| **Epochs 2-20** | 40-60 min | Rapid improvement |
| **Epochs 21-50** | 60-90 min | Fine-tuning |
| **Total** | **2-3 hours** | Complete training |

**Result**: Trained model with 80-88% accuracy (vs. 20-50 hours on CPU!)

---

## Verification Checklist

Before starting training, verify:

- [ ] Virtual environment is activated (`(venv)` prefix visible)
- [ ] `python check_gpu.py` shows `CUDA Available: True`
- [ ] GPU name shows: `NVIDIA GeForce GTX 1650`
- [ ] PyTorch version shows: `2.5.1+cu121` (or similar with `+cu`)
- [ ] Training output shows: `Device: cuda` (not `cpu`)

**If all checked**, you're ready to train! 🚀

---

## Summary

**What you need to do:**

1. Open PowerShell in your project folder
2. Activate venv: `.\venv\Scripts\Activate.ps1`
3. Uninstall CPU PyTorch: `pip uninstall -y torch torchvision`
4. Install GPU PyTorch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
5. Verify: `python check_gpu.py`
6. Train: `python src\train_resnet_improved.py`

**Time**: ~15 minutes setup + 2-3 hours training = **~3 hours total**

**Result**: Trained model, 10-15x faster than CPU!

---

## Need Help?

If you get stuck at any step:

1. Check the **Troubleshooting** section above
2. Run `python check_gpu.py` to diagnose the issue
3. Check `nvidia-smi` to verify GPU is working
4. Make sure you're in the correct venv (`(venv)` prefix)

**Most common issue**: Forgetting to activate venv!

---

**Ready to start? Follow the installation commands above! 🚀**
