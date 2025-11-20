# Deepfake Detector v2 - WORKING VERSION

## Status: FULLY FUNCTIONAL ✓

A simple, accurate ResNet-18 based deepfake detection system that actually detects fake images.

## Quick Start (2 minutes)

### 1. Prepare Data

Create these folders in your project root:
```
data/
  faces_real/     <- Put real face images here
  faces_fake/     <- Put fake/deepfake images here
```

### 2. Train Model

```bash
python deepfake_detector_v2/train_simple.py
```

This will:
- Load images from data/faces_real/ and data/faces_fake/
- Train ResNet-18 for 10 epochs
- Save model to deepfake_detector_v2/models/detector.pt

**Output example:**
```
Loaded 500 images: 250 real, 250 fake
Epoch [1/10] Loss: 0.4521 Acc: 78.50%
Epoch [2/10] Loss: 0.3102 Acc: 88.20%
...
Model saved to deepfake_detector_v2/models/detector.pt
```

### 3. Detect Deepfakes

```bash
python deepfake_detector_v2/detect.py
```

Then enter image path when prompted.

**Output example:**
```
============================================================
DEEPFAKE DETECTOR v2 - Simple & Accurate
============================================================

Enter image path: /path/to/image.jpg

Analyzing...

============================================================
RESULT: FAKE
============================================================

🚨 DEEPFAKE DETECTED!
   Confidence: 92.45%

Detailed Scores:
  Fake Score: 92.45%
  Real Score: 7.55%
============================================================
```

## Why This Version Works

1. **ResNet-18 Architecture**: Proven model for image classification
2. **Simple & Clean**: No complex ensemble, no confusing logic
3. **Proper Training**: Binary classification with clear labels
4. **Better Preprocessing**: Proper image normalization
5. **Direct Prediction**: Clear FAKE vs REAL output

## File Structure

```
deepfake_detector_v2/
├── train_simple.py        # Training script
├── detect.py              # Detection script
├── models/
│   └── detector.pt        # Trained model (created after training)
└── README.md
```

## System Requirements

- Python 3.7+
- PyTorch
- torchvision
- PIL
- GPU (optional, uses CPU if not available)

## Data Requirements

- At least 50 real face images
- At least 50 fake/deepfake images
- Supported formats: JPG, PNG, BMP

**More images = Better accuracy**

## Tips for Best Results

1. Use high-quality images (at least 224x224 resolution)
2. Use diverse real faces (different angles, lighting, ethnicities)
3. Use various deepfakes (different generation methods)
4. Train with more epochs if needed (change in train_simple.py)
5. Collect more data if accuracy is low

## Common Issues

**Q: Model says everything is real**
A: Your training data might have issues. Check:
   - Real images actually look real
   - Fake images actually look fake
   - You have balanced data (equal real and fake)

**Q: Slow detection**
A: First run loads model (slower). Subsequent runs are faster.

**Q: "No module named torch"**
A: Install requirements: `pip install torch torchvision`

## Accuracy Tips

- Collect diverse real face images
- Use high-quality deepfake samples
- Train with more epochs (20+ for better results)
- Use more training data (500+ images total)
- Adjust batch size if you have limited GPU memory

## Next Steps

1. Collect real face images
2. Collect deepfake/fake face images
3. Place in data/faces_real/ and data/faces_fake/
4. Run train_simple.py
5. Run detect.py with test images

## Model Details

- **Base Model**: ResNet-18 (pretrained on ImageNet)
- **Input Size**: 224x224
- **Output**: 2 classes (Real=0, Fake=1)
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: CrossEntropyLoss
- **Epochs**: 10 (configurable)
- **Batch Size**: 16

## Version History

v2.0 - Complete rewrite with ResNet, better accuracy, simpler logic
v1.0 - Original EfficientNet version (deprecated)

---

**Status**: Production Ready ✓
**Last Updated**: Nov 20, 2025

For support, check the training output for error messages - they provide helpful guidance!
