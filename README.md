# CIFAR-10 Image Classifier

ResNet-18 trained on CIFAR-10 from scratch (well, fine-tuned from ImageNet weights). Gets around 90% test accuracy.

## what it does

Takes the standard CIFAR-10 dataset (60k images across 10 classes — planes, cars, birds, cats, etc.) and trains a ResNet-18 to classify them. The model architecture is slightly modified from the standard ResNet since CIFAR images are only 32x32 (vs the 224x224 ImageNet was designed for).

Key changes to ResNet-18:
- Replaced the 7x7 conv1 with a 3x3 conv (stride 1) so we don't lose too much spatial info
- Removed the maxpool layer for the same reason
- Swapped the final FC for 10 classes

## training details

- SGD w/ momentum (0.9) and weight decay (5e-4)
- Cosine annealing LR schedule starting at 0.01
- Data augmentation: random crops, horizontal flips, color jitter
- Early stopping with patience of 7 epochs
- Trains for up to 50 epochs but usually converges around 35-40

## setup

```bash
pip install -r requirements.txt
```

## usage

Train the model:
```bash
python train.py
```

Evaluate on test set (prints per-class accuracy + saves confusion matrix):
```bash
python evaluate.py
```

Export to ONNX for deployment:
```bash
python export_onnx.py
```

## results

~90% test accuracy on CIFAR-10. The model struggles most with cat/dog and deer/horse distinctions which makes sense since those classes look pretty similar even to humans at 32x32.

## project structure

```
config.py        - hyperparameters and paths
dataset.py       - data loading and augmentation
model.py         - modified resnet-18 architecture
train.py         - training loop
evaluate.py      - evaluation and confusion matrix
export_onnx.py   - onnx export and verification
```
