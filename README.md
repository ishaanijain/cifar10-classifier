# cifar10-classifier

Trained a ResNet-18 on CIFAR-10 to learn more about CNNs and transfer learning. Getting ~90% accuracy on the test set.

I modified the ResNet-18 architecture a bit since CIFAR images are 32x32 and the default ResNet expects 224x224. Basically swapped the first conv layer for a smaller 3x3 one and got rid of the maxpool so it doesn't downsample too aggressively early on.

Uses cosine annealing for the learning rate and early stopping so it doesn't overfit. Also added some data augmentation (random crops, flips, color jitter) which helped a lot — was only getting like 85% without it.

After training, the model gets exported to ONNX so it can be used for inference without needing pytorch.

## how to run

```
pip install -r requirements.txt
python train.py
python evaluate.py
python export_onnx.py
```

`train.py` trains the model and saves checkpoints. `evaluate.py` gives you per-class accuracy and a confusion matrix. `export_onnx.py` converts the best model to ONNX format.
