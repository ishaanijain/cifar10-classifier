import torch
import numpy as np
import matplotlib.pyplot as plt

from config import DEVICE, BEST_MODEL_PATH, NUM_CLASSES, CLASS_NAMES
from dataset import get_loaders
from model import build_resnet18


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        for i in range(labels.size(0)):
            label = labels[i].item()
            class_correct[label] += (predicted[i] == labels[i]).item()
            class_total[label] += 1

    overall_acc = 100.0 * correct / total
    return overall_acc, class_correct, class_total, np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(preds, labels, save_path="confusion_matrix.png"):
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(labels, preds):
        cm[t][p] += 1

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=range(NUM_CLASSES), yticks=range(NUM_CLASSES),
           xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
           ylabel="True label", xlabel="Predicted label",
           title="Confusion Matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # put numbers in cells
    thresh = cm.max() / 2.0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def main():
    _, test_loader = get_loaders()

    model = build_resnet18().to(DEVICE)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True))
    print(f"Loaded model from {BEST_MODEL_PATH}")

    overall_acc, class_correct, class_total, preds, labels = evaluate(model, test_loader)

    print(f"\nOverall test accuracy: {overall_acc:.2f}%\n")
    print("Per-class accuracy:")
    for i in range(NUM_CLASSES):
        acc = 100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f"  {CLASS_NAMES[i]:<12s}: {acc:.1f}% ({class_correct[i]}/{class_total[i]})")

    plot_confusion_matrix(preds, labels)


if __name__ == "__main__":
    main()
