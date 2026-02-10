import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import (DEVICE, EPOCHS, LR, MOMENTUM, WEIGHT_DECAY,
                    PATIENCE, CHECKPOINT_DIR, BEST_MODEL_PATH)
from dataset import get_loaders
from model import build_resnet18


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  val  ", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc


def main():
    print(f"Using device: {DEVICE}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    train_loader, test_loader = get_loaders()
    model = build_resnet18().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR,
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    # for plotting later
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, EPOCHS + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{EPOCHS}  (lr={current_lr:.6f})")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, test_loader, criterion)
        scheduler.step()

        print(f"  Train loss: {train_loss:.4f}  acc: {train_acc:.2f}%")
        print(f"  Val   loss: {val_loss:.4f}  acc: {val_acc:.2f}%")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  -> saved best model (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  no improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print("\nTraining done.")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to {BEST_MODEL_PATH}")

    plot_curves(history)


def plot_curves(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["val_acc"], label="val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("Saved training_curves.png")
    plt.close()


if __name__ == "__main__":
    main()
