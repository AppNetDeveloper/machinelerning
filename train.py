"""
Entrenador mejorado de modelo de confecciones de frutas.
Usa transfer learning con ResNet50 para clasificar imagenes de cajas de frutas.

Mejoras:
- Pesos de clase para manejar desbalance
- Early stopping para evitar sobreajuste
- Callback de progreso para el panel web
- Mejor data augmentation
- Guarda historial en la base de datos

Uso directo:
    python train.py
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms, models
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    DATASET_DIR, MODEL_PATH, CLASES_PATH, IMG_SIZE, BATCH_SIZE,
    EPOCHS_HEAD, EPOCHS_FINETUNE, LR_HEAD, LR_FINETUNE,
    VALIDATION_SPLIT, LABEL_SMOOTHING, EARLY_STOP_PATIENCE,
)


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def load_dataset(dataset_dir, train_transform, val_transform):
    full_dataset = datasets.ImageFolder(root=dataset_dir, transform=train_transform)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset = datasets.ImageFolder(root=dataset_dir, transform=val_transform)

    train_targets = [full_dataset.targets[i] for i in train_dataset.indices]
    class_counts = Counter(train_targets)
    sample_weights = [1.0 / class_counts[full_dataset.targets[i]] for i in train_dataset.indices]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)

    class_counts_all = Counter(full_dataset.targets)
    print(f"""
{"="*50}
Dataset cargado:
  Clases: {class_names}
  Total imagenes: {len(full_dataset)}""")
    for i, name in enumerate(class_names):
        print(f"    {name}: {class_counts_all[i]} imagenes")
    print(f"  Entrenamiento: {train_size} | Validacion: {val_size}")
    print(f"  Weighted sampler activo")
    print(f"{"="*50}")
    return train_loader, val_loader, class_names, num_classes


def create_model(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(num_features, 256),
        nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, num_classes),
    )
    for param in model.fc.parameters():
        param.requires_grad = True
    return model


def unfreeze_layers(model):
    for name, param in model.named_parameters():
        if "layer3" in name or "layer4" in name or "fc" in name:
            param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parametros entrenables: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    accuracy = 100.0 * correct / total
    avg_loss = running_loss / len(train_loader)
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    accuracy = 100.0 * correct / total
    avg_loss = running_loss / len(val_loader)
    return avg_loss, accuracy


def generar_graficas(history, class_names):
    epochs = range(1, len(history["train_loss"]) + 1)
    split_epoch = len([l for l in history["phase"] if l == 1])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Entrenamiento de Confecciones (2 fases)", fontsize=14, fontweight="bold")
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=3)
    axes[0].plot(epochs, history["val_loss"], "r-o", label="Val Loss", markersize=3)
    axes[0].axvline(x=split_epoch, color="green", linestyle="--", alpha=0.7, label="Fine-tune start")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoca"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Acc", markersize=3)
    axes[1].plot(epochs, history["val_acc"], "r-o", label="Val Acc", markersize=3)
    axes[1].axvline(x=split_epoch, color="green", linestyle="--", alpha=0.7, label="Fine-tune start")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoca"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 105)
    plt.tight_layout()
    grafica_path = "entrenamiento_grafica.png"
    plt.savefig(grafica_path, dpi=150, bbox_inches="tight")
    plt.close()
    return grafica_path


def run_training_sync(progress_callback=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def report(msg, **kwargs):
        if progress_callback:
            return progress_callback({"message": msg, **kwargs})
        print(msg)
        return True

    report(f"Usando dispositivo: {device}", phase="init")
    train_transform, val_transform = get_transforms()
    train_loader, val_loader, class_names, num_classes = load_dataset(
        str(DATASET_DIR), train_transform, val_transform
    )
    model = create_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    epochs_no_improve = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": [], "phase": []}
    total_epochs = EPOCHS_HEAD + EPOCHS_FINETUNE

    # FASE 1
    report(f"FASE 1: Entrenando capa clasificadora ({EPOCHS_HEAD} epocas)", phase="phase1")
    optimizer = optim.Adam(model.fc.parameters(), lr=LR_HEAD)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    for epoch in range(EPOCHS_HEAD):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss); history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss); history["val_acc"].append(val_acc)
        history["lr"].append(lr); history["phase"].append(1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc; epochs_no_improve = 0
            torch.save(model.state_dict(), str(MODEL_PATH))
        else:
            epochs_no_improve += 1

        cont = report(
            f"Epoca {epoch+1:3d}/{total_epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}% | LR: {lr:.6f}",
            phase="phase1", epoch=epoch+1, total_epochs=total_epochs,
            train_loss=train_loss, train_acc=train_acc, val_loss=val_loss, val_acc=val_acc, lr=lr, best_val_acc=best_val_acc,
        )
        if cont is False:
            return {"status": "cancelled", "history": history, "best_val_acc": best_val_acc}

    # FASE 2
    report(f"FASE 2: Fine-tuning layer3+4 ({EPOCHS_FINETUNE} epocas)", phase="phase2")
    unfreeze_layers(model)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_FINETUNE)
    scheduler_ft = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=EPOCHS_FINETUNE, eta_min=1e-7)

    for epoch in range(EPOCHS_FINETUNE):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer_ft, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler_ft.step()
        lr = optimizer_ft.param_groups[0]["lr"]
        epoch_num = EPOCHS_HEAD + epoch + 1
        history["train_loss"].append(train_loss); history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss); history["val_acc"].append(val_acc)
        history["lr"].append(lr); history["phase"].append(2)

        if val_acc > best_val_acc:
            best_val_acc = val_acc; epochs_no_improve = 0
            torch.save(model.state_dict(), str(MODEL_PATH))
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            report(f"Early stopping en epoca {epoch_num}", phase="early_stop", epoch=epoch_num, total_epochs=total_epochs)
            break

        cont = report(
            f"Epoca {epoch_num:3d}/{total_epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}% | LR: {lr:.6f}",
            phase="phase2", epoch=epoch_num, total_epochs=total_epochs,
            train_loss=train_loss, train_acc=train_acc, val_loss=val_loss, val_acc=val_acc, lr=lr, best_val_acc=best_val_acc,
        )
        if cont is False:
            return {"status": "cancelled", "history": history, "best_val_acc": best_val_acc}

    with open(str(CLASES_PATH), "w") as f:
        json.dump(class_names, f, indent=2)
    grafica_path = generar_graficas(history, class_names)
    report(f"Entrenamiento completado! Mejor val acc: {best_val_acc:.1f}%", phase="done", best_val_acc=best_val_acc, grafica_path=grafica_path)

    return {
        "status": "completed", "best_val_acc": best_val_acc,
        "final_train_acc": history["train_acc"][-1],
        "grafica_path": grafica_path, "history": history, "class_names": class_names,
    }


def main():
    result = run_training_sync()
    print(f"\n{"="*50}\nEntrenamiento completado!\n  Mejor val acc: {result["best_val_acc"]:.1f}%\n{"="*50}")

if __name__ == "__main__":
    main()
