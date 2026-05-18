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
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import datasets, transforms, models
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from config import (
    DATASET_DIR, MODEL_PATH, CLASES_PATH, IMG_SIZE, BATCH_SIZE,
    EPOCHS_HEAD, EPOCHS_FINETUNE, LR_HEAD, LR_FINETUNE,
    VALIDATION_SPLIT, LABEL_SMOOTHING, EARLY_STOP_PATIENCE,
    WEIGHT_DECAY, MIXUP_ALPHA, GRAD_CLIP_MAX_NORM, WARMUP_EPOCHS,
    USE_AMP, NUM_WORKERS, PREFETCH_FACTOR, CHECKPOINT_EVERY, CHECKPOINT_PATH,
)


def create_fc_head(num_features=2048, num_classes=3):
    """Cabeza FC compartida entre entrenamiento e inferencia."""
    return nn.Sequential(
        nn.Dropout(0.3), nn.Linear(num_features, 256),
        nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, num_classes),
    )


def mixup_data(images, labels, alpha=MIXUP_ALPHA):
    """Aplica MixUp: mezcla pares de imágenes para crear ejemplos virtuales."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)
    mixed_images = lam * images + (1 - lam) * images[index]
    return mixed_images, labels, labels[index], lam


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
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


def stratified_split(dataset, val_ratio=0.2, seed=42):
    """Divide el dataset preservando la proporcion de clases en train y val."""
    targets = np.array(dataset.targets)
    rng = np.random.RandomState(seed)
    train_indices, val_indices = [], []
    for cls in np.unique(targets):
        cls_indices = np.where(targets == cls)[0]
        rng.shuffle(cls_indices)
        n_val = max(1, int(len(cls_indices) * val_ratio))
        val_indices.extend(cls_indices[:n_val].tolist())
        train_indices.extend(cls_indices[n_val:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def clean_empty_classes(dataset_dir):
    """Elimina carpetas de clases sin imágenes válidas."""
    valid_ext = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'}
    dataset_path = Path(dataset_dir)
    removed = []
    for class_dir in sorted(dataset_path.iterdir()):
        if not class_dir.is_dir():
            continue
        images = [f for f in class_dir.iterdir() if f.suffix.lower() in valid_ext]
        if not images:
            class_dir.rmdir()
            removed.append(class_dir.name)
    if removed:
        print(f"Clases vacias eliminadas: {removed}")


def load_dataset(dataset_dir, train_transform, val_transform):
    clean_empty_classes(dataset_dir)
    # Verificar que queden clases con imágenes
    valid_ext = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'}
    dataset_path = Path(dataset_dir)
    valid_classes = [d.name for d in sorted(dataset_path.iterdir())
                     if d.is_dir() and any(f.suffix.lower() in valid_ext for f in d.iterdir())]
    if len(valid_classes) < 2:
        raise ValueError(
            f"Se necesitan al menos 2 clases con imagenes. "
            f"Clases encontradas: {valid_classes}. "
            f"Sube imagenes desde el panel web antes de entrenar."
        )
    train_dataset_full = datasets.ImageFolder(root=dataset_dir, transform=train_transform)
    val_dataset_full = datasets.ImageFolder(root=dataset_dir, transform=val_transform)
    class_names = train_dataset_full.classes
    num_classes = len(class_names)

    train_subset, val_subset = stratified_split(train_dataset_full, val_ratio=VALIDATION_SPLIT)
    # Reemplazar el dataset interno del val_subset para usar val_transform
    val_subset = Subset(val_dataset_full, val_subset.indices)

    train_targets = [train_dataset_full.targets[i] for i in train_subset.indices]
    class_counts = Counter(train_targets)
    sample_weights = [1.0 / class_counts[train_dataset_full.targets[i]] for i in train_subset.indices]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=NUM_WORKERS > 0, prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            persistent_workers=NUM_WORKERS > 0)

    class_counts_all = Counter(train_dataset_full.targets)
    print(f"""
{"="*50}
Dataset cargado:
  Clases: {class_names}
  Total imagenes: {len(train_dataset_full)}""")
    for i, name in enumerate(class_names):
        print(f"    {name}: {class_counts_all[i]} imagenes")
    print(f"  Entrenamiento: {len(train_subset)} | Validacion: {len(val_subset)}")
    print(f"  Split estratificado activo")
    print(f"  Weighted sampler activo")
    print(f"{"="*50}")
    return train_loader, val_loader, class_names, num_classes


def create_model(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = create_fc_head(model.fc.in_features, num_classes)
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

def train_epoch(model, train_loader, criterion, optimizer, device, use_mixup=True, max_grad_norm=None, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    use_amp = scaler is not None
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            if use_mixup:
                mixed_images, labels_a, labels_b, lam = mixup_data(images, labels)
                outputs = model(mixed_images)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
        if use_amp:
            scaler.scale(loss).backward()
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
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


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, phase, best_val_acc, history, path):
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "phase": phase,
        "best_val_acc": best_val_acc,
        "history": history,
    }, str(path))


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    if not Path(path).exists():
        return None
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if scaler and ckpt.get("scaler_state"):
        scaler.load_state_dict(ckpt["scaler_state"])
    return ckpt


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


def save_history_json(history, path="training_history.json"):
    with open(path, "w") as f:
        json.dump(history, f)


def run_training_sync(progress_callback=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def report(msg, **kwargs):
        if progress_callback:
            return progress_callback({"message": msg, **kwargs})
        print(msg)
        return True

    use_amp = USE_AMP and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    report(f"Usando dispositivo: {device} | AMP: {'activo' if use_amp else 'inactivo (CPU)'}", phase="init")

    train_transform, val_transform = get_transforms()
    train_loader, val_loader, class_names, num_classes = load_dataset(
        str(DATASET_DIR), train_transform, val_transform
    )
    model = create_model(num_classes).to(device)
    best_val_acc = 0.0
    epochs_no_improve = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": [], "phase": []}
    total_epochs = EPOCHS_HEAD + EPOCHS_FINETUNE

    # Intentar reanudar desde checkpoint
    start_phase, start_epoch = 1, 0
    if Path(str(CHECKPOINT_PATH)).exists():
        report(f"Checkpoint encontrado, reanudando...", phase="resume")
        optimizer_tmp = optim.AdamW(model.fc.parameters(), lr=LR_HEAD)
        scheduler_tmp = optim.lr_scheduler.ReduceLROnPlateau(optimizer_tmp, patience=3, factor=0.5)
        ckpt = load_checkpoint(CHECKPOINT_PATH, model, optimizer_tmp, scheduler_tmp, scaler)
        if ckpt:
            start_phase = ckpt.get("phase", 1)
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_acc = ckpt.get("best_val_acc", 0.0)
            history = ckpt.get("history", history)
            report(f"Reanudando desde fase {start_phase}, epoca {start_epoch}, mejor val_acc: {best_val_acc:.1f}%", phase="resume")

    # FASE 1
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.fc.parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    phase1_range = range(start_epoch if start_phase == 1 else 0, EPOCHS_HEAD)
    if start_phase == 1:
        report(f"FASE 1: Entrenando capa clasificadora ({EPOCHS_HEAD} epocas)", phase="phase1")
        for epoch in phase1_range:
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            scheduler.step(val_loss)
            lr = optimizer.param_groups[0]["lr"]
            history["train_loss"].append(train_loss); history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss); history["val_acc"].append(val_acc)
            history["lr"].append(lr); history["phase"].append(1)
            save_history_json(history)

            if val_acc > best_val_acc:
                best_val_acc = val_acc; epochs_no_improve = 0
                torch.save(model.state_dict(), str(MODEL_PATH))
            else:
                epochs_no_improve += 1

            if (epoch + 1) % CHECKPOINT_EVERY == 0:
                save_checkpoint(model, optimizer, scheduler, scaler, epoch, 1, best_val_acc, history, CHECKPOINT_PATH)

            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                report(f"Early stopping Fase 1 en epoca {epoch+1}", phase="early_stop", epoch=epoch+1, total_epochs=total_epochs)
                break

            cont = report(
                f"Epoca {epoch+1:3d}/{total_epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}% | LR: {lr:.6f}",
                phase="phase1", epoch=epoch+1, total_epochs=total_epochs,
                train_loss=train_loss, train_acc=train_acc, val_loss=val_loss, val_acc=val_acc, lr=lr, best_val_acc=best_val_acc,
            )
            if cont is False:
                save_checkpoint(model, optimizer, scheduler, scaler, epoch, 1, best_val_acc, history, CHECKPOINT_PATH)
                return {"status": "cancelled", "history": history, "best_val_acc": best_val_acc}
        start_epoch = 0

    # FASE 2
    report(f"FASE 2: Fine-tuning layer3+4 ({EPOCHS_FINETUNE} epocas)", phase="phase2")
    unfreeze_layers(model)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer_ft = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY)
    scheduler_warmup = optim.lr_scheduler.LinearLR(optimizer_ft, start_factor=0.1, total_iters=WARMUP_EPOCHS)
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=EPOCHS_FINETUNE - WARMUP_EPOCHS, eta_min=1e-7)
    scheduler_ft = optim.lr_scheduler.SequentialLR(optimizer_ft, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[WARMUP_EPOCHS])

    for epoch in range(start_epoch if start_phase == 2 else 0, EPOCHS_FINETUNE):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer_ft, device, max_grad_norm=GRAD_CLIP_MAX_NORM, scaler=scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler_ft.step()
        lr = optimizer_ft.param_groups[0]["lr"]
        epoch_num = EPOCHS_HEAD + epoch + 1
        history["train_loss"].append(train_loss); history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss); history["val_acc"].append(val_acc)
        history["lr"].append(lr); history["phase"].append(2)
        save_history_json(history)

        if val_acc > best_val_acc:
            best_val_acc = val_acc; epochs_no_improve = 0
            torch.save(model.state_dict(), str(MODEL_PATH))
        else:
            epochs_no_improve += 1

        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(model, optimizer_ft, scheduler_ft, scaler, epoch, 2, best_val_acc, history, CHECKPOINT_PATH)

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            report(f"Early stopping en epoca {epoch_num}", phase="early_stop", epoch=epoch_num, total_epochs=total_epochs)
            break

        cont = report(
            f"Epoca {epoch_num:3d}/{total_epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}% | LR: {lr:.6f}",
            phase="phase2", epoch=epoch_num, total_epochs=total_epochs,
            train_loss=train_loss, train_acc=train_acc, val_loss=val_loss, val_acc=val_acc, lr=lr, best_val_acc=best_val_acc,
        )
        if cont is False:
            save_checkpoint(model, optimizer_ft, scheduler_ft, scaler, epoch, 2, best_val_acc, history, CHECKPOINT_PATH)
            return {"status": "cancelled", "history": history, "best_val_acc": best_val_acc}

    # Limpieza final
    with open(str(CLASES_PATH), "w") as f:
        json.dump(class_names, f, indent=2)
    grafica_path = generar_graficas(history, class_names)
    if Path(str(CHECKPOINT_PATH)).exists():
        Path(str(CHECKPOINT_PATH)).unlink()
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
