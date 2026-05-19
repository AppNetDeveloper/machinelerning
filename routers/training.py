"""Rutas de entrenamiento del modelo."""

import json
import asyncio
import base64
from io import BytesIO
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    DATASET_DIR, MODEL_PATH, TEMPLATES_DIR,
    EPOCHS_HEAD, EPOCHS_FINETUNE, LR_HEAD, LR_FINETUNE,
    BATCH_SIZE, EARLY_STOP_PATIENCE, LABEL_SMOOTHING, WARMUP_EPOCHS,
)
from database import save_training_run
from auth import get_current_user
from ml_model import model_manager


def get_dataset_stats():
    """Obtiene estadisticas del dataset: clases, conteos, total, tamano."""
    stats = {}
    total_images = 0
    total_size = 0
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif'}

    if not DATASET_DIR.exists():
        return {"classes": stats, "total_images": 0, "total_size_mb": 0, "num_classes": 0}

    for class_dir in sorted(DATASET_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        images = [f for f in class_dir.iterdir()
                  if f.is_file() and f.suffix.lower() in extensions]
        count = len(images)
        size = sum(f.stat().st_size for f in images)
        stats[class_dir.name] = {"count": count, "size_bytes": size}
        total_images += count
        total_size += size

    return {
        "classes": stats,
        "total_images": total_images,
        "total_size_mb": round(total_size / (1024 * 1024), 1),
        "num_classes": len(stats),
    }


def generate_dataset_chart(stats):
    """Genera grafica de barras del dataset y retorna base64 PNG."""
    if not stats["classes"]:
        return None

    classes = list(stats["classes"].keys())
    counts = [stats["classes"][c]["count"] for c in classes]

    fig, ax = plt.subplots(figsize=(8, max(3, len(classes) * 0.6)))
    colors = plt.cm.viridis([i / len(classes) for i in range(len(classes))])
    bars = ax.barh(classes, counts, color=colors, height=0.6)
    ax.set_xlabel("Numero de imagenes", fontsize=11)
    ax.set_title(f"Distribucion del Dataset ({stats['total_images']} imagenes, {stats['num_classes']} clases)",
                 fontsize=12, fontweight="bold")
    ax.invert_yaxis()

    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                str(count), va='center', fontsize=10)

    ax.set_xlim(0, max(counts) * 1.2 if counts else 10)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ─── Estado global del entrenamiento ─────────────────────────────
training_state = {
    "running": False,
    "progress": [],
    "current_epoch": 0,
    "total_epochs": 0,
    "best_val_acc": 0.0,
    "run_id": None,
}
training_lock = asyncio.Lock()

# ─── Estado global de augmentacion ──────────────────────────────
augment_state = {
    "running": False,
    "progress": [],
    "done": False,
}
augment_lock = asyncio.Lock()


async def get_user_or_redirect(request: Request):
    user = await get_current_user(request)
    if not user:
        return None
    return user


@router.get("/train", response_class=HTMLResponse)
async def train_page(request: Request):
    user = await get_user_or_redirect(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    dataset_stats = get_dataset_stats()
    dataset_chart = generate_dataset_chart(dataset_stats)

    return templates.TemplateResponse(request, "train.html", {
        "request": request,
        "user": user,
        "training_state": training_state,
        "model_loaded": model_manager.is_loaded,
        "dataset_stats": dataset_stats,
        "dataset_chart": dataset_chart,
        "cfg_epochs_head": EPOCHS_HEAD,
        "cfg_epochs_finetune": EPOCHS_FINETUNE,
        "cfg_lr_head": LR_HEAD,
        "cfg_lr_finetune": LR_FINETUNE,
        "cfg_batch_size": BATCH_SIZE,
        "cfg_early_stop": EARLY_STOP_PATIENCE,
        "cfg_label_smoothing": LABEL_SMOOTHING,
        "cfg_warmup_epochs": WARMUP_EPOCHS,
    })


@router.post("/train/start")
async def train_start(request: Request):
    user = await get_user_or_redirect(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    async with training_lock:
        if training_state["running"]:
            return RedirectResponse("/train?error=already_running", status_code=303)

        training_state["running"] = True
        training_state["progress"] = []
        training_state["current_epoch"] = 0
        training_state["total_epochs"] = 0
        training_state["best_val_acc"] = 0.0

    run_id = await save_training_run("running", EPOCHS_HEAD, EPOCHS_FINETUNE)
    training_state["run_id"] = run_id

    asyncio.create_task(_run_training_background(run_id))

    return RedirectResponse("/train?started=1", status_code=303)


async def _run_training_background(run_id: int):
    """Ejecuta el entrenamiento en segundo plano."""
    from train import run_training_sync

    loop = asyncio.get_event_loop()

    def progress_callback(info):
        msg = info.get("message", "")
        training_state["progress"].append(info)

        if "epoch" in info:
            training_state["current_epoch"] = info["epoch"]
        if "total_epochs" in info:
            training_state["total_epochs"] = info["total_epochs"]
        if "best_val_acc" in info:
            training_state["best_val_acc"] = info["best_val_acc"]

        return True  # continuar

    try:
        result = await loop.run_in_executor(
            None, lambda: run_training_sync(progress_callback=progress_callback)
        )

        if result["status"] == "completed":
            history_json = json.dumps(result["history"])
            await save_training_run(
                "completed", 0, 0,
                best_val_acc=result["best_val_acc"],
                final_train_acc=result.get("final_train_acc"),
                model_path=str(MODEL_PATH),
                history_json=history_json,
                run_id=run_id,
            )
            model_manager.load()
        elif result["status"] == "cancelled":
            await save_training_run("cancelled", 0, 0, run_id=run_id)

    except Exception as e:
        await save_training_run("error", 0, 0, error_message=str(e), run_id=run_id)
        training_state["progress"].append({"message": f"ERROR: {str(e)}", "phase": "error"})

    finally:
        training_state["running"] = False


@router.get("/train/progress")
async def train_progress():
    """SSE endpoint para progreso del entrenamiento en tiempo real."""
    async def event_generator():
        last_idx = 0
        while training_state["running"] or last_idx < len(training_state["progress"]):
            while last_idx < len(training_state["progress"]):
                info = training_state["progress"][last_idx]
                data = json.dumps(info)
                yield f"data: {data}\n\n"
                last_idx += 1

            if not training_state["running"] and last_idx >= len(training_state["progress"]):
                yield f"data: {json.dumps({'message': '[DONE]', 'phase': 'done'})}\n\n"
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/train/augment")
async def train_augment(request: Request):
    """Augmenta el dataset generando variaciones de imagenes existentes."""
    user = await get_user_or_redirect(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    async with augment_lock:
        if augment_state["running"]:
            return RedirectResponse("/train?error=augment_running", status_code=303)
        augment_state["running"] = True
        augment_state["progress"] = []
        augment_state["done"] = False

    form = await request.form()
    target = int(form.get("target", 400))

    asyncio.create_task(_run_augment_background(target))

    return RedirectResponse("/train?augment=started", status_code=303)


async def _run_augment_background(target: int):
    """Ejecuta la augmentacion en segundo plano."""
    import random
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps

    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    def get_images(class_dir):
        return [f for f in class_dir.iterdir()
                if f.is_file() and f.suffix.lower() in extensions]

    def augment_image(img):
        augmentations = [
            lambda im: im.transpose(Image.FLIP_LEFT_RIGHT),
            lambda im: im.rotate(random.uniform(-25, 25), fillcolor=(0, 0, 0)),
            lambda im: ImageEnhance.Brightness(im).enhance(random.uniform(0.7, 1.3)),
            lambda im: ImageEnhance.Contrast(im).enhance(random.uniform(0.7, 1.3)),
            lambda im: ImageEnhance.Color(im).enhance(random.uniform(0.7, 1.3)),
            lambda im: ImageEnhance.Sharpness(im).enhance(random.uniform(0.5, 1.5)),
            lambda im: im.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5))),
            lambda im: ImageOps.posterize(im, bits=random.choice([4, 5, 6])),
            lambda im: im.transform(im.size, Image.AFFINE,
                                    (1, random.uniform(-0.1, 0.1), random.randint(-10, 10),
                                     random.uniform(-0.1, 0.1), 1, random.randint(-10, 10)),
                                    fillcolor=(0, 0, 0)),
        ]
        num_aug = random.randint(2, 4)
        selected = random.sample(augmentations, min(num_aug, len(augmentations)))
        result = img.copy()
        for aug in selected:
            try:
                result = aug(result)
            except Exception:
                continue
        return result

    try:
        classes = [d for d in sorted(DATASET_DIR.iterdir()) if d.is_dir()]
        total_generated = 0

        for class_dir in classes:
            existing = get_images(class_dir)
            current = len(existing)

            if current >= target:
                augment_state["progress"].append({
                    "message": f"{class_dir.name}: {current} imagenes (ya cumple)",
                    "phase": "augment"
                })
                continue

            needed = target - current
            augment_state["progress"].append({
                "message": f"{class_dir.name}: {current} -> generando {needed} imagenes...",
                "phase": "augment"
            })

            generated = 0
            while generated < needed:
                src = random.choice(existing)
                try:
                    img = Image.open(src).convert('RGB')
                    aug_img = augment_image(img)
                    aug_name = f"aug_{generated:04d}_{src.stem}.jpg"
                    aug_path = class_dir / aug_name
                    if not aug_path.exists():
                        aug_img.save(str(aug_path), 'JPEG', quality=90)
                        generated += 1
                        total_generated += 1
                except Exception:
                    continue

            final = len(get_images(class_dir))
            augment_state["progress"].append({
                "message": f"{class_dir.name}: {final} imagenes completado",
                "phase": "augment"
            })

        augment_state["progress"].append({
            "message": f"Augmentacion completada! {total_generated} imagenes generadas",
            "phase": "done"
        })
        augment_state["done"] = True

    except Exception as e:
        augment_state["progress"].append({
            "message": f"ERROR: {str(e)}",
            "phase": "error"
        })

    finally:
        augment_state["running"] = False
