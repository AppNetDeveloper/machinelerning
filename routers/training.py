"""Rutas de entrenamiento del modelo."""

import json
import asyncio
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from config import (
    DATASET_DIR, MODEL_PATH, TEMPLATES_DIR,
    EPOCHS_HEAD, EPOCHS_FINETUNE, LR_HEAD, LR_FINETUNE,
    BATCH_SIZE, EARLY_STOP_PATIENCE, LABEL_SMOOTHING, WARMUP_EPOCHS,
)
from database import save_training_run
from auth import get_current_user
from ml_model import model_manager

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

    return templates.TemplateResponse(request, "train.html", {
        "request": request,
        "user": user,
        "training_state": training_state,
        "model_loaded": model_manager.is_loaded,
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
