"""Rutas de historial y dashboard."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from config import TEMPLATES_DIR
from database import get_training_runs, get_predictions, get_dataset_stats
from auth import get_current_user
from ml_model import model_manager
from routers.training import training_state

router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


async def get_user_or_redirect(request: Request):
    user = await get_current_user(request)
    if not user:
        return None
    return user


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    user = await get_user_or_redirect(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    dataset_stats = await get_dataset_stats()
    runs = await get_training_runs(5)

    return templates.TemplateResponse(request, "dashboard.html", {
        "request": request,
        "user": user,
        "model_loaded": model_manager.is_loaded,
        "class_names": model_manager.class_names,
        "dataset_stats": dataset_stats,
        "total_images": sum(dataset_stats.values()),
        "recent_runs": runs,
        "training_running": training_state["running"],
    })


@router.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    user = await get_user_or_redirect(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    runs = await get_training_runs(50)
    predictions = await get_predictions(100)

    return templates.TemplateResponse(request, "history.html", {
        "request": request,
        "user": user,
        "runs": runs,
        "predictions": predictions,
    })


@router.get("/api-docs", response_class=HTMLResponse)
async def api_docs_page(request: Request):
    user = await get_user_or_redirect(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    return templates.TemplateResponse(request, "api_docs.html", {
        "request": request,
        "user": user,
    })
