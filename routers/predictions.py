"""Rutas de predicciones."""

import io
from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from PIL import Image

from config import TEMPLATES_DIR
from database import save_prediction
from auth import get_current_user
from ml_model import model_manager

router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


async def get_user_or_redirect(request: Request):
    user = await get_current_user(request)
    if not user:
        return None
    return user


@router.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    user = await get_user_or_redirect(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    return templates.TemplateResponse(request, "predict.html", {
        "request": request,
        "user": user,
        "model_loaded": model_manager.is_loaded,
        "class_names": model_manager.class_names,
    })


@router.post("/predict")
async def predict_submit(request: Request, file: UploadFile = File(...),
                         use_tta: bool = Form(True)):
    user = await get_user_or_redirect(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    if not model_manager.is_loaded:
        return templates.TemplateResponse(request, "predict.html", {
            "request": request,
            "user": user,
            "model_loaded": False,
            "error": "No hay modelo cargado. Entrena primero.",
        })

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        result = model_manager.predict(image, use_tta=use_tta)

        await save_prediction(
            file.filename or "unknown",
            result["confeccion"],
            result["confianza"],
            result["probabilidades"],
        )

        return templates.TemplateResponse(request, "predict.html", {
            "request": request,
            "user": user,
            "model_loaded": True,
            "class_names": model_manager.class_names,
            "result": result,
            "filename": file.filename,
        })

    except Exception as e:
        return templates.TemplateResponse(request, "predict.html", {
            "request": request,
            "user": user,
            "model_loaded": True,
            "class_names": model_manager.class_names,
            "error": f"Error procesando imagen: {str(e)}",
        })
