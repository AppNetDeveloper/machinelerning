"""Rutas de gestion de dataset."""

import uuid
from pathlib import Path
from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from config import DATASET_DIR, TEMPLATES_DIR
from database import get_dataset_stats
from auth import get_current_user

router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


async def get_user_or_redirect(request: Request):
    user = await get_current_user(request)
    if not user:
        return None
    return user


@router.get("/dataset", response_class=HTMLResponse)
async def dataset_page(request: Request):
    user = await get_user_or_redirect(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    dataset_stats = await get_dataset_stats()
    images_by_class = {}
    for class_dir in sorted(DATASET_DIR.iterdir()):
        if class_dir.is_dir():
            images = sorted([
                f.name for f in class_dir.iterdir()
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')
            ])
            images_by_class[class_dir.name] = images

    return templates.TemplateResponse(request, "dataset.html", {
        "request": request,
        "user": user,
        "dataset_stats": dataset_stats,
        "images_by_class": images_by_class,
        "total_images": sum(dataset_stats.values()),
    })


@router.post("/dataset/upload")
async def dataset_upload(request: Request, class_name: str = Form(...),
                         files: list[UploadFile] = File(...)):
    user = await get_user_or_redirect(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    class_dir = DATASET_DIR / class_name
    class_dir.mkdir(exist_ok=True)

    saved = 0
    for file in files:
        if file.content_type and file.content_type.startswith("image/"):
            content = await file.read()
            filename = f"{uuid.uuid4().hex[:8]}_{file.filename}"
            (class_dir / filename).write_bytes(content)
            saved += 1

    return RedirectResponse(f"/dataset?uploaded={saved}", status_code=303)


@router.post("/dataset/delete")
async def dataset_delete(request: Request, class_name: str = Form(...),
                         filename: str = Form(...)):
    user = await get_user_or_redirect(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    filepath = DATASET_DIR / class_name / filename
    if filepath.exists() and filepath.parent.parent == DATASET_DIR:
        filepath.unlink()

    return RedirectResponse(f"/dataset?deleted=1", status_code=303)


@router.post("/dataset/new-class")
async def dataset_new_class(request: Request, class_name: str = Form(...)):
    user = await get_user_or_redirect(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    class_dir = DATASET_DIR / class_name.strip().replace(" ", "_").lower()
    class_dir.mkdir(exist_ok=True)

    return RedirectResponse(f"/dataset?created={class_name}", status_code=303)
