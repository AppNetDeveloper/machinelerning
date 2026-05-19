"""Rutas de camaras y escaner QR."""

import cv2
import numpy as np
from fastapi import APIRouter, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates

import asyncio
from config import TEMPLATES_DIR
from database import (
    get_dataset_stats, save_qr_scan, get_qr_scans,
    create_camera, get_camera_by_id, get_camera_by_slug,
    get_all_cameras, update_camera_db, delete_camera_db,
)
from auth import get_current_user
from camera import camera_manager

router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


async def get_user_or_redirect(request: Request):
    user = await get_current_user(request)
    if not user:
        return None
    return user


# ─── Camaras ─────────────────────────────────────────────────────

@router.get("/camera", response_class=HTMLResponse)
async def camera_page(request: Request):
    user = await get_user_or_redirect(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    dataset_stats = await get_dataset_stats()
    cameras = camera_manager.list_cameras()
    db_cameras = await get_all_cameras()
    return templates.TemplateResponse(request, "camera.html", {
        "request": request,
        "user": user,
        "cameras": cameras,
        "db_cameras": db_cameras,
        "dataset_stats": dataset_stats,
    })


@router.post("/camera/scan")
async def camera_scan(request: Request):
    user = await get_user_or_redirect(request)
    if not user:
        return JSONResponse({"error": "No autorizado"}, status_code=401)

    found = camera_manager.scan_usb_cameras()
    return JSONResponse({"cameras": found, "total": len(found)})


@router.post("/camera/add-ip")
async def camera_add_ip(request: Request):
    user = await get_user_or_redirect(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    form = await request.form()
    name = form.get("name", "").strip()
    url = form.get("url", "").strip()
    callback_url = form.get("callback_url", "").strip()
    callback_active = "true" if form.get("callback_active") else "false"

    if not name or not url:
        return RedirectResponse("/camera?error=Nombre+y+URL+requeridos", status_code=303)

    test = camera_manager.test_ip_camera(url)
    if "error" in test:
        return RedirectResponse(f"/camera?error={test['error']}", status_code=303)

    resolution = test.get("resolution", "")

    cam = await create_camera(
        name=name, camera_type="ip", source=url,
        resolution=resolution, callback_url=callback_url,
        callback_active=callback_active,
    )

    camera_manager.register_camera(
        str(cam["id"]), "ip", url, name, cam["slug"],
        resolution, callback_url, callback_active,
    )

    return RedirectResponse(f"/camera?success=Camara+{name}+registrada", status_code=303)


@router.post("/camera/register-usb")
async def camera_register_usb(request: Request):
    user = await get_user_or_redirect(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    form = await request.form()
    name = form.get("name", "").strip()
    source = form.get("source", "").strip()
    resolution = form.get("resolution", "").strip()
    callback_url = form.get("callback_url", "").strip()
    callback_active = "true" if form.get("callback_active") else "false"

    if not name or not source:
        return RedirectResponse("/camera?error=Datos+incompletos", status_code=303)

    existing = await get_all_cameras()
    for c in existing:
        if c["camera_type"] == "usb" and c["source"] == source:
            return RedirectResponse("/camera?error=Camara+USB+ya+registrada", status_code=303)

    cam = await create_camera(
        name=name, camera_type="usb", source=source,
        resolution=resolution, callback_url=callback_url,
        callback_active=callback_active,
    )

    camera_manager.register_camera(
        str(cam["id"]), "usb", source, name, cam["slug"],
        resolution, callback_url, callback_active,
    )

    return RedirectResponse(f"/camera?success=Camara+{name}+registrada", status_code=303)


@router.post("/camera/remove")
async def camera_remove(request: Request):
    user = await get_user_or_redirect(request)
    if not user:
        return JSONResponse({"error": "No autorizado"}, status_code=401)

    form = await request.form()
    cam_id = form.get("camera_id", "")

    try:
        await delete_camera_db(int(cam_id))
    except (ValueError, TypeError):
        pass

    camera_manager.remove_camera(cam_id)
    return RedirectResponse("/camera?success=Camara+eliminada", status_code=303)


@router.post("/camera/edit")
async def camera_edit(request: Request):
    user = await get_user_or_redirect(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    form = await request.form()
    cam_id = form.get("camera_id", "").strip()
    name = form.get("name", "").strip()
    source = form.get("source", "").strip()
    callback_url = form.get("callback_url", "").strip()
    callback_active = "true" if form.get("callback_active") else "false"

    if not cam_id or not name or not source:
        return RedirectResponse("/camera?error=Datos+incompletos", status_code=303)

    cam = await get_camera_by_id(int(cam_id))
    if not cam:
        return RedirectResponse("/camera?error=Camara+no+encontrada", status_code=303)

    cam_type = cam["camera_type"]
    resolution = cam.get("resolution", "")

    if cam_type == "ip":
        test = camera_manager.test_ip_camera(source)
        if "error" in test:
            return RedirectResponse(f"/camera?error={test['error']}", status_code=303)
        resolution = test.get("resolution", resolution)

    await update_camera_db(
        int(cam_id), name=name, source=source,
        resolution=resolution, callback_url=callback_url,
        callback_active=callback_active,
    )

    camera_manager.update_camera(
        cam_id, name, source, cam_type,
        resolution, callback_url, callback_active,
    )

    return RedirectResponse(f"/camera?success=Camara+{name}+actualizada", status_code=303)


@router.get("/camera/stream/{camera_id}")
async def camera_stream(camera_id: str):
    """MJPEG streaming de la camara en tiempo real."""

    async def generate():
        while True:
            frame_bytes = camera_manager.get_frame_jpeg(camera_id, quality=60)
            if frame_bytes is None:
                break
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            await asyncio.sleep(0.05)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@router.post("/camera/capture")
async def camera_capture(request: Request):
    user = await get_user_or_redirect(request)
    if not user:
        return JSONResponse({"error": "No autorizado"}, status_code=401)

    form = await request.form()
    cam_id = form.get("camera_id", "")
    class_name = form.get("class_name", "")

    if not cam_id or not class_name:
        return JSONResponse({"error": "Camara y clase requeridas"}, status_code=400)

    result = camera_manager.capture_and_save(cam_id, class_name)
    return JSONResponse(result)


@router.get("/camera/snapshot/{camera_id}")
async def camera_snapshot(camera_id: str):
    """Captura un solo frame como imagen JPEG."""
    frame_bytes = camera_manager.get_frame_jpeg(camera_id, quality=90)
    if frame_bytes is None:
        return Response(status_code=503, content="Camara no disponible")
    return Response(content=frame_bytes, media_type="image/jpeg")


# ─── Escaner QR ─────────────────────────────────────────────────

@router.get("/qr", response_class=HTMLResponse)
async def qr_page(request: Request):
    user = await get_user_or_redirect(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    cameras = camera_manager.list_cameras()
    history = await get_qr_scans(limit=50)
    return templates.TemplateResponse(request, "qr.html", {
        "request": request,
        "user": user,
        "cameras": cameras,
        "history": history,
    })


@router.post("/qr/scan")
async def qr_scan(request: Request):
    user = await get_user_or_redirect(request)
    if not user:
        return JSONResponse({"error": "No autenticado"}, status_code=401)

    data = await request.json()
    camera_id = data.get("camera_id", "")

    if not camera_id:
        return JSONResponse({"error": "camera_id requerido"}, status_code=400)

    results = camera_manager.scan_qr(camera_id)

    for r in results:
        await save_qr_scan(r["type"], r["data"], camera_id)

    return JSONResponse({"results": results, "count": len(results)})
