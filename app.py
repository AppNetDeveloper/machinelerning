"""
Panel web para gestionar el sistema de clasificacion de confecciones de frutas.
Punto de entrada principal que monta todos los routers.

Uso:
    python app.py

Accede a: http://localhost:8000
Acceso: admin / 123456789
"""

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import (
    DATASET_DIR, STATIC_DIR,
    get_host, get_port,
)
from database import init_db, change_password, get_camera_by_id
from auth import get_current_user, get_csrf_token, verify_csrf
from ml_model import model_manager
from camera import camera_manager
from mqtt_manager import mqtt_manager

from routers.auth import router as auth_router
from routers.pages import router as pages_router
from routers.dataset import router as dataset_router
from routers.training import router as training_router
from routers.predictions import router as predictions_router
from routers.settings import router as settings_router
from routers.cameras import router as cameras_router
from routers.api import router as api_router
from routers.mqtt import router as mqtt_router


def _mqtt_trigger_sync(camera_id):
    """Callback sincrono para triggers MQTT (ejecutado en hilo daemon)."""
    import asyncio

    async def _do_trigger():
        from routers.api import _run_trigger
        cam = await get_camera_by_id(int(camera_id))
        if not cam:
            return {"error": f"Camara ID {camera_id} no encontrada"}
        result = await _run_trigger(
            camera_id=str(cam["id"]),
            camera_slug=cam["slug"],
            camera_name=cam["name"],
            callback_url=cam.get("callback_url", ""),
            callback_active=cam.get("callback_active", "false"),
        )
        if hasattr(result, 'body'):
            return json.loads(result.body)
        return result

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_do_trigger())
    finally:
        loop.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup y shutdown de la aplicacion."""
    await init_db()
    from config import ADMIN_USERNAME, ADMIN_PASSWORD
    await change_password(ADMIN_USERNAME, ADMIN_PASSWORD)
    loaded = model_manager.load()
    if loaded:
        print(f"Modelo cargado: {len(model_manager.class_names)} clases")
    else:
        print("Sin modelo entrenado. Ve a Entrenar para crear uno.")
    await camera_manager.init_from_db()
    db_cameras = camera_manager.list_cameras()
    if db_cameras:
        print(f"  Camaras cargadas desde BD: {len(db_cameras)}")
    mqtt_manager.set_trigger_callback(_mqtt_trigger_sync)
    mqtt_manager.start()
    yield
    mqtt_manager.stop()
    camera_manager.release_all()


app = FastAPI(
    title="Panel de Confecciones de Frutas",
    description="Sistema completo de clasificacion y gestion de confecciones",
    version="3.1.0",
    lifespan=lifespan,
)

# ─── Static files ────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─── CSRF Middleware ─────────────────────────────────────────────
CSRF_EXEMPT_PATHS = {"/login"}
CSRF_EXEMPT_PREFIXES = ("/api/", "/camera/stream", "/camera/snapshot")


@app.middleware("http")
async def csrf_middleware(request: Request, call_next):
    """Verifica CSRF header en POST y establece cookie CSRF."""
    # Verificar CSRF en POST (excepto API y login)
    if request.method == "POST":
        path = request.url.path
        is_exempt = (
            path in CSRF_EXEMPT_PATHS
            or any(path.startswith(p) for p in CSRF_EXEMPT_PREFIXES)
        )
        if not is_exempt:
            session = request.cookies.get("session", "")
            if session and not await verify_csrf(request):
                return JSONResponse(
                    {"error": "Token CSRF invalido. Recarga la pagina."},
                    status_code=403,
                )

    response = await call_next(request)

    # Establecer cookie CSRF si hay sesion
    session = request.cookies.get("session", "")
    if session:
        csrf = get_csrf_token(request)
        response.set_cookie(
            "csrf_token", csrf, max_age=86400 * 7,
            httponly=False, samesite="lax",
        )

    return response


# ─── Dataset files con autenticacion ─────────────────────────────
@app.get("/dataset-files/{filepath:path}")
async def serve_dataset_file(request: Request, filepath: str):
    """Sirve archivos del dataset solo a usuarios autenticados."""
    user = await get_current_user(request)
    if not user:
        return JSONResponse({"error": "No autenticado"}, status_code=401)

    file_path = DATASET_DIR / filepath
    if not file_path.exists() or not file_path.is_file():
        return JSONResponse({"error": "Archivo no encontrado"}, status_code=404)

    try:
        file_path.resolve().relative_to(DATASET_DIR.resolve())
    except ValueError:
        return JSONResponse({"error": "Acceso denegado"}, status_code=403)

    return FileResponse(str(file_path))


# ─── Legacy redirect ─────────────────────────────────────────────
@app.post("/settings/save-trigger")
async def save_trigger_settings(request: Request):
    return JSONResponse({"redirect": "/camera"}, status_code=303, headers={"Location": "/camera"})


# ─── Montar routers ──────────────────────────────────────────────
app.include_router(auth_router)
app.include_router(pages_router)
app.include_router(dataset_router)
app.include_router(training_router)
app.include_router(predictions_router)
app.include_router(settings_router)
app.include_router(cameras_router)
app.include_router(api_router)
app.include_router(mqtt_router)


# ─── Main ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    host, port = get_host(), get_port()
    print("="*50)
    print("  Panel de Confecciones de Frutas v3.1")
    print(f"  http://localhost:{port}")
    print("  Acceso: admin / 123456789")
    print("="*50)
    uvicorn.run(app, host=host, port=port)
