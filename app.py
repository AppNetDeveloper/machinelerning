"""
Panel web para gestionar el sistema de clasificacion de confecciones de frutas.
Incluye: login, dashboard, gestion de dataset, entrenamiento, predicciones y API docs.

Uso:
    python app.py

Accede a: http://localhost:8000
Login: admin / REDACTED
"""

import io
import json
import asyncio
import shutil
import uuid
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

from config import (
    DATASET_DIR, MODEL_PATH, CLASES_PATH, STATIC_DIR, TEMPLATES_DIR,
    HOST, PORT, CONFIDENCE_THRESHOLD, IMG_SIZE,
)
from database import (
    init_db, save_training_run, get_training_runs, get_training_run,
    save_prediction, get_predictions, get_dataset_stats,
    get_all_users, add_user, change_password, delete_user,
    save_qr_scan, get_qr_scans,
)
from auth import authenticate, get_current_user, require_auth
from ml_model import model_manager
from camera import camera_manager


# ─── Estado global del entrenamiento ─────────────────────────────
training_state = {
    "running": False,
    "progress": [],
    "current_epoch": 0,
    "total_epochs": 0,
    "best_val_acc": 0.0,
    "run_id": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup y shutdown de la aplicacion."""
    await init_db()
    loaded = model_manager.load()
    if loaded:
        print(f"Modelo cargado: {len(model_manager.class_names)} clases")
    else:
        print("Sin modelo entrenado. Ve a Entrenar para crear uno.")
    yield
    camera_manager.release_all()


app = FastAPI(
    title="Panel de Confecciones de Frutas",
    description="Sistema completo de clasificacion y gestion de confecciones",
    version="3.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/dataset-files", StaticFiles(directory=str(DATASET_DIR)), name="dataset-files")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ─── Helpers ─────────────────────────────────────────────────────
async def get_user_or_redirect(request: Request):
    """Obtiene el usuario o retorna redirect a login."""
    user = await get_current_user(request)
    if not user:
        return None
    return user


def flash(request: Request, message: str, category: str = "info"):
    """Guarda un mensaje flash en la sesion."""
    if not hasattr(request.state, '_flash'):
        request.state._flash = []
    request.state._flash.append({"message": message, "category": category})


# ─── Rutas de autenticacion ─────────────────────────────────────
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    user = await get_current_user(request)
    if user:
        return RedirectResponse("/", status_code=303)
    return templates.TemplateResponse(request, "login.html", {"request": request})


@app.post("/login")
async def login_submit(request: Request):
    form = await request.form()
    username = form.get("username", "")
    password = form.get("password", "")

    token = await authenticate(username, password)
    if not token:
        return templates.TemplateResponse(request, "login.html", {
            "request": request,
            "error": "Usuario o contrasena incorrectos",
        })

    response = RedirectResponse("/", status_code=303)
    response.set_cookie("session", token, max_age=86400 * 7, httponly=True)
    return response


@app.get("/logout")
async def logout():
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie("session")
    return response


# ─── Dashboard ───────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
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


# ─── Dataset ─────────────────────────────────────────────────────
@app.get("/dataset", response_class=HTMLResponse)
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


@app.post("/dataset/upload")
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
            ext = Path(file.filename or "image.jpg").suffix or ".jpg"
            filename = f"{uuid.uuid4().hex[:8]}_{file.filename}"
            (class_dir / filename).write_bytes(content)
            saved += 1

    return RedirectResponse(f"/dataset?uploaded={saved}", status_code=303)


@app.post("/dataset/delete")
async def dataset_delete(request: Request, class_name: str = Form(...),
                         filename: str = Form(...)):
    user = await get_user_or_redirect(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    filepath = DATASET_DIR / class_name / filename
    if filepath.exists() and filepath.parent.parent == DATASET_DIR:
        filepath.unlink()

    return RedirectResponse(f"/dataset?deleted=1", status_code=303)


@app.post("/dataset/new-class")
async def dataset_new_class(request: Request, class_name: str = Form(...)):
    user = await get_user_or_redirect(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    class_dir = DATASET_DIR / class_name.strip().replace(" ", "_").lower()
    class_dir.mkdir(exist_ok=True)

    return RedirectResponse(f"/dataset?created={class_name}", status_code=303)


# ─── Entrenamiento ───────────────────────────────────────────────
@app.get("/train", response_class=HTMLResponse)
async def train_page(request: Request):
    user = await get_user_or_redirect(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    return templates.TemplateResponse(request, "train.html", {
        "request": request,
        "user": user,
        "training_state": training_state,
        "model_loaded": model_manager.is_loaded,
    })


@app.post("/train/start")
async def train_start(request: Request):
    user = await get_user_or_redirect(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    if training_state["running"]:
        return RedirectResponse("/train?error=already_running", status_code=303)

    # Resetear estado
    training_state["running"] = True
    training_state["progress"] = []
    training_state["current_epoch"] = 0
    training_state["total_epochs"] = 0
    training_state["best_val_acc"] = 0.0

    # Guardar registro en DB
    from config import EPOCHS_HEAD, EPOCHS_FINETUNE
    run_id = await save_training_run("running", EPOCHS_HEAD, EPOCHS_FINETUNE)
    training_state["run_id"] = run_id

    # Ejecutar entrenamiento en background
    asyncio.create_task(_run_training_background(run_id))

    return RedirectResponse("/train?started=1", status_code=303)


async def _run_training_background(run_id: int):
    """Ejecuta el entrenamiento en segundo plano."""
    from train import run_training_sync

    loop = asyncio.get_event_loop()

    def progress_callback(info):
        """Callback que se llama por cada epoca."""
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
            # Recargar modelo
            model_manager.load()
        elif result["status"] == "cancelled":
            await save_training_run("cancelled", 0, 0, run_id=run_id)

    except Exception as e:
        await save_training_run("error", 0, 0, error_message=str(e), run_id=run_id)
        training_state["progress"].append({"message": f"ERROR: {str(e)}", "phase": "error"})

    finally:
        training_state["running"] = False


@app.get("/train/progress")
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


# ─── Predicciones ────────────────────────────────────────────────
@app.get("/predict", response_class=HTMLResponse)
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


@app.post("/predict")
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

        # Guardar prediccion en DB
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


# ─── Historial ───────────────────────────────────────────────────
@app.get("/history", response_class=HTMLResponse)
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


# ─── API Docs ────────────────────────────────────────────────────
@app.get("/api-docs", response_class=HTMLResponse)
async def api_docs_page(request: Request):
    user = await get_user_or_redirect(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    return templates.TemplateResponse(request, "api_docs.html", {
        "request": request,
        "user": user,
    })


# ─── Ajustes ─────────────────────────────────────────────────────
@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    user = await get_user_or_redirect(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    users = await get_all_users()
    return templates.TemplateResponse(request, "settings.html", {
        "request": request,
        "user": user,
        "users": users,
        "host": HOST,
        "port": PORT,
        "success": request.query_params.get("success"),
        "error": request.query_params.get("error"),
    })


@app.post("/settings/change-password")
async def settings_change_password(request: Request):
    user = await get_user_or_redirect(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    form = await request.form()
    target_user = form.get("username", "")
    new_pass = form.get("new_password", "")
    confirm_pass = form.get("confirm_password", "")

    if not new_pass or len(new_pass) < 4:
        return RedirectResponse("/settings?error=La+contrasena+debe+tener+al+menos+4+caracteres", status_code=303)

    if new_pass != confirm_pass:
        return RedirectResponse("/settings?error=Las+contrasenas+no+coinciden", status_code=303)

    ok, msg = await change_password(target_user, new_pass)
    if ok:
        return RedirectResponse(f"/settings?success={msg}", status_code=303)
    else:
        return RedirectResponse(f"/settings?error={msg}", status_code=303)


@app.post("/settings/add-user")
async def settings_add_user(request: Request):
    user = await get_user_or_redirect(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    form = await request.form()
    new_user = form.get("new_username", "").strip()
    new_pass = form.get("new_user_password", "")

    if not new_user or not new_pass:
        return RedirectResponse("/settings?error=Usuario+y+contrasena+requeridos", status_code=303)

    if len(new_pass) < 4:
        return RedirectResponse("/settings?error=La+contrasena+debe+tener+al+menos+4+caracteres", status_code=303)

    ok, msg = await add_user(new_user, new_pass)
    if ok:
        return RedirectResponse(f"/settings?success={msg}", status_code=303)
    else:
        return RedirectResponse(f"/settings?error={msg}", status_code=303)


@app.post("/settings/delete-user")
async def settings_delete_user(request: Request):
    user = await get_user_or_redirect(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    form = await request.form()
    target = form.get("username", "")

    if target == user:
        return RedirectResponse("/settings?error=No+puedes+eliminarte+a+ti+mismo", status_code=303)

    ok, msg = await delete_user(target)
    if ok:
        return RedirectResponse(f"/settings?success={msg}", status_code=303)
    else:
        return RedirectResponse(f"/settings?error={msg}", status_code=303)


@app.post("/settings/save-server")
async def settings_save_server(request: Request):
    user = await get_user_or_redirect(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    form = await request.form()
    new_host = form.get("host", "0.0.0.0").strip()
    new_port = form.get("port", "8000").strip()

    try:
        new_port = int(new_port)
        if not (1024 <= new_port <= 65535):
            raise ValueError()
    except ValueError:
        return RedirectResponse("/settings?error=Puerto+debe+ser+un+numero+entre+1024+y+65535", status_code=303)

    # Actualizar config.py
    config_path = Path(__file__).parent / "config.py"
    content = config_path.read_text()
    import re
    content = re.sub(r'^HOST\s*=.*$', f'HOST = "{new_host}"', content, flags=re.MULTILINE)
    content = re.sub(r'^PORT\s*=.*$', f'PORT = {new_port}', content, flags=re.MULTILINE)
    config_path.write_text(content)

    return RedirectResponse(
        f"/settings?success=Configuracion+guardada.+Reinicia+el+servidor+para+aplicar+(puerto={new_port},+host={new_host})",
        status_code=303
    )


# ─── Camaras ─────────────────────────────────────────────────────
@app.get("/camera", response_class=HTMLResponse)
async def camera_page(request: Request):
    user = await get_user_or_redirect(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    dataset_stats = await get_dataset_stats()
    cameras = camera_manager.list_cameras()
    return templates.TemplateResponse(request, "camera.html", {
        "request": request,
        "user": user,
        "cameras": cameras,
        "dataset_stats": dataset_stats,
    })


@app.post("/camera/scan")
async def camera_scan(request: Request):
    user = await get_user_or_redirect(request)
    if not user:
        return JSONResponse({"error": "No autorizado"}, status_code=401)

    found = camera_manager.scan_usb_cameras()
    return JSONResponse({"cameras": found, "total": len(found)})


@app.post("/camera/add-ip")
async def camera_add_ip(request: Request):
    user = await get_user_or_redirect(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    form = await request.form()
    name = form.get("name", "").strip()
    url = form.get("url", "").strip()

    if not name or not url:
        return RedirectResponse("/camera?error=Nombre+y+URL+requeridos", status_code=303)

    result = camera_manager.add_ip_camera(name, url)
    if "error" in result:
        return RedirectResponse(f"/camera?error={result['error']}", status_code=303)

    return RedirectResponse(f"/camera?success=Camara+{name}+anadida", status_code=303)


@app.post("/camera/remove")
async def camera_remove(request: Request):
    user = await get_user_or_redirect(request)
    if not user:
        return JSONResponse({"error": "No autorizado"}, status_code=401)

    form = await request.form()
    cam_id = form.get("camera_id", "")
    camera_manager.remove_camera(cam_id)
    return RedirectResponse("/camera?success=Camara+eliminada", status_code=303)


@app.get("/camera/stream/{camera_id}")
async def camera_stream(camera_id: str):
    """MJPEG streaming de la camara en tiempo real."""
    import asyncio
    from fastapi.responses import StreamingResponse

    async def generate():
        while True:
            frame_bytes = camera_manager.get_frame_jpeg(camera_id, quality=60)
            if frame_bytes is None:
                break
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            await asyncio.sleep(0.05)  # ~20 FPS

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/camera/capture")
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


@app.get("/camera/snapshot/{camera_id}")
async def camera_snapshot(camera_id: str):
    """Captura un solo frame como imagen JPEG."""
    from fastapi.responses import Response
    frame_bytes = camera_manager.get_frame_jpeg(camera_id, quality=90)
    if frame_bytes is None:
        return Response(status_code=503, content="Camara no disponible")
    return Response(content=frame_bytes, media_type="image/jpeg")


# ─── Escaner QR ─────────────────────────────────────────────────
@app.get("/qr", response_class=HTMLResponse)
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


@app.post("/qr/scan")
async def qr_scan(request: Request):
    user = await get_user_or_redirect(request)
    if not user:
        return JSONResponse({"error": "No autenticado"}, status_code=401)

    data = await request.json()
    camera_id = data.get("camera_id", "")

    if not camera_id:
        return JSONResponse({"error": "camera_id requerido"}, status_code=400)

    results = camera_manager.scan_qr(camera_id)

    # Guardar en historial
    for r in results:
        await save_qr_scan(r["type"], r["data"], camera_id)

    return JSONResponse({"results": results, "count": len(results)})


# ─── API REST (para uso programatico) ───────────────────────────
@app.get("/api")
async def api_root():
    return {
        "status": "ok",
        "modelo_cargado": model_manager.is_loaded,
        "clases_disponibles": model_manager.class_names,
        "version": "3.0.0",
    }


@app.get("/api/clases")
async def api_clases():
    return {"clases": model_manager.class_names, "total": len(model_manager.class_names)}


@app.post("/api/predecir")
async def api_predecir(file: UploadFile = File(...)):
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Modelo no cargado. Ejecuta el entrenamiento primero.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        result = model_manager.predict(image, use_tta=True)

        await save_prediction(
            file.filename or "unknown",
            result["confeccion"],
            result["confianza"],
            result["probabilidades"],
        )

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando imagen: {str(e)}")


@app.post("/api/predecir-lote")
async def api_predecir_lote(files: list[UploadFile] = File(...)):
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Modelo no cargado.")

    resultados = []
    for file in files:
        if not file.content_type or not file.content_type.startswith("image/"):
            resultados.append({"archivo": file.filename, "error": "No es una imagen valida"})
            continue
        try:
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            result = model_manager.predict(image, use_tta=True)
            resultados.append({"archivo": file.filename, **result})
        except Exception as e:
            resultados.append({"archivo": file.filename, "error": str(e)})

    return {"resultados": resultados, "total": len(resultados)}


@app.get("/api/modelo/status")
async def api_modelo_status():
    return {
        "cargado": model_manager.is_loaded,
        "clases": model_manager.class_names,
        "modelo_existe": MODEL_PATH.exists(),
        "modelo_tamano": MODEL_PATH.stat().st_size if MODEL_PATH.exists() else 0,
    }


@app.get("/api/dataset/stats")
async def api_dataset_stats():
    stats = await get_dataset_stats()
    return {"clases": stats, "total": sum(stats.values())}


@app.get("/api/entrenamientos")
async def api_entrenamientos():
    runs = await get_training_runs(20)
    return {"entrenamientos": runs}


@app.get("/api/predicciones")
async def api_predicciones():
    preds = await get_predictions(50)
    return {"predicciones": preds}


# ─── Main ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("="*50)
    print("  Panel de Confecciones de Frutas v3.0")
    print(f"  http://localhost:{PORT}")
    print("  Login: admin / REDACTED")
    print("="*50)
    uvicorn.run(app, host=HOST, port=PORT)
