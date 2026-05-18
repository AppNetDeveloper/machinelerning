"""API REST para uso programatico y triggers de sensores."""

import io
import cv2
import numpy as np
from datetime import datetime
from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

from config import (
    MODEL_PATH, DATASET_DIR, get_trigger_api_key,
)
from database import (
    save_prediction, save_qr_scan, get_qr_scans,
    get_training_runs, get_predictions, get_dataset_stats,
    get_camera_by_slug, get_all_cameras,
)
from ml_model import model_manager
from camera import camera_manager

router = APIRouter()


# ─── API REST (para uso programatico) ───────────────────────────

@router.get("/api")
async def api_root():
    return {
        "status": "ok",
        "modelo_cargado": model_manager.is_loaded,
        "clases_disponibles": model_manager.class_names,
        "version": "3.0.0",
    }


@router.get("/api/clases")
async def api_clases():
    return {"clases": model_manager.class_names, "total": len(model_manager.class_names)}


@router.post("/api/predecir")
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


@router.post("/api/predecir-lote")
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


@router.get("/api/modelo/status")
async def api_modelo_status():
    return {
        "cargado": model_manager.is_loaded,
        "clases": model_manager.class_names,
        "modelo_existe": MODEL_PATH.exists(),
        "modelo_tamano": MODEL_PATH.stat().st_size if MODEL_PATH.exists() else 0,
    }


@router.get("/api/dataset/stats")
async def api_dataset_stats():
    stats = await get_dataset_stats()
    return {"clases": stats, "total": sum(stats.values())}


@router.get("/api/entrenamientos")
async def api_entrenamientos():
    runs = await get_training_runs(20)
    return {"entrenamientos": runs}


@router.get("/api/predicciones")
async def api_predicciones():
    preds = await get_predictions(50)
    return {"predicciones": preds}


@router.get("/api/qr/camaras")
async def api_qr_camaras():
    cameras = camera_manager.list_cameras()
    return {"cameras": cameras, "total": len(cameras)}


@router.post("/api/qr/escanear")
async def api_qr_escanear(camera_id: str = ""):
    if not camera_id:
        return JSONResponse({"error": "camera_id requerido"}, status_code=400)

    results = camera_manager.scan_qr(camera_id)

    for r in results:
        await save_qr_scan(r["type"], r["data"], camera_id)

    return {"resultados": results, "total": len(results)}


@router.post("/api/qr/escanear-imagen")
async def api_qr_escanear_imagen(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse({"error": "Imagen no valida"}, status_code=400)

    results = []
    try:
        qr = cv2.QRCodeDetector()
        retval, decoded_info, points, _ = qr.detectAndDecodeMulti(frame)
        if retval and decoded_info:
            for text in decoded_info:
                if text:
                    results.append({"type": "QR", "data": text})
    except Exception:
        try:
            qr = cv2.QRCodeDetector()
            decoded, _, _ = qr.detectAndDecode(frame)
            if decoded:
                results.append({"type": "QR", "data": decoded})
        except Exception:
            pass

    try:
        barcode = cv2.barcode_BarcodeDetector()
        decoded, _, _ = barcode.detectAndDecode(frame)
        if decoded:
            results.append({"type": "BARCODE", "data": decoded})
    except Exception:
        pass

    for r in results:
        await save_qr_scan(r["type"], r["data"], "upload")

    return {"resultados": results, "total": len(results)}


@router.get("/api/qr/historial")
async def api_qr_historial(limit: int = 50):
    scans = await get_qr_scans(limit)
    return {"escaneos": scans, "total": len(scans)}


# ─── Trigger / Sensor por camara ─────────────────────────────────

async def _run_trigger(camera_id: str, camera_slug: str, camera_name: str,
                       callback_url: str = '', callback_active: str = 'false'):
    """Logica comun de trigger: captura, ML, QR, callback."""
    import httpx
    import asyncio

    frame, error = camera_manager.get_frame(camera_id)
    if error:
        return JSONResponse({"error": f"Error de camara: {error}"}, status_code=503)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    ml_result = {}
    try:
        if model_manager.is_loaded:
            ml_result = model_manager.predict(pil_image, use_tta=True)
            await save_prediction(
                f"trigger_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                ml_result.get("confeccion", ""),
                ml_result.get("confianza", 0),
                ml_result.get("probabilidades", {})
            )
    except Exception as e:
        ml_result = {"error": str(e)}

    qr_results = []
    try:
        qr_results = camera_manager.scan_qr(camera_id)
        for r in qr_results:
            await save_qr_scan(r["type"], r["data"], camera_id)
    except Exception as e:
        qr_results = [{"error": str(e)}]

    timestamp = datetime.now().isoformat()
    result = {
        "timestamp": timestamp,
        "camera_slug": camera_slug,
        "camera_name": camera_name,
        "ml": ml_result,
        "qr": qr_results,
        "qr_count": len([r for r in qr_results if "error" not in r]),
    }

    try:
        trigger_dir = DATASET_DIR.parent / "trigger_captures"
        trigger_dir.mkdir(exist_ok=True)
        filename = f"trigger_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = trigger_dir / filename
        cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        result["capture_path"] = str(filepath)
    except Exception:
        pass

    if callback_active == "true" and callback_url:

        async def send_callback():
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    await client.post(callback_url, json=result)
            except Exception as e:
                print(f"Callback error: {e}")

        asyncio.create_task(send_callback())
        return JSONResponse({
            "status": "processing",
            "callback": True,
            "callback_url": callback_url,
            "camera": camera_slug,
            "timestamp": timestamp,
        })
    else:
        return JSONResponse(result)


@router.post("/api/disparar/{camera_slug}")
async def api_disparar_camara(camera_slug: str, request: Request):
    """Trigger por camara especifica. Requiere API key si esta configurada."""
    api_key = get_trigger_api_key()
    if api_key:
        provided = request.headers.get("X-API-Key", "") or request.query_params.get("api_key", "")
        if not provided or provided != api_key:
            return JSONResponse({"error": "API key requerida o invalida"}, status_code=401)

    cam = await get_camera_by_slug(camera_slug)
    if not cam:
        return JSONResponse({"error": f"Camara '{camera_slug}' no encontrada"}, status_code=404)

    return await _run_trigger(
        str(cam["id"]), cam["slug"], cam["name"],
        cam.get("callback_url", ""), cam.get("callback_active", "false"),
    )


@router.post("/api/disparar")
async def api_disparar_legacy():
    """Endpoint legacy. Lista camaras disponibles."""
    db_cams = await get_all_cameras()
    if not db_cams:
        return JSONResponse({
            "error": "No hay camaras registradas. Registra camaras en la pagina de Camara.",
            "camaras_disponibles": [],
        }, status_code=404)

    endpoints = []
    for c in db_cams:
        endpoints.append({
            "slug": c["slug"],
            "name": c["name"],
            "endpoint": f"/api/disparar/{c['slug']}",
            "callback_active": c["callback_active"],
        })

    return JSONResponse({
        "mensaje": "Usa /api/disparar/{slug} para activar una camara especifica.",
        "camaras_disponibles": endpoints,
    })
