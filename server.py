"""
API server standalone para clasificacion de confecciones de frutas.
Usa el modulo ml_model para la logica de prediccion.

Este servidor es independiente del panel web (app.py).
Para el panel web completo, usa: python app.py

Uso:
    source venv/bin/activate
    python server.py

El servidor arranca en: http://localhost:8000
Documentacion automatica en: http://localhost:8000/docs
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io

from ml_model import model_manager
from config import CONFIDENCE_THRESHOLD

app = FastAPI(
    title="Clasificador de Confecciones de Frutas",
    description="API standalone para clasificacion. Para el panel web, usa app.py.",
    version="3.0.0",
)


@app.on_event("startup")
async def startup():
    loaded = model_manager.load()
    if loaded:
        print(f"Modelo cargado: {len(model_manager.class_names)} clases")
        print(f"TTA activado: {model_manager.tta_transforms.__len__()} augmentaciones")
    else:
        print("ADVERTENCIA: Modelo no encontrado. Ejecuta 'python train.py' primero.")


@app.get("/")
async def root():
    return {
        "status": "ok",
        "modelo_cargado": model_manager.is_loaded,
        "clases_disponibles": model_manager.class_names,
        "tta_augmentations": len(model_manager.tta_transforms),
    }


@app.get("/clases")
async def get_clases():
    return {"clases": model_manager.class_names, "total": len(model_manager.class_names)}


@app.post("/predecir")
async def predecir(file: UploadFile = File(...)):
    """Recibe una foto y devuelve la confeccion predicha (con TTA)."""
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Modelo no cargado. Ejecuta train.py primero.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen (JPG, PNG, etc.)")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        result = model_manager.predict(image, use_tta=True)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando imagen: {str(e)}")


@app.post("/predecir-lote")
async def predecir_lote(files: list[UploadFile] = File(...)):
    """Recibe multiples fotos y devuelve la prediccion para cada una."""
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


if __name__ == "__main__":
    import uvicorn
    print("Iniciando servidor standalone de clasificacion...")
    print("Documentacion API: http://localhost:8000/docs")
    print("Para el panel web completo: python app.py")
    uvicorn.run(app, host="0.0.0.0", port=8000)
