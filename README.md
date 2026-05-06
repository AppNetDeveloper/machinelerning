# Confecciones - Clasificador de Frutas v3.0

Sistema completo de clasificacion de imagenes de frutas utilizando deep learning (ResNet50 transfer learning) con un panel web integrado para gestion del dataset, entrenamiento, predicciones y escaneo QR.

## Caracteristicas

- **Clasificacion de imagenes** con ResNet50 y Test-Time Augmentation (TTA)
- **Panel web** con login, dashboard, gestion de dataset, entrenamiento en vivo, predicciones y documentacion API
- **Captura de fotos** via camaras USB e IP (RTSP, HTTP, MJPEG)
- **Escaner QR/codigos de barras** integrado con deteccion en tiempo real
- **API REST** para integracion con otros sistemas
- **Historial completo** de entrenamientos y predicciones en SQLite
- **Gestion de usuarios** con autenticacion por sesiones firmadas

## Instalacion

```bash
# Clonar el repositorio
git clone <url-del-repo>
cd machinelerning

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar
python app.py
```

El servidor inicia en `http://localhost:8000`.

**Login por defecto:** `admin` / `REDACTED`

## Estructura del Proyecto

```
machinelerning/
├── app.py                  # Aplicacion principal (FastAPI)
├── config.py               # Configuracion centralizada
├── database.py             # Base de datos SQLite
├── auth.py                 # Autenticacion por sesiones
├── ml_model.py             # Carga y prediccion del modelo
├── camera.py               # Gestion de camaras USB/IP
├── train.py                # Entrenamiento del modelo
├── server.py               # API server standalone
├── requirements.txt        # Dependencias
├── templates/              # Templates Jinja2
│   ├── base.html           # Layout con sidebar
│   ├── login.html          # Pagina de login
│   ├── dashboard.html      # Dashboard principal
│   ├── dataset.html        # Gestion del dataset
│   ├── camera.html         # Captura de fotos
│   ├── qr.html             # Escaner QR
│   ├── train.html          # Control de entrenamiento
│   ├── predict.html        # Probar modelo
│   ├── history.html        # Historial
│   ├── api_docs.html       # Documentacion API
│   └── settings.html       # Ajustes
├── static/
│   └── style.css           # Estilos dark theme
├── dataset/                # Dataset de imagenes
│   ├── confeccion1/
│   ├── confeccion2/
│   └── confeccion3/
├── modelo_confecciones.pth # Modelo entrenado
└── clases.json             # Nombres de clases
```

## API REST

| Endpoint | Metodo | Descripcion |
|----------|--------|-------------|
| `/api` | GET | Estado del modelo |
| `/api/clases` | GET | Clases disponibles |
| `/api/predecir` | POST | Clasificar una imagen |
| `/api/predecir-lote` | POST | Clasificar multiples imagenes |
| `/api/modelo/status` | GET | Estado detallado del modelo |
| `/api/dataset/stats` | GET | Estadisticas del dataset |
| `/api/entrenamientos` | GET | Historial de entrenamientos |
| `/api/predicciones` | GET | Historial de predicciones |

### Ejemplo de prediccion

```bash
curl -X POST http://localhost:8000/api/predecir -F "file=@imagen.jpg"
```

Respuesta:
```json
{
    "confeccion": "confeccion2",
    "confianza": 89.86,
    "probabilidades": {
        "confeccion2": 89.86,
        "confeccion3": 7.21,
        "confeccion1": 2.93
    },
    "confiable": true,
    "tta": true
}
```

## Configuracion

Edite `config.py` para ajustar:

- **Modelo ML:** `IMG_SIZE`, `BATCH_SIZE`, `EPOCHS_HEAD`, `EPOCHS_FINETUNE`, `LR_*`
- **Admin:** `ADMIN_USERNAME`, `ADMIN_PASSWORD`
- **Servidor:** `HOST`, `PORT`
- **Rutas:** `DATASET_DIR`, `MODEL_PATH`, `DB_PATH`

## Entrenamiento

El modelo usa ResNet50 con transfer learning en 2 fases:

1. **Fase 1 (Head):** Entrena solo la capa final (15 epocas, lr=0.001)
2. **Fase 2 (Fine-tune):** Descongela layer3+4 de ResNet (40 epocas, lr=0.00005)

Incluye: WeightedRandomSampler para clases desbalanceadas, early stopping, data augmentation avanzada, y label smoothing.

## Camaras

Soporte para:
- **Camaras USB** via OpenCV (escaneo automatico de indices 0-4)
- **Camaras IP** via RTSP, HTTP, HTTPS, MJPEG con autenticacion

Captura directa a carpetas del dataset con nombres timestamped.

## Escaner QR

Deteccion de codigos QR y de barras (EAN, UPC, Code128) usando OpenCV integrado. Modo auto-escaneo continuo o escaneo manual.

## Licencia

Este proyecto esta licenciado bajo la [Licencia Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](LICENSE).

**No se permite el uso comercial.**
