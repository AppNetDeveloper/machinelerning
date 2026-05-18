# Confecciones - Clasificador de Frutas v3.1

Sistema industrial de vision por computadora para clasificacion de cajas de frutas. Pipeline IoT completo: sensor fisico → captura de camara → clasificacion ML + escaneo QR → publicacion MQTT / webhook callback.

## Caracteristicas

- **Clasificacion ML** con EfficientNet-B0 (transfer learning), EMA, MixUp, TTA determinista
- **Entrenamiento en 2 fases** con early stopping, warmup + cosine annealing, gradient clipping
- **Panel web** dark theme con login, dashboard, dataset, entrenamiento en vivo (SSE), predicciones
- **Captura de fotos** via camaras USB e IP (RTSP, HTTP, MJPEG)
- **Escaner QR/codigos de barras** integrado con deteccion en tiempo real
- **API REST** para integracion con sistemas externos
- **Triggers IoT** por camara con webhook callbacks y API key auth
- **MQTT** para integracion con brokers IoT (publicacion de resultados)
- **Historial completo** de entrenamientos y predicciones en SQLite
- **Seguridad** bcrypt, CSRF (HMAC tokens), sesiones firmadas, auto-migracion SHA-256→bcrypt

## Stack

| Componente | Tecnologia |
|------------|------------|
| Backend | FastAPI + Jinja2 (server-side rendering) |
| ML | PyTorch EfficientNet-B0, EMA, MixUp, TTA |
| Base de datos | SQLite async (aiosqlite) |
| Auth | bcrypt, itsdangerous cookies, HMAC CSRF |
| MQTT | paho-mqtt |
| Camaras | OpenCV (USB, IP/MJPEG) |
| Frontend | Bootstrap 5 dark theme |

## Instalacion

```bash
git clone https://github.com/AppNetDeveloper/machinelerning.git
cd machinelerning

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

python app.py
```

El servidor inicia en `http://localhost:8000`.

**Login por defecto:** `admin` / `123456789`

## Estructura del Proyecto

```
app.py                  → Monta routers, lifespan, CSRF middleware, MQTT sync
config.py               → Constantes, rutas, runtime config (host/port/API key)
database.py             → SQLite async, bcrypt, auto-migracion SHA-256
auth.py                 → Sesiones firmadas, tokens CSRF (HMAC)
ml_model.py             → ModelManager singleton, TTA determinista, hot-reload
train.py                → EfficientNet-B0, EMA, MixUp, 2-phase training
mqtt_manager.py         → Cliente MQTT, trigger topic, extraccion de payload
camera.py               → Gestion de camaras USB/IP
routers/
  auth.py               → Login/logout
  pages.py              → Dashboard, historial, API docs
  dataset.py            → Upload, eliminar, nueva clase
  training.py           → Estado de entrenamiento, SSE progreso
  predictions.py        → Pagina y submit de prediccion
  settings.py           → Gestion de usuarios, config servidor, trigger key
  cameras.py            → Gestion de camaras, escaner QR
  api.py                → API REST + endpoints de trigger (API key)
  mqtt.py               → Configuracion MQTT
templates/              → HTML Jinja2 (Bootstrap 5 dark)
static/style.css        → CSS dark theme personalizado
dataset/                → Imagenes por clase (ImageFolder layout)
```

## Entrenamiento

El modelo usa EfficientNet-B0 con transfer learning en 2 fases:

1. **Fase 1 (Head):** Entrena solo la capa clasificadora (10 epocas, lr=0.001)
2. **Fase 2 (Fine-tune):** Descongela capas altas de EfficientNet (50 epocas, lr=0.0002)

Incluye:
- **EMA** (Exponential Moving Average) para mejor generalizacion
- **MixUp** para augmentation de datos
- **Label smoothing** (0.1) para regularizacion
- **WeightedRandomSampler** para clases desbalanceadas
- **Early stopping** con reset entre fases (paciencia: 10)
- **Warmup + CosineAnnealing** LR scheduler
- **Gradient clipping** (max norm 1.0)
- **AMP** para entrenamiento en GPU

## API REST

### Clasificacion ML

| Endpoint | Metodo | Descripcion |
|----------|--------|-------------|
| `/api/status` | GET | Estado del modelo |
| `/api/clases` | GET | Clases disponibles |
| `/api/predecir` | POST | Clasificar imagen (campo: `file`) |
| `/api/predecir-lote` | POST | Clasificar multiples imagenes |
| `/api/modelo/status` | GET | Estado detallado del modelo |
| `/api/dataset/stats` | GET | Estadisticas del dataset |
| `/api/entrenamientos` | GET | Historial de entrenamientos |
| `/api/predicciones` | GET | Historial de predicciones |

### Escaner QR

| Endpoint | Metodo | Descripcion |
|----------|--------|-------------|
| `/api/qr/camaras` | GET | Camaras disponibles para QR |
| `/api/qr/escanear` | POST | Escanea frame de camara (`camera_id`) |
| `/api/qr/escanear-imagen` | POST | Escanea imagen subida (`file`) |
| `/api/qr/historial` | GET | Historial de escaneos QR/barcode |

### Triggers IoT (por camara)

Cada camara registrada tiene su propio endpoint. Requiere API key si esta configurada.

| Endpoint | Metodo | Descripcion |
|----------|--------|-------------|
| `/api/disparar/{slug}` | POST | Captura → ML + QR → resultado o webhook |
| `/api/disparar` | POST | Lista endpoints disponibles |

```bash
# Disparar camara por slug:
curl -X POST http://localhost:8000/api/disparar/camara-almacen

# Con API key:
curl -X POST http://localhost:8000/api/disparar/camara-almacen \
  -H "X-API-Key: tu-api-key"
```

Respuesta:
```json
{
    "timestamp": "2026-05-18T22:30:00",
    "camera_slug": "camara-almacen",
    "camera_name": "Camara Almacen",
    "ml": {
        "confeccion": "confeccion2",
        "confianza": 89.86,
        "probabilidades": {"confeccion2": 89.86, "confeccion3": 7.21, "confeccion1": 2.93},
        "confiable": true
    },
    "qr": [{"type": "QR", "data": "https://example.com"}],
    "qr_count": 1
}
```

### Ejemplo de prediccion

```bash
curl -X POST http://localhost:8000/api/predecir -F "file=@imagen.jpg"
```

## Configuracion

Host, puerto y API key se configuran via panel web (`/settings`) y se guardan en `data/runtime_config.json`.

Parametros ML en `config.py`:

| Parametro | Valor | Descripcion |
|-----------|-------|-------------|
| `IMG_SIZE` | 224 | Tamano de imagen |
| `BATCH_SIZE` | 32 | Tamano de lote |
| `EPOCHS_HEAD` | 10 | Epocas fase 1 |
| `EPOCHS_FINETUNE` | 50 | Epocas fase 2 |
| `LR_HEAD` | 0.001 | Learning rate fase 1 |
| `LR_FINETUNE` | 0.0002 | Learning rate fase 2 |
| `EMA_DECAY` | 0.999 | Decay del EMA |
| `MIXUP_ALPHA` | 0.2 | Intensidad MixUp |
| `LABEL_SMOOTHING` | 0.1 | Suavizado de etiquetas |
| `EARLY_STOP_PATIENCE` | 10 | Epocas sin mejora para parar |
| `TTA_AUGMENTATIONS` | 7 | Augmentaciones TTA por prediccion |

## Camaras

- **USB** via OpenCV (escaneo automatico indices 0-4)
- **IP** via RTSP, HTTP, HTTPS, MJPEG con autenticacion

Captura directa a carpetas del dataset con nombres timestamped.

## Licencia

[Licencia Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](LICENSE)

**No se permite el uso comercial.**
