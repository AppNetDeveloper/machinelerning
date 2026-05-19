# Confecciones de Frutas - Industrial Vision System

Low-cost IoT industrial vision system for fruit box classification. Physical sensor trigger → camera capture → ML classification + QR/barcode scan → MQTT publish/webhook callback.

## Stack

- **Backend**: FastAPI + Jinja2 (server-side rendering, not SPA)
- **ML**: PyTorch EfficientNet-B0 transfer learning, 2-phase training (head → fine-tune)
- **Database**: SQLite async (aiosqlite)
- **Auth**: bcrypt passwords, signed session cookies (itsdangerous), HMAC CSRF tokens
- **MQTT**: paho-mqtt for IoT trigger integration
- **Camera**: OpenCV (USB, IP/MJPEG streams)

## Project Structure

```
app.py              → Mounts routers, lifespan, CSRF middleware, MQTT sync callback
config.py           → All constants, paths, runtime config (host/port/API key JSON)
database.py         → SQLite async layer, bcrypt hashing, auto-migration from SHA-256
auth.py             → Session cookies, CSRF token generation/verification
ml_model.py         → ModelManager singleton, TTA predictions, hot-reload
train.py            → Training pipeline (EfficientNet-B0, EMA, MixUp, 2-phase)
mqtt_manager.py     → MQTT client, trigger topic, payload extraction
camera.py           → Camera manager (USB, IP streams)
routers/
  auth.py           → Login/logout
  pages.py          → Dashboard, history, api-docs
  dataset.py        → Upload, delete, new-class
  training.py       → Training state, SSE progress endpoint
  predictions.py    → Predict page and submit
  settings.py       → User management, server config, trigger key
  cameras.py        → Camera management, QR scanner
  api.py            → REST API + trigger endpoints (API key auth)
  mqtt.py           → MQTT configuration page
templates/          → Jinja2 HTML (Bootstrap 5 dark theme)
static/style.css    → Custom dark theme CSS
```

## Key Patterns

- **Routers**: All routes in `routers/` via `APIRouter`. `app.py` only mounts them.
- **CSRF**: Middleware checks `X-CSRF-Token` header on POST. Exempt: `/login`, `/api/*`, `/camera/stream/*`. JS in `base.html` intercepts all forms/fetch to add header.
- **Training state**: `routers/training.py` owns `training_state` dict + `training_lock` (asyncio.Lock). `routers/pages.py` imports it.
- **Runtime config**: Host, port, API key stored in `data/runtime_config.json` (not in code). Functions: `_load_runtime_config()`, `_save_runtime_config()`.
- **Model hot-reload**: `ModelManager.check_reload()` detects file mtime change and reloads.
- **EMA**: Exponential Moving Average of model weights during training. Applied before validation and final model save.

## Running

```bash
./venv/bin/python app.py          # Starts on port 8000
# Admin: admin / 123456789
```

## Training

```bash
./venv/bin/python train.py        # Standalone training
# Or via web panel: /train page → "Iniciar Entrenamiento"
```

Dataset: `dataset/` folder with subfolders per class (standard ImageFolder layout).

## Key Config (config.py)

- `BATCH_SIZE = 32`, `IMG_SIZE = 224`
- `EPOCHS_HEAD = 10`, `EPOCHS_FINETUNE = 50`
- `LR_HEAD = 0.001`, `LR_FINETUNE = 0.0002`
- `EARLY_STOP_PATIENCE = 10`
- `EMA_DECAY = 0.999`
- `MIXUP_ALPHA = 0.2`, `LABEL_SMOOTHING = 0.1`
- `TTA_AUGMENTATIONS = 7`

## Conventions

- Language: Spanish for UI, comments, and user-facing strings
- Python 3.14+ (venv at `./venv/`)
- No type hints used consistently — match existing style
- Database tables auto-created on server startup via `init_db()`
- SECRET_KEY: env var or random per startup (sessions invalidate on restart)
- Git: `.gitignore` excludes `dataset/`, `*.pth`, `__pycache__/`, `data/`

## Testing

No test suite. Test manually:
- Server: `curl http://localhost:8000/login` (200)
- API: `curl http://localhost:8000/api/status`
- CSRF: POST without header → 403
- Auth: `/dataset/files/*` without session → 401
