"""
Configuracion centralizada del sistema de confecciones.
Todas las constantes, rutas y credenciales en un solo lugar.
"""

import os
import json
import secrets
from pathlib import Path

# ─── Rutas del proyecto ──────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_PATH = BASE_DIR / "modelo_confecciones.pth"
CLASES_PATH = BASE_DIR / "clases.json"
DB_PATH = BASE_DIR / "data" / "app.db"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
RUNTIME_CONFIG_PATH = BASE_DIR / "data" / "runtime_config.json"

# ─── Modelo ML ───────────────────────────────────────────────────
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_HEAD = 10
EPOCHS_FINETUNE = 50
LR_HEAD = 0.001
LR_FINETUNE = 0.0002
VALIDATION_SPLIT = 0.2
LABEL_SMOOTHING = 0.1
CONFIDENCE_THRESHOLD = 0.5
TTA_AUGMENTATIONS = 7
EARLY_STOP_PATIENCE = 10
WEIGHT_DECAY = 1e-4
MIXUP_ALPHA = 0.2
GRAD_CLIP_MAX_NORM = 1.0
WARMUP_EPOCHS = 5
EMA_DECAY = 0.999

# ─── Rendimiento ─────────────────────────────────────────────────
USE_AMP = True
NUM_WORKERS = 4
PREFETCH_FACTOR = 4
CHECKPOINT_EVERY = 5
CHECKPOINT_PATH = BASE_DIR / "checkpoint.pth"

# ─── Admin por defecto ──────────────────────────────────────────
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "123456789"

# SECRET_KEY: variable de entorno o generado aleatoriamente por arranque
SECRET_KEY = os.environ.get("SECRET_KEY") or secrets.token_hex(32)

# ─── Servidor (valores por defecto, overridable via runtime config) ──
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000


def _load_runtime_config() -> dict:
    """Carga configuracion runtime desde JSON (HOST, PORT, etc.)."""
    if RUNTIME_CONFIG_PATH.exists():
        try:
            return json.loads(RUNTIME_CONFIG_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_runtime_config(cfg: dict):
    """Guarda configuracion runtime a JSON."""
    RUNTIME_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_runtime_config()
    existing.update(cfg)
    RUNTIME_CONFIG_PATH.write_text(json.dumps(existing, indent=2))


def get_host() -> str:
    cfg = _load_runtime_config()
    return cfg.get("host", DEFAULT_HOST)


def get_port() -> int:
    cfg = _load_runtime_config()
    return cfg.get("port", DEFAULT_PORT)


def get_trigger_api_key() -> str:
    """API key para endpoints de trigger. Vacio = sin proteccion (legacy)."""
    cfg = _load_runtime_config()
    return cfg.get("trigger_api_key", "")


# Valores iniciales (para compatibilidad con imports existentes)
HOST = get_host()
PORT = get_port()
