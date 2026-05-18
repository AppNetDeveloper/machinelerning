"""
Configuracion centralizada del sistema de confecciones.
Todas las constantes, rutas y credenciales en un solo lugar.
"""

import os
from pathlib import Path

# ─── Rutas del proyecto ──────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_PATH = BASE_DIR / "modelo_confecciones.pth"
CLASES_PATH = BASE_DIR / "clases.json"
DB_PATH = BASE_DIR / "data" / "app.db"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

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

# ─── Rendimiento ─────────────────────────────────────────────────
USE_AMP = True
NUM_WORKERS = 4
PREFETCH_FACTOR = 4
CHECKPOINT_EVERY = 5
CHECKPOINT_PATH = BASE_DIR / "checkpoint.pth"

# ─── Admin por defecto ──────────────────────────────────────────
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "123456789"
SECRET_KEY = "confecciones-frutas-secret-key-2026"

# ─── Servidor ────────────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 8000
