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
BATCH_SIZE = 8
EPOCHS_HEAD = 8
EPOCHS_FINETUNE = 40
LR_HEAD = 0.0003
LR_FINETUNE = 0.00005
VALIDATION_SPLIT = 0.2
LABEL_SMOOTHING = 0.1
CONFIDENCE_THRESHOLD = 0.5
TTA_AUGMENTATIONS = 7
EARLY_STOP_PATIENCE = 8
WEIGHT_DECAY = 1e-4
MIXUP_ALPHA = 0.2

# ─── Admin por defecto ──────────────────────────────────────────
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "123456789"
SECRET_KEY = "confecciones-frutas-secret-key-2026"

# ─── Servidor ────────────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 8000
