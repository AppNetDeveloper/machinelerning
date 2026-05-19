"""
Base de datos SQLite para el panel de administracion.
Maneja usuarios, historial de entrenamientos y predicciones.
"""

import aiosqlite
import hashlib
import os
import json
import re
import unicodedata
from datetime import datetime
from config import DB_PATH, ADMIN_USERNAME, ADMIN_PASSWORD
import bcrypt


def hash_password(password: str) -> str:
    """Hashea una contraseña con bcrypt. Retorna el hash como string."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _verify_bcrypt(password: str, hashed: str) -> bool:
    """Verifica password contra hash bcrypt."""
    return bcrypt.checkpw(password.encode(), hashed.encode())


def _verify_sha256(password: str, stored_hash: str, salt: str) -> bool:
    """Verifica password contra hash SHA-256 legacy."""
    h = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
    return h == stored_hash


async def init_db():
    """Crea las tablas si no existen y el usuario admin por defecto."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                role TEXT DEFAULT 'admin',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS training_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                status TEXT DEFAULT 'pending',
                epochs_head INTEGER,
                epochs_finetune INTEGER,
                best_val_acc REAL,
                final_train_acc REAL,
                model_path TEXT,
                history_json TEXT,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_name TEXT,
                predicted_class TEXT,
                confidence REAL,
                probabilities_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS qr_scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_type TEXT,
                data TEXT NOT NULL,
                camera_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE TABLE IF NOT EXISTS cameras (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                slug TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                camera_type TEXT NOT NULL,
                source TEXT NOT NULL,
                resolution TEXT DEFAULT '',
                callback_url TEXT DEFAULT '',
                callback_active TEXT DEFAULT 'false',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS mqtt_triggers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                camera_id INTEGER NOT NULL,
                result_topic TEXT DEFAULT '',
                payload_vars TEXT DEFAULT '',
                active TEXT DEFAULT 'true',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE CASCADE
            );
        """)

        # Migracion: agregar payload_vars si no existe
        try:
            await db.execute("SELECT payload_vars FROM mqtt_triggers LIMIT 1")
        except Exception:
            await db.execute("ALTER TABLE mqtt_triggers ADD COLUMN payload_vars TEXT DEFAULT ''")

        # Crear admin si no existe
        cursor = await db.execute("SELECT id FROM users WHERE username = ?", (ADMIN_USERNAME,))
        if not await cursor.fetchone():
            pw_hash = hash_password(ADMIN_PASSWORD)
            await db.execute(
                "INSERT INTO users (username, password_hash, password_salt, role) VALUES (?, ?, ?, ?)",
                (ADMIN_USERNAME, pw_hash, "", "admin")
            )

        await db.commit()


async def get_user(username: str) -> dict | None:
    """Busca un usuario por nombre."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = await cursor.fetchone()
        return dict(row) if row else None


async def verify_user(username: str, password: str) -> bool:
    """Verifica credenciales de usuario. Soporta bcrypt y SHA-256 legacy."""
    user = await get_user(username)
    if not user:
        return False

    stored_hash = user["password_hash"]

    # bcrypt hashes empiezan con $2b$
    if stored_hash.startswith("$2b$"):
        if _verify_bcrypt(password, stored_hash):
            return True
    else:
        # Legacy SHA-256: verificar y migrar a bcrypt
        if _verify_sha256(password, stored_hash, user.get("password_salt", "")):
            # Migrar a bcrypt silenciosamente
            new_hash = hash_password(password)
            async with aiosqlite.connect(str(DB_PATH)) as db:
                await db.execute(
                    "UPDATE users SET password_hash = ?, password_salt = '' WHERE username = ?",
                    (new_hash, username)
                )
                await db.commit()
            return True

    return False


async def save_training_run(status: str, epochs_head: int, epochs_finetune: int,
                            best_val_acc: float = None, final_train_acc: float = None,
                            model_path: str = None, history_json: str = None,
                            error_message: str = None, run_id: int = None) -> int:
    """Guarda o actualiza un registro de entrenamiento."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        if run_id:
            await db.execute("""
                UPDATE training_runs SET status=?, best_val_acc=?, final_train_acc=?,
                model_path=?, history_json=?, error_message=?,
                completed_at=CASE WHEN ? IN ('completed','error') THEN CURRENT_TIMESTAMP ELSE completed_at END
                WHERE id=?
            """, (status, best_val_acc, final_train_acc, model_path, history_json,
                  error_message, status, run_id))
            await db.commit()
            return run_id
        else:
            cursor = await db.execute("""
                INSERT INTO training_runs (status, epochs_head, epochs_finetune)
                VALUES (?, ?, ?)
            """, (status, epochs_head, epochs_finetune))
            await db.commit()
            return cursor.lastrowid


async def get_training_runs(limit: int = 20) -> list[dict]:
    """Obtiene el historial de entrenamientos."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM training_runs ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def get_training_run(run_id: int) -> dict | None:
    """Obtiene un entrenamiento especifico."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM training_runs WHERE id = ?", (run_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None


async def save_prediction(image_name: str, predicted_class: str,
                          confidence: float, probabilities: dict):
    """Guarda una prediccion en el historial."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute("""
            INSERT INTO predictions (image_name, predicted_class, confidence, probabilities_json)
            VALUES (?, ?, ?, ?)
        """, (image_name, predicted_class, confidence, json.dumps(probabilities)))
        await db.commit()


async def get_predictions(limit: int = 50) -> list[dict]:
    """Obtiene el historial de predicciones."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def get_all_users() -> list[dict]:
    """Obtiene todos los usuarios."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT id, username, role, created_at FROM users ORDER BY id")
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def add_user(username: str, password: str, role: str = "admin") -> tuple[bool, str]:
    """Anade un nuevo usuario. Retorna (exito, mensaje)."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        cursor = await db.execute("SELECT id FROM users WHERE username = ?", (username,))
        if await cursor.fetchone():
            return False, "El usuario ya existe"

        pw_hash = hash_password(password)
        await db.execute(
            "INSERT INTO users (username, password_hash, password_salt, role) VALUES (?, ?, ?, ?)",
            (username, pw_hash, "", role)
        )
        await db.commit()
        return True, f"Usuario '{username}' creado correctamente"


async def change_password(username: str, new_password: str) -> tuple[bool, str]:
    """Cambia la contrasena de un usuario. Retorna (exito, mensaje)."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        cursor = await db.execute("SELECT id FROM users WHERE username = ?", (username,))
        if not await cursor.fetchone():
            return False, "Usuario no encontrado"

        pw_hash = hash_password(new_password)
        await db.execute(
            "UPDATE users SET password_hash = ?, password_salt = '' WHERE username = ?",
            (pw_hash, username)
        )
        await db.commit()
        return True, f"Contrasena de '{username}' actualizada"


async def delete_user(username: str) -> tuple[bool, str]:
    """Elimina un usuario (no puede eliminar al ultimo admin)."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        # Contar admins
        cursor = await db.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
        count = (await cursor.fetchone())[0]

        cursor = await db.execute("SELECT role FROM users WHERE username = ?", (username,))
        row = await cursor.fetchone()
        if not row:
            return False, "Usuario no encontrado"

        if row[0] == 'admin' and count <= 1:
            return False, "No se puede eliminar el ultimo administrador"

        await db.execute("DELETE FROM users WHERE username = ?", (username,))
        await db.commit()
        return True, f"Usuario '{username}' eliminado"


async def get_dataset_stats() -> dict:
    """Obtiene estadisticas del dataset desde la base de datos y el filesystem."""
    from config import DATASET_DIR
    stats = {}
    if DATASET_DIR.exists():
        for class_dir in sorted(DATASET_DIR.iterdir()):
            if class_dir.is_dir():
                images = [f for f in class_dir.iterdir()
                          if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')]
                stats[class_dir.name] = len(images)
    return stats


async def save_qr_scan(scan_type: str, data: str, camera_id: str = None):
    """Guarda un escaneo QR/barcode en el historial."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute(
            "INSERT INTO qr_scans (scan_type, data, camera_id) VALUES (?, ?, ?)",
            (scan_type, data, camera_id)
        )
        await db.commit()


async def get_qr_scans(limit: int = 100) -> list[dict]:
    """Obtiene el historial de escaneos QR/barcode."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM qr_scans ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def get_setting(key: str) -> str | None:
    """Obtiene un valor de configuracion."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        cursor = await db.execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = await cursor.fetchone()
        return row[0] if row else None


async def set_setting(key: str, value: str):
    """Guarda un valor de configuracion."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
            (key, value)
        )
        await db.commit()


async def get_trigger_settings() -> dict:
    """Obtiene todas las configuraciones de trigger (legacy)."""
    return {
        "default_camera": await get_setting("default_camera") or "",
        "callback_url": await get_setting("callback_url") or "",
        "callback_active": await get_setting("callback_active") or "false",
    }


def _make_slug(name: str) -> str:
    """Genera un slug URL-safe a partir de un nombre.
    'Camara Almacen' -> 'camara-almacen'
    'Cámara Línea 1' -> 'camara-linea-1'
    """
    # Normalizar y quitar acentos
    nfkd = unicodedata.normalize('NFKD', name)
    ascii_name = ''.join(c for c in nfkd if not unicodedata.combining(c))
    # Minusculas, reemplazar no-alfanumericos con guiones
    slug = re.sub(r'[^a-z0-9]+', '-', ascii_name.lower()).strip('-')
    # Colapsar guiones multiples
    slug = re.sub(r'-+', '-', slug)
    return slug or 'camara'


async def create_camera(name: str, camera_type: str, source: str,
                        resolution: str = '', callback_url: str = '',
                        callback_active: str = 'false') -> dict:
    """Crea una camara en la BD. Retorna el dict de la camara creada."""
    base_slug = _make_slug(name)
    slug = base_slug
    # Asegurar slug unico
    async with aiosqlite.connect(str(DB_PATH)) as db:
        counter = 1
        while True:
            cursor = await db.execute("SELECT id FROM cameras WHERE slug = ?", (slug,))
            if not await cursor.fetchone():
                break
            counter += 1
            slug = f"{base_slug}-{counter}"

        cursor = await db.execute("""
            INSERT INTO cameras (slug, name, camera_type, source, resolution, callback_url, callback_active)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (slug, name, camera_type, source, resolution, callback_url, callback_active))
        await db.commit()
        cam_id = cursor.lastrowid

    return {
        "id": cam_id, "slug": slug, "name": name, "camera_type": camera_type,
        "source": source, "resolution": resolution,
        "callback_url": callback_url, "callback_active": callback_active,
    }


async def get_camera_by_slug(slug: str) -> dict | None:
    """Busca una camara por su slug."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM cameras WHERE slug = ?", (slug,))
        row = await cursor.fetchone()
        return dict(row) if row else None


async def get_camera_by_id(cam_id: int) -> dict | None:
    """Busca una camara por su ID."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM cameras WHERE id = ?", (cam_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None


async def get_all_cameras() -> list[dict]:
    """Retorna todas las camaras registradas."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM cameras ORDER BY id")
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def update_camera_db(cam_id: int, **fields) -> bool:
    """Actualiza campos especificos de una camara."""
    if not fields:
        return False
    allowed = {'slug', 'name', 'camera_type', 'source', 'resolution', 'callback_url', 'callback_active'}
    filtered = {k: v for k, v in fields.items() if k in allowed}
    if not filtered:
        return False
    sets = ', '.join(f"{k} = ?" for k in filtered)
    values = list(filtered.values()) + [cam_id]
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute(f"UPDATE cameras SET {sets} WHERE id = ?", values)
        await db.commit()
    return True


async def delete_camera_db(cam_id: int) -> bool:
    """Elimina una camara de la BD."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        cursor = await db.execute("DELETE FROM cameras WHERE id = ?", (cam_id,))
        await db.commit()
        return cursor.rowcount > 0


# ─── MQTT Triggers ─────────────────────────────────────────────────


async def create_mqtt_trigger(topic: str, camera_id: int,
                              result_topic: str = '', payload_vars: str = '',
                              active: str = 'true') -> dict:
    """Crea un trigger MQTT (topico -> camara)."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        cursor = await db.execute(
            "INSERT INTO mqtt_triggers (topic, camera_id, result_topic, payload_vars, active) VALUES (?, ?, ?, ?, ?)",
            (topic, camera_id, result_topic, payload_vars, active)
        )
        await db.commit()
        return {"id": cursor.lastrowid, "topic": topic, "camera_id": camera_id,
                "result_topic": result_topic, "payload_vars": payload_vars, "active": active}


async def get_all_mqtt_triggers_async() -> list[dict]:
    """Obtiene todos los triggers MQTT (version async)."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("""
            SELECT t.*, c.name as camera_name, c.slug as camera_slug
            FROM mqtt_triggers t
            LEFT JOIN cameras c ON t.camera_id = c.id
            ORDER BY t.id
        """)
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def update_mqtt_trigger_db(trigger_id: int, **fields) -> bool:
    """Actualiza un trigger MQTT."""
    if not fields:
        return False
    allowed = {'topic', 'camera_id', 'result_topic', 'payload_vars', 'active'}
    filtered = {k: v for k, v in fields.items() if k in allowed}
    if not filtered:
        return False
    sets = ', '.join(f"{k} = ?" for k in filtered)
    values = list(filtered.values()) + [trigger_id]
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute(f"UPDATE mqtt_triggers SET {sets} WHERE id = ?", values)
        await db.commit()
    return True


async def delete_mqtt_trigger_db(trigger_id: int) -> bool:
    """Elimina un trigger MQTT."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        cursor = await db.execute("DELETE FROM mqtt_triggers WHERE id = ?", (trigger_id,))
        await db.commit()
        return cursor.rowcount > 0


# ─── Funciones sincronas para el hilo MQTT ─────────────────────────

import sqlite3 as _sqlite3


def get_setting_sync(key: str) -> str | None:
    """Obtiene un setting de forma sincrona (para hilo MQTT)."""
    try:
        conn = _sqlite3.connect(str(DB_PATH))
        cursor = conn.execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None
    except Exception:
        return None


def get_all_mqtt_triggers() -> list[dict]:
    """Obtiene todos los triggers MQTT de forma sincrona (para hilo MQTT)."""
    try:
        conn = _sqlite3.connect(str(DB_PATH))
        conn.row_factory = _sqlite3.Row
        cursor = conn.execute("""
            SELECT t.*, c.name as camera_name, c.slug as camera_slug
            FROM mqtt_triggers t
            LEFT JOIN cameras c ON t.camera_id = c.id
            ORDER BY t.id
        """)
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []
