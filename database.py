"""
Base de datos SQLite para el panel de administracion.
Maneja usuarios, historial de entrenamientos y predicciones.
"""

import aiosqlite
import hashlib
import os
import json
from datetime import datetime
from config import DB_PATH, ADMIN_USERNAME, ADMIN_PASSWORD


def hash_password(password: str, salt: str = None) -> tuple[str, str]:
    """Hashea una contraseña con salt. Retorna (hash, salt)."""
    if salt is None:
        salt = os.urandom(16).hex()
    h = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
    return h, salt


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
        """)

        # Crear admin si no existe
        cursor = await db.execute("SELECT id FROM users WHERE username = ?", (ADMIN_USERNAME,))
        if not await cursor.fetchone():
            pw_hash, pw_salt = hash_password(ADMIN_PASSWORD)
            await db.execute(
                "INSERT INTO users (username, password_hash, password_salt, role) VALUES (?, ?, ?, ?)",
                (ADMIN_USERNAME, pw_hash, pw_salt, "admin")
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
    """Verifica credenciales de usuario."""
    user = await get_user(username)
    if not user:
        return False
    h, _ = hash_password(password, user["password_salt"])
    return h == user["password_hash"]


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

        pw_hash, pw_salt = hash_password(password)
        await db.execute(
            "INSERT INTO users (username, password_hash, password_salt, role) VALUES (?, ?, ?, ?)",
            (username, pw_hash, pw_salt, role)
        )
        await db.commit()
        return True, f"Usuario '{username}' creado correctamente"


async def change_password(username: str, new_password: str) -> tuple[bool, str]:
    """Cambia la contrasena de un usuario. Retorna (exito, mensaje)."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        cursor = await db.execute("SELECT id FROM users WHERE username = ?", (username,))
        if not await cursor.fetchone():
            return False, "Usuario no encontrado"

        pw_hash, pw_salt = hash_password(new_password)
        await db.execute(
            "UPDATE users SET password_hash = ?, password_salt = ? WHERE username = ?",
            (pw_hash, pw_salt, username)
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
