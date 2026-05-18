"""
Sistema de autenticacion basado en sesiones con cookies firmadas.
Incluye proteccion CSRF con double-submit cookie.
"""

import hmac
import hashlib
from fastapi import Request, HTTPException
from fastapi.responses import RedirectResponse
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from config import SECRET_KEY
from database import verify_user

serializer = URLSafeTimedSerializer(SECRET_KEY)
SESSION_MAX_AGE = 86400 * 7  # 7 dias


def create_session_token(username: str) -> str:
    """Crea un token de sesion firmado."""
    return serializer.dumps({"user": username})


def get_username_from_token(token: str) -> str | None:
    """Extrae el usuario de un token de sesion. Retorna None si expiro o es invalido."""
    try:
        data = serializer.loads(token, max_age=SESSION_MAX_AGE)
        return data.get("user")
    except (BadSignature, SignatureExpired):
        return None


async def get_current_user(request: Request) -> str | None:
    """Obtiene el usuario actual desde la cookie de sesion."""
    token = request.cookies.get("session")
    if not token:
        return None
    return get_username_from_token(token)


async def require_auth(request: Request) -> str:
    """Dependencia que requiere autenticacion. Retorna el username o redirige al login."""
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=303, headers={"Location": "/login"})
    return user


async def authenticate(username: str, password: str) -> str | None:
    """Verifica credenciales y retorna un token de sesion o None."""
    if await verify_user(username, password):
        return create_session_token(username)
    return None


# ─── CSRF Protection ──────────────────────────────────────────────

def generate_csrf_token(session_token: str) -> str:
    """Genera un CSRF token deterministico a partir del session token."""
    return hmac.new(
        SECRET_KEY.encode(), session_token.encode(), hashlib.sha256
    ).hexdigest()


def get_csrf_token(request: Request) -> str:
    """Obtiene el CSRF token para el request actual."""
    session = request.cookies.get("session", "")
    return generate_csrf_token(session)


async def verify_csrf(request: Request) -> bool:
    """Verifica CSRF via header X-CSRF-Token (no consume el body del form)."""
    header_token = request.headers.get("x-csrf-token", "")
    if not header_token:
        return False

    session = request.cookies.get("session", "")
    expected = generate_csrf_token(session)
    return hmac.compare_digest(header_token, expected)
