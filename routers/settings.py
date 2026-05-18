"""Rutas de ajustes y gestion de usuarios."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from config import TEMPLATES_DIR, get_host, get_port, get_trigger_api_key, _save_runtime_config
from database import get_all_users, add_user, change_password, delete_user
from auth import get_current_user

router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


async def get_user_or_redirect(request: Request):
    user = await get_current_user(request)
    if not user:
        return None
    return user


@router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    user = await get_user_or_redirect(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    users = await get_all_users()
    return templates.TemplateResponse(request, "settings.html", {
        "request": request,
        "user": user,
        "users": users,
        "host": get_host(),
        "port": get_port(),
        "trigger_api_key": get_trigger_api_key(),
        "success": request.query_params.get("success"),
        "error": request.query_params.get("error"),
    })


@router.post("/settings/change-password")
async def settings_change_password(request: Request):
    user = await get_user_or_redirect(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    form = await request.form()
    target_user = form.get("username", "")
    new_pass = form.get("new_password", "")
    confirm_pass = form.get("confirm_password", "")

    if not new_pass or len(new_pass) < 4:
        return RedirectResponse("/settings?error=La+contrasena+debe+tener+al+menos+4+caracteres", status_code=303)

    if new_pass != confirm_pass:
        return RedirectResponse("/settings?error=Las+contrasenas+no+coinciden", status_code=303)

    ok, msg = await change_password(target_user, new_pass)
    if ok:
        return RedirectResponse(f"/settings?success={msg}", status_code=303)
    else:
        return RedirectResponse(f"/settings?error={msg}", status_code=303)


@router.post("/settings/add-user")
async def settings_add_user(request: Request):
    user = await get_user_or_redirect(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    form = await request.form()
    new_user = form.get("new_username", "").strip()
    new_pass = form.get("new_user_password", "")

    if not new_user or not new_pass:
        return RedirectResponse("/settings?error=Usuario+y+contrasena+requeridos", status_code=303)

    if len(new_pass) < 4:
        return RedirectResponse("/settings?error=La+contrasena+debe+tener+al+menos+4+caracteres", status_code=303)

    ok, msg = await add_user(new_user, new_pass)
    if ok:
        return RedirectResponse(f"/settings?success={msg}", status_code=303)
    else:
        return RedirectResponse(f"/settings?error={msg}", status_code=303)


@router.post("/settings/delete-user")
async def settings_delete_user(request: Request):
    user = await get_user_or_redirect(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    form = await request.form()
    target = form.get("username", "")

    if target == user:
        return RedirectResponse("/settings?error=No+puedes+eliminarte+a+ti+mismo", status_code=303)

    ok, msg = await delete_user(target)
    if ok:
        return RedirectResponse(f"/settings?success={msg}", status_code=303)
    else:
        return RedirectResponse(f"/settings?error={msg}", status_code=303)


@router.post("/settings/save-server")
async def settings_save_server(request: Request):
    user = await get_user_or_redirect(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    form = await request.form()
    new_host = form.get("host", "0.0.0.0").strip()
    new_port = form.get("port", "8000").strip()

    try:
        new_port = int(new_port)
        if not (1024 <= new_port <= 65535):
            raise ValueError()
    except ValueError:
        return RedirectResponse("/settings?error=Puerto+debe+ser+un+numero+entre+1024+y+65535", status_code=303)

    _save_runtime_config({"host": new_host, "port": new_port})

    return RedirectResponse(
        f"/settings?success=Configuracion+guardada.+Reinicia+el+servidor+para+aplicar+(puerto={new_port},+host={new_host})",
        status_code=303
    )


@router.post("/settings/save-trigger-key")
async def settings_save_trigger_key(request: Request):
    user = await get_user_or_redirect(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    form = await request.form()
    api_key = form.get("trigger_api_key", "").strip()

    _save_runtime_config({"trigger_api_key": api_key})

    msg = "API+Key+guardada" if api_key else "API+Key+eliminada+(triggers+sin+proteccion)"
    return RedirectResponse(f"/settings?success={msg}", status_code=303)
