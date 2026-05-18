"""Rutas de configuracion MQTT."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from config import TEMPLATES_DIR
from database import (
    get_setting, set_setting,
    get_all_mqtt_triggers_async, create_mqtt_trigger, delete_mqtt_trigger_db,
    get_all_cameras,
)
from auth import get_current_user
from mqtt_manager import mqtt_manager

router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


async def get_user_or_redirect(request: Request):
    user = await get_current_user(request)
    if not user:
        return None
    return user


@router.get("/mqtt", response_class=HTMLResponse)
async def mqtt_page(request: Request):
    user = await get_user_or_redirect(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    triggers = await get_all_mqtt_triggers_async()
    cameras = await get_all_cameras()

    return templates.TemplateResponse(request, "mqtt.html", {
        "request": request,
        "user": user,
        "triggers": triggers,
        "cameras": cameras,
        "mqtt_host": await get_setting("mqtt_host") or "",
        "mqtt_port": await get_setting("mqtt_port") or "1883",
        "mqtt_user": await get_setting("mqtt_user") or "",
        "mqtt_pass": await get_setting("mqtt_pass") or "",
        "mqtt_enabled": await get_setting("mqtt_enabled") or "false",
        "mqtt_connected": mqtt_manager.is_connected,
        "success": request.query_params.get("success"),
        "error": request.query_params.get("error"),
    })


@router.post("/mqtt/save-broker")
async def mqtt_save_broker(request: Request):
    user = await get_user_or_redirect(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    form = await request.form()
    host = form.get("mqtt_host", "").strip()
    port = form.get("mqtt_port", "1883").strip()
    user_val = form.get("mqtt_user", "").strip()
    pass_val = form.get("mqtt_pass", "").strip()
    enabled = "true" if form.get("mqtt_enabled") == "true" else "false"

    await set_setting("mqtt_host", host)
    await set_setting("mqtt_port", port)
    await set_setting("mqtt_user", user_val)
    await set_setting("mqtt_pass", pass_val)
    await set_setting("mqtt_enabled", enabled)

    mqtt_manager.reload()

    return RedirectResponse("/mqtt?success=Broker MQTT guardado correctamente", status_code=303)


@router.post("/mqtt/add-trigger")
async def mqtt_add_trigger(request: Request):
    user = await get_user_or_redirect(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    form = await request.form()
    topic = form.get("topic", "").strip()
    camera_id = form.get("camera_id", "").strip()
    result_topic = form.get("result_topic", "").strip()
    payload_vars = form.get("payload_vars", "").strip()

    if not topic or not camera_id:
        return RedirectResponse("/mqtt?error=Topico y camara son obligatorios", status_code=303)

    try:
        await create_mqtt_trigger(topic, int(camera_id), result_topic, payload_vars)
        mqtt_manager.reload()
        return RedirectResponse(f"/mqtt?success=Trigger agregado: {topic}", status_code=303)
    except Exception as e:
        return RedirectResponse(f"/mqtt?error={e}", status_code=303)


@router.post("/mqtt/delete-trigger")
async def mqtt_delete_trigger(request: Request):
    user = await get_user_or_redirect(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    form = await request.form()
    trigger_id = form.get("trigger_id", "").strip()

    if trigger_id:
        await delete_mqtt_trigger_db(int(trigger_id))
        mqtt_manager.reload()

    return RedirectResponse("/mqtt?success=Trigger eliminado", status_code=303)
