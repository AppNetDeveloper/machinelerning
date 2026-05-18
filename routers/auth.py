"""Rutas de autenticacion: login, logout."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from auth import authenticate, get_current_user
from config import TEMPLATES_DIR

router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    user = await get_current_user(request)
    if user:
        return RedirectResponse("/", status_code=303)
    return templates.TemplateResponse(request, "login.html", {"request": request})


@router.post("/login")
async def login_submit(request: Request):
    form = await request.form()
    username = form.get("username", "")
    password = form.get("password", "")

    token = await authenticate(username, password)
    if not token:
        return templates.TemplateResponse(request, "login.html", {
            "request": request,
            "error": "Usuario o contrasena incorrectos",
        })

    response = RedirectResponse("/", status_code=303)
    response.set_cookie("session", token, max_age=86400 * 7, httponly=True)
    return response


@router.get("/logout")
async def logout():
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie("session")
    return response
