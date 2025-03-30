import logging
from datetime import datetime
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from pathlib import Path
import os

from jose import jwt
from sqlalchemy import select

from backend.app.api import auth, admin, transcriptions, users
from backend.app.core.config import settings
from backend.app.core.security import get_current_active_user_combined
from backend.app.db.session import async_session
from backend.app.models.user import User

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


app = FastAPI(
    title=settings.APP_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Set up CORS
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Mount static directory
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="backend/app/templates")
templates.env.globals["settings"] = settings
templates.env.globals["now"] = datetime.now()

# Include API routers
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(admin.router, prefix="/admin", tags=["admin"])
app.include_router(users.router, prefix="/users", tags=["users"])
app.include_router(transcriptions.router, prefix="/transcriptions", tags=["transcriptions"])

# API version prefix
api_v1 = FastAPI(title=f"{settings.APP_NAME} API")
api_v1.include_router(auth.router, prefix="/auth", tags=["auth"])
api_v1.include_router(admin.router, prefix="/admin", tags=["admin"])
api_v1.include_router(users.router, prefix="/users", tags=["users"])
api_v1.include_router(transcriptions.router, prefix="/transcriptions", tags=["transcriptions"])

app.mount(settings.API_V1_STR, api_v1)


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """
    Root endpoint, redirects to user dashboard or login page
    """
    # Проверяем токен из cookie
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse(url="/auth/login")

    # Если есть токен, пытаемся получить пользователя
    try:
        token_type, _, token_value = token.partition(" ")
        payload = jwt.decode(
            token_value, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
        user_id = payload.get("sub")

        # Получаем сессию
        async with async_session() as db:
            result = await db.execute(select(User).filter(User.id == int(user_id)))
            user = result.scalars().first()

        if user.is_admin:
            return RedirectResponse(url="/admin")
        else:
            return RedirectResponse(url="/users")
    except Exception:
        return RedirectResponse(url="/auth/login")


@app.exception_handler(403)
async def not_found_exception_handler(request: Request, exc: HTTPException):
    """
    404 page handler
    """
    return templates.TemplateResponse(
        "errors/403.html", {"request": request}, status_code=403
    )


@app.exception_handler(404)
async def not_found_exception_handler(request: Request, exc: HTTPException):
    """
    404 page handler
    """
    return templates.TemplateResponse(
        "errors/404.html", {"request": request}, status_code=404
    )


@app.exception_handler(500)
async def server_error_exception_handler(request: Request, exc: HTTPException):
    """
    500 page handler
    """
    return templates.TemplateResponse(
        "errors/500.html", {"request": request}, status_code=500
    )


@app.exception_handler(401)
async def server_error_exception_handler(request: Request, exc: HTTPException):
    """
    500 page handler
    """
    return templates.TemplateResponse(
        "errors/401.html", {"request": request}, status_code=401
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
