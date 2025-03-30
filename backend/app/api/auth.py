from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from backend.app.core.config import settings
from backend.app.core.security import (
    create_access_token,
    create_refresh_token,
    verify_password,
    get_current_user
)
from backend.app.db.session import get_db
from backend.app.models.user import User
from backend.app.core.errors import InvalidCredentialsError, service_error_handler

router = APIRouter()
templates = Jinja2Templates(directory="backend/app/templates")
templates.env.globals["settings"] = settings
templates.env.globals["now"] = datetime.now()


@router.post("/token")
async def login_access_token(
        form_data: OAuth2PasswordRequestForm = Depends(),
        db: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """OAuth2 compatible token login, get an access token for future requests"""
    try:
        result = await db.execute(
            select(User).filter(
                (User.email == form_data.username) | (User.username == form_data.username)
            )
        )
        user = result.scalars().first()

        if not user or not verify_password(form_data.password, user.hashed_password):
            raise InvalidCredentialsError()

        if not user.is_active:
            raise InvalidCredentialsError("Inactive user")

        access_token_expires = timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)

        return {
            "access_token": create_access_token(user.id, access_token_expires),
            "refresh_token": create_refresh_token(user.id),
            "token_type": "bearer",
        }
    except InvalidCredentialsError as e:
        service_error_handler(e)


@router.post("/refresh-token")
async def refresh_token(
        refresh_token: str,
        db: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """Get a new access token using refresh token"""
    try:
        # Verify refresh token (implementation needed)
        # For now, we'll just delegate to get_current_user
        user = await get_current_user(token=refresh_token, db=db)

        access_token_expires = timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)

        return {
            "access_token": create_access_token(user.id, access_token_expires),
            "token_type": "bearer",
        }
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Login page"""
    return templates.TemplateResponse("auth/login.html", {"request": request})


@router.post("/login")
async def login(
        request: Request,
        response: Response,
        db: AsyncSession = Depends(get_db)
):
    """Process login form submission"""
    try:
        form_data = await request.form()
        username = form_data.get("username")
        password = form_data.get("password")

        result = await db.execute(
            select(User).filter(
                (User.email == username) | (User.username == username)
            )
        )
        user = result.scalars().first()

        if not user or not verify_password(password, user.hashed_password):
            raise InvalidCredentialsError()

        if not user.is_active:
            raise InvalidCredentialsError("Inactive user")

        access_token_expires = timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(user.id, access_token_expires)

        response = RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
        response.set_cookie(
            key="access_token",
            value=f"Bearer {access_token}",
            httponly=True,
            max_age=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            expires=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            samesite="lax",
            secure=settings.APP_ENV == "production",
        )

        return response
    except InvalidCredentialsError:
        return templates.TemplateResponse(
            "auth/login.html",
            {
                "request": request,
                "error": "Invalid credentials",
                "username": username,
            },
            status_code=status.HTTP_401_UNAUTHORIZED,
        )


@router.get("/logout")
async def logout():
    """Logout user"""
    response = RedirectResponse(url="/auth/login", status_code=status.HTTP_302_FOUND)
    response.delete_cookie(key="access_token")
    return response
