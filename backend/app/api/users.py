from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from backend.app.core.config import settings
from backend.app.core.security import get_current_active_user_combined, get_password_hash
from backend.app.db.session import get_db
from backend.app.models.user import User
from backend.app.schemas.user import User as UserSchema, UserUpdate
from backend.app.core.errors import UserNotFoundError, service_error_handler

router = APIRouter()
templates = Jinja2Templates(directory="backend/app/templates")
templates.env.globals["settings"] = settings
templates.env.globals["now"] = datetime.now()


# API endpoints for user operations

@router.get("/me", response_model=UserSchema)
async def read_user_me(
        current_user: User = Depends(get_current_active_user_combined)
) -> Any:
    """Get current user"""
    return current_user


@router.put("/me", response_model=UserSchema)
async def update_user_me(
        user_in: UserUpdate,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user_combined)
) -> Any:
    """Update current user"""
    update_data = user_in.dict(exclude_unset=True)

    if "password" in update_data and update_data["password"]:
        update_data["hashed_password"] = get_password_hash(update_data.pop("password"))

    for field, value in update_data.items():
        if field != "is_admin":  # Don't allow users to change their admin status
            setattr(current_user, field, value)

    await db.commit()
    await db.refresh(current_user)
    return current_user


# Web UI endpoints for user

@router.get("/", response_class=HTMLResponse)
async def user_dashboard(
        request: Request,
        current_user: User = Depends(get_current_active_user_combined)
):
    """User dashboard page"""
    return templates.TemplateResponse(
        "user/dashboard.html",
        {
            "request": request,
            "current_user": current_user
        }
    )


@router.get("/about", response_class=HTMLResponse)
async def about_page(
        request: Request,
        current_user: User = Depends(get_current_active_user_combined)
):
    """About page"""
    return templates.TemplateResponse(
        "user/about.html",
        {
            "request": request,
            "current_user": current_user
        }
    )


@router.get("/profile", response_class=HTMLResponse)
async def user_profile(
        request: Request,
        current_user: User = Depends(get_current_active_user_combined)
):
    """User profile page"""
    return templates.TemplateResponse(
        "user/profile.html",
        {
            "request": request,
            "current_user": current_user
        }
    )
