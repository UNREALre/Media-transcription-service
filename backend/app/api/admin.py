import os
import logging

from datetime import datetime, timedelta
from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func

from backend.app.core.config import settings
from backend.app.core.security import get_current_active_admin_combined, get_password_hash
from backend.app.db.session import get_db
from backend.app.models.user import User
from backend.app.models.transcription import Transcription
from backend.app.schemas.user import User as UserSchema, UserCreate, UserUpdate
from backend.app.schemas.transcription import Transcription as TranscriptionSchema
from backend.app.core.errors import UserNotFoundError, service_error_handler, TranscriptionNotFoundError

router = APIRouter()
templates = Jinja2Templates(directory="backend/app/templates")
templates.env.globals["settings"] = settings
templates.env.globals["now"] = datetime.now()

logger = logging.getLogger(__name__)


# API endpoints for admin operations

@router.get("/users", response_model=List[UserSchema])
async def read_users(
        skip: int = 0,
        limit: int = 100,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_admin_combined)
) -> Any:
    """Get all users"""
    result = await db.execute(select(User).offset(skip).limit(limit))
    users = result.scalars().all()
    return users


@router.post("/users", response_model=UserSchema)
async def create_user(
        user_in: UserCreate,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_admin_combined)
) -> Any:
    """Create new user"""
    # Check if user with this email or username exists
    result = await db.execute(
        select(User).filter(
            (User.email == user_in.email) | (User.username == user_in.username)
        )
    )
    user = result.scalars().first()
    if user:
        raise HTTPException(
            status_code=400,
            detail="A user with this email or username already exists"
        )

    user = User(
        email=user_in.email,
        username=user_in.username,
        hashed_password=get_password_hash(user_in.password),
        is_active=user_in.is_active,
        is_admin=user_in.is_admin,
        first_name=user_in.first_name,
        last_name=user_in.last_name
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


@router.get("/users/{user_id}", response_model=UserSchema)
async def read_user(
        user_id: int,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_admin_combined)
) -> Any:
    """Get a specific user by id"""
    try:
        result = await db.execute(select(User).filter(User.id == user_id))
        user = result.scalars().first()
        if not user:
            raise UserNotFoundError()
        return user
    except UserNotFoundError as e:
        service_error_handler(e)


@router.put("/users/{user_id}", response_model=UserSchema)
async def update_user(
        user_id: int,
        user_in: UserUpdate,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_admin_combined)
) -> Any:
    """Update a user"""
    try:
        result = await db.execute(select(User).filter(User.id == user_id))
        user = result.scalars().first()
        if not user:
            raise UserNotFoundError()

        update_data = user_in.dict(exclude_unset=True)

        if "password" in update_data and update_data["password"]:
            update_data["hashed_password"] = get_password_hash(update_data.pop("password"))

        for field, value in update_data.items():
            setattr(user, field, value)

        await db.commit()
        await db.refresh(user)
        return user
    except UserNotFoundError as e:
        service_error_handler(e)


@router.delete("/users/{user_id}", response_model=UserSchema)
async def delete_user(
        user_id: int,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_admin_combined)
) -> Any:
    """Delete a user"""
    try:
        result = await db.execute(select(User).filter(User.id == user_id))
        user = result.scalars().first()
        if not user:
            raise UserNotFoundError()

        await db.delete(user)
        await db.commit()
        return user
    except UserNotFoundError as e:
        service_error_handler(e)


# Web UI endpoints for admin

@router.get("/", response_class=HTMLResponse)
async def admin_dashboard(
        request: Request,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_admin_combined)
):
    """Admin dashboard page"""
    # Get counts for dashboard
    users_count = await db.execute(select(func.count(User.id)))
    users_count = users_count.scalar()

    transcriptions_count = await db.execute(select(func.count(Transcription.id)))
    transcriptions_count = transcriptions_count.scalar()

    return templates.TemplateResponse(
        "admin/index.html",
        {
            "request": request,
            "current_user": current_user,
            "users_count": users_count,
            "transcriptions_count": transcriptions_count
        }
    )


@router.get("/users-management", response_class=HTMLResponse)
async def users_management(
        request: Request,
        skip: int = 0,
        limit: int = 100,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_admin_combined)
):
    """Users management page"""
    result = await db.execute(select(User).offset(skip).limit(limit))
    users = result.scalars().all()

    return templates.TemplateResponse(
        "admin/users.html",
        {
            "request": request,
            "current_user": current_user,
            "users": users
        }
    )


@router.get("/transcriptions-management", response_class=HTMLResponse)
async def transcriptions_management(
        request: Request,
        skip: int = 0,
        limit: int = 100,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_admin_combined)
):
    """Transcriptions management page"""
    # Используем опцию selectinload для предварительной загрузки связанного пользователя
    from sqlalchemy.orm import selectinload

    result = await db.execute(
        select(Transcription)
        .options(selectinload(Transcription.user))  # Предварительно загружаем пользователя
        .offset(skip)
        .limit(limit)
    )
    transcriptions = result.scalars().all()

    return templates.TemplateResponse(
        "admin/transcriptions.html",
        {
            "request": request,
            "current_user": current_user,
            "transcriptions": transcriptions
        }
    )


@router.delete("/transcriptions/{transcription_id}", response_model=TranscriptionSchema)
async def delete_transcription(
        transcription_id: int,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_admin_combined)
) -> Any:
    """Delete a transcription"""
    try:
        result = await db.execute(select(Transcription).filter(Transcription.id == transcription_id))
        transcription = result.scalars().first()
        if not transcription:
            raise TranscriptionNotFoundError()

        # Удаляем файл, если он существует
        if transcription.file_path and os.path.exists(transcription.file_path):
            try:
                os.remove(transcription.file_path)
            except Exception as e:
                logger.error(f"Error deleting file: {str(e)}")

        await db.delete(transcription)
        await db.commit()
        return transcription
    except TranscriptionNotFoundError as e:
        service_error_handler(e)
