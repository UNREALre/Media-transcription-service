import os
import re
import uuid
from datetime import datetime
from typing import Any, List

import markdown
from fastapi import APIRouter, Depends, HTTPException, Request, status, UploadFile, File, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import desc, func

from backend.app.core.config import settings
from backend.app.core.security import get_current_active_user_combined
from backend.app.db.session import get_db
from backend.app.models.user import User
from backend.app.models.transcription import Transcription, TranscriptionStatus, TranscriptionType
from backend.app.schemas.transcription import (
    Transcription as TranscriptionSchema,
    TranscriptionCreate,
    TranscriptionUpdate,
    TranscriptionList
)
from backend.app.core.errors import (
    TranscriptionNotFoundError,
    FileUploadError,
    PermissionDeniedError,
    service_error_handler
)
from backend.app.worker.tasks import send_transcription_task

router = APIRouter()
templates = Jinja2Templates(directory="backend/app/templates")
templates.env.globals["settings"] = settings
templates.env.globals["now"] = datetime.now()


# API endpoints for transcriptions

@router.get("/", response_model=TranscriptionList)
async def read_transcriptions(
        skip: int = 0,
        limit: int = 100,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user_combined)
) -> Any:
    """Get all transcriptions for the current user"""
    result = await db.execute(
        select(Transcription)
        .filter(Transcription.user_id == current_user.id)
        .order_by(desc(Transcription.created_at))
        .offset(skip)
        .limit(limit)
    )
    transcriptions = result.scalars().all()

    # Get total count - исправленная версия
    count_result = await db.execute(
        select(func.count(Transcription.id))
        .filter(Transcription.user_id == current_user.id)
    )
    total = count_result.scalar()

    return {"transcriptions": transcriptions, "total": total}


@router.post("/", response_model=TranscriptionSchema)
async def create_transcription(
        transcription_in: TranscriptionCreate,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user_combined)
) -> Any:
    """Create new transcription task"""
    transcription = Transcription(
        user_id=current_user.id,
        original_filename=transcription_in.original_filename,
        file_path="",  # Will be filled later when file is uploaded
        transcription_type=transcription_in.transcription_type,
        custom_prompt=transcription_in.custom_prompt,
        status=TranscriptionStatus.PENDING
    )

    db.add(transcription)
    await db.commit()
    await db.refresh(transcription)
    return transcription


# Web UI endpoints for transcriptions

@router.get("/history", response_class=HTMLResponse)
async def transcription_history(
        request: Request,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user_combined)
):
    """User's transcription history page"""
    result = await db.execute(
        select(Transcription)
        .filter(Transcription.user_id == current_user.id)
        .order_by(desc(Transcription.created_at))
    )
    transcriptions = result.scalars().all()

    return templates.TemplateResponse(
        "user/history.html",
        {
            "request": request,
            "current_user": current_user,
            "transcriptions": transcriptions
        }
    )


@router.get("/new", response_class=HTMLResponse)
async def new_transcription_page(
        request: Request,
        current_user: User = Depends(get_current_active_user_combined)
):
    """New transcription upload page"""
    return templates.TemplateResponse(
        "user/transcribe.html",
        {
            "request": request,
            "current_user": current_user
        }
    )


@router.post("/upload")
async def upload_file(
        request: Request,
        file: UploadFile = File(...),
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user_combined)
):
    """Handle file upload for transcription"""
    try:
        # Validate file
        if not file.filename:
            raise FileUploadError("No file selected")

        # Check file size
        file_size = 0
        contents = await file.read()
        file_size = len(contents)
        await file.seek(0)  # Reset file position

        if file_size > settings.MAX_UPLOAD_SIZE:
            raise FileUploadError(f"File too large. Maximum size is {settings.MAX_UPLOAD_SIZE / (1024 * 1024)} MB")

        # Create unique filename
        file_ext = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = os.path.join(settings.UPLOAD_DIR, unique_filename)

        # Save file
        with open(file_path, "wb") as f:
            f.write(contents)

        # Create transcription record
        transcription = Transcription(
            user_id=current_user.id,
            original_filename=file.filename,
            file_path=file_path,
            transcription_type=TranscriptionType.SUMMARY,  # Default, will be updated later
            status=TranscriptionStatus.PENDING
        )

        db.add(transcription)
        await db.commit()
        await db.refresh(transcription)

        # Redirect to form for selecting transcription type
        return RedirectResponse(
            url=f"/transcriptions/select-type/{transcription.id}",
            status_code=status.HTTP_302_FOUND
        )
    except FileUploadError as e:
        service_error_handler(e)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading file: {str(e)}"
        )


@router.get("/{transcription_id}", response_model=TranscriptionSchema)
async def read_transcription(
        transcription_id: int,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user_combined)
) -> Any:
    """Get a specific transcription by id"""
    try:
        result = await db.execute(
            select(Transcription).filter(Transcription.id == transcription_id)
        )
        transcription = result.scalars().first()

        if not transcription:
            raise TranscriptionNotFoundError()

        if transcription.user_id != current_user.id and not current_user.is_admin:
            raise PermissionDeniedError()

        return transcription
    except (TranscriptionNotFoundError, PermissionDeniedError) as e:
        service_error_handler(e)


@router.put("/{transcription_id}", response_model=TranscriptionSchema)
async def update_transcription(
        transcription_id: int,
        transcription_in: TranscriptionUpdate,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user_combined)
) -> Any:
    """Update a transcription status"""
    try:
        result = await db.execute(
            select(Transcription).filter(Transcription.id == transcription_id)
        )
        transcription = result.scalars().first()

        if not transcription:
            raise TranscriptionNotFoundError()

        if transcription.user_id != current_user.id and not current_user.is_admin:
            raise PermissionDeniedError()

        update_data = transcription_in.dict(exclude_unset=True)

        # Only allow certain fields to be updated
        allowed_fields = ["status", "error_message", "raw_transcript", "processed_text", "completed_at"]
        update_fields = {k: v for k, v in update_data.items() if k in allowed_fields}

        for field, value in update_fields.items():
            setattr(transcription, field, value)

        await db.commit()
        await db.refresh(transcription)
        return transcription
    except (TranscriptionNotFoundError, PermissionDeniedError) as e:
        service_error_handler(e)


@router.get("/select-type/{transcription_id}", response_class=HTMLResponse)
async def select_transcription_type(
        transcription_id: int,
        request: Request,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user_combined)
):
    """Page for selecting transcription type"""
    try:
        result = await db.execute(
            select(Transcription).filter(Transcription.id == transcription_id)
        )
        transcription = result.scalars().first()

        if not transcription:
            raise TranscriptionNotFoundError()

        if transcription.user_id != current_user.id:
            raise PermissionDeniedError()

        return templates.TemplateResponse(
            "user/select_type.html",
            {
                "request": request,
                "current_user": current_user,
                "transcription": transcription
            }
        )
    except (TranscriptionNotFoundError, PermissionDeniedError) as e:
        service_error_handler(e)


@router.post("/select-type/{transcription_id}")
async def process_transcription_type(
        transcription_id: int,
        request: Request,
        transcription_type: str = Form(...),
        custom_prompt: str = Form(None),
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user_combined)
):
    """Process the selected transcription type and start the task"""
    try:
        result = await db.execute(
            select(Transcription).filter(Transcription.id == transcription_id)
        )
        transcription = result.scalars().first()

        if not transcription:
            raise TranscriptionNotFoundError()

        if transcription.user_id != current_user.id:
            raise PermissionDeniedError()

        # Update transcription type and custom prompt
        if transcription_type == "summary":
            transcription.transcription_type = TranscriptionType.SUMMARY
        elif transcription_type == "technical_spec":
            transcription.transcription_type = TranscriptionType.TECHNICAL_SPEC
        elif transcription_type == "custom":
            transcription.transcription_type = TranscriptionType.CUSTOM
            transcription.custom_prompt = custom_prompt
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid transcription type"
            )

        await db.commit()
        await db.refresh(transcription)

        # Send task to queue
        await send_transcription_task(transcription.id)

        # Redirect to history page
        return RedirectResponse(
            url="/transcriptions/history",
            status_code=status.HTTP_302_FOUND
        )
    except (TranscriptionNotFoundError, PermissionDeniedError) as e:
        service_error_handler(e)


@router.get("/view/{transcription_id}", response_class=HTMLResponse)
async def view_transcription(
        transcription_id: int,
        request: Request,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user_combined)
):
    def is_markdown(text):
        """
        Проверяет, содержит ли текст типичные элементы Markdown форматирования.
        Возвращает True, если текст похож на Markdown.
        """
        # Паттерны для распознавания типичных элементов Markdown
        markdown_patterns = [
            r'#{1,6}\s+\w+',  # Заголовки (# Заголовок)
            r'\*\*.+?\*\*',  # Жирный текст (**текст**)
            r'\*.+?\*',  # Курсив (*текст*)
            r'```[\s\S]+?```',  # Блоки кода (```код```)
            r'`[^`]+`',  # Встроенный код (`код`)
            r'!\[.+?\]\(.+?\)',  # Изображения (![alt](url))
            r'\[.+?\]\(.+?\)',  # Ссылки ([текст](url))
            r'^-\s+.+$',  # Ненумерованные списки (- элемент)
            r'^\d+\.\s+.+$',  # Нумерованные списки (1. элемент)
            r'^>\s+.+$',  # Цитаты (> цитата)
            r'^---+$',  # Горизонтальные линии (---)
        ]

        # Проверяем, соответствует ли текст хотя бы одному паттерну
        for pattern in markdown_patterns:
            if re.search(pattern, text, re.MULTILINE):
                return True

        return False

    """View transcription details page"""
    try:
        result = await db.execute(
            select(Transcription).filter(Transcription.id == transcription_id)
        )
        transcription = result.scalars().first()

        if not transcription:
            raise TranscriptionNotFoundError()

        if transcription.user_id != current_user.id and not current_user.is_admin:
            raise PermissionDeniedError()

        # Преобразуем в HTML
        html_content = markdown.markdown(transcription.processed_text) if transcription.processed_text else ""

        return templates.TemplateResponse(
            "user/transcription_detail.html",
            {
                "request": request,
                "current_user": current_user,
                "transcription": transcription,
                "processed_text_html": html_content,
            }
        )
    except (TranscriptionNotFoundError, PermissionDeniedError) as e:
        service_error_handler(e)
