import asyncio
import concurrent.futures
import json
import logging
import os
import sys
import tempfile
import whisper

from datetime import datetime
from typing import Optional

import aio_pika
import ffmpeg
import openai
from fastapi import FastAPI, Depends
from functools import lru_cache
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Add parent directory to path so we can import from backend.app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.core.config import settings
from backend.app.models.transcription import Transcription, TranscriptionStatus, TranscriptionType
from backend.app.models.user import User
from backend.app.utils.email import send_notification_email

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Setup database connection
engine = create_async_engine(settings.DATABASE_URL)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

app = FastAPI()

# Создаем ThreadPoolExecutor для I/O-bound операций
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Список обрабатываемых задач для контроля
active_tasks = set()


class WhisperModel:
    def __init__(self, model_size: str = "large"):
        """Инициализация модели Whisper.
        "tiny": самая быстрая модель, но менее точная
        "base": хороший баланс между скоростью и точностью
        "small", "medium", "large": более точные, но требуют больше ресурсов
        """
        self.model = whisper.load_model(model_size)

    def transcribe(self, audio_path: str) -> str:
        result = self.model.transcribe(audio_path)
        return result["text"]


@lru_cache()
def get_whisper_model(model_size: str = "large"):
    """Инициализация модели Whisper (синглтон)"""
    return WhisperModel(model_size)


# При старте приложения модель загрузится один раз
whisper_model = get_whisper_model()


async def get_db():
    """Get DB session"""
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def update_transcription_status(
        db: AsyncSession,
        transcription: Transcription,
        status: TranscriptionStatus,
        **kwargs
):
    """Update transcription status"""
    transcription.status = status
    for key, value in kwargs.items():
        if hasattr(transcription, key):
            setattr(transcription, key, value)
    await db.commit()
    await db.refresh(transcription)
    logger.info(f"Updated transcription {transcription.id} status to {status.value}")


def run_whisper_transcription(audio_path: str, model_size: str = "large"):
    """Блокирующая функция для запуска в отдельном потоке"""
    try:
        model = whisper.load_model(model_size)
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        logger.error(f"Error in Whisper transcription: {str(e)}")
        raise


async def extract_audio(video_path: str) -> str:
    """Extract audio from video"""
    temp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(temp_dir, "audio.mp3")

    try:
        # Используем неблокирующую версию с executor
        loop = asyncio.get_running_loop()

        def extract_audio_blocking():
            try:
                # Extract audio using ffmpeg
                (
                    ffmpeg
                    .input(video_path)
                    .output(audio_path, acodec="libmp3lame", ac=1, ar="16000")
                    .run(quiet=True, overwrite_output=True)
                )
                return audio_path
            except ffmpeg.Error as e:
                error_msg = e.stderr.decode() if e.stderr else str(e)
                logger.error(f"Error extracting audio: {error_msg}")
                raise

        # Запускаем в thread pool
        result = await loop.run_in_executor(thread_pool, extract_audio_blocking)
        return result
    except Exception as e:
        logger.error(f"Error in extract_audio: {str(e)}")
        raise


async def transcribe_audio_local(audio_path: str, model_size: str = "large") -> str:
    """Transcribe audio using locally installed Whisper model - non-blocking version"""
    try:
        # Запускаем в отдельном потоке
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            thread_pool,
            run_whisper_transcription,
            audio_path,
            model_size
        )
        return result
    except Exception as e:
        logger.error(f"Error in transcribe_audio_local: {str(e)}")
        raise


async def process_with_llm(
        transcript: str,
        transcription_type: TranscriptionType,
        custom_prompt: Optional[str] = None,
        use_local: bool = False
) -> str:
    """Process transcript with LLM"""
    # Define prompts for different transcription types
    prompts = {
        TranscriptionType.SUMMARY: "Пожалуйста, ознакомься с предоставленным текстом, который является "
                                   "расшифровкой голосового обсуждения."
                                   "Составь краткое и структурированное memo по его содержанию. Включи: "
                                   "1.	Основные темы обсуждения. "
                                   "2.	Суть мнений или позиций каждого участника. "
                                   "3.	Ключевые выводы, договорённости или открытые вопросы. "
                                   "4.	Если уместно — указания на настроение или тон общения. "
                                   "Не пересказывай весь диалог, а выдели суть. Используй деловой или нейтральный "
                                   "стиль, избегай лишней воды.",
        TranscriptionType.TECHNICAL_SPEC: "Пожалуйста, ознакомься с предоставленным текстом, который является "
                                          "расшифровкой голосового обсуждения. В этом обсуждении есть технические, "
                                          "на основании которых нужно сформировать техническое задание. Постарайся "
                                          "вычленить из текста все технические требования, особенности и детали, "
                                          "прочитав которые специалист мог бы начать работу.",
    }

    # Get prompt
    if transcription_type == TranscriptionType.CUSTOM and custom_prompt:
        prompt_text = custom_prompt
    else:
        prompt_text = prompts.get(transcription_type, prompts[TranscriptionType.SUMMARY])

    try:
        if not use_local:
            logger.info(f"OpenAI request start ...")
            llm = ChatOpenAI(
                model_name="gpt-4o",
                openai_api_key=settings.OPENAI_API_KEY,
                temperature=0.7,
            )
            prompt = PromptTemplate(
                input_variables=["transcript"],
                template=f"{prompt_text}\n\nТранскрипт:\n{{transcript}}"
            )
            chain = LLMChain(llm=llm, prompt=prompt)
            response = await chain.arun(transcript=transcript)
            return response
        else:
            logger.info(f"Local model query start...")

            # Запускаем в отдельном потоке
            loop = asyncio.get_running_loop()

            def run_ollama():
                try:
                    llm = Ollama(
                        model=settings.OLLAMA_MODEL,
                        base_url=settings.OLLAMA_BASE_URL
                    )
                    prompt = PromptTemplate(
                        input_variables=["transcript"],
                        template=f"{prompt_text}\n\nТранскрипт:\n{{transcript}}"
                    )
                    chain = LLMChain(llm=llm, prompt=prompt)
                    result = chain.invoke({"transcript": transcript})
                    return result.get("text", "")
                except Exception as e:
                    logger.error(f"Error in run_ollama: {str(e)}")
                    raise

            response = await loop.run_in_executor(thread_pool, run_ollama)
            return response
    except Exception as e:
        logger.error(f"Error in process_with_llm: {str(e)}")
        raise


async def process_transcription(transcription_id: int):
    """Process a transcription task"""
    # Если задача уже обрабатывается, пропускаем
    if transcription_id in active_tasks:
        logger.info(f"Transcription {transcription_id} is already being processed. Skipping.")
        return

    # Добавляем задачу в список активных
    active_tasks.add(transcription_id)

    db = None
    try:
        # Get database session
        db_gen = get_db()
        db = await anext(db_gen)

        # Get transcription
        result = await db.execute(select(Transcription).filter(Transcription.id == transcription_id))
        transcription = result.scalars().first()

        if not transcription:
            logger.error(f"Transcription {transcription_id} not found")
            return

        # Проверяем, не завершена ли уже задача
        if transcription.status in [TranscriptionStatus.COMPLETED, TranscriptionStatus.FAILED]:
            logger.info(f"Transcription {transcription_id} is already in final state: {transcription.status.value}")
            return

        # Get user email
        result = await db.execute(select(User).filter(User.id == transcription.user_id))
        user = result.scalars().first()

        if not user:
            logger.error(f"User {transcription.user_id} not found")
            return

        # Update status to extracting audio
        await update_transcription_status(db, transcription, TranscriptionStatus.EXTRACTING_AUDIO)

        # Extract audio
        logger.info(f"Extracting audio for transcription {transcription_id}...")
        audio_path = await extract_audio(transcription.file_path)

        # Update status to transcribing
        await update_transcription_status(db, transcription, TranscriptionStatus.TRANSCRIBING)

        # Transcribe audio
        logger.info(f"Transcribing audio for transcription {transcription_id}...")
        # Используем model_size="small" для более быстрой обработки больших файлов
        transcript = await transcribe_audio_local(audio_path, model_size="large")

        # Update raw transcript
        await update_transcription_status(
            db,
            transcription,
            TranscriptionStatus.PROCESSING_WITH_LLM,
            raw_transcript=transcript
        )

        # Process with LLM
        logger.info(f"Processing with LLM for transcription {transcription_id}...")
        processed_text = await process_with_llm(
            transcript,
            transcription.transcription_type,
            transcription.custom_prompt,
            use_local=True
        )

        # Update status to completed
        await update_transcription_status(
            db,
            transcription,
            TranscriptionStatus.COMPLETED,
            processed_text=processed_text,
            completed_at=datetime.utcnow()
        )

        # Send email notification
        # await send_notification_email(
        #     user.email,
        #     f"Преобразование '{transcription.original_filename}' завершено",
        #     f"Задача преобразования видео в текст выполнена. Вы можете посмотреть результаты тут: "
        #     f"{settings.APP_URL}/transcriptions/view/{transcription.id}"
        # )

        logger.info(f"Transcription {transcription_id} completed successfully")

        # Delete temporary audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)

    except Exception as e:
        logger.error(f"Error processing transcription {transcription_id}: {str(e)}")
        if db and transcription:
            # Update status to failed
            await update_transcription_status(
                db,
                transcription,
                TranscriptionStatus.FAILED,
                error_message=str(e)
            )

            # Send error notification
            if user:
                await send_notification_email(
                    user.email,
                    f"Transcription '{transcription.original_filename}' failed",
                    f"Your transcription task has failed: {str(e)}"
                )
    finally:
        # Удаляем задачу из списка активных
        active_tasks.discard(transcription_id)


async def consume_task_queue():
    """Consume tasks from RabbitMQ queue"""
    # Connect to RabbitMQ with increased heartbeat
    connection = await aio_pika.connect_robust(
        host=settings.RABBITMQ_HOST,
        port=settings.RABBITMQ_PORT,
        login=settings.RABBITMQ_USER,
        password=settings.RABBITMQ_PASSWORD,
        virtualhost=settings.RABBITMQ_VHOST,
        heartbeat=600  # 10 minutes
    )

    # Create channel with QoS settings
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=1)

    # Declare queue with increased message TTL
    queue = await channel.declare_queue(
        settings.RABBITMQ_QUEUE_TRANSCRIPTION,
        durable=True,
        arguments={
            "x-message-ttl": 7200000  # 2 hours
        }
    )

    logger.info("Worker started, waiting for messages...")

    async def process_message(message: aio_pika.IncomingMessage):
        """Process message from queue with immediate ACK"""
        try:
            # Decode message
            body = message.body.decode()
            data = json.loads(body)

            # Get transcription ID
            transcription_id = data.get("transcription_id")
            if not transcription_id:
                logger.error("No transcription_id in message")
                await message.ack()  # Acknowledge anyway to remove from queue
                return

            logger.info(f"Processing transcription {transcription_id}")

            # СРАЗУ подтверждаем получение сообщения, чтобы RabbitMQ не пытался переотправить его
            await message.ack()

            # Запускаем обработку в отдельной задаче
            # Не ожидаем завершения (fire and forget)
            asyncio.create_task(process_transcription(transcription_id))

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            # Подтверждаем сообщение даже в случае ошибки
            await message.ack()

    # Start consuming with explicit acknowledgment
    await queue.consume(process_message)

    # Keep connection open
    await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(consume_task_queue())
    finally:
        # Закрываем thread pool при завершении
        thread_pool.shutdown(wait=False)