import json
import logging
from typing import Any, Dict

import aio_pika
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.db.session import async_session

logger = logging.getLogger(__name__)


async def send_transcription_task(transcription_id: int) -> None:
    """Send a transcription task to the RabbitMQ queue"""
    try:
        # Connect to RabbitMQ
        connection = await aio_pika.connect_robust(
            host=settings.RABBITMQ_HOST,
            port=settings.RABBITMQ_PORT,
            login=settings.RABBITMQ_USER,
            password=settings.RABBITMQ_PASSWORD,
            virtualhost=settings.RABBITMQ_VHOST
        )

        async with connection:
            # Create channel
            channel = await connection.channel()

            # Declare queue
            queue = await channel.declare_queue(
                settings.RABBITMQ_QUEUE_TRANSCRIPTION,
                durable=True,
                arguments={
                    "x-message-ttl": 7200000  # 2 hours
                },
            )

            # Create message
            message_body = json.dumps({"transcription_id": transcription_id})

            # Send message
            await channel.default_exchange.publish(
                aio_pika.Message(
                    body=message_body.encode(),
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                ),
                routing_key=queue.name,
            )

            logger.info(f"Sent transcription task {transcription_id} to queue")
    except Exception as e:
        logger.error(f"Error sending transcription task to queue: {str(e)}")
        # Handle error, maybe update transcription status to failed
        async with async_session() as db:
            await update_transcription_status(db, transcription_id, "failed", error_message=str(e))


async def update_transcription_status(
        db: AsyncSession,
        transcription_id: int,
        status: str,
        **kwargs
) -> None:
    """Update transcription status in database"""
    try:
        # Import here to avoid circular imports
        from backend.app.models.transcription import Transcription, TranscriptionStatus

        # Get transcription
        transcription = await db.get(Transcription, transcription_id)
        if not transcription:
            logger.error(f"Transcription {transcription_id} not found")
            return

        # Update status
        transcription.status = TranscriptionStatus(status)

        # Update other fields
        for key, value in kwargs.items():
            if hasattr(transcription, key):
                setattr(transcription, key, value)

        # Commit changes
        await db.commit()
        logger.info(f"Updated transcription {transcription_id} status to {status}")
    except Exception as e:
        logger.error(f"Error updating transcription status: {str(e)}")
        await db.rollback()
