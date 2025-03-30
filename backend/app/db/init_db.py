import logging
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.core.security import get_password_hash
from backend.app.schemas.user import UserCreate
from backend.app.models.user import User

logger = logging.getLogger(__name__)


async def init_db(db: AsyncSession) -> None:
    """Initialize the database with a super admin user."""

    # Check if super admin user already exists
    query = await db.execute(
        text("SELECT id FROM users WHERE email = :email"),
        {"email": settings.FIRST_SUPERUSER_EMAIL}
    )
    user = query.first()

    if not user:
        logger.info("Creating initial super admin user")
        user_in = UserCreate(
            email=settings.FIRST_SUPERUSER_EMAIL,
            username=settings.FIRST_SUPERUSER_USERNAME,
            password=settings.FIRST_SUPERUSER_PASSWORD,
            is_admin=True,
            first_name="Admin",
            last_name="User"
        )

        db_user = User(
            email=user_in.email,
            username=user_in.username,
            hashed_password=get_password_hash(user_in.password),
            is_admin=user_in.is_admin,
            first_name=user_in.first_name,
            last_name=user_in.last_name
        )

        db.add(db_user)
        await db.commit()
        logger.info("Super admin user created")
    else:
        logger.info("Super admin user already exists")
