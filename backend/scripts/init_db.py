import asyncio
import sys
import os

# Добавляем корневую директорию проекта в PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.db.session import async_session
from backend.app.db.init_db import init_db


async def main():
    """Initialize the database"""
    async with async_session() as db:
        await init_db(db)
    print("Database initialized")

if __name__ == "__main__":
    asyncio.run(main())
