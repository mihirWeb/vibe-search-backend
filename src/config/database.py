from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from src.config.settings import settings

# For PostgreSQL (using asyncpg)
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,  # Changed from True to False
    future=True
)

# For SQLite (using aiosqlite)
# engine = create_async_engine(
#     "sqlite+aiosqlite:///./app.db",
#     echo=False,  # Also change this if you use SQLite
#     future=True
# )

AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

Base = declarative_base()


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()