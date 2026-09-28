from datetime import datetime, timezone

from sqlalchemy import DateTime, Integer, JSON, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .config import get_settings


class Base(DeclarativeBase):
    pass


class ForecastLog(Base):
    __tablename__ = "forecast_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    request_task: Mapped[str] = mapped_column(Text, nullable=False)
    requested_quarters: Mapped[list] = mapped_column(JSON, nullable=False)
    request_payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    final_output: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="received", nullable=False)


settings = get_settings()
engine = create_engine(settings.mysql_url, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(
    bind=engine, autoflush=False, autocommit=False, expire_on_commit=False
)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
