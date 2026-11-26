from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "sqlite:///./returns.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ReturnRecord(Base):
    __tablename__ = "return_records"

    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(String, index=True, default="ORD-SINGLE")
    comment = Column(String, nullable=False)
    predicted_reason = Column(String, index=True, nullable=False)
    confidence = Column(Float, nullable=False)
    model_used = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
