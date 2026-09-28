import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .agent import generate_forecast
from .db import ForecastLog, SessionLocal, init_db
from .schemas import ForecastRequest, ForecastResponse, HealthResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("tcs_forecasting_agent")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="TCS Financial Forecasting Agent",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="tcs-financial-forecasting-agent")


@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest) -> ForecastResponse:
    db = SessionLocal()
    log_row = ForecastLog(
        request_task=request.task,
        requested_quarters=request.quarters,
        request_payload=request.model_dump(),
        status="received",
    )
    db.add(log_row)
    db.commit()

    try:
        logger.info("Forecast request received: quarters=%s", request.quarters)
        result = generate_forecast(request.task, request.quarters)
        log_row.final_output = result.model_dump()
        log_row.status = "succeeded"
        db.commit()
        logger.info("Forecast completed successfully")
        return result
    except Exception as exc:
        log_row.error = str(exc)
        log_row.status = "failed"
        db.commit()
        logger.exception("Forecast failed")
        raise HTTPException(
            status_code=500,
            detail="Forecast generation failed. Check server logs.",
        ) from exc
    finally:
        db.close()
