from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from observability.metrics import MetricsRegistry

router = APIRouter()
metrics = MetricsRegistry()


@router.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
async def prometheus_metrics() -> str:
    return metrics.prometheus_text()
