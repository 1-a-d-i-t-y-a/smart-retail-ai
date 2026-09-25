from fastapi import APIRouter

from app.core import snapshot
from app.schemas.api import DashboardStats

router = APIRouter(tags=["Dashboard"])


@router.get("/dashboard/stats", response_model=DashboardStats)
def dashboard_stats():
    return snapshot()
