from fastapi import status, APIRouter
from fastapi.responses import JSONResponse

health_router = APIRouter()

@health_router.get("/health")
async def health_check():
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "healthy"}
    )