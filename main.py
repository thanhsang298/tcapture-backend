import uvicorn
from fastapi import FastAPI
from src.api import api_router, health_router
from src.config import TaiyoConfig

app = FastAPI(title=TaiyoConfig.API.PROJECT_NAME)
app.include_router(api_router, prefix=TaiyoConfig.API.API_V1_STR, tags=["Taiyo TCapture"])
app.include_router(health_router, prefix=TaiyoConfig.API.API_V1_STR, tags=["Taiyo TCapture"])

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=TaiyoConfig.API.PORT)