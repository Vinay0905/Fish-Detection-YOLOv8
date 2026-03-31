from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Fish Detection API",
    description="API for fish detection and model inference.",
    version="1.0.0"
)


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Fish Detection API is running",
        "docs": "/docs",
        "redoc": "/redoc",
        "status": "success"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "service": "fish-detection-api"
        }
    )
