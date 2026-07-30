from fastapi import FastAPI

app = FastAPI(
    title="Industrial Defect Detection API",
    description="Week 4 FastAPI Backend",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "Industrial Defect Detection API is running."
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "Industrial Defect Detection",
        "version": "1.0.0"
    }