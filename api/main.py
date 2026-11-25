from fastapi import FastAPI
from api.routes import router

app = FastAPI(
    title="ResearchMind AI API",
    description="REST API for the ResearchMind AI engine.",
    version="2.0.0"
)

app.include_router(router, prefix="/api/v1")

@app.get("/health")
def health_check():
    return {"status": "ok", "app": "ResearchMind AI"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
