from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers import leaderboard, models, runs

app = FastAPI(title="VIP Benchmark API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(leaderboard.router)
app.include_router(models.router)
app.include_router(runs.router)


@app.get("/")
def root():
    return {"message": "VIP Benchmark API is running"}