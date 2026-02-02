from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from src.agents import solve_async


class SolveRequest(BaseModel):
    question: str


class SolveResponse(BaseModel):
    answer: str


app = FastAPI()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/solve", response_model=SolveResponse)
async def solve(req: SolveRequest) -> SolveResponse:
    answer = await solve_async(req.question)
    return SolveResponse(answer=answer)
