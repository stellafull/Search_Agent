from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.agents import solve

app = FastAPI(title="PAI-LangStudio Research Agent", version="0.1.0")


class QuestionRequest(BaseModel):
    question: str = Field(..., description="Input question string")


class AnswerResponse(BaseModel):
    answer: str = Field(..., description="Normalized answer")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/")
async def root_solve(payload: QuestionRequest) -> AnswerResponse:
    return AnswerResponse(answer=solve(payload.question))


@app.post("/solve")
async def solve_question(payload: QuestionRequest) -> AnswerResponse:
    return AnswerResponse(answer=solve(payload.question))


@app.post("/predict")
async def predict(payload: QuestionRequest) -> AnswerResponse:
    return AnswerResponse(answer=solve(payload.question))
