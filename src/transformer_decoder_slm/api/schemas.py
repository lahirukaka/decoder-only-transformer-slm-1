"""Request and response schemas for inference endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    temperature: float = Field(gt=0)


class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
