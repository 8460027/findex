from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SearchResultItem(BaseModel):
    doc_id: int
    title: str
    score: float
    snippet: str


class SearchResponse(BaseModel):
    query: str
    scorer: Literal["bm25", "tfidf"]
    page: int = Field(ge=1)
    k: int = Field(ge=1, le=100)
    total: int = Field(ge=0)
    total_pages: int = Field(ge=0)
    took_ms: float = Field(ge=0.0)
    results: list[SearchResultItem]


class DocumentResponse(BaseModel):
    doc_id: int
    title: str
    path: str
    length: int = Field(ge=0)
    text: str


class StatsResponse(BaseModel):
    doc_count: int = Field(ge=0)
    vocabulary_size: int = Field(ge=0)
    total_postings: int = Field(ge=0)
    index_size_bytes: int = Field(ge=0)
    uptime_s: float = Field(ge=0.0)


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    doc_count: int = Field(ge=0)
    uptime_s: float = Field(ge=0.0)
