from __future__ import annotations

import json
import logging
import math
import re
import time
import uuid
from collections.abc import AsyncGenerator, Callable, Coroutine
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import escape

from findex.index import Index, build_index
from findex.query import QuerySyntaxError, parse_query
from findex.search import SearchEngine
from findex.snippets import build_snippet
from findex.store import open_index
from findex.web.config import Settings
from findex.web.schemas import (
    DocumentResponse,
    HealthResponse,
    SearchResponse,
    SearchResultItem,
    StatsResponse,
)

log = logging.getLogger("findex.web")

_WEB_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(_WEB_DIR / "templates"))


def _load_index_from_path(index_path: str) -> tuple[Index, int]:
    path = Path(index_path)
    if not path.exists():
        msg = f"INDEX_PATH does not exist: {path}"
        raise FileNotFoundError(msg)

    if path.suffix in {".jsonl", ".txt"} or path.is_dir():
        idx = build_index(path, executor="serial", with_positions=True)
        return idx, path.stat().st_size if path.is_file() else 0

    with open_index(path) as idx:
        return idx, path.stat().st_size


def _read_document_text(doc_path: str) -> str:
    """Resolve document body either from a .jsonl#line reference or a plain file path."""
    if "#" in doc_path:
        file_part, line_str = doc_path.rsplit("#", 1)
        p = Path(file_part)
        if p.is_file() and line_str.isdigit():
            target_idx = int(line_str)
            with p.open("r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if idx == target_idx:
                        raw_obj: object = json.loads(line)
                        if isinstance(raw_obj, dict):
                            data = cast("dict[str, object]", raw_obj)
                            return str(data.get("text", ""))
    p = Path(doc_path)
    if p.is_file():
        return p.read_text(encoding="utf-8", errors="replace")
    return ""


def _highlight_terms(snippet: str, query_str: str) -> str:
    escaped = str(escape(snippet))
    try:
        terms = sorted(set(parse_query(query_str).positive_terms()), key=len, reverse=True)
    except QuerySyntaxError:
        terms = []
    if not terms:
        return escaped
    pattern = re.compile(r"(" + "|".join(re.escape(t) for t in terms) + r")", re.IGNORECASE)
    return pattern.sub(r"<mark>\1</mark>", escaped)


def get_settings(request: Request) -> Settings:
    settings: Settings | None = getattr(request.app.state, "settings", None)
    return settings if settings is not None else Settings()


def get_optional_index(request: Request) -> Index | None:
    return getattr(request.app.state, "index", None)


def get_index(
    idx: Annotated[Index | None, Depends(get_optional_index)] = None,
) -> Index:
    if idx is None:
        raise HTTPException(status_code=503, detail="Search index is not loaded")
    return idx


def get_engine(request: Request) -> SearchEngine:
    engine: SearchEngine | None = getattr(request.app.state, "engine", None)
    if engine is None:
        idx = get_index(get_optional_index(request))
        engine = SearchEngine(idx)
        request.app.state.engine = engine
    return engine


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings if settings is not None else Settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncGenerator[None]:
        logging.basicConfig(
            level=getattr(logging, cfg.log_level.upper(), logging.INFO),
            format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":%(message)s}',
        )
        application.state.settings = cfg
        application.state.started_at = time.monotonic()

        if getattr(application.state, "index", None) is None:
            idx, size_bytes = _load_index_from_path(cfg.index_path)
            application.state.index = idx
            application.state.engine = SearchEngine(idx)
            application.state.index_size_bytes = size_bytes
            log.info(
                json.dumps(
                    {
                        "event": "startup",
                        "index_path": cfg.index_path,
                        "docs": len(idx.doc_meta),
                    }
                )
            )
        yield
        log.info(json.dumps({"event": "shutdown"}))

    application = FastAPI(
        title="findex Search API",
        version="0.7.0",
        description="Ranked full-text search API and web UI powered by FastAPI and Pydantic.",
        lifespan=lifespan,
    )
    application.state.settings = cfg
    application.state.started_at = time.monotonic()
    application.state.index_size_bytes = 0

    application.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.parsed_cors_origins(),
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @application.middleware("http")
    async def request_logging_middleware(
        request: Request,
        call_next: Callable[[Request], Coroutine[Any, Any, Response]],
    ) -> Response:
        req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = req_id
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            log.exception(
                json.dumps(
                    {
                        "event": "unhandled_error",
                        "request_id": req_id,
                        "method": request.method,
                        "path": request.url.path,
                        "duration_ms": round(elapsed_ms, 2),
                    }
                )
            )
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error", "request_id": req_id},
                headers={"X-Request-ID": req_id},
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        response.headers["X-Request-ID"] = req_id
        log.info(
            json.dumps(
                {
                    "event": "request",
                    "request_id": req_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": round(elapsed_ms, 2),
                }
            )
        )
        return response

    application.mount("/static", StaticFiles(directory=str(_WEB_DIR / "static")), name="static")

    @application.get("/health", response_model=HealthResponse, tags=["ops"])
    def health(
        request: Request,
        idx: Annotated[Index | None, Depends(get_optional_index)] = None,
    ) -> JSONResponse:
        started: float = getattr(request.app.state, "started_at", time.monotonic())
        uptime = round(max(0.0, time.monotonic() - started), 3)
        if idx is None:
            payload = HealthResponse(status="unavailable", doc_count=0, uptime_s=uptime)
            return JSONResponse(status_code=503, content=payload.model_dump())
        payload = HealthResponse(status="ok", doc_count=len(idx.doc_meta), uptime_s=uptime)
        return JSONResponse(status_code=200, content=payload.model_dump())

    @application.get("/stats", response_model=StatsResponse, tags=["ops"])
    def stats(request: Request, idx: Annotated[Index, Depends(get_index)]) -> StatsResponse:
        started: float = getattr(request.app.state, "started_at", time.monotonic())
        size_bytes: int = getattr(request.app.state, "index_size_bytes", 0)
        total_postings = sum(len(plist) for plist in idx.postings.values())
        return StatsResponse(
            doc_count=len(idx.doc_meta),
            vocabulary_size=len(idx.postings),
            total_postings=total_postings,
            index_size_bytes=size_bytes,
            uptime_s=round(max(0.0, time.monotonic() - started), 3),
        )

    @application.get("/search", response_model=SearchResponse, tags=["search"])
    def search(
        q: Annotated[str, Query(min_length=1, max_length=200, description="Search query string")],
        k: Annotated[int, Query(ge=1, le=100, description="Results per page")] = 10,
        scorer: Annotated[Literal["bm25", "tfidf"], Query(description="Ranking model")] = "bm25",
        page: Annotated[int, Query(ge=1, le=1000, description="1-based page number")] = 1,
        engine: Annotated[SearchEngine, Depends(get_engine)] = ...,  # type: ignore[assignment]
    ) -> SearchResponse:
        t0 = time.perf_counter()
        try:
            ast = parse_query(q)
            matched_docs = ast.evaluate(engine.index)
            total = len(matched_docs)
            total_pages = math.ceil(total / k) if total > 0 else 0
            needed = page * k
            all_top = engine.search(q, scorer=scorer, k=max(needed, k), with_snippets=False)
            start = (page - 1) * k
            page_slice = all_top[start : start + k]
            terms = ast.positive_terms()
        except (QuerySyntaxError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"Invalid query syntax: {exc}") from exc

        items: list[SearchResultItem] = []
        for hit in page_slice:
            meta = engine.index.doc_meta.get(hit.doc_id)
            snippet = build_snippet(meta.path, terms) if meta is not None else ""
            items.append(
                SearchResultItem(
                    doc_id=hit.doc_id,
                    title=hit.title,
                    score=round(hit.score, 6),
                    snippet=snippet,
                )
            )

        took_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return SearchResponse(
            query=q,
            scorer=scorer,
            page=page,
            k=k,
            total=total,
            total_pages=total_pages,
            took_ms=took_ms,
            results=items,
        )

    @application.get("/docs/{doc_id}", response_model=DocumentResponse, tags=["search"])
    def get_document(
        doc_id: int,
        idx: Annotated[Index, Depends(get_index)],
    ) -> DocumentResponse:
        meta = idx.doc_meta.get(doc_id)
        if meta is None:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        text = _read_document_text(meta.path)
        return DocumentResponse(
            doc_id=meta.doc_id,
            title=meta.title,
            path=meta.path,
            length=idx.doc_lengths.get(doc_id, 0),
            text=text,
        )

    @application.get("/", response_class=HTMLResponse, include_in_schema=False)
    def ui_home(
        request: Request,
        q: str = "",
        scorer: Literal["bm25", "tfidf"] = "bm25",
        k: Annotated[int, Query(ge=1, le=100)] = 10,
        page: Annotated[int, Query(ge=1, le=1000)] = 1,
        engine: Annotated[SearchEngine, Depends(get_engine)] = ...,  # type: ignore[assignment]
    ) -> HTMLResponse:
        cleaned_q = q.strip()
        response_obj: SearchResponse | None = None
        highlighted: list[dict[str, object]] = []
        error_msg: str | None = None

        if cleaned_q:
            try:
                response_obj = search(q=cleaned_q, k=k, scorer=scorer, page=page, engine=engine)
                for item in response_obj.results:
                    highlighted.append(
                        {
                            "doc_id": item.doc_id,
                            "title": item.title,
                            "score": item.score,
                            "highlighted_snippet": _highlight_terms(item.snippet, cleaned_q),
                        }
                    )
            except HTTPException as exc:
                error_msg = str(exc.detail)

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "q": cleaned_q,
                "scorer": scorer,
                "response": response_obj,
                "highlighted_results": highlighted,
                "error": error_msg,
            },
        )

    @application.get("/doc/{doc_id}", response_class=HTMLResponse, include_in_schema=False)
    def ui_document(
        request: Request,
        doc_id: int,
        idx: Annotated[Index, Depends(get_index)],
    ) -> HTMLResponse:
        doc = get_document(doc_id=doc_id, idx=idx)
        return templates.TemplateResponse(
            request=request,
            name="doc.html",
            context={"doc": doc},
        )

    return application


app = create_app()
