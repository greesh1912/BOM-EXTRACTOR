"""Drawing upload route handler."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from ..services.pipeline_service import run_extraction_sync
from ..storage import storage

router = APIRouter(prefix="/api/upload", tags=["Upload"])

from bomx.ingest import SUPPORTED_EXT, validate_format

# single source of truth: the formats the extraction engine actually reads
ALLOWED_EXTENSIONS = set(SUPPORTED_EXT)


@router.post("")
async def upload_drawings(
    files: List[UploadFile] = File(...),
    auto_extract: bool = Form(default=True),
    force_ocr: bool = Form(default=False),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    uploaded_items = []

    for file in files:
        filename = file.filename or "drawing.pdf"
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"File extension '{ext}' for file '{filename}' is not supported. Supported extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            )

        content = await file.read()
        try:  # content must match a supported format and the extension
            validate_format(Path(filename), header=content[:2048])
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        drawing_id = storage.create_drawing(filename, content)
        meta = storage.get_metadata(drawing_id)

        if auto_extract:
            # Trigger extraction in thread pool
            loop = asyncio.get_event_loop()
            loop.run_in_executor(None, run_extraction_sync, drawing_id, None)

        uploaded_items.append(meta)

    return {"uploaded": uploaded_items, "count": len(uploaded_items)}

