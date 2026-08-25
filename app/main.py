from __future__ import annotations

import logging
import shutil
import uuid
from functools import partial
from pathlib import Path
from threading import Lock

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
)
from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.services.llm_pipeline import LLMExtractionPipeline


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-user Excel locks
# ---------------------------------------------------------------------------

# Each browser/user gets one persistent workbook. Requests for the same user
# are serialized so two simultaneous runs cannot write the same XLSX at once.
# Different users still process independently.
_USER_EXCEL_LOCKS: dict[str, Lock] = {}
_USER_EXCEL_LOCKS_GUARD = Lock()


def _get_user_excel_lock(user_key: str) -> Lock:
    with _USER_EXCEL_LOCKS_GUARD:
        lock = _USER_EXCEL_LOCKS.get(user_key)
        if lock is None:
            lock = Lock()
            _USER_EXCEL_LOCKS[user_key] = lock
        return lock


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LinkedIn Job Recorder and Excel Extractor",
    version="2.3.0",
    description=(
        "Record a LinkedIn browser tab or upload a manually recorded "
        "LinkedIn video/image. The application extracts visible job "
        "information using Tesseract OCR and generates an Excel file."
    ),
)


# ---------------------------------------------------------------------------
# Supported file formats
# ---------------------------------------------------------------------------

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
}

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
}

ALLOWED_EXTENSIONS = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS


# ---------------------------------------------------------------------------
# Recorder page
# ---------------------------------------------------------------------------

RECORDER_HTML_PATH = (
    Path(__file__).resolve().parent
    / "recorder.html"
)


# ---------------------------------------------------------------------------
# General endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root() -> dict[str, object]:
    return {
        "service": (
            "LinkedIn Job Recorder and Excel Extractor"
        ),
        "recorder": "/recorder",
        "swagger": "/docs",
        "health": "/health",
        "supported_video_formats": sorted(
            VIDEO_EXTENSIONS
        ),
        "supported_image_formats": sorted(
            IMAGE_EXTENSIONS
        ),
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
    }


@app.get(
    "/recorder",
    response_class=HTMLResponse,
)
def recorder_page() -> HTMLResponse:
    """
    Display the browser-based screen recorder.

    The recorder page must exist at:

        app/recorder.html
    """

    if not RECORDER_HTML_PATH.is_file():
        raise HTTPException(
            status_code=500,
            detail=(
                "Recorder page was not found. "
                f"Expected file: {RECORDER_HTML_PATH}"
            ),
        )

    try:
        html_content = (
            RECORDER_HTML_PATH.read_text(
                encoding="utf-8"
            )
        )

    except OSError as exc:
        LOGGER.exception(
            "Unable to read recorder page"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to read recorder page: "
                f"{exc}"
            ),
        ) from exc

    return HTMLResponse(
        content=html_content,
        status_code=200,
    )


# ---------------------------------------------------------------------------
# Job extraction endpoint
# ---------------------------------------------------------------------------

@app.post(
    "/api/jobs/extract",
    response_class=FileResponse,
)
async def extract_jobs(
    background_tasks: BackgroundTasks,

    file: UploadFile = File(
        ...,
        description=(
            "LinkedIn video recording or screenshot. "
            "Supported formats: MP4, MOV, MKV, AVI, WEBM, "
            "PNG, JPG, JPEG, WEBP and BMP."
        ),
    ),

    user_id: str = Form(
        ...,
        description=(
            "Stable browser user ID. The same browser keeps the same ID so "
            "future runs append to that user's workbook only."
        ),
    ),

    sample_interval_seconds: float = Form(
        1.0,
        ge=0.25,
        le=10.0,
        description=(
            "How often to capture a frame from a video. "
            "This value is ignored for PNG/JPG images."
        ),
    ),

    crop_left_ratio: float = Form(
        0.075,
        ge=0.0,
        le=0.95,
        description=(
            "Left crop ratio. Use 0.0 when the image "
            "already contains only the job post."
        ),
    ),

    crop_right_ratio: float = Form(
        0.505,
        ge=0.05,
        le=1.0,
        description=(
            "Right crop ratio."
        ),
    ),

    crop_top_ratio: float = Form(
        0.145,
        ge=0.0,
        le=0.95,
        description=(
            "Top crop ratio."
        ),
    ),

    crop_bottom_ratio: float = Form(
        0.985,
        ge=0.05,
        le=1.0,
        description=(
            "Bottom crop ratio."
        ),
    ),

    keep_artifacts: bool = Form(
        False,
        description=(
            "Keep extracted frames and run files "
            "after the Excel file is returned."
        ),
    ),
) -> FileResponse:
    """
    Process a LinkedIn recording or screenshot.

    Workflow:

        Upload
            → Save locally
            → Extract video frames or load image
            → Run Tesseract OCR
            → Parse job information
            → Create Job_Postings Excel sheet
            → Return Excel file
    """

    # -----------------------------------------------------------------------
    # Validate and normalize browser user ID
    # -----------------------------------------------------------------------

    try:
        # Store only a canonical UUID-derived key on disk. This prevents path
        # traversal and keeps one stable workbook per browser/user.
        user_key = uuid.UUID(str(user_id).strip()).hex
    except (ValueError, AttributeError, TypeError) as exc:
        raise HTTPException(
            status_code=400,
            detail="user_id must be a valid UUID.",
        ) from exc

    # -----------------------------------------------------------------------
    # Validate crop values
    # -----------------------------------------------------------------------

    if crop_left_ratio >= crop_right_ratio:
        raise HTTPException(
            status_code=400,
            detail=(
                "crop_left_ratio must be less than "
                "crop_right_ratio"
            ),
        )

    if crop_top_ratio >= crop_bottom_ratio:
        raise HTTPException(
            status_code=400,
            detail=(
                "crop_top_ratio must be less than "
                "crop_bottom_ratio"
            ),
        )

    # -----------------------------------------------------------------------
    # Validate uploaded filename and extension
    # -----------------------------------------------------------------------

    original_name = Path(
        file.filename
        or "linkedin_recording.webm"
    ).name

    suffix = Path(
        original_name
    ).suffix.lower()

    if not suffix:
        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded file does not have "
                "a file extension."
            ),
        )

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported input format. "
                "Allowed formats: "
                + ", ".join(
                    sorted(ALLOWED_EXTENSIONS)
                )
            ),
        )

    # -----------------------------------------------------------------------
    # Create run folder
    # -----------------------------------------------------------------------

    run_id = uuid.uuid4().hex

    run_dir = (
        settings.runs_dir
        / run_id
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    input_path = (
        run_dir
        / f"input{suffix}"
    )

    # One persistent Excel file PER USER.
    # Example:
    #   User A run 1 -> user_excels/<A>.xlsx
    #   User A run 2 -> appends to the same <A>.xlsx
    #   User B run 1 -> user_excels/<B>.xlsx
    #
    # The workbook is outside run_dir so normal run cleanup never deletes it.
    user_excel_dir = settings.runs_dir / "user_excels"
    user_excel_dir.mkdir(parents=True, exist_ok=True)

    # One persistent LLM Excel file per browser/user.
    # Future runs for the same user append to this workbook.
    llm_output_path = (
        user_excel_dir
        / f"linkedin_results_{user_key}_llm.xlsx"
    )

    user_excel_lock = _get_user_excel_lock(user_key)

    result = None

    try:
        # -------------------------------------------------------------------
        # Save uploaded file
        # -------------------------------------------------------------------

        await _save_upload(
            file=file,
            destination=input_path,
            max_upload_mb=(
                settings.max_upload_mb
            ),
        )

        LOGGER.info(
            (
                "Starting LLM extraction "
                "run=%s source=%s"
            ),
            run_id,
            original_name,
        )

        # ---------------------------------------------------------------
        # LLM pipeline only
        # ---------------------------------------------------------------
        llm_pipeline_call = partial(
            LLMExtractionPipeline(
                settings
            ).run,

            input_path=input_path,
            run_dir=run_dir,
            output_path=llm_output_path,

            sample_interval_seconds=(
                sample_interval_seconds
            ),

            crop_left_ratio=(
                crop_left_ratio
            ),

            crop_right_ratio=(
                crop_right_ratio
            ),

            crop_top_ratio=(
                crop_top_ratio
            ),

            crop_bottom_ratio=(
                crop_bottom_ratio
            ),
        )

        def run_llm_pipeline_for_user():
            # Serialize writes to this user's persistent LLM workbook.
            with user_excel_lock:
                return llm_pipeline_call()

        result = await run_in_threadpool(
            run_llm_pipeline_for_user
        )

        if not llm_output_path.is_file():
            raise RuntimeError(
                "LLM extraction completed but the Excel file was not created."
            )

        LOGGER.info(
            (
                "Completed LLM extraction run=%s "
                "jobs=%s frames=%s errors=%s "
                "source=%s"
            ),
            run_id,
            len(result.jobs),
            result.frame_count,
            len(result.errors),
            original_name,
        )

    except HTTPException:
        shutil.rmtree(
            run_dir,
            ignore_errors=True,
        )

        raise

    except Exception as exc:
        LOGGER.exception(
            "LLM extraction failed for run=%s",
            run_id,
        )

        shutil.rmtree(
            run_dir,
            ignore_errors=True,
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    finally:
        await file.close()

    # -----------------------------------------------------------------------
    # Delete temporary run folder only after the Excel response has been sent
    # -----------------------------------------------------------------------

    if not (
        keep_artifacts
        or settings.keep_run_artifacts
    ):
        background_tasks.add_task(
            shutil.rmtree,
            run_dir,
            True,
        )

    source_type = (
        "image"
        if suffix in IMAGE_EXTENSIONS
        else "video"
    )

    return FileResponse(
        path=llm_output_path,
        media_type=(
            "application/vnd.openxmlformats-"
            "officedocument.spreadsheetml.sheet"
        ),
        filename="linkedin_results.xlsx",
        background=background_tasks,
        headers={
            "X-User-Id": str(user_id).strip(),
            "X-Jobs-Detected": str(
                len(result.jobs)
            ),
            "X-Frames-Processed": str(
                result.frame_count
            ),
            "X-Extraction-Errors": str(
                len(result.errors)
            ),
            "X-Pipeline-Type": "llm",
            "X-Source-Type": source_type,
            "X-Run-Id": run_id,
        },
    )


# ---------------------------------------------------------------------------
# Upload helper
# ---------------------------------------------------------------------------

async def _save_upload(
    file: UploadFile,
    destination: Path,
    max_upload_mb: int,
) -> None:
    """
    Save an uploaded file in chunks.

    The chunked approach prevents the complete video from being loaded
    into memory at once.
    """

    max_bytes = (
        max_upload_mb
        * 1024
        * 1024
    )

    bytes_written = 0

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        with destination.open("wb") as output:
            while True:
                chunk = await file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                bytes_written += len(chunk)

                if bytes_written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Uploaded file exceeds the "
                            f"{max_upload_mb} MB limit."
                        ),
                    )

                output.write(chunk)

    except Exception:
        destination.unlink(
            missing_ok=True
        )

        raise

    if bytes_written == 0:
        destination.unlink(
            missing_ok=True
        )

        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded file is empty."
            ),
        )

    LOGGER.info(
        (
            "Saved upload file=%s "
            "size_bytes=%s"
        ),
        destination.name,
        bytes_written,
    )