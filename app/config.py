from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass(slots=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    uploads_dir: Path = Path(
        os.getenv(
            "UPLOADS_DIR",
            BASE_DIR / "uploads",
        )
    )

    outputs_dir: Path = Path(
        os.getenv(
            "OUTPUTS_DIR",
            BASE_DIR / "outputs",
        )
    )

    runs_dir: Path = Path(
        os.getenv(
            "RUNS_DIR",
            BASE_DIR / "runs",
        )
    )

    skills_file: Path = Path(
        os.getenv(
            "SKILLS_FILE",
            BASE_DIR / "data" / "skills.txt",
        )
    )

    max_upload_mb: int = int(
        os.getenv("MAX_UPLOAD_MB", "500")
    )

    sample_interval_seconds: float = float(
        os.getenv(
            "SAMPLE_INTERVAL_SECONDS",
            "1.0",
        )
    )

    blur_threshold: float = float(
        os.getenv(
            "BLUR_THRESHOLD",
            "85.0",
        )
    )

    duplicate_hash_distance: int = int(
        os.getenv(
            "DUPLICATE_HASH_DISTANCE",
            "4",
        )
    )

    min_ocr_confidence: float = float(
        os.getenv(
            "MIN_OCR_CONFIDENCE",
            "0.38",
        )
    )

    # Crop values tuned for the uploaded 1366 × 720 recording.
    # This keeps the centre LinkedIn post feed.
    crop_left_ratio: float = float(
        os.getenv(
            "CROP_LEFT_RATIO",
            "0.075",
        )
    )

    crop_right_ratio: float = float(
        os.getenv(
            "CROP_RIGHT_RATIO",
            "0.505",
        )
    )

    crop_top_ratio: float = float(
        os.getenv(
            "CROP_TOP_RATIO",
            "0.145",
        )
    )

    crop_bottom_ratio: float = float(
        os.getenv(
            "CROP_BOTTOM_RATIO",
            "0.985",
        )
    )

    ocr_language: str = os.getenv(
        "OCR_LANGUAGE",
        "eng",
    )

    ocr_cpu_threads: int = int(
        os.getenv(
            "OCR_CPU_THREADS",
            "4",
        )
    )

    tesseract_cmd: str = os.getenv(
        "TESSERACT_CMD",
        "",
    ).strip()

    keep_run_artifacts: bool = _env_bool(
        "KEEP_RUN_ARTIFACTS",
        False,
    )

    def ensure_directories(self) -> None:
        for directory in (
            self.uploads_dir,
            self.outputs_dir,
            self.runs_dir,
        ):
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

    def validate_crop(self) -> None:
        values = (
            self.crop_left_ratio,
            self.crop_right_ratio,
            self.crop_top_ratio,
            self.crop_bottom_ratio,
        )

        if any(
            value < 0.0 or value > 1.0
            for value in values
        ):
            raise ValueError(
                "Crop ratios must be between 0.0 and 1.0"
            )

        if self.crop_left_ratio >= self.crop_right_ratio:
            raise ValueError(
                "CROP_LEFT_RATIO must be less than "
                "CROP_RIGHT_RATIO"
            )

        if self.crop_top_ratio >= self.crop_bottom_ratio:
            raise ValueError(
                "CROP_TOP_RATIO must be less than "
                "CROP_BOTTOM_RATIO"
            )


settings = Settings()
settings.validate_crop()
settings.ensure_directories()