from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path

import cv2
from dotenv import load_dotenv

from app.config import Settings
from app.models import (
    ExtractionError,
    ExtractionResult,
    FrameOcrResult,
    FrameRecord,
)
from app.services.apollo_service import (
    ApolloEnrichmentResult,
    ApolloService,
)
from app.services.excel_service import ExcelService
from app.services.llm_job_extractor import LLMJobExtractor
from app.services.ocr_service import (
    OcrRuntimeCompatibilityError,
    OcrService,
)
from app.services.video_processor import VideoProcessor


load_dotenv()

LOGGER = logging.getLogger(__name__)


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


class LLMExtractionPipeline:
    def __init__(
        self,
        settings: Settings,
    ) -> None:
        self.settings = settings

    def run(
        self,
        input_path: Path,
        run_dir: Path,
        output_path: Path,
        sample_interval_seconds: float | None = None,
        crop_left_ratio: float | None = None,
        crop_right_ratio: float | None = None,
        crop_top_ratio: float | None = None,
        crop_bottom_ratio: float | None = None,
    ) -> ExtractionResult:
        input_path = Path(input_path)
        run_dir = Path(run_dir)
        output_path = Path(output_path)

        if not input_path.is_file():
            raise FileNotFoundError(
                f"Input file was not found: {input_path}"
            )

        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        frames_dir = run_dir / "frames"

        errors: list[ExtractionError] = []

        suffix = input_path.suffix.lower()

        # ---------------------------------------------------------------
        # Prepare frames
        # ---------------------------------------------------------------

        if suffix in IMAGE_EXTENSIONS:
            frames, source_errors = (
                self._prepare_image(
                    input_path=input_path,
                    frames_dir=frames_dir,
                )
            )

            left = 0.0
            right = 1.0
            top = 0.0
            bottom = 1.0

        elif suffix in VIDEO_EXTENSIONS:
            frames, source_errors = (
                self._prepare_video(
                    input_path=input_path,
                    frames_dir=frames_dir,
                    sample_interval_seconds=(
                        sample_interval_seconds
                    ),
                )
            )

            left = (
                self.settings.crop_left_ratio
                if crop_left_ratio is None
                else crop_left_ratio
            )

            right = (
                self.settings.crop_right_ratio
                if crop_right_ratio is None
                else crop_right_ratio
            )

            top = (
                self.settings.crop_top_ratio
                if crop_top_ratio is None
                else crop_top_ratio
            )

            bottom = (
                self.settings.crop_bottom_ratio
                if crop_bottom_ratio is None
                else crop_bottom_ratio
            )

        else:
            raise ValueError(
                (
                    "Unsupported input format: "
                    f"{suffix}"
                )
            )

        errors.extend(source_errors)

        self._validate_crop(
            left,
            right,
            top,
            bottom,
        )

        # ---------------------------------------------------------------
        # OCR
        # ---------------------------------------------------------------

        ocr_service = OcrService(
            language=self.settings.ocr_language,
            min_confidence=(
                self.settings.min_ocr_confidence
            ),
            cpu_threads=(
                self.settings.ocr_cpu_threads
            ),
            tesseract_cmd=(
                self.settings.tesseract_cmd
            ),
        )

        ocr_frames: list[FrameOcrResult] = []

        for frame in frames:
            try:
                result = ocr_service.extract(
                    frame,
                    left,
                    right,
                    top,
                    bottom,
                )

                ocr_frames.append(result)

            except OcrRuntimeCompatibilityError as exc:
                LOGGER.exception(
                    "OCR runtime error frame=%s",
                    frame.path,
                )

                errors.append(
                    ExtractionError(
                        "ocr_runtime",
                        frame.timestamp,
                        frame.path.name,
                        str(exc),
                    )
                )

                break

            except Exception as exc:
                LOGGER.exception(
                    "OCR failed frame=%s",
                    frame.path,
                )

                errors.append(
                    ExtractionError(
                        "ocr",
                        frame.timestamp,
                        frame.path.name,
                        str(exc),
                    )
                )

        # ---------------------------------------------------------------
        # LLM semantic extraction (SEPARATE from existing regex pipeline)
        # ---------------------------------------------------------------

        jobs = []
        llm_usage: dict[str, int | str] = {
            "request_count": 0,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "model": os.getenv("OPENAI_JOB_MODEL", "gpt-5-mini"),
        }

        try:
            with LLMJobExtractor(
                skills_file=self.settings.skills_file,
            ) as llm_extractor:
                jobs = llm_extractor.extract(
                    frames=ocr_frames,
                    source_video=input_path.name,
                )
                llm_usage = llm_extractor.get_usage_summary()

        except Exception as exc:
            LOGGER.exception(
                "LLM LinkedIn extraction failed source=%s",
                input_path.name,
            )

            errors.append(
                ExtractionError(
                    "llm_extraction",
                    "",
                    input_path.name,
                    str(exc),
                )
            )

        if not jobs:
            errors.append(
                ExtractionError(
                    "parsing",
                    "",
                    input_path.name,
                    "No valid job posts were detected by the LLM pipeline",
                )
            )

        # ---------------------------------------------------------------
        # Cross-run duplicate filtering
        # ---------------------------------------------------------------

        excel_service = ExcelService()
        jobs_for_export = excel_service.filter_new_jobs(
            jobs=jobs,
            output_path=output_path,
        )

        skipped_duplicate_jobs = len(jobs) - len(jobs_for_export)
        if skipped_duplicate_jobs:
            LOGGER.info(
                "Skipped duplicate LinkedIn posts before Apollo count=%s",
                skipped_duplicate_jobs,
            )

        # ---------------------------------------------------------------
        # Apollo enrichment
        # ---------------------------------------------------------------

        apollo_results = (
            self._enrich_with_apollo(
                jobs=jobs_for_export,
                errors=errors,
                source_file=input_path.name,
            )
        )

        # ---------------------------------------------------------------
        # Excel
        # ---------------------------------------------------------------

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        excel_service.create_workbook(
            jobs=jobs_for_export,
            errors=errors,
            output_path=output_path,
            apollo_results=apollo_results,
        )

        # ---------------------------------------------------------------
        # Manifest
        # ---------------------------------------------------------------

        apollo_summary = {}

        for job_id, result in (
            apollo_results.items()
        ):
            apollo_summary[job_id] = {
                "match_status": (
                    result.company.match_status
                ),
                "company": (
                    result.company.name
                ),
                "domain": (
                    result.company.domain
                ),
                "officials_found": len(
                    result.officials
                ),
                "error": result.error,
            }

        manifest = {
            "pipeline_type": "llm",
            "llm_model": str(llm_usage.get("model") or os.getenv("OPENAI_JOB_MODEL", "gpt-5-mini")),
            "llm_token_usage": llm_usage,
            "source_file": input_path.name,
            "source_type": (
                "image"
                if suffix in IMAGE_EXTENSIONS
                else "video"
            ),
            "output_file": output_path.name,
            "retained_frames": len(frames),
            "ocr_frames": len(ocr_frames),
            "jobs_detected": len(jobs),
            "jobs_new_for_export": len(jobs_for_export),
            "jobs_skipped_duplicate": skipped_duplicate_jobs,
            "apollo_enabled": (
                self._apollo_enabled()
            ),
            "apollo": apollo_summary,
            "crop": {
                "left": left,
                "right": right,
                "top": top,
                "bottom": bottom,
            },
            "errors": [
                asdict(error)
                for error in errors
            ],
        }

        (
            run_dir / "manifest.json"
        ).write_text(
            json.dumps(
                manifest,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        estimated_llm_cost_usd = self._estimate_openai_cost_usd(
            model=str(llm_usage.get("model") or ""),
            input_tokens=int(llm_usage.get("input_tokens") or 0),
            cached_input_tokens=int(llm_usage.get("cached_input_tokens") or 0),
            output_tokens=int(llm_usage.get("output_tokens") or 0),
        )

        cost_text = (
            f"${estimated_llm_cost_usd:.6f}"
            if estimated_llm_cost_usd is not None
            else "N/A"
        )

        LOGGER.info(
            (
                "TOTAL OPENAI LLM USAGE FOR THIS RUN = %s TOKENS | "
                "INPUT=%s | CACHED_INPUT=%s | OUTPUT=%s | "
                "REASONING=%s | REQUESTS=%s | MODEL=%s | "
                "ESTIMATED_COST_USD=%s"
            ),
            int(llm_usage.get("total_tokens") or 0),
            int(llm_usage.get("input_tokens") or 0),
            int(llm_usage.get("cached_input_tokens") or 0),
            int(llm_usage.get("output_tokens") or 0),
            int(llm_usage.get("reasoning_tokens") or 0),
            int(llm_usage.get("request_count") or 0),
            str(llm_usage.get("model") or "unknown"),
            cost_text,
        )

        LOGGER.info(
            (
                "Pipeline completed source=%s "
                "frames=%s jobs=%s "
                "apollo_results=%s errors=%s"
            ),
            input_path.name,
            len(frames),
            len(jobs),
            len(apollo_results),
            len(errors),
        )

        return ExtractionResult(
            jobs,
            errors,
            output_path,
            len(frames),
        )

    # ------------------------------------------------------------------
    # Apollo
    # ------------------------------------------------------------------

    def _enrich_with_apollo(
        self,
        jobs: list,
        errors: list[ExtractionError],
        source_file: str,
    ) -> dict[
        str,
        ApolloEnrichmentResult,
    ]:
        results: dict[
            str,
            ApolloEnrichmentResult,
        ] = {}

        if not self._apollo_enabled():
            LOGGER.info(
                "Apollo enrichment is disabled"
            )
            return results

        api_key = os.getenv(
            "APOLLO_API_KEY",
            "",
        ).strip()

        if not api_key:
            errors.append(
                ExtractionError(
                    "apollo_configuration",
                    "",
                    source_file,
                    (
                        "APOLLO_ENABLED is true but "
                        "APOLLO_API_KEY is missing"
                    ),
                )
            )

            return results

        # Avoid charging repeatedly for the same
        # company in one video.
        company_cache: dict[
            str,
            ApolloEnrichmentResult,
        ] = {}

        try:
            with ApolloService(
                api_key=api_key
            ) as apollo:
                for job in jobs:
                    company_name = str(
                        job.company_name
                        or ""
                    ).strip()

                    known_email = (
                        job.contact_emails[0]
                        if job.contact_emails
                        else ""
                    )

                    cache_key = (
                        apollo.build_company_cache_key(
                            company_name=(
                                company_name
                            ),
                            known_email=known_email,
                        )
                    )

                    if not cache_key:
                        results[job.job_id] = (
                            ApolloEnrichmentResult(
                                company=(
                                    self._missing_company()
                                ),
                                error=(
                                    "No reliable company "
                                    "name or domain"
                                ),
                            )
                        )

                        continue

                    if cache_key not in company_cache:
                        company_cache[cache_key] = (
                            apollo
                            .enrich_company_and_find_talent(
                                company_name=(
                                    company_name
                                ),
                                known_email=(
                                    known_email
                                ),
                            )
                        )

                    results[job.job_id] = (
                        company_cache[cache_key]
                    )

        except Exception as exc:
            LOGGER.exception(
                "Apollo enrichment process failed"
            )

            errors.append(
                ExtractionError(
                    "apollo",
                    "",
                    source_file,
                    str(exc),
                )
            )

        return results

    @staticmethod
    def _missing_company():
        from app.services.apollo_service import (
            ApolloCompany,
        )

        return ApolloCompany(
            match_status=(
                "MISSING_COMPANY_IDENTIFIER"
            )
        )

    @staticmethod
    def _estimate_openai_cost_usd(
        model: str,
        input_tokens: int,
        cached_input_tokens: int,
        output_tokens: int,
    ) -> float | None:
        """Estimate text-token cost in USD using rates per 1M tokens."""
        normalized_model = str(model or "").strip().casefold()

        # (normal input, cached input, output), USD per 1M tokens.
        rates: dict[str, tuple[float, float, float]] = {
            "gpt-5-mini": (0.25, 0.025, 2.00),
            "gpt-5.4-mini": (0.75, 0.075, 4.50),
            "gpt-5.6-luna": (0.50, 0.05, 3.00),
            "gpt-5.6-terra": (1.25, 0.125, 7.50),
            "gpt-5.6-sol": (2.50, 0.25, 15.00),
        }

        selected_rates = None
        for model_name, model_rates in rates.items():
            if (
                normalized_model == model_name
                or normalized_model.startswith(model_name + "-")
            ):
                selected_rates = model_rates
                break

        if selected_rates is None:
            return None

        input_rate, cached_rate, output_rate = selected_rates
        total_input = max(0, int(input_tokens))
        cached_input = min(total_input, max(0, int(cached_input_tokens)))
        uncached_input = total_input - cached_input
        output = max(0, int(output_tokens))

        return (
            uncached_input * input_rate
            + cached_input * cached_rate
            + output * output_rate
        ) / 1_000_000.0

    @staticmethod
    def _apollo_enabled() -> bool:
        value = os.getenv(
            "APOLLO_ENABLED",
            "false",
        ).strip().casefold()

        return value in {
            "1",
            "true",
            "yes",
            "on",
        }

    # ------------------------------------------------------------------
    # Source preparation
    # ------------------------------------------------------------------

    def _prepare_video(
        self,
        input_path: Path,
        frames_dir: Path,
        sample_interval_seconds: float | None,
    ) -> tuple[
        list[FrameRecord],
        list[ExtractionError],
    ]:
        interval = (
            self.settings.sample_interval_seconds
            if sample_interval_seconds is None
            else sample_interval_seconds
        )

        processor = VideoProcessor(
            sample_interval_seconds=interval,
            blur_threshold=(
                self.settings.blur_threshold
            ),
            duplicate_hash_distance=(
                self.settings
                .duplicate_hash_distance
            ),
        )

        return processor.extract_frames(
            input_path,
            frames_dir,
        )

    def _prepare_image(
        self,
        input_path: Path,
        frames_dir: Path,
    ) -> tuple[
        list[FrameRecord],
        list[ExtractionError],
    ]:
        frames_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        image = cv2.imread(
            str(input_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise ValueError(
                (
                    "OpenCV could not read image: "
                    f"{input_path}"
                )
            )

        height, width = image.shape[:2]

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        blur_score = float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F,
            ).var()
        )

        normalized_path = (
            frames_dir
            / "frame_00000_00-00-00.png"
        )

        if not cv2.imwrite(
            str(normalized_path),
            image,
        ):
            raise RuntimeError(
                "Failed to save image frame"
            )

        frame = FrameRecord(
            frame_index=0,
            source_frame_number=0,
            timestamp_seconds=0.0,
            path=normalized_path,
            width=width,
            height=height,
            blur_score=blur_score,
        )

        return [frame], []

    @staticmethod
    def _validate_crop(
        left: float,
        right: float,
        top: float,
        bottom: float,
    ) -> None:
        for value in (
            left,
            right,
            top,
            bottom,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    (
                        "Crop values must be "
                        "between 0 and 1"
                    )
                )

        if left >= right:
            raise ValueError(
                (
                    "crop_left_ratio must be "
                    "less than crop_right_ratio"
                )
            )

        if top >= bottom:
            raise ValueError(
                (
                    "crop_top_ratio must be "
                    "less than crop_bottom_ratio"
                )
            )