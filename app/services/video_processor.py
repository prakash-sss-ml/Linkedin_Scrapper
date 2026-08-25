from __future__ import annotations

import logging
import math
from pathlib import Path

import cv2

from app.models import (
    ExtractionError,
    FrameRecord,
)


LOGGER = logging.getLogger(__name__)


class VideoProcessor:
    """
    Extract frames sequentially from MP4, WebM, MOV, AVI and MKV files.

    This implementation is designed for browser-generated WebM recordings,
    which may contain variable or incorrect FPS metadata.

    Important behavior:
    - Reads every frame sequentially.
    - Does not seek using CAP_PROP_POS_MSEC.
    - Uses both timestamps and frame numbers for sampling.
    - Does not discard frames as visual duplicates.
    - Keeps low-quality frames while recording a warning.
    """

    def __init__(
        self,
        sample_interval_seconds: float = 1.0,
        blur_threshold: float = 85.0,
        duplicate_hash_distance: int = 4,
    ) -> None:
        if sample_interval_seconds <= 0:
            raise ValueError(
                "sample_interval_seconds must be greater than zero"
            )

        self.sample_interval_seconds = float(
            sample_interval_seconds
        )

        self.blur_threshold = float(
            blur_threshold
        )

        # Kept for compatibility with pipeline.py.
        # Duplicate filtering is intentionally disabled because scrolling
        # LinkedIn frames may look similar but contain different job posts.
        self.duplicate_hash_distance = int(
            duplicate_hash_distance
        )

    def extract_frames(
        self,
        input_path: Path,
        frames_dir: Path,
    ) -> tuple[
        list[FrameRecord],
        list[ExtractionError],
    ]:
        input_path = Path(input_path)
        frames_dir = Path(frames_dir)

        if not input_path.is_file():
            raise FileNotFoundError(
                f"Video was not found: {input_path}"
            )

        frames_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        capture = cv2.VideoCapture(
            str(input_path)
        )

        if not capture.isOpened():
            raise RuntimeError(
                f"OpenCV could not open the video: {input_path}"
            )

        errors: list[ExtractionError] = []
        retained_frames: list[FrameRecord] = []

        reported_fps = float(
            capture.get(cv2.CAP_PROP_FPS)
        )

        fps = self._normalize_fps(
            reported_fps
        )

        reported_frame_count = int(
            capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
            or 0
        )

        reported_width = int(
            capture.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
            or 0
        )

        reported_height = int(
            capture.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
            or 0
        )

        LOGGER.info(
            (
                "Opening video=%s reported_fps=%s "
                "effective_fps=%s reported_frames=%s "
                "size=%sx%s interval=%s"
            ),
            input_path.name,
            reported_fps,
            fps,
            reported_frame_count,
            reported_width,
            reported_height,
            self.sample_interval_seconds,
        )

        sample_every_frames = max(
            1,
            int(
                round(
                    fps
                    * self.sample_interval_seconds
                )
            ),
        )

        next_sample_frame = 0
        next_sample_time = 0.0

        decoded_frame_number = 0
        previous_timestamp_seconds = -1.0

        try:
            while True:
                read_success, frame = capture.read()

                if not read_success:
                    break

                if frame is None or frame.size == 0:
                    decoded_frame_number += 1
                    continue

                timestamp_seconds = (
                    self._get_timestamp_seconds(
                        capture=capture,
                        frame_number=decoded_frame_number,
                        fps=fps,
                        previous_timestamp_seconds=(
                            previous_timestamp_seconds
                        ),
                    )
                )

                previous_timestamp_seconds = (
                    timestamp_seconds
                )

                should_sample_by_frame = (
                    decoded_frame_number
                    >= next_sample_frame
                )

                should_sample_by_time = (
                    timestamp_seconds + 0.001
                    >= next_sample_time
                )

                if (
                    not should_sample_by_frame
                    and not should_sample_by_time
                ):
                    decoded_frame_number += 1
                    continue

                height, width = frame.shape[:2]

                blur_score = self._calculate_blur_score(
                    frame
                )

                retained_index = len(
                    retained_frames
                )

                timestamp_label = (
                    self._timestamp_for_filename(
                        timestamp_seconds
                    )
                )

                frame_path = (
                    frames_dir
                    / (
                        f"frame_{retained_index:05d}_"
                        f"{timestamp_label}.jpg"
                    )
                )

                saved = cv2.imwrite(
                    str(frame_path),
                    frame,
                    [
                        cv2.IMWRITE_JPEG_QUALITY,
                        94,
                    ],
                )

                if not saved:
                    errors.append(
                        ExtractionError(
                            "frame_write",
                            self._format_timestamp(
                                timestamp_seconds
                            ),
                            frame_path.name,
                            (
                                "OpenCV failed to save "
                                "the extracted frame."
                            ),
                        )
                    )

                    decoded_frame_number += 1
                    next_sample_frame += (
                        sample_every_frames
                    )
                    next_sample_time += (
                        self.sample_interval_seconds
                    )
                    continue

                frame_record = FrameRecord(
                    frame_index=retained_index,
                    source_frame_number=(
                        decoded_frame_number
                    ),
                    timestamp_seconds=(
                        timestamp_seconds
                    ),
                    path=frame_path,
                    width=width,
                    height=height,
                    blur_score=blur_score,
                )

                retained_frames.append(
                    frame_record
                )

                if (
                    blur_score
                    < self.blur_threshold
                ):
                    LOGGER.warning(
                        (
                            "Retaining blurry frame=%s "
                            "timestamp=%s blur=%.2f "
                            "threshold=%.2f"
                        ),
                        frame_path.name,
                        frame_record.timestamp,
                        blur_score,
                        self.blur_threshold,
                    )

                LOGGER.info(
                    (
                        "Retained frame=%s "
                        "source_frame=%s timestamp=%s "
                        "blur=%.2f"
                    ),
                    frame_path.name,
                    decoded_frame_number,
                    frame_record.timestamp,
                    blur_score,
                )

                # Advance frame-based sampling.
                while (
                    next_sample_frame
                    <= decoded_frame_number
                ):
                    next_sample_frame += (
                        sample_every_frames
                    )

                # Advance timestamp-based sampling.
                while (
                    next_sample_time
                    <= timestamp_seconds
                ):
                    next_sample_time += (
                        self.sample_interval_seconds
                    )

                decoded_frame_number += 1

        finally:
            capture.release()

        LOGGER.info(
            (
                "Video extraction completed "
                "video=%s decoded_frames=%s "
                "retained_frames=%s"
            ),
            input_path.name,
            decoded_frame_number,
            len(retained_frames),
        )

        if decoded_frame_number == 0:
            raise RuntimeError(
                (
                    "OpenCV opened the video but could "
                    "not decode any frames. The WebM "
                    "codec may not be supported by the "
                    "installed OpenCV build."
                )
            )

        if not retained_frames:
            raise RuntimeError(
                (
                    "The video was decoded, but no "
                    "frames were saved."
                )
            )

        if (
            len(retained_frames) == 1
            and decoded_frame_number > 1
        ):
            errors.append(
                ExtractionError(
                    "video_sampling",
                    retained_frames[0].timestamp,
                    retained_frames[0].path.name,
                    (
                        "Only one frame was retained "
                        f"from {decoded_frame_number} "
                        "decoded frames. Reduce "
                        "sample_interval_seconds."
                    ),
                )
            )

        return retained_frames, errors

    @staticmethod
    def _normalize_fps(
        reported_fps: float,
    ) -> float:
        """
        Browser WebM recordings occasionally report invalid values such as
        zero, NaN, 1000 FPS or extremely large frame rates.

        The recorder page requests approximately 15 FPS, so 15 FPS is used
        as a safe fallback.
        """

        if not math.isfinite(
            reported_fps
        ):
            return 15.0

        if (
            reported_fps < 1.0
            or reported_fps > 120.0
        ):
            return 15.0

        return reported_fps

    @staticmethod
    def _get_timestamp_seconds(
        capture: cv2.VideoCapture,
        frame_number: int,
        fps: float,
        previous_timestamp_seconds: float,
    ) -> float:
        """
        Prefer the decoder timestamp when it is valid and increasing.

        Otherwise, calculate the timestamp using the sequential frame number.
        """

        position_ms = float(
            capture.get(
                cv2.CAP_PROP_POS_MSEC
            )
            or 0.0
        )

        decoded_timestamp = (
            position_ms / 1000.0
        )

        calculated_timestamp = (
            frame_number / fps
        )

        if (
            math.isfinite(
                decoded_timestamp
            )
            and decoded_timestamp >= 0
            and decoded_timestamp
            > previous_timestamp_seconds
        ):
            return decoded_timestamp

        return calculated_timestamp

    @staticmethod
    def _calculate_blur_score(
        frame,
    ) -> float:
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        return float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F,
            ).var()
        )

    @staticmethod
    def _format_timestamp(
        seconds: float,
    ) -> str:
        safe_seconds = max(
            0,
            int(seconds)
        )

        hours = (
            safe_seconds // 3600
        )

        minutes = (
            safe_seconds % 3600
        ) // 60

        remaining_seconds = (
            safe_seconds % 60
        )

        return (
            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{remaining_seconds:02d}"
        )

    @staticmethod
    def _timestamp_for_filename(
        seconds: float,
    ) -> str:
        milliseconds = max(
            0,
            int(
                round(
                    seconds * 1000
                )
            ),
        )

        hours = (
            milliseconds // 3_600_000
        )

        minutes = (
            milliseconds % 3_600_000
        ) // 60_000

        remaining_seconds = (
            milliseconds % 60_000
        ) // 1000

        remaining_milliseconds = (
            milliseconds % 1000
        )

        return (
            f"{hours:02d}-"
            f"{minutes:02d}-"
            f"{remaining_seconds:02d}-"
            f"{remaining_milliseconds:03d}"
        )