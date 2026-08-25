from __future__ import annotations

import logging
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import pytesseract
from pytesseract import Output

from app.models import (
    FrameOcrResult,
    FrameRecord,
    OcrLine,
)
from app.services.text_utils import clean_text


LOGGER = logging.getLogger(__name__)


class OcrInitializationError(RuntimeError):
    """Raised when Tesseract cannot be initialized."""


class OcrRuntimeCompatibilityError(RuntimeError):
    """Compatibility exception for pipeline handling."""


class OcrService:
    """Tesseract OCR with four-sided cropping."""

    def __init__(
        self,
        language: str = "eng",
        min_confidence: float = 0.38,
        cpu_threads: int = 4,
        tesseract_cmd: str = "",
    ) -> None:
        language = language.strip().lower()

        self.language = (
            "eng"
            if language in {
                "en",
                "english",
            }
            else language
        )

        self.min_confidence = max(
            0.0,
            min(
                1.0,
                float(min_confidence),
            ),
        )

        os.environ.setdefault(
            "OMP_THREAD_LIMIT",
            str(
                max(
                    1,
                    int(cpu_threads),
                )
            ),
        )

        executable = self._find_tesseract(
            tesseract_cmd
        )

        pytesseract.pytesseract.tesseract_cmd = (
            executable
        )

        try:
            version = (
                pytesseract.get_tesseract_version()
            )
        except Exception as exc:
            raise OcrInitializationError(
                (
                    f"Tesseract was found at "
                    f"'{executable}' but could not "
                    f"start: {exc}"
                )
            ) from exc

        LOGGER.info(
            (
                "Initialized Tesseract OCR "
                "version=%s language=%s "
                "executable=%s"
            ),
            version,
            self.language,
            executable,
        )

    @staticmethod
    def _find_tesseract(
        configured_path: str = "",
    ) -> str:
        candidates = [
            configured_path.strip(),
            os.getenv(
                "TESSERACT_CMD",
                "",
            ).strip(),
            (
                r"C:\Program Files\Tesseract-OCR"
                r"\tesseract.exe"
            ),
            (
                r"C:\Program Files (x86)"
                r"\Tesseract-OCR\tesseract.exe"
            ),
            shutil.which("tesseract") or "",
        ]

        for candidate in candidates:
            if (
                candidate
                and Path(candidate).is_file()
            ):
                return str(Path(candidate))

        raise OcrInitializationError(
            (
                "Tesseract executable was not found. "
                "Install Tesseract OCR to "
                r"C:\Program Files\Tesseract-OCR"
                r"\tesseract.exe "
                "or set the TESSERACT_CMD "
                "environment variable."
            )
        )

    def extract(
        self,
        frame: FrameRecord,
        crop_left_ratio: float,
        crop_right_ratio: float,
        crop_top_ratio: float,
        crop_bottom_ratio: float,
    ) -> FrameOcrResult:
        image = cv2.imread(
            str(frame.path)
        )

        if image is None:
            raise RuntimeError(
                (
                    "Could not read extracted frame: "
                    f"{frame.path}"
                )
            )

        height, width = image.shape[:2]

        x1 = max(
            0,
            min(
                width - 1,
                int(
                    width
                    * crop_left_ratio
                ),
            ),
        )

        x2 = max(
            x1 + 1,
            min(
                width,
                int(
                    width
                    * crop_right_ratio
                ),
            ),
        )

        y1 = max(
            0,
            min(
                height - 1,
                int(
                    height
                    * crop_top_ratio
                ),
            ),
        )

        y2 = max(
            y1 + 1,
            min(
                height,
                int(
                    height
                    * crop_bottom_ratio
                ),
            ),
        )

        cropped = image[
            y1:y2,
            x1:x2,
        ]

        if cropped.size == 0:
            raise RuntimeError(
                (
                    "Invalid OCR crop for frame "
                    f"{frame.path.name}: "
                    f"x=({x1},{x2}) "
                    f"y=({y1},{y2}) "
                    f"image={width}x{height}"
                )
            )

        processed, scale = (
            self._preprocess(cropped)
        )

        try:
            data = pytesseract.image_to_data(
                processed,
                lang=self.language,
                config="--oem 3 --psm 6",
                output_type=Output.DICT,
                timeout=120,
            )

        except RuntimeError as exc:
            raise RuntimeError(
                (
                    "Tesseract timed out for frame "
                    f"{frame.path.name}: {exc}"
                )
            ) from exc

        except (
            pytesseract.TesseractNotFoundError
        ) as exc:
            raise OcrInitializationError(
                (
                    "Tesseract executable is "
                    "unavailable. Check "
                    "TESSERACT_CMD."
                )
            ) from exc

        except pytesseract.TesseractError as exc:
            raise RuntimeError(
                (
                    "Tesseract failed for frame "
                    f"{frame.path.name}: {exc}"
                )
            ) from exc

        lines = self._build_lines(
            data,
            scale,
        )

        lines.sort(
            key=lambda line: (
                round(line.y1 / 6),
                line.x1,
            )
        )

        LOGGER.info(
            (
                "OCR frame=%s timestamp=%s "
                "lines=%s crop=%sx%s"
            ),
            frame.path.name,
            frame.timestamp,
            len(lines),
            cropped.shape[1],
            cropped.shape[0],
        )

        return FrameOcrResult(
            frame=frame,
            lines=lines,
            cropped_width=cropped.shape[1],
            cropped_height=cropped.shape[0],
        )

    @staticmethod
    def _preprocess(
        image: Any,
    ) -> tuple[Any, float]:
        """
        Enlarge browser text while preserving
        anti-aliased characters.
        """

        scale = 2.0

        enlarged = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

        gray = cv2.cvtColor(
            enlarged,
            cv2.COLOR_BGR2GRAY,
        )

        gray = cv2.bilateralFilter(
            gray,
            5,
            45,
            45,
        )

        return gray, scale

    def _build_lines(
        self,
        data: dict[str, list[Any]],
        scale: float,
    ) -> list[OcrLine]:
        groups: dict[
            tuple[int, int, int],
            list[dict[str, Any]],
        ] = defaultdict(list)

        count = len(
            data.get(
                "text",
                [],
            )
        )

        for index in range(count):
            texts = data.get(
                "text",
                [],
            )

            text = clean_text(
                str(texts[index])
            )

            confidence_values = data.get(
                "conf",
                [],
            )

            confidence = self._safe_float(
                (
                    confidence_values[index]
                    if index
                    < len(confidence_values)
                    else -1
                ),
                -1.0,
            )

            if (
                not text
                or confidence < 0
            ):
                continue

            normalized_confidence = (
                confidence / 100.0
            )

            if (
                normalized_confidence
                < self.min_confidence
            ):
                continue

            block = self._value_as_int(
                data,
                "block_num",
                index,
            )

            paragraph = self._value_as_int(
                data,
                "par_num",
                index,
            )

            line = self._value_as_int(
                data,
                "line_num",
                index,
            )

            left = self._value_as_float(
                data,
                "left",
                index,
            )

            top = self._value_as_float(
                data,
                "top",
                index,
            )

            word_width = self._value_as_float(
                data,
                "width",
                index,
            )

            word_height = self._value_as_float(
                data,
                "height",
                index,
            )

            groups[
                (
                    block,
                    paragraph,
                    line,
                )
            ].append(
                {
                    "text": text,
                    "confidence": (
                        normalized_confidence
                    ),
                    "x1": left / scale,
                    "y1": top / scale,
                    "x2": (
                        left + word_width
                    ) / scale,
                    "y2": (
                        top + word_height
                    ) / scale,
                }
            )

        result: list[OcrLine] = []

        for words in groups.values():
            words.sort(
                key=lambda item: item["x1"]
            )

            text = clean_text(
                " ".join(
                    item["text"]
                    for item in words
                )
            )

            if not text:
                continue

            result.append(
                OcrLine(
                    text=text,
                    confidence=(
                        sum(
                            item["confidence"]
                            for item in words
                        )
                        / len(words)
                    ),
                    x1=min(
                        item["x1"]
                        for item in words
                    ),
                    y1=min(
                        item["y1"]
                        for item in words
                    ),
                    x2=max(
                        item["x2"]
                        for item in words
                    ),
                    y2=max(
                        item["y2"]
                        for item in words
                    ),
                )
            )

        return result

    @staticmethod
    def _value_as_float(
        data: dict[str, list[Any]],
        key: str,
        index: int,
    ) -> float:
        values = data.get(
            key,
            [],
        )

        if index >= len(values):
            return 0.0

        return OcrService._safe_float(
            values[index]
        )

    @staticmethod
    def _value_as_int(
        data: dict[str, list[Any]],
        key: str,
        index: int,
    ) -> int:
        values = data.get(
            key,
            [],
        )

        if index >= len(values):
            return 0

        return OcrService._safe_int(
            values[index]
        )

    @staticmethod
    def _safe_float(
        value: Any,
        default: float = 0.0,
    ) -> float:
        try:
            return float(value)
        except (
            TypeError,
            ValueError,
        ):
            return default

    @staticmethod
    def _safe_int(
        value: Any,
    ) -> int:
        try:
            return int(
                float(value)
            )
        except (
            TypeError,
            ValueError,
        ):
            return 0