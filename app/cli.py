from __future__ import annotations

import argparse
import logging
import uuid
from pathlib import Path

from app.config import settings
from app.services.pipeline import ExtractionPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract LinkedIn job posts from a "
            "manually recorded video."
        )
    )

    parser.add_argument(
        "video",
        type=Path,
        help="Input video path",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Output Excel path",
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=settings.sample_interval_seconds,
    )

    parser.add_argument(
        "--crop-left",
        type=float,
        default=settings.crop_left_ratio,
    )

    parser.add_argument(
        "--crop-right",
        type=float,
        default=settings.crop_right_ratio,
    )

    parser.add_argument(
        "--crop-top",
        type=float,
        default=settings.crop_top_ratio,
    )

    parser.add_argument(
        "--crop-bottom",
        type=float,
        default=settings.crop_bottom_ratio,
    )

    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    args = build_parser().parse_args()

    video_path = args.video.resolve()

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    run_dir = (
        settings.runs_dir
        / uuid.uuid4().hex
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    output_path = (
        args.output.resolve()
        if args.output
        else settings.outputs_dir
        / f"{video_path.stem}_jobs.xlsx"
    )

    result = ExtractionPipeline(
        settings
    ).run(
        video_path,
        run_dir,
        output_path,
        args.interval,
        args.crop_left,
        args.crop_right,
        args.crop_top,
        args.crop_bottom,
    )

    print(
        f"Output: {result.output_path}"
    )
    print(
        f"Jobs detected: {len(result.jobs)}"
    )
    print(
        f"Frames retained: {result.frame_count}"
    )
    print(
        f"Errors: {len(result.errors)}"
    )


if __name__ == "__main__":
    main()