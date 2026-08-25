from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class FrameRecord:
    frame_index: int
    source_frame_number: int
    timestamp_seconds: float
    path: Path
    width: int
    height: int
    blur_score: float

    @property
    def timestamp(self) -> str:
        total_seconds = max(0, int(round(self.timestamp_seconds)))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


@dataclass(slots=True)
class OcrLine:
    text: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)


@dataclass(slots=True)
class FrameOcrResult:
    frame: FrameRecord
    lines: list[OcrLine]
    cropped_width: int
    cropped_height: int

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


@dataclass(slots=True)
class EvidenceItem:
    job_id: str
    field: str
    value: str
    timestamp: str
    frame_file: str
    confidence: float


@dataclass(slots=True)
class JobRecord:
    job_id: str
    job_title: str = ""
    company_name: str = ""
    location: str = ""
    workplace_type: str = ""
    employment_type: str = ""
    experience: str = ""
    skills: list[str] = field(default_factory=list)
    posted_date: str = ""
    applicant_count: str = ""
    contact_name: str = ""
    contact_emails: list[str] = field(default_factory=list)
    company_website: str = ""
    application_url: str = ""
    linkedin_job_url: str = ""
    job_description: str = ""
    source_video: str = ""
    first_timestamp: str = ""
    last_timestamp: str = ""
    confidence: float = 0.0
    review_status: str = "Needs Review"
    evidence: list[EvidenceItem] = field(default_factory=list)


@dataclass(slots=True)
class ExtractionError:
    stage: str
    timestamp: str
    frame_file: str
    message: str
    raw_text: str = ""


@dataclass(slots=True)
class ExtractionResult:
    jobs: list[JobRecord]
    errors: list[ExtractionError]
    output_path: Path
    frame_count: int
