from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean

from openai import OpenAI
from pydantic import BaseModel

from app.models import EvidenceItem, FrameOcrResult, JobRecord
from app.services.text_utils import clean_text, normalized_key


LOGGER = logging.getLogger(__name__)

EMAIL_RE = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    re.I,
)

URL_RE = re.compile(
    r"\b(?:https?://|www\.)[^\s<>()]+",
    re.I,
)


class ExtractedField(BaseModel):
    value: str
    evidence: str


class ExtractedJob(BaseModel):
    is_job_post: bool
    job_title: ExtractedField
    company_name: ExtractedField
    location: ExtractedField
    workplace_type: ExtractedField
    employment_type: ExtractedField
    experience: ExtractedField
    post_contact_name: ExtractedField
    post_contact_email: ExtractedField
    application_url: ExtractedField
    posted_date: ExtractedField
    job_description_lines: list[str]


class ExtractedJobBatch(BaseModel):
    jobs: list[ExtractedJob]


@dataclass(slots=True)
class SourceLine:
    text: str
    confidence: float
    timestamp: str
    frame_file: str
    frame_index: int
    order: int


SYSTEM_INSTRUCTIONS = """
You extract structured job-post data from OCR text captured while scrolling LinkedIn.

Important rules:
1. Extract only genuine hiring/job posts. Ignore comments, reactions, navigation, profile menus,
   educational posts, unrelated LinkedIn UI, and social-media noise.
2. A job title can be any legitimate role, including generic titles such as "Quality Lead",
   "QA Lead", "Head of Quality", "Finance Manager", etc. Do NOT require D365 or any product keyword.
3. Do not invent or infer missing facts from world knowledge. Use only the supplied OCR text.
4. If a field is not clearly supported by the OCR text, return value="" and evidence="".
5. For each non-empty scalar field, evidence must be copied from the OCR text and must directly
   support that value. Evidence should be the smallest useful phrase or line.
6. company_name means the employer/hiring organization, not the LinkedIn author/recruiter name.
7. post_contact_name means the author/recruiter/contact person only when clearly connected to that
   job post. Do not treat random commenters as the post contact.
8. job_description_lines must contain only job-related content copied from OCR text. Exclude
   reactions, counts, comments, hashtags-only lines, navigation, and unrelated UI. Do not rewrite
   or summarize the description; preserve the source wording as much as possible.
9. If the same post appears repeatedly across adjacent frames because the page is scrolling,
   return it only once.
10. If multiple different job posts are visible in the OCR batch, return each one separately.
11. workplace_type should be one of the source-supported values such as Remote, Hybrid, On-site,
    or empty. employment_type should be source-supported such as Full-time, Part-time, Contract,
    Temporary, Internship, Freelance, or empty.
12. is_job_post must be false for anything that is not clearly a job/hiring post.
""".strip()


class LLMJobExtractor:
    """
    OpenAI-based semantic mapper for LinkedIn OCR.

    OCR remains deterministic. This class only replaces the fragile regex-heavy
    semantic mapping from OCR text -> JobRecord.

    Normal videos are sent in one model call. Very large recordings are split
    into bounded batches with small overlap, then deduplicated afterwards.
    """

    def __init__(
        self,
        skills_file: Path,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
        max_batch_chars: int | None = None,
        overlap_frames: int = 2,
    ) -> None:
        self.api_key = (
            api_key
            or os.getenv("OPENAI_API_KEY", "")
        ).strip()

        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY is missing. Add it to your .env file."
            )

        self.model = (
            model
            or os.getenv("OPENAI_JOB_MODEL", "gpt-5-mini")
        ).strip()

        timeout = (
            float(timeout_seconds)
            if timeout_seconds is not None
            else self._env_float("OPENAI_TIMEOUT_SECONDS", 180.0)
        )

        self.timeout_seconds = max(30.0, float(timeout))

        # Smaller OCR payloads reduce latency and make timeout recovery safer.
        self.max_batch_chars = max(
            8000,
            int(
                max_batch_chars
                if max_batch_chars is not None
                else self._env_int("OPENAI_JOB_MAX_BATCH_CHARS", 28000)
            ),
        )

        self.overlap_frames = max(0, int(overlap_frames))

        self.max_retries = max(
            0,
            self._env_int("OPENAI_MAX_RETRIES", 1),
        )

        self.max_recovery_depth = max(
            0,
            self._env_int("OPENAI_TIMEOUT_RECOVERY_DEPTH", 2),
        )

        self.skill_catalog = self._load_skills(Path(skills_file))

        self.client = OpenAI(
            api_key=self.api_key,
            timeout=self.timeout_seconds,
            max_retries=self.max_retries,
        )

        # Successful response usage comes from the Responses API.
        # Client calls/failures are tracked separately because timed-out calls
        # do not return a response object containing token usage.
        self._client_call_count = 0
        self._failed_request_count = 0
        self._usage_request_count = 0
        self._usage_input_tokens = 0
        self._usage_cached_input_tokens = 0
        self._usage_output_tokens = 0
        self._usage_reasoning_tokens = 0
        self._usage_total_tokens = 0
        self._usage_response_model = self.model

        LOGGER.info(
            (
                "Initialized OpenAI LinkedIn extractor model=%s "
                "max_batch_chars=%s timeout_seconds=%s max_retries=%s "
                "timeout_recovery_depth=%s"
            ),
            self.model,
            self.max_batch_chars,
            self.timeout_seconds,
            self.max_retries,
            self.max_recovery_depth,
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "LLMJobExtractor":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def extract(
        self,
        frames: list[FrameOcrResult],
        source_video: str,
    ) -> list[JobRecord]:
        if not frames:
            return []

        all_source_lines = self._source_lines(frames)
        if not all_source_lines:
            return []

        batches = self._build_batches(frames)
        all_jobs: list[JobRecord] = []

        for batch_index, batch_frames in enumerate(batches, start=1):
            self._process_batch_with_recovery(
                batch_frames=batch_frames,
                source_video=source_video,
                all_jobs=all_jobs,
                batch_label=f"{batch_index}/{len(batches)}",
                recovery_depth=0,
            )

        jobs = self._deduplicate(all_jobs)

        # Re-number after cross-batch dedupe so IDs are stable within the run.
        for index, job in enumerate(jobs, start=1):
            job.job_id = f"JOB-{index:04d}"
            for evidence in job.evidence:
                evidence.job_id = job.job_id

        LOGGER.info(
            (
                "OpenAI extraction completed source=%s batches=%s jobs=%s "
                "client_calls=%s successful_responses=%s failed_requests=%s"
            ),
            source_video,
            len(batches),
            len(jobs),
            self._client_call_count,
            self._usage_request_count,
            self._failed_request_count,
        )

        return jobs

    def _process_batch_with_recovery(
        self,
        batch_frames: list[FrameOcrResult],
        source_video: str,
        all_jobs: list[JobRecord],
        batch_label: str,
        recovery_depth: int,
    ) -> None:
        """Extract one batch and split it into smaller pieces if it times out."""
        if not batch_frames:
            return

        batch_lines = self._source_lines(batch_frames)
        transcript = self._build_transcript(batch_frames)

        if not transcript.strip():
            return

        LOGGER.info(
            (
                "OpenAI extraction batch=%s recovery_depth=%s "
                "frames=%s chars=%s model=%s"
            ),
            batch_label,
            recovery_depth,
            len(batch_frames),
            len(transcript),
            self.model,
        )

        try:
            parsed = self._extract_structured(transcript)

        except Exception as exc:
            # The OpenAI SDK raises APITimeoutError when the read operation
            # exceeds the configured timeout. Checking by class name keeps this
            # compatible across SDK import layouts.
            if exc.__class__.__name__ != "APITimeoutError":
                raise

            self._failed_request_count += 1

            can_split = (
                recovery_depth < self.max_recovery_depth
                and len(batch_frames) >= 4
            )

            if can_split:
                split_batches = self._split_batch_for_timeout(batch_frames)

                if len(split_batches) == 2:
                    LOGGER.warning(
                        (
                            "OpenAI timeout batch=%s frames=%s chars=%s. "
                            "Splitting into smaller retry batches depth=%s/%s"
                        ),
                        batch_label,
                        len(batch_frames),
                        len(transcript),
                        recovery_depth + 1,
                        self.max_recovery_depth,
                    )

                    for split_index, split_frames in enumerate(
                        split_batches,
                        start=1,
                    ):
                        self._process_batch_with_recovery(
                            batch_frames=split_frames,
                            source_video=source_video,
                            all_jobs=all_jobs,
                            batch_label=f"{batch_label}.{split_index}",
                            recovery_depth=recovery_depth + 1,
                        )

                    return

            LOGGER.error(
                (
                    "OpenAI timeout could not be recovered batch=%s "
                    "frames=%s chars=%s depth=%s error=%s"
                ),
                batch_label,
                len(batch_frames),
                len(transcript),
                recovery_depth,
                exc,
            )
            return

        for raw_job in parsed.jobs:
            if not raw_job.is_job_post:
                continue

            job = self._to_job_record(
                raw_job=raw_job,
                source_lines=batch_lines,
                source_video=source_video,
                provisional_index=len(all_jobs) + 1,
            )

            if job is not None:
                all_jobs.append(job)

    def _split_batch_for_timeout(
        self,
        frames: list[FrameOcrResult],
    ) -> list[list[FrameOcrResult]]:
        """Split a timed-out batch in half with a small boundary overlap."""
        frame_count = len(frames)

        if frame_count < 4:
            return [frames]

        midpoint = frame_count // 2

        overlap = min(
            self.overlap_frames,
            max(0, midpoint - 1),
            max(0, frame_count - midpoint - 1),
        )

        left_end = midpoint + overlap
        right_start = midpoint - overlap

        left = frames[:left_end]
        right = frames[right_start:]

        # Defensive fallback: never recurse with a child equal to the parent.
        if (
            not left
            or not right
            or len(left) >= frame_count
            or len(right) >= frame_count
        ):
            left = frames[:midpoint]
            right = frames[midpoint:]

        if not left or not right:
            return [frames]

        return [left, right]

    def _extract_structured(self, transcript: str) -> ExtractedJobBatch:
        self._client_call_count += 1

        response = self.client.responses.parse(
            model=self.model,
            instructions=SYSTEM_INSTRUCTIONS,
            input=(
                "Extract all genuine LinkedIn job posts from this OCR transcript.\n\n"
                "OCR TRANSCRIPT:\n"
                f"{transcript}"
            ),
            text_format=ExtractedJobBatch,
            store=False,
        )

        # Capture actual token usage returned by OpenAI for this request.
        self._record_response_usage(response)

        # Avoid depending on a convenience property that may differ between SDK
        # versions. Read the parsed structured output directly from the message.
        for output in response.output:
            if getattr(output, "type", "") != "message":
                continue

            for content in getattr(output, "content", []):
                parsed = getattr(content, "parsed", None)
                if isinstance(parsed, ExtractedJobBatch):
                    return parsed

        raise RuntimeError(
            "OpenAI returned no parsed structured LinkedIn extraction result."
        )

    def _record_response_usage(self, response) -> None:
        """Accumulate actual token usage returned by one Responses API call."""
        self._usage_request_count += 1

        response_model = str(
            getattr(response, "model", "") or self.model
        ).strip()
        if response_model:
            self._usage_response_model = response_model

        usage = getattr(response, "usage", None)
        if usage is None:
            LOGGER.warning(
                "OpenAI response did not include token usage response_id=%s",
                getattr(response, "id", ""),
            )
            return

        input_tokens = self._safe_int(getattr(usage, "input_tokens", 0))
        output_tokens = self._safe_int(getattr(usage, "output_tokens", 0))
        total_tokens = self._safe_int(getattr(usage, "total_tokens", 0))

        input_details = getattr(usage, "input_tokens_details", None)
        cached_input_tokens = self._safe_int(
            getattr(input_details, "cached_tokens", 0)
            if input_details is not None else 0
        )

        output_details = getattr(usage, "output_tokens_details", None)
        reasoning_tokens = self._safe_int(
            getattr(output_details, "reasoning_tokens", 0)
            if output_details is not None else 0
        )

        self._usage_input_tokens += input_tokens
        self._usage_cached_input_tokens += cached_input_tokens
        self._usage_output_tokens += output_tokens
        self._usage_reasoning_tokens += reasoning_tokens
        self._usage_total_tokens += (
            total_tokens if total_tokens > 0 else input_tokens + output_tokens
        )

    def get_usage_summary(self) -> dict[str, int | str]:
        """Return accumulated OpenAI usage for the current recording."""
        return {
            "client_call_count": self._client_call_count,
            "request_count": self._usage_request_count,
            "failed_request_count": self._failed_request_count,
            "input_tokens": self._usage_input_tokens,
            "cached_input_tokens": self._usage_cached_input_tokens,
            "output_tokens": self._usage_output_tokens,
            "reasoning_tokens": self._usage_reasoning_tokens,
            "total_tokens": self._usage_total_tokens,
            "model": self._usage_response_model or self.model,
        }

    @staticmethod
    def _safe_int(value) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    def _to_job_record(
        self,
        raw_job: ExtractedJob,
        source_lines: list[SourceLine],
        source_video: str,
        provisional_index: int,
    ) -> JobRecord | None:
        if not source_lines:
            return None

        matched_lines: list[SourceLine] = []
        evidence: list[EvidenceItem] = []
        job_id = f"JOB-{provisional_index:04d}"

        def validated_field(
            field_name: str,
            field: ExtractedField,
        ) -> str:
            value = clean_text(field.value)
            evidence_text = clean_text(field.evidence)

            if not value:
                return ""

            source_line = self._find_evidence_line(
                evidence_text=evidence_text,
                source_lines=source_lines,
            )

            if source_line is None:
                LOGGER.warning(
                    "Rejecting unsupported LLM field field=%s value=%r evidence=%r",
                    field_name,
                    value,
                    evidence_text,
                )
                return ""

            if not self._value_supported_by_evidence(value, evidence_text):
                LOGGER.warning(
                    "Rejecting LLM field not supported by its evidence field=%s value=%r evidence=%r",
                    field_name,
                    value,
                    evidence_text,
                )
                return ""

            matched_lines.append(source_line)
            evidence.append(
                EvidenceItem(
                    job_id=job_id,
                    field=field_name,
                    value=value,
                    timestamp=source_line.timestamp,
                    frame_file=source_line.frame_file,
                    confidence=round(source_line.confidence, 3),
                )
            )

            return value

        job_title = validated_field(
            "Job Title",
            raw_job.job_title,
        )

        # Job title is mandatory. Do not require any hard-coded role keyword.
        if not self._valid_job_title(job_title):
            return None

        company = validated_field(
            "Company Name",
            raw_job.company_name,
        )
        location = validated_field(
            "Location",
            raw_job.location,
        )
        workplace = validated_field(
            "Workplace Type",
            raw_job.workplace_type,
        )
        employment = validated_field(
            "Employment Type",
            raw_job.employment_type,
        )
        experience = validated_field(
            "Experience",
            raw_job.experience,
        )
        contact_name = validated_field(
            "Contact Name",
            raw_job.post_contact_name,
        )

        contact_email = self._validated_email_field(
            raw_job.post_contact_email,
            source_lines,
        )
        if contact_email:
            email_line = self._find_evidence_line(
                raw_job.post_contact_email.evidence,
                source_lines,
            )
            if email_line is not None:
                matched_lines.append(email_line)
                evidence.append(
                    EvidenceItem(
                        job_id=job_id,
                        field="Contact Email",
                        value=contact_email,
                        timestamp=email_line.timestamp,
                        frame_file=email_line.frame_file,
                        confidence=round(email_line.confidence, 3),
                    )
                )

        application_url = self._validated_url_field(
            raw_job.application_url,
            source_lines,
        )

        posted_date = validated_field(
            "Posted Date",
            raw_job.posted_date,
        )

        description, description_matches = self._validated_description(
            raw_job.job_description_lines,
            source_lines,
        )
        matched_lines.extend(description_matches)

        if description and description_matches:
            first_description_line = min(
                description_matches,
                key=lambda item: item.order,
            )
            evidence.append(
                EvidenceItem(
                    job_id=job_id,
                    field="Job Description",
                    value=description,
                    timestamp=first_description_line.timestamp,
                    frame_file=first_description_line.frame_file,
                    confidence=round(
                        mean(line.confidence for line in description_matches),
                        3,
                    ),
                )
            )

        # A valid title plus some job-specific body/support is required.
        supporting_fields = sum(
            bool(value)
            for value in (
                company,
                location,
                workplace,
                employment,
                experience,
                contact_email,
                description,
            )
        )

        if supporting_fields < 1:
            return None

        if matched_lines:
            first_match = min(matched_lines, key=lambda item: item.order)
            last_match = max(matched_lines, key=lambda item: item.order)
            average_ocr_confidence = mean(
                line.confidence for line in matched_lines
            )
        else:
            first_match = source_lines[0]
            last_match = source_lines[-1]
            average_ocr_confidence = mean(
                line.confidence for line in source_lines
            )

        completeness = (
            0.35
            + 0.20 * bool(company)
            + 0.20 * bool(description)
            + 0.08 * bool(contact_email)
            + 0.07 * bool(location)
            + 0.05 * bool(workplace)
            + 0.05 * bool(experience)
        )

        confidence = round(
            min(
                1.0,
                0.70 * average_ocr_confidence
                + 0.30 * completeness,
            ),
            3,
        )

        skills = self._extract_skills(
            f"{job_title}\n{description}"
        )

        return JobRecord(
            job_id=job_id,
            job_title=job_title,
            company_name=company,
            location=location,
            workplace_type=workplace,
            employment_type=employment,
            experience=experience,
            skills=skills,
            posted_date=posted_date,
            contact_name=contact_name,
            contact_emails=(
                [contact_email]
                if contact_email
                else []
            ),
            application_url=application_url,
            job_description=description,
            source_video=source_video,
            first_timestamp=first_match.timestamp,
            last_timestamp=last_match.timestamp,
            confidence=confidence,
            review_status=(
                "Verified"
                if (
                    confidence >= 0.68
                    and len(description) >= 40
                )
                else "Needs Review"
            ),
            evidence=evidence,
        )

    def _source_lines(
        self,
        frames: list[FrameOcrResult],
    ) -> list[SourceLine]:
        lines: list[SourceLine] = []
        order = 0

        for frame_index, frame in enumerate(frames):
            for line in frame.lines:
                text = clean_text(line.text)
                if len(text) < 2:
                    continue

                lines.append(
                    SourceLine(
                        text=text,
                        confidence=float(line.confidence),
                        timestamp=frame.frame.timestamp,
                        frame_file=frame.frame.path.name,
                        frame_index=frame_index,
                        order=order,
                    )
                )
                order += 1

        return lines

    def _build_transcript(
        self,
        frames: list[FrameOcrResult],
    ) -> str:
        output: list[str] = []

        # Suppress only exact/near-exact OCR repetition from immediately adjacent
        # frames. Reappearance later in the recording is preserved because it may
        # belong to a different post.
        last_seen_frame: dict[str, int] = {}

        for frame_index, frame in enumerate(frames):
            frame_lines: list[str] = []

            for line in frame.lines:
                text = clean_text(line.text)
                if len(text) < 2:
                    continue

                key = normalized_key(text)
                previous_frame = last_seen_frame.get(key)

                if (
                    key
                    and previous_frame is not None
                    and frame_index - previous_frame <= 1
                ):
                    continue

                if key:
                    last_seen_frame[key] = frame_index

                frame_lines.append(
                    f"- ({float(line.confidence):.2f}) {text}"
                )

            if not frame_lines:
                continue

            output.append(
                f"[FRAME {frame_index:04d} | {frame.frame.timestamp} | {frame.frame.path.name}]"
            )
            output.extend(frame_lines)
            output.append("")

        return "\n".join(output).strip()

    def _build_batches(
        self,
        frames: list[FrameOcrResult],
    ) -> list[list[FrameOcrResult]]:
        if not frames:
            return []

        # Fast path: normal recording -> exactly one OpenAI request.
        whole_transcript = self._build_transcript(frames)
        if len(whole_transcript) <= self.max_batch_chars:
            return [frames]

        batches: list[list[FrameOcrResult]] = []
        current: list[FrameOcrResult] = []

        for frame in frames:
            candidate = [*current, frame]
            candidate_text = self._build_transcript(candidate)

            if current and len(candidate_text) > self.max_batch_chars:
                batches.append(current)

                overlap = (
                    current[-self.overlap_frames :]
                    if self.overlap_frames
                    else []
                )
                current = [*overlap, frame]
            else:
                current = candidate

        if current:
            batches.append(current)

        return batches

    def _find_evidence_line(
        self,
        evidence_text: str,
        source_lines: list[SourceLine],
    ) -> SourceLine | None:
        evidence = clean_text(evidence_text)
        if not evidence:
            return None

        evidence_key = normalized_key(evidence)
        if not evidence_key:
            return None

        # Strong exact/substring match first.
        for line in source_lines:
            line_key = normalized_key(line.text)
            if (
                evidence_key == line_key
                or evidence_key in line_key
                or line_key in evidence_key
            ):
                return line

        # OCR may contain a small typo. Permit a conservative fuzzy fallback.
        best_line: SourceLine | None = None
        best_score = 0.0

        for line in source_lines:
            line_key = normalized_key(line.text)
            if not line_key:
                continue

            score = SequenceMatcher(
                None,
                evidence_key,
                line_key,
            ).ratio()

            if score > best_score:
                best_score = score
                best_line = line

        return best_line if best_score >= 0.76 else None

    @staticmethod
    def _value_supported_by_evidence(
        value: str,
        evidence: str,
    ) -> bool:
        value_key = normalized_key(value)
        evidence_key = normalized_key(evidence)

        if not value_key or not evidence_key:
            return False

        if value_key in evidence_key or evidence_key in value_key:
            return True

        value_tokens = {
            token
            for token in value_key.split()
            if len(token) >= 2
        }
        evidence_tokens = set(evidence_key.split())

        if value_tokens:
            token_coverage = (
                len(value_tokens & evidence_tokens)
                / len(value_tokens)
            )
            if token_coverage >= 0.60:
                return True

        return (
            SequenceMatcher(
                None,
                value_key,
                evidence_key,
            ).ratio()
            >= 0.58
        )

    def _validated_email_field(
        self,
        field: ExtractedField,
        source_lines: list[SourceLine],
    ) -> str:
        value = clean_text(field.value).rstrip(".,;:])")
        evidence_text = clean_text(field.evidence)

        if not value:
            return ""

        source_emails = {
            email.casefold(): email
            for line in source_lines
            for email in EMAIL_RE.findall(line.text)
        }

        # Prefer the exact email selected by the model if it exists in OCR.
        if value.casefold() in source_emails:
            return source_emails[value.casefold()]

        # Otherwise use an email present in the supplied evidence line.
        for email in EMAIL_RE.findall(evidence_text):
            if email.casefold() in source_emails:
                return source_emails[email.casefold()]

        return ""

    def _validated_url_field(
        self,
        field: ExtractedField,
        source_lines: list[SourceLine],
    ) -> str:
        value = clean_text(field.value).rstrip(".,;:])")
        if not value:
            return ""

        source_urls = {
            url.rstrip(".,;:])")
            for line in source_lines
            for url in URL_RE.findall(line.text)
        }

        value_key = value.casefold()
        for url in source_urls:
            if url.casefold() == value_key:
                return url

        return ""

    def _validated_description(
        self,
        raw_lines: list[str],
        source_lines: list[SourceLine],
    ) -> tuple[str, list[SourceLine]]:
        if not raw_lines:
            return "", []

        plain_source_key = normalized_key(
            " ".join(line.text for line in source_lines)
        )

        accepted: list[str] = []
        matches: list[SourceLine] = []
        seen: set[str] = set()

        for raw_line in raw_lines:
            text = clean_text(raw_line)
            key = normalized_key(text)

            if not key or key in seen:
                continue

            source_line = self._find_evidence_line(
                text,
                source_lines,
            )

            # A model may combine two adjacent OCR lines into one line. Allow it
            # only if that combined wording still occurs in the normalized OCR.
            if source_line is None and key not in plain_source_key:
                LOGGER.debug(
                    "Dropping unsupported description line=%r",
                    text,
                )
                continue

            accepted.append(text)
            seen.add(key)

            if source_line is not None:
                matches.append(source_line)

        return "\n".join(accepted), self._unique_source_lines(matches)

    @staticmethod
    def _valid_job_title(title: str) -> bool:
        title = clean_text(title)
        if len(title) < 3 or len(title) > 160:
            return False

        key = normalized_key(title)
        if not key:
            return False

        invalid = {
            "linkedin",
            "jobs",
            "job",
            "hiring",
            "we are hiring",
            "we re hiring",
            "view job",
            "comment",
            "repost",
            "like",
            "follow",
        }
        return key not in invalid

    def _extract_skills(self, text: str) -> list[str]:
        if not text or not self.skill_catalog:
            return []

        return [
            skill
            for skill in self.skill_catalog
            if re.search(
                rf"(?<![A-Za-z0-9]){re.escape(skill)}(?![A-Za-z0-9])",
                text,
                re.I,
            )
        ]

    def _deduplicate(
        self,
        jobs: list[JobRecord],
    ) -> list[JobRecord]:
        result: list[JobRecord] = []

        for job in jobs:
            duplicate = next(
                (
                    existing
                    for existing in result
                    if self._same_job(existing, job)
                ),
                None,
            )

            if duplicate is None:
                result.append(job)
                continue

            # Keep the richer version if overlapping LLM batches returned the
            # same post with different completeness.
            if self._job_richness(job) > self._job_richness(duplicate):
                result[result.index(duplicate)] = job

        return result

    @staticmethod
    def _same_job(left: JobRecord, right: JobRecord) -> bool:
        left_title = normalized_key(left.job_title)
        right_title = normalized_key(right.job_title)

        if not left_title or not right_title:
            return False

        title_score = SequenceMatcher(
            None,
            left_title,
            right_title,
        ).ratio()

        if title_score < 0.90:
            return False

        left_company = normalized_key(left.company_name)
        right_company = normalized_key(right.company_name)

        if left_company and right_company:
            company_score = SequenceMatcher(
                None,
                left_company,
                right_company,
            ).ratio()
            if company_score < 0.90:
                return False

        left_location = normalized_key(left.location)
        right_location = normalized_key(right.location)
        if left_location and right_location:
            location_score = SequenceMatcher(
                None,
                left_location,
                right_location,
            ).ratio()
            if location_score < 0.88:
                return False

        left_emails = {
            email.casefold()
            for email in (left.contact_emails or [])
        }
        right_emails = {
            email.casefold()
            for email in (right.contact_emails or [])
        }

        if left_emails and right_emails and left_emails & right_emails:
            return True

        if left_company and right_company:
            return True

        left_desc = normalized_key(left.job_description)
        right_desc = normalized_key(right.job_description)

        if left_desc and right_desc:
            return (
                SequenceMatcher(
                    None,
                    left_desc[:1200],
                    right_desc[:1200],
                ).ratio()
                >= 0.82
            )

        return False

    @staticmethod
    def _job_richness(job: JobRecord) -> tuple[int, int, float]:
        populated = sum(
            bool(value)
            for value in (
                job.company_name,
                job.location,
                job.workplace_type,
                job.employment_type,
                job.experience,
                job.contact_name,
                job.contact_emails,
                job.job_description,
            )
        )
        return (
            populated,
            len(job.job_description or ""),
            float(job.confidence or 0.0),
        )

    @staticmethod
    def _unique_source_lines(
        lines: list[SourceLine],
    ) -> list[SourceLine]:
        result: list[SourceLine] = []
        seen: set[tuple[str, str, str]] = set()

        for line in lines:
            key = (
                line.frame_file,
                line.timestamp,
                normalized_key(line.text),
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(line)

        return result

    @staticmethod
    def _load_skills(path: Path) -> list[str]:
        if not path.exists():
            return []

        values = [
            clean_text(line)
            for line in path.read_text(
                encoding="utf-8"
            ).splitlines()
        ]

        return sorted(
            {
                value
                for value in values
                if value and not value.startswith("#")
            },
            key=lambda value: (-len(value), value.casefold()),
        )

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        raw = os.getenv(name, "").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        raw = os.getenv(name, "").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            return default