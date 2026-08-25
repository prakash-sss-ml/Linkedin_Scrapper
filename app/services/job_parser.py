from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean

from app.models import (
    EvidenceItem,
    FrameOcrResult,
    JobRecord,
)
from app.services.text_utils import (
    clean_text,
    normalized_key,
)


# ---------------------------------------------------------------------------
# General extraction patterns
# ---------------------------------------------------------------------------

EMAIL_RE = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    re.I,
)

URL_RE = re.compile(
    r"\b(?:https?://|www\.)[^\s<>()]+",
    re.I,
)

EXPERIENCE_RE = re.compile(
    r"\b(?:minimum\s+|at\s+least\s+)?"
    r"\d+(?:\.\d+)?\s*"
    r"(?:\+|[-–]|\bto\b)?\s*"
    r"(?:\d+(?:\.\d+)?\s*)?"
    r"years?(?:\s+of\s+experience)?\b",
    re.I,
)

POSTED_RE = re.compile(
    r"\b(?:"
    r"\d+\s*(?:m|h|d|w|mo)"
    r"|"
    r"\d+\s+(?:minutes?|hours?|days?|weeks?|months?)"
    r"\s+ago"
    r")\b",
    re.I,
)


# ---------------------------------------------------------------------------
# Job-title patterns
# ---------------------------------------------------------------------------

ROLE_HINTS = (
    "engineer",
    "developer",
    "consultant",
    "architect",
    "specialist",
    "analyst",
    "manager",
    "lead",
    "administrator",
    "functional",
    "technical",
    "migration",
)

TITLE_CONTEXT_HINTS = ROLE_HINTS + (
    "dynamics",
    "d365",
    "power platform",
    "finance",
    "supply chain",
    "crm",
    "erp",
    "revenue recognition",
)

TITLE_PREFIX_RE = re.compile(
    r"^(?:"
    r"#*\s*we(?:'|’)?re\s+hiring"
    r"|"
    r"#*\s*we\s+are\s+hiring"
    r"|"
    r"#*\s*now\s+hiring"
    r"|"
    r"#*\s*job\s*title"
    r"|"
    r"#*\s*role"
    r"|"
    r"#*\s*position"
    r"|"
    r"#*\s*opening"
    r"|"
    r"#*\s*hiring"
    r")"
    r"\s*[|:\-–—]*\s*",
    re.I,
)


# ---------------------------------------------------------------------------
# Job-description formatting patterns
# ---------------------------------------------------------------------------

DESCRIPTION_SECTION_RE = re.compile(
    r"^#*\s*("
    r"job\s+description"
    r"|key\s+requirements"
    r"|required\s+skills"
    r"|skills\s+required"
    r"|responsibilities"
    r"|qualifications"
    r"|requirements"
    r"|preferred\s+qualifications"
    r"|what\s+you(?:'|’)?ll\s+do"
    r"|what\s+you\s+will\s+do"
    r")\s*:?\s*(.*)$",
    re.I,
)

DESCRIPTION_SECTION_NAMES = {
    "job description": "Job Description",
    "key requirements": "Key Requirements",
    "required skills": "Required Skills",
    "skills required": "Required Skills",
    "responsibilities": "Responsibilities",
    "qualifications": "Qualifications",
    "requirements": "Requirements",
    "preferred qualifications": "Preferred Qualifications",
    "what you'll do": "Responsibilities",
    "what you’ll do": "Responsibilities",
    "what you will do": "Responsibilities",
}

DESCRIPTION_STOP_RE = re.compile(
    r"^(?:"
    r"#*\s*interested\b"
    r"|apply\s+now\b"
    r"|apply\b"
    r"|share\s+your\s+updated\s+resume\b"
    r"|share\s+your\s+resume\b"
    r"|send\s+(?:your|the)\s+resume\b"
    r"|candidates?\s+share\s+resume\b"
    r"|please\s+(?:send|share)\b"
    r"|phone\s*(?:no|number)?\s*[:\-]"
    r"|contact\s*[:\-]"
    r"|email\s*[:\-]"
    r"|reach\s+out\b"
    r"|#(?:hiring|jobs?|recruitment)\b"
    r")",
    re.I,
)

JOB_METADATA_RE = re.compile(
    r"^(?:"
    r"job\s+title"
    r"|position"
    r"|role"
    r"|experience"
    r"|experience\s+level"
    r"|time\s*zone"
    r"|work\s+mode"
    r"|workplace\s+type"
    r"|work\s+location"
    r"|location"
    r"|employment\s+type"
    r"|type"
    r"|shift"
    r"|budget"
    r"|duration"
    r"|notice\s+period"
    r")\s*[:\-–—]",
    re.I,
)

DESCRIPTION_BULLET_RE = re.compile(
    r"^\s*(?:"
    r"[-–—•●▪◦✓✔►]"
    r"|\d+[.)]"
    r")\s*"
)

HASHTAG_LIST_RE = re.compile(
    r"^(?:#[A-Za-z0-9_]+\s*){2,}$"
)

SOCIAL_NOISE_RE = re.compile(
    r"^[\s\dO°©®+./|,_\-–—%]*$",
    re.I,
)

HASHTAG_TOKEN_RE = re.compile(
    r"#[A-Za-z0-9_]+"
)

COMMENT_TITLE_RE = re.compile(
    r"\b(?:"
    r"comment"
    r"|repost"
    r"|like"
    r"|reply"
    r"|view\s+profile"
    r"|certified\s+d365"
    r"|microsoft\s+certified"
    r"|financial\s+dimensions"
    r"|what\s+they\s+are"
    r"|teams\s+go\s+wrong"
    r")\b",
    re.I,
)

JOB_CONTENT_SIGNAL_RE = re.compile(
    r"\b(?:"
    r"job\s+description"
    r"|key\s+requirements"
    r"|required\s+skills"
    r"|responsibilities"
    r"|qualifications"
    r"|experience"
    r"|years"
    r"|apply"
    r"|resume"
    r"|hiring"
    r"|work\s+mode"
    r"|location"
    r"|contract"
    r"|full[ -]?time"
    r"|part[ -]?time"
    r")\b",
    re.I,
)

INVALID_COMPANY_NAMES = {
    "the",
    "and",
    "more",
    "linkedin",
    "remote",
    "hiring",
    "jobs",
    "job",
    "company",
    "consultant",
}


# ---------------------------------------------------------------------------
# LinkedIn hashtag/comment cleanup
# ---------------------------------------------------------------------------

GENERIC_HIRING_HASHTAGS = {
    "hiring",
    "hiringnow",
    "wearehiring",
    "nowhiring",
    "job",
    "jobs",
    "jobopening",
    "jobopenings",
    "career",
    "careers",
    "opportunity",
    "opportunities",
    "recruitment",
    "recruiting",
    "urgentopening",
    "urgentrequirement",
    "immediatehiring",
    "openposition",
    "openings",
}

COMMENT_PHRASES = (
    "i am interested",
    "i'm interested",
    "interested in this role",
    "interested candidate",
    "please check my profile",
    "please review my profile",
    "thanks for sharing",
    "thank you for sharing",
    "great opportunity",
    "nice opportunity",
    "commenting for better reach",
    "commenting for reach",
    "better reach",
    "cfbr",
)

COMPANY_NOISE_PHRASES = (
    "interpersonal skills",
    "communication skills",
    "time management",
    "problem solving",
    "problem-solving",
    "client centric",
    "client-centric",
    "customer centric",
    "customer-centric",
    "hands on experience",
    "hands-on experience",
    "years of experience",
    "strong knowledge",
    "good knowledge",
    "technical expertise",
    "mandatory skills",
    "preferred skills",
    "required skills",
    "key skills",
    "responsibilities",
    "requirements",
    "qualifications",
    "job description",
    "experience required",
    "notice period",
    "immediate joiner",
    "apply now",
    "share resume",
    "send resume",
    "send your resume",
)

FREE_EMAIL_COMPANY_DOMAINS = {
    "gmail.com",
    "googlemail.com",
    "yahoo.com",
    "yahoo.co.in",
    "outlook.com",
    "hotmail.com",
    "live.com",
    "icloud.com",
    "protonmail.com",
    "proton.me",
    "aol.com",
    "rediffmail.com",
}


# ---------------------------------------------------------------------------
# LinkedIn layout patterns
# ---------------------------------------------------------------------------

AUTHOR_MARKER_RE = re.compile(
    r"\b(?:"
    r"follow"
    r"|1st"
    r"|2nd"
    r"|3rd"
    r"|3rd\+"
    r"|2nd\+"
    r")\b",
    re.I,
)

LOCATION_LABEL_RE = re.compile(
    r"\b(?:work\s+location|location)"
    r"\s*[:\-–—]\s*([^\n]{2,100})",
    re.I,
)

NOISE_EXACT = {
    "home",
    "my network",
    "jobs",
    "messaging",
    "notifications",
    "search",
    "posts",
    "sort by",
    "date posted",
    "content type",
    "from member",
    "all filters",
    "follow",
    "like",
    "comment",
    "repost",
    "send",
    "view job",
    "see who's hiring on linkedin",
    "about",
    "accessibility",
    "help center",
    "privacy & terms",
    "ad choices",
    "advertising",
    "business services",
    "get the linkedin app",
    "more",
    "linkedin corporation",
}

NOISE_PREFIXES = (
    "linkedin corporation",
    "privacy",
    "see who",
    "are these results helpful",
    "your feedback helps",
    "posts sort by",
    "in q d365",
)


# ---------------------------------------------------------------------------
# Employment and workplace patterns
# ---------------------------------------------------------------------------

EMPLOYMENT_PATTERNS = (
    (
        re.compile(
            r"\bfull[ -]?time\b",
            re.I,
        ),
        "Full-time",
    ),
    (
        re.compile(
            r"\bpart[ -]?time\b",
            re.I,
        ),
        "Part-time",
    ),
    (
        re.compile(
            r"\bcontract\b",
            re.I,
        ),
        "Contract",
    ),
    (
        re.compile(
            r"\btemporary\b",
            re.I,
        ),
        "Temporary",
    ),
    (
        re.compile(
            r"\binternship\b",
            re.I,
        ),
        "Internship",
    ),
    (
        re.compile(
            r"\bfreelance\b",
            re.I,
        ),
        "Freelance",
    ),
)

WORKPLACE_PATTERNS = (
    (
        re.compile(
            r"\bremote\b",
            re.I,
        ),
        "Remote",
    ),
    (
        re.compile(
            r"\bhybrid\b",
            re.I,
        ),
        "Hybrid",
    ),
    (
        re.compile(
            r"\bon[ -]?site\b",
            re.I,
        ),
        "On-site",
    ),
    (
        re.compile(
            r"\bwork\s+from\s+home\b",
            re.I,
        ),
        "Remote",
    ),
)


# ---------------------------------------------------------------------------
# Internal records
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Candidate:
    title: str
    title_company: str
    lines: list[dict]
    contact_name: str = ""


@dataclass(slots=True)
class Accumulator:
    title: str
    title_company: str = ""
    contact_name: str = ""
    lines: list[dict] = field(
        default_factory=list
    )
    first_timestamp: str = ""
    last_timestamp: str = ""


class JobParser:
    """
    Rule-based LinkedIn job-post parser.

    It combines OCR text from nearby video frames while preventing different
    LinkedIn posts from being merged into the same job description.
    """

    def __init__(
        self,
        skills_file: Path,
    ) -> None:
        self.skill_catalog = self._load_skills(
            skills_file
        )

    # -----------------------------------------------------------------------
    # Public parser
    # -----------------------------------------------------------------------

    def parse(
        self,
        frames: list[FrameOcrResult],
        source_video: str,
    ) -> list[JobRecord]:
        accumulators: list[Accumulator] = []

        for frame in frames:
            frame_items = self._frame_items(
                frame
            )

            candidates = self._extract_candidates(
                frame_items
            )

            for candidate in candidates:
                accumulator = self._find_accumulator(
                    accumulators,
                    candidate,
                )

                if accumulator is None:
                    first_timestamp = (
                        candidate.lines[0]["timestamp"]
                        if candidate.lines
                        else frame.frame.timestamp
                    )

                    accumulator = Accumulator(
                        title=candidate.title,
                        title_company=(
                            candidate.title_company
                        ),
                        contact_name=(
                            candidate.contact_name
                        ),
                        first_timestamp=first_timestamp,
                        last_timestamp=first_timestamp,
                    )

                    accumulators.append(
                        accumulator
                    )

                else:
                    if (
                        not accumulator.title_company
                        and candidate.title_company
                    ):
                        accumulator.title_company = (
                            candidate.title_company
                        )

                    if (
                        not accumulator.contact_name
                        and candidate.contact_name
                    ):
                        accumulator.contact_name = (
                            candidate.contact_name
                        )

                    if candidate.lines:
                        accumulator.last_timestamp = (
                            candidate.lines[-1]["timestamp"]
                        )

                self._merge_lines(
                    accumulator.lines,
                    candidate.lines,
                )

        jobs: list[JobRecord] = []

        for index, accumulator in enumerate(
            accumulators,
            start=1,
        ):
            job = self._to_job(
                accumulator=accumulator,
                source_video=source_video,
                index=index,
            )

            if job is None:
                continue

            if not self._is_valid_job_record(job):
                continue

            jobs.append(job)

        return self._deduplicate(
            jobs
        )

    # -----------------------------------------------------------------------
    # OCR frame preparation
    # -----------------------------------------------------------------------

    def _frame_items(
        self,
        frame: FrameOcrResult,
    ) -> list[dict]:
        items: list[dict] = []

        for line in frame.lines:
            text = clean_text(
                line.text
            )

            if (
                len(text) < 3
                or self._is_noise(text)
            ):
                continue

            items.append(
                {
                    "text": text,
                    "confidence": float(
                        line.confidence
                    ),
                    "timestamp": (
                        frame.frame.timestamp
                    ),
                    "frame_file": (
                        frame.frame.path.name
                    ),
                    "y": line.y1,
                }
            )

        return items

    # -----------------------------------------------------------------------
    # Candidate detection
    # -----------------------------------------------------------------------

    def _extract_candidates(
        self,
        items: list[dict],
    ) -> list[Candidate]:
        starts: list[
            tuple[int, str, str]
        ] = []

        for index, item in enumerate(
            items
        ):
            parsed = self._parse_title_line(
                item["text"]
            )

            if parsed is None:
                continue

            title, company = parsed

            # Ignore immediate OCR repetition of the same heading.
            if starts:
                previous_index = starts[-1][0]
                previous_title = starts[-1][1]

                if (
                    self._same_title(
                        title,
                        previous_title,
                    )
                    and index - previous_index <= 2
                ):
                    continue

            starts.append(
                (
                    index,
                    title,
                    company,
                )
            )

        candidates: list[Candidate] = []

        for position, (
            start_index,
            title,
            company,
        ) in enumerate(starts):
            if position + 1 < len(starts):
                next_index = starts[
                    position + 1
                ][0]
            else:
                next_index = len(items)

            # A new LinkedIn author line indicates another post.
            for possible_end in range(
                start_index + 1,
                next_index,
            ):
                if self._is_author_line(
                    items[possible_end]["text"]
                ):
                    next_index = possible_end
                    break

            previous_context = items[
                max(
                    0,
                    start_index - 6,
                ):start_index
            ]

            block = items[
                start_index:next_index
            ]

            contact_name = self._extract_contact_name(
                previous_context
            )

            profile_company = self._extract_profile_company(
                previous_context
            )

            candidates.append(
                Candidate(
                    title=title,
                    title_company=(
                        company
                        or profile_company
                    ),
                    lines=block,
                    contact_name=contact_name,
                )
            )

        return candidates

    # -----------------------------------------------------------------------
    # Job-title parsing
    # -----------------------------------------------------------------------

    def _parse_title_line(
        self,
        text: str,
    ) -> tuple[str, str] | None:
        original = clean_text(
            text
        )

        if not original:
            return None

        # Preserve hashtag tokens initially because a valid heading can be:
        # "#Hiring #D365 F&O Technical Consultant".
        prefix_ready = re.sub(
            r"^[^A-Za-z0-9#]+",
            "",
            original,
        ).strip()

        # OCR sometimes adds 1-2 junk characters before a real title prefix,
        # for example: x” #Hiring: D365 Developer.
        prefix_match = re.search(
            r"(?:"
            r"#*\s*we(?:'|’)?re\s+hiring"
            r"|#*\s*we\s+are\s+hiring"
            r"|#*\s*now\s+hiring"
            r"|#*\s*job\s*title"
            r"|#*\s*role"
            r"|#*\s*position"
            r"|#*\s*opening"
            r"|#*\s*hiring"
            r")",
            prefix_ready,
            re.I,
        )

        if (
            prefix_match is not None
            and 0 < prefix_match.start() <= 8
        ):
            prefix_ready = prefix_ready[
                prefix_match.start():
            ].strip()

        # OCR may interpret a role icon as "@", "f@" or another prefix.
        prefix_ready = re.sub(
            r"^[A-Za-z]?@\s*"
            r"(?=(?:role|job\s*title|position)\b)",
            "",
            prefix_ready,
            flags=re.I,
        )

        # Remove a trailing LinkedIn/OCR counter such as "#4".
        prefix_ready = re.sub(
            r"\s+#\d+\s*$",
            "",
            prefix_ready,
            flags=re.I,
        ).strip()

        explicit = bool(
            TITLE_PREFIX_RE.search(
                prefix_ready
            )
        )

        # A comment/hashtag collection must never become an implicit title.
        if (
            not explicit
            and self._looks_like_hashtag_or_comment(
                prefix_ready
            )
        ):
            return None

        lowered = prefix_ready.casefold()

        contains_product = (
            "d365" in lowered
            or "dynamics 365" in lowered
            or "dynamics ax" in lowered
        )

        contains_role = any(
            hint in lowered
            for hint in ROLE_HINTS
        )

        sentence_markers = (
            "is looking for",
            "looking for",
            "experience with",
            "years of experience",
            "to support",
            "previous d365",
            "strong ",
            "knowledge of",
            "in this role",
            "we're looking for",
            "we are looking for",
            "interested?",
            "please send",
            "view job",
            "share resume",
            "commenting for",
            "thanks for",
        )

        if (
            not explicit
            and any(
                marker in lowered
                for marker in sentence_markers
            )
        ):
            return None

        if not explicit:
            if not (
                contains_product
                and contains_role
            ):
                return None

            role_at_end = re.search(
                r"\b(?:"
                r"engineer"
                r"|developer"
                r"|consultant"
                r"|architect"
                r"|specialist"
                r"|analyst"
                r"|manager"
                r"|lead"
                r"|administrator"
                r")"
                r"(?:"
                r"\s*\|.*"
                r"|"
                r"\s*[-–—]\s*"
                r"(?:remote|hybrid|on[ -]?site).*"
                r")?$",
                prefix_ready,
                re.I,
            )

            if not role_at_end:
                return None

        cleaned = TITLE_PREFIX_RE.sub(
            "",
            prefix_ready,
        ).strip(
            " |:-–—"
        )

        # Remove generic hashtags while preserving meaningful hashtag words:
        # "#Hiring #Dynamics365 Developer" -> "Dynamics365 Developer".
        cleaned = self._clean_job_title_text(
            cleaned
        )

        if not cleaned:
            return None

        parts = [
            part.strip(
                " |:-–—"
            )
            for part in re.split(
                r"\s*\|\s*",
                cleaned,
            )
            if part.strip()
        ]

        company = ""

        if len(parts) > 1:
            role_parts = [
                part
                for part in parts
                if any(
                    hint in part.casefold()
                    for hint in ROLE_HINTS
                )
            ]

            title = (
                role_parts[0]
                if role_parts
                else parts[0]
            )

            title_index = parts.index(
                title
            )

            if title_index + 1 < len(parts):
                possible_company = parts[
                    title_index + 1
                ]

                has_title_context = any(
                    hint
                    in possible_company.casefold()
                    for hint in TITLE_CONTEXT_HINTS
                )

                looks_like_location = bool(
                    re.search(
                        r"\b(?:"
                        r"remote"
                        r"|hybrid"
                        r"|on[ -]?site"
                        r"|europe"
                        r"|india"
                        r"|usa"
                        r"|uk"
                        r")\b",
                        possible_company,
                        re.I,
                    )
                )

                if (
                    not has_title_context
                    and not looks_like_location
                ):
                    company = self._clean_company_candidate(
                        possible_company
                    )
        else:
            title = cleaned

        title = self._clean_job_title_text(
            title
        )

        title = re.sub(
            r"^(?:a|an)\s+",
            "",
            title,
            flags=re.I,
        )

        title = re.sub(
            r"[.!]+$",
            "",
            title,
        ).strip(
            " |:-–—"
        )

        title = re.sub(
            r"^RED\s*[-–—]\s*",
            "",
            title,
            flags=re.I,
        )

        title = re.sub(
            r"\s*[-–—]\s*"
            r"(?:Remote|Hybrid|On[ -]?site)"
            r"(?:\s*\([^)]*\))?.*$",
            "",
            title,
            flags=re.I,
        )

        # One final cleanup after suffix removal.
        title = self._clean_job_title_text(
            title
        )

        if (
            len(title) < 5
            or len(title) > 150
        ):
            return None

        if self._looks_like_hashtag_or_comment(
            title
        ):
            return None

        # A valid job title must contain a role indicator.
        if not any(
            role_hint in title.casefold()
            for role_hint in ROLE_HINTS
        ):
            return None

        return title, company
    def _to_job(
        self,
        accumulator: Accumulator,
        source_video: str,
        index: int,
    ) -> JobRecord | None:
        if not accumulator.lines:
            return None

        texts = [
            item["text"]
            for item in accumulator.lines
        ]

        full_text = "\n".join(
            texts
        )

        job_id = f"JOB-{index:04d}"

        emails = self._unique(
            [
                value.rstrip(
                    ".,;:])"
                )
                for value in EMAIL_RE.findall(
                    full_text
                )
            ]
        )

        urls = self._unique(
            [
                value.rstrip(
                    ".,;:])"
                )
                for value in URL_RE.findall(
                    full_text
                )
            ]
        )

        skills = [
            skill
            for skill in self.skill_catalog
            if self._skill_present(
                skill,
                full_text,
            )
        ]

        company = self._clean_company_candidate(
            accumulator.title_company
        )

        if not company:
            company = self._extract_company(
                texts=texts,
                full_text=full_text,
                emails=emails,
            )

        company = self._clean_company_candidate(
            company
        )

        location = self._extract_location(
            texts,
            full_text,
        )

        experience = self._extract_experience(
            full_text
        )

        employment = self._extract_pattern_value(
            EMPLOYMENT_PATTERNS,
            full_text,
        )

        workplace = self._extract_pattern_value(
            WORKPLACE_PATTERNS,
            full_text,
        )

        posted_date = self._extract_posted_date(
            texts
        )

        application_url = next(
            (
                url
                for url in urls
                if "linkedin.com"
                not in url.casefold()
            ),
            "",
        )

        clean_title = self._clean_job_title_text(
            accumulator.title
        )

        description = self._build_job_description(
            texts=texts,
            current_title=clean_title,
            company=company,
            contact_name=(
                accumulator.contact_name
            ),
        )

        confidences = [
            float(
                item["confidence"]
            )
            for item in accumulator.lines
        ]

        average_confidence = (
            mean(confidences)
            if confidences
            else 0.0
        )

        completeness = (
            0.35
            + 0.15 * bool(company)
            + 0.15 * bool(description)
            + 0.10 * bool(emails)
            + 0.10 * bool(location)
            + 0.10 * bool(skills)
            + 0.05 * bool(experience)
        )

        confidence = round(
            min(
                1.0,
                (
                    0.65
                    * average_confidence
                    + 0.35
                    * completeness
                ),
            ),
            3,
        )

        first_item = accumulator.lines[0]

        evidence: list[EvidenceItem] = []

        def add_evidence(
            field_name: str,
            value: str,
        ) -> None:
            if not value:
                return

            evidence.append(
                EvidenceItem(
                    job_id=job_id,
                    field=field_name,
                    value=value,
                    timestamp=(
                        first_item["timestamp"]
                    ),
                    frame_file=(
                        first_item["frame_file"]
                    ),
                    confidence=round(
                        average_confidence,
                        3,
                    ),
                )
            )

        add_evidence(
            "Job Title",
            clean_title,
        )

        add_evidence(
            "Company Name",
            company,
        )

        add_evidence(
            "Contact Name",
            accumulator.contact_name,
        )

        add_evidence(
            "Location",
            location,
        )

        add_evidence(
            "Experience",
            experience,
        )

        if emails:
            add_evidence(
                "Contact Email",
                ", ".join(
                    emails
                ),
            )

        if skills:
            add_evidence(
                "Skills",
                ", ".join(
                    skills
                ),
            )

        return JobRecord(
            job_id=job_id,
            job_title=clean_title,
            company_name=company,
            location=location,
            workplace_type=workplace,
            employment_type=employment,
            experience=experience,
            skills=skills,
            posted_date=posted_date,
            contact_name=(
                accumulator.contact_name
            ),
            contact_emails=emails,
            application_url=(
                application_url
            ),
            job_description=description,
            source_video=source_video,
            first_timestamp=(
                accumulator.first_timestamp
            ),
            last_timestamp=(
                accumulator.last_timestamp
            ),
            confidence=confidence,
            review_status=(
                "Verified"
                if (
                    len(description) >= 80
                    and confidence >= 0.68
                )
                else "Needs Review"
            ),
            evidence=evidence,
        )

    # -----------------------------------------------------------------------
    # Clean Job Description
    # -----------------------------------------------------------------------

    def _build_job_description(
        self,
        texts: list[str],
        current_title: str,
        company: str,
        contact_name: str,
    ) -> str:
        """
        Build a clean, formatted job description.

        Removes:
        - LinkedIn reactions and counts
        - author/profile information
        - repeated OCR lines
        - metadata already stored in other Excel columns
        - email/application instructions
        - hashtags
        - content belonging to a later job post
        """

        if not texts:
            return ""

        cleaned_lines = [
            clean_text(text)
            for text in texts
            if clean_text(text)
        ]

        start_index = self._find_description_start(
            cleaned_lines
        )

        has_explicit_section = any(
            DESCRIPTION_SECTION_RE.match(
                text
            )
            for text in cleaned_lines
        )

        description_lines: list[str] = []
        seen_lines: list[str] = []

        inside_section = False

        if (
            not has_explicit_section
            and start_index < len(cleaned_lines)
        ):
            description_lines.append(
                "Job Description:"
            )
            inside_section = True

        for text in cleaned_lines[
            start_index:
        ]:
            text = clean_text(
                text
            )

            if not text:
                continue

            section_match = DESCRIPTION_SECTION_RE.match(
                text
            )

            if section_match:
                raw_section = clean_text(
                    section_match.group(1)
                ).casefold()

                remaining_text = clean_text(
                    section_match.group(2)
                )

                section_name = DESCRIPTION_SECTION_NAMES.get(
                    raw_section,
                    raw_section.title(),
                )

                section_heading = (
                    f"{section_name}:"
                )

                if section_heading not in description_lines:
                    if (
                        description_lines
                        and description_lines[-1] != ""
                    ):
                        description_lines.append(
                            ""
                        )

                    description_lines.append(
                        section_heading
                    )

                inside_section = True

                if remaining_text:
                    formatted = self._format_description_line(
                        remaining_text
                    )

                    self._append_description_line(
                        output=description_lines,
                        seen_lines=seen_lines,
                        line=formatted,
                        force_bullet=True,
                    )

                continue

            if self._should_stop_description(
                text=text,
                current_title=current_title,
            ):
                break

            if self._should_skip_description_line(
                text=text,
                current_title=current_title,
                company=company,
                contact_name=contact_name,
            ):
                continue

            formatted = self._format_description_line(
                text
            )

            self._append_description_line(
                output=description_lines,
                seen_lines=seen_lines,
                line=formatted,
                force_bullet=inside_section,
            )

        while (
            description_lines
            and not description_lines[0]
        ):
            description_lines.pop(
                0
            )

        while (
            description_lines
            and not description_lines[-1]
        ):
            description_lines.pop()

        return "\n".join(
            description_lines
        )

    def _find_description_start(
        self,
        texts: list[str],
    ) -> int:
        for index, text in enumerate(
            texts
        ):
            if DESCRIPTION_SECTION_RE.match(
                text
            ):
                return index

        last_metadata_index = -1

        for index, text in enumerate(
            texts
        ):
            probe = DESCRIPTION_BULLET_RE.sub(
                "",
                text,
            ).strip()

            if JOB_METADATA_RE.match(
                probe
            ):
                last_metadata_index = index

        if last_metadata_index >= 0:
            return min(
                last_metadata_index + 1,
                len(texts),
            )

        return min(
            1,
            len(texts),
        )

    def _should_stop_description(
        self,
        text: str,
        current_title: str,
    ) -> bool:
        probe = DESCRIPTION_BULLET_RE.sub(
            "",
            text,
        ).strip()

        if DESCRIPTION_STOP_RE.search(
            probe
        ):
            return True

        if HASHTAG_LIST_RE.fullmatch(
            probe
        ):
            return True

        parsed_title = self._parse_title_line(
            probe
        )

        if parsed_title is not None:
            next_title = parsed_title[0]

            if not self._same_title(
                current_title,
                next_title,
            ):
                return True

        return False

    def _should_skip_description_line(
        self,
        text: str,
        current_title: str,
        company: str,
        contact_name: str,
    ) -> bool:
        probe = DESCRIPTION_BULLET_RE.sub(
            "",
            text,
        ).strip()

        key = normalized_key(
            probe
        )

        if not key:
            return True

        if self._is_noise(
            probe
        ):
            return True

        if self._is_author_line(
            probe
        ):
            return True

        if SOCIAL_NOISE_RE.fullmatch(
            probe
        ):
            return True

        meaningful_words = re.findall(
            r"[A-Za-z]{3,}",
            probe,
        )

        if (
            len(meaningful_words) == 0
            and len(probe) <= 60
        ):
            return True

        excluded_values = {
            normalized_key(
                current_title
            ),
            normalized_key(
                company
            ),
            normalized_key(
                contact_name
            ),
        }

        if key in excluded_values:
            return True

        parsed_title = self._parse_title_line(
            probe
        )

        if (
            parsed_title is not None
            and self._same_title(
                current_title,
                parsed_title[0],
            )
        ):
            return True

        if JOB_METADATA_RE.match(
            probe
        ):
            return True

        if EMAIL_RE.search(
            probe
        ):
            return True

        if URL_RE.search(
            probe
        ):
            return True

        social_phrases = {
            "connect",
            "follow",
            "linkedin dm",
            "view job",
            "see more",
            "more",
            "like",
            "comment",
            "repost",
            "send",
        }

        if key in social_phrases:
            return True

        return False

    @staticmethod
    def _format_description_line(
        text: str,
    ) -> str:
        text = clean_text(
            text
        )

        has_bullet = bool(
            DESCRIPTION_BULLET_RE.match(
                text
            )
        )

        text = DESCRIPTION_BULLET_RE.sub(
            "",
            text,
        ).strip()

        text = re.sub(
            r"\b(\d+)\s*\+\s*years?\b",
            r"\1+ years",
            text,
            flags=re.I,
        )

        text = re.sub(
            r"\b(\d+)\s*[-–—]\s*(\d+)\b",
            r"\1-\2",
            text,
        )

        text = re.sub(
            r"\s+([,.;:])",
            r"\1",
            text,
        )

        text = re.sub(
            r"\s{2,}",
            " ",
            text,
        ).strip()

        if not text:
            return ""

        if has_bullet:
            return f"• {text}"

        return text

    def _append_description_line(
        self,
        output: list[str],
        seen_lines: list[str],
        line: str,
        force_bullet: bool,
    ) -> None:
        if not line:
            return

        plain_line = line.removeprefix(
            "• "
        ).strip()

        normalized_line = normalized_key(
            plain_line
        )

        if not normalized_line:
            return

        for existing in seen_lines:
            similarity = SequenceMatcher(
                None,
                existing,
                normalized_line,
            ).ratio()

            if similarity >= 0.90:
                return

        # Join wrapped OCR lines to the previous bullet.
        if (
            output
            and output[-1].startswith("• ")
            and not line.startswith("• ")
            and (
                output[-1].endswith(",")
                or not output[-1].endswith(
                    (
                        ".",
                        ";",
                        ":",
                        "!",
                        "?",
                    )
                )
            )
        ):
            output[-1] = clean_text(
                f"{output[-1]} {plain_line}"
            )

            if seen_lines:
                seen_lines[-1] = normalized_key(
                    output[-1].removeprefix(
                        "• "
                    )
                )

            return

        if (
            force_bullet
            and not line.startswith("• ")
            and not line.endswith(":")
        ):
            line = f"• {line}"

        output.append(
            line
        )

        seen_lines.append(
            normalized_key(
                line.removeprefix(
                    "• "
                )
            )
        )

    # -----------------------------------------------------------------------
    # Field extraction
    # -----------------------------------------------------------------------

    def _extract_company(
        self,
        texts: list[str],
        full_text: str,
        emails: list[str],
    ) -> str:
        explicit_patterns = (
            re.compile(
                r"\bcompany\s*[:\-–—]\s*"
                r"([^\n]{2,100})",
                re.I,
            ),
            re.compile(
                r"\bclient\s*[:\-–—]\s*"
                r"([^\n]{2,100})",
                re.I,
            ),
            re.compile(
                r"\borganization\s*[:\-–—]\s*"
                r"([^\n]{2,100})",
                re.I,
            ),
            re.compile(
                r"\borganisation\s*[:\-–—]\s*"
                r"([^\n]{2,100})",
                re.I,
            ),
        )

        for pattern in explicit_patterns:
            match = pattern.search(
                full_text
            )

            if not match:
                continue

            candidate = (
                self._clean_company_candidate(
                    match.group(1)
                )
            )

            if candidate:
                return candidate

        # Company-like words are useful, but only after the candidate line
        # passes strict validation. This prevents skill/JD/comment sentences
        # containing "solutions", "systems", etc. from becoming companies.
        company_tokens = (
            "global",
            "labs",
            "solutions",
            "technologies",
            "technology",
            "consulting",
            "consultancy",
            "systems",
            "services",
            "corporation",
            "limited",
            "pvt",
            "private limited",
            "inc",
            "llc",
            "group",
            "digital",
            "software",
        )

        for text in texts:
            key = text.casefold()

            if not (
                2 <= len(text) <= 100
                and any(
                    token in key
                    for token in company_tokens
                )
            ):
                continue

            if any(
                hint in key
                for hint in TITLE_CONTEXT_HINTS
            ):
                continue

            if EMAIL_RE.search(
                text
            ):
                continue

            if self._is_author_line(
                text
            ):
                continue

            candidate = (
                self._clean_company_candidate(
                    text
                )
            )

            if not candidate:
                continue

            if normalized_key(
                candidate
            ).startswith(
                "red globa"
            ):
                return "RED Global"

            return candidate

        # Email domain is safer than inventing a company from a comment line.
        if emails:
            raw_domain = (
                emails[0]
                .split("@", 1)[1]
                .strip()
                .casefold()
            )

            if (
                raw_domain
                and raw_domain
                not in FREE_EMAIL_COMPANY_DOMAINS
            ):
                domain_label = (
                    raw_domain.split(".", 1)[0]
                )

                known_domains = {
                    "redglobal": "RED Global",
                    "fidelituscorp": "Fidelitus HR Labs",
                    "korcomptenz": "Korcomptenz",
                    "cortracker369": "Cortracker 369",
                    "cortracker360": "Cortracker 360",
                }

                candidate = known_domains.get(
                    domain_label,
                    domain_label.replace(
                        "-",
                        " ",
                    ).title(),
                )

                candidate = (
                    self._clean_company_candidate(
                        candidate
                    )
                )

                if candidate:
                    return candidate

        return ""
    @staticmethod
    def _extract_location(
        texts: list[str],
        full_text: str,
    ) -> str:
        match = LOCATION_LABEL_RE.search(
            full_text
        )

        if match:
            value = clean_text(
                match.group(1)
            )

            value = re.split(
                r"\b(?:"
                r"full[ -]?time"
                r"|part[ -]?time"
                r"|contract"
                r")\b",
                value,
                flags=re.I,
            )[0]

            return value.strip(
                " ._-|"
            )

        for text in texts:
            if re.match(
                r"^\s*work\s+mode\s*[:\-–—]",
                text,
                re.I,
            ):
                continue

            if re.search(
                r"\b(?:"
                r"remote"
                r"|hybrid"
                r"|on[ -]?site"
                r")\b",
                text,
                re.I,
            ):
                if len(text) <= 100:
                    return text.strip(
                        " ."
                    )

        return ""

    @staticmethod
    def _extract_experience(
        full_text: str,
    ) -> str:
        match = EXPERIENCE_RE.search(
            full_text
        )

        if not match:
            return ""

        value = clean_text(
            match.group(0)
        )

        value = re.sub(
            r"\s+to\s+",
            "-",
            value,
            flags=re.I,
        )

        value = re.sub(
            r"\s*[-–—]\s*",
            "-",
            value,
        )

        value = re.sub(
            r"(\d+)\s*\+\s*years?",
            r"\1+ years",
            value,
            flags=re.I,
        )

        value = re.sub(
            r"\s+of\s+experience$",
            "",
            value,
            flags=re.I,
        )

        return value

    @staticmethod
    def _extract_posted_date(
        texts: list[str],
    ) -> str:
        for text in texts[:10]:
            match = POSTED_RE.search(
                text
            )

            if match:
                return clean_text(
                    match.group(0)
                )

        return ""

    @staticmethod
    def _extract_pattern_value(
        patterns: tuple[
            tuple[re.Pattern, str],
            ...
        ],
        text: str,
    ) -> str:
        for pattern, value in patterns:
            if pattern.search(
                text
            ):
                return value

        return ""

    def _extract_profile_company(
        self,
        previous_items: list[dict],
    ) -> str:
        """
        Extract a company only from a short profile/headline-style line.

        A random sentence containing the word "at" must not become a company.
        """

        for item in reversed(
            previous_items
        ):
            text = clean_text(
                item["text"]
            )

            if (
                not text
                or "@" in text
                or EMAIL_RE.search(text)
            ):
                continue

            # Profile headline context is usually short.
            if len(text) > 140:
                continue

            match = re.search(
                r"\bat\s+([^|\n]{2,80})",
                text,
                re.I,
            )

            if not match:
                continue

            company = clean_text(
                match.group(1)
            )

            # Stop at LinkedIn/profile separators and common suffix text.
            company = re.split(
                r"\s*(?:\||•|·)\s*",
                company,
                maxsplit=1,
            )[0]

            company = re.sub(
                r"\s*\bHiring\b.*$",
                "",
                company,
                flags=re.I,
            )

            company = company.rstrip(
                "., "
            )

            candidate = (
                self._clean_company_candidate(
                    company
                )
            )

            if not candidate:
                continue

            if any(
                hint in candidate.casefold()
                for hint in TITLE_CONTEXT_HINTS
            ):
                continue

            return candidate

        return ""
    @staticmethod
    def _extract_contact_name(
        previous_items: list[dict],
    ) -> str:
        for item in reversed(
            previous_items
        ):
            text = item["text"]

            if not AUTHOR_MARKER_RE.search(
                text
            ):
                continue

            candidate = re.sub(
                r"^[^A-Za-z]+",
                "",
                text,
            )

            candidate = candidate.replace(
                "+",
                " ",
            )

            candidate = re.split(
                r"\s+(?:@|•|-)?\s*"
                r"(?:"
                r"1st"
                r"|2nd"
                r"|3rd"
                r"|3rd\+"
                r"|2nd\+"
                r"|Follow"
                r")\b",
                candidate,
                maxsplit=1,
                flags=re.I,
            )[0]

            candidate = clean_text(
                candidate
            ).strip(
                " -@•+"
            )

            if re.fullmatch(
                r"[A-Z][A-Za-z.'-]*"
                r"(?:\s+"
                r"[A-Za-z][A-Za-z.'-]*)"
                r"{1,3}",
                candidate,
            ):
                return " ".join(
                    (
                        part.upper()
                        if len(part) == 1
                        else part
                    )
                    for part in candidate.split()
                )

        return ""

    # -----------------------------------------------------------------------
    # Candidate merging
    # -----------------------------------------------------------------------

    def _find_accumulator(
        self,
        accumulators: list[Accumulator],
        candidate: Candidate,
    ) -> Accumulator | None:
        """
        Merge nearby frames belonging to the same LinkedIn post.

        Different recruiters, companies or distant timestamps are treated
        as separate postings even when the job titles are similar.
        """

        candidate_timestamp = (
            candidate.lines[0]["timestamp"]
            if candidate.lines
            else ""
        )

        candidate_seconds = self._timestamp_to_seconds(
            candidate_timestamp
        )

        for accumulator in reversed(
            accumulators
        ):
            if not self._same_title(
                accumulator.title,
                candidate.title,
            ):
                continue

            if (
                candidate.contact_name
                and accumulator.contact_name
                and normalized_key(
                    candidate.contact_name
                )
                != normalized_key(
                    accumulator.contact_name
                )
            ):
                continue

            if (
                candidate.title_company
                and accumulator.title_company
                and normalized_key(
                    candidate.title_company
                )
                != normalized_key(
                    accumulator.title_company
                )
            ):
                continue

            accumulator_seconds = self._timestamp_to_seconds(
                accumulator.last_timestamp
            )

            time_gap = abs(
                candidate_seconds
                - accumulator_seconds
            )

            has_identity = bool(
                candidate.contact_name
                or candidate.title_company
                or accumulator.contact_name
                or accumulator.title_company
            )

            if (
                not has_identity
                and time_gap > 4
            ):
                continue

            return accumulator

        return None

    @staticmethod
    def _timestamp_to_seconds(
        timestamp: str,
    ) -> int:
        try:
            parts = [
                int(part)
                for part in timestamp.split(
                    ":"
                )
            ]

            if len(parts) != 3:
                return 0

            hours, minutes, seconds = parts

            return (
                hours * 3600
                + minutes * 60
                + seconds
            )

        except (
            TypeError,
            ValueError,
        ):
            return 0

    @staticmethod
    def _merge_lines(
        target: list[dict],
        incoming: list[dict],
    ) -> None:
        existing = {
            normalized_key(
                item["text"]
            )
            for item in target
        }

        for item in incoming:
            key = normalized_key(
                item["text"]
            )

            if (
                key
                and key not in existing
            ):
                target.append(
                    item
                )
                existing.add(
                    key
                )

    # -----------------------------------------------------------------------
    # General helpers
    # -----------------------------------------------------------------------


    @staticmethod
    def _clean_job_title_text(
        value: str,
    ) -> str:
        """
        Remove LinkedIn hashtag/social noise without deleting useful role text.

        Examples:
            #Hiring #D365 F&O Technical Consultant
            -> D365 F&O Technical Consultant

            D365 Technical Consultant #4
            -> D365 Technical Consultant
        """

        value = clean_text(
            value
        )

        if not value:
            return ""

        value = re.sub(
            r"\s+#\d+\s*$",
            "",
            value,
            flags=re.I,
        )

        def replace_hashtag(
            match: re.Match,
        ) -> str:
            token = match.group(1)

            normalized = re.sub(
                r"[^a-z0-9]",
                "",
                token.casefold(),
            )

            if normalized in GENERIC_HIRING_HASHTAGS:
                return " "

            return " " + token.replace(
                "_",
                " ",
            ) + " "

        value = re.sub(
            r"#([A-Za-z][A-Za-z0-9_+.-]*)",
            replace_hashtag,
            value,
        )

        value = re.sub(
            r"^\s*(?:"
            r"we(?:'|’)?re\s+hiring"
            r"|we\s+are\s+hiring"
            r"|now\s+hiring"
            r"|hiring\s+alert"
            r"|hiring"
            r"|job\s+opening"
            r"|job\s+opportunity"
            r"|urgent\s+hiring"
            r"|urgent\s+requirement"
            r")"
            r"\s*[|:\-–—]*\s*",
            "",
            value,
            flags=re.I,
        )

        value = re.sub(
            r"^\s*(?:"
            r"job\s*title"
            r"|role"
            r"|position"
            r")"
            r"\s*[|:\-–—]+\s*",
            "",
            value,
            flags=re.I,
        )

        value = re.sub(
            r"\s+",
            " ",
            value,
        ).strip(
            " #|:-–—•·"
        )

        return value

    @staticmethod
    def _clean_company_candidate(
        value: str,
    ) -> str:
        """
        Return a plausible company name or blank.

        Skill sentences, comments, hashtag blocks, URLs and application
        instructions must never become the company sent to Apollo.
        """

        value = clean_text(
            value
        ).strip(
            " |:-–—•·"
        )

        if not value:
            return ""

        lowered = value.casefold()

        if EMAIL_RE.search(
            value
        ):
            return ""

        if URL_RE.search(
            value
        ):
            return ""

        if any(
            phrase in lowered
            for phrase in COMMENT_PHRASES
        ):
            return ""

        if COMMENT_TITLE_RE.search(
            value
        ):
            return ""

        if any(
            phrase in lowered
            for phrase in COMPANY_NOISE_PHRASES
        ):
            return ""

        hashtag_matches = (
            HASHTAG_TOKEN_RE.findall(
                value
            )
        )

        if len(hashtag_matches) >= 2:
            return ""

        if hashtag_matches:
            hashtag = hashtag_matches[0]
            normalized = re.sub(
                r"[^a-z0-9]",
                "",
                hashtag[1:].casefold(),
            )

            if normalized in GENERIC_HIRING_HASHTAGS:
                return ""

            value = value.replace(
                hashtag,
                hashtag[1:].replace(
                    "_",
                    " ",
                ),
            )

        lowered = value.casefold()

        sentence_markers = (
            " is looking for ",
            " are looking for ",
            " we are ",
            " we're ",
            " should ",
            " must ",
            " required ",
            " responsible for ",
            " experience in ",
            " experience with ",
            " knowledge of ",
            " skills ",
        )

        padded = f" {lowered} "

        if any(
            marker in padded
            for marker in sentence_markers
        ):
            return ""

        if (
            len(value) > 100
            or len(value.split()) > 12
            or value.count(",") >= 2
            or value.count(";") >= 1
        ):
            return ""

        normalized = normalized_key(
            value
        )

        if (
            not normalized
            or normalized in INVALID_COMPANY_NAMES
        ):
            return ""

        value = re.sub(
            r"^(?:"
            r"company(?:\s+name)?"
            r"|organization"
            r"|organisation"
            r"|client"
            r")"
            r"\s*[:\-–—]\s*",
            "",
            value,
            flags=re.I,
        ).strip(
            " |:-–—"
        )

        if not value:
            return ""

        return value

    @staticmethod
    def _is_author_line(
        text: str,
    ) -> bool:
        return (
            bool(
                AUTHOR_MARKER_RE.search(
                    text
                )
            )
            and len(text) <= 120
        )

    @staticmethod
    def _same_title(
        left: str,
        right: str,
    ) -> bool:
        left_key = normalized_key(
            left
        )

        right_key = normalized_key(
            right
        )

        if left_key == right_key:
            return True

        if (
            left_key in right_key
            or right_key in left_key
        ):
            return (
                min(
                    len(left_key),
                    len(right_key),
                )
                >= 12
            )

        return (
            SequenceMatcher(
                None,
                left_key,
                right_key,
            ).ratio()
            >= 0.84
        )

    @staticmethod
    def _skill_present(
        skill: str,
        text: str,
    ) -> bool:
        return (
            re.search(
                rf"(?<![A-Za-z0-9])"
                rf"{re.escape(skill)}"
                rf"(?![A-Za-z0-9])",
                text,
                re.I,
            )
            is not None
        )

    @staticmethod
    def _load_skills(
        path: Path,
    ) -> list[str]:
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
                if (
                    value
                    and not value.startswith(
                        "#"
                    )
                )
            },
            key=lambda value: (
                -len(value),
                value.casefold(),
            ),
        )

    @staticmethod
    def _is_noise(
        text: str,
    ) -> bool:
        key = normalized_key(
            text
        )

        if key in NOISE_EXACT:
            return True

        return any(
            key.startswith(
                prefix
            )
            for prefix in NOISE_PREFIXES
        )

    @staticmethod
    def _unique(
        values: list[str],
    ) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()

        for value in values:
            key = normalized_key(
                value
            )

            if (
                value
                and key not in seen
            ):
                result.append(
                    value
                )
                seen.add(
                    key
                )

        return result


    @staticmethod
    def _looks_like_hashtag_or_comment(
        text: str,
    ) -> bool:
        """
        Reject LinkedIn comments, social controls, article headings and
        hashtag collections.

        A real title using hashtags is allowed only when it contains a
        recognizable role signal after generic hashtag noise is removed.
        """

        cleaned = clean_text(
            text
        )

        if not cleaned:
            return True

        lowered = cleaned.casefold()

        if any(
            phrase in lowered
            for phrase in COMMENT_PHRASES
        ):
            return True

        if COMMENT_TITLE_RE.search(
            cleaned
        ):
            return True

        normalized = normalized_key(
            cleaned
        )

        ui_values = {
            "like",
            "comment",
            "comments",
            "repost",
            "send",
            "reply",
            "replies",
            "view profile",
            "see more",
            "show more",
            "more",
            "follow",
            "share",
        }

        if normalized in ui_values:
            return True

        hashtag_matches = (
            HASHTAG_TOKEN_RE.findall(
                cleaned
            )
        )

        if hashtag_matches:
            # Remove hashtags and inspect remaining natural-language text.
            remaining_text = (
                HASHTAG_TOKEN_RE.sub(
                    "",
                    cleaned,
                )
            )

            remaining_text = re.sub(
                r"[^A-Za-z0-9]+",
                " ",
                remaining_text,
            ).strip()

            # Clean hashtag words themselves and see whether the resulting
            # text still contains an actual role.
            title_probe = (
                JobParser._clean_job_title_text(
                    cleaned
                )
            ).casefold()

            has_role = any(
                hint in title_probe
                for hint in ROLE_HINTS
            )

            has_product = (
                "d365" in title_probe
                or "dynamics 365" in title_probe
                or "dynamics365" in title_probe
                or "dynamics ax" in title_probe
            )

            # Pure hashtag collections such as:
            # #D365 #ERP #HiringNow #Jobs
            # should never create a job.
            if (
                len(hashtag_matches) >= 2
                and len(remaining_text) < 20
                and not (
                    has_role
                    and has_product
                )
            ):
                return True

            # A hashtag-starting line with no recognizable role is social
            # tagging, not a title.
            if (
                cleaned.startswith("#")
                and not has_role
            ):
                return True

        return False
    def _is_valid_job_record(
        self,
        job: JobRecord,
    ) -> bool:
        """
        Final validation before creating an Excel row.

        Hashtag/comment rows are rejected. A missing company is allowed because
        Apollo can recover the organization from a professional email domain.
        """

        title = self._clean_job_title_text(
            job.job_title or ""
        )

        if not title:
            return False

        # Store the cleaned title back on the final record.
        job.job_title = title

        description = clean_text(
            job.job_description or ""
        )

        if self._looks_like_hashtag_or_comment(
            title
        ):
            return False

        if not any(
            role_hint in title.casefold()
            for role_hint in ROLE_HINTS
        ):
            return False

        # Hashtag/comment rows normally contain only a tiny description.
        if len(description) < 45:
            return False

        # Sanitize company rather than trusting OCR/profile text.
        company = self._clean_company_candidate(
            job.company_name or ""
        )

        job.company_name = company

        signal_count = 0

        if company:
            signal_count += 1

        if job.contact_emails:
            signal_count += 1

        if clean_text(
            job.experience or ""
        ):
            signal_count += 1

        if job.skills:
            signal_count += 1

        if JOB_CONTENT_SIGNAL_RE.search(
            description
        ):
            signal_count += 1

        if clean_text(
            job.location or ""
        ):
            signal_count += 1

        # Require at least two supporting job signals.
        if signal_count < 2:
            return False

        return True
    def _deduplicate(
        self,
        jobs: list[JobRecord],
    ) -> list[JobRecord]:
        result: list[JobRecord] = []

        

        for job in jobs:
            if not self._is_valid_job_record(
                job
            ):
                continue
            duplicate = next(   
                (
                    existing
                    for existing in result
                    if (
                        self._same_title(
                            existing.job_title,
                            job.job_title,
                        )
                        and (
                            not existing.company_name
                            or not job.company_name
                            or normalized_key(
                                existing.company_name
                            )
                            == normalized_key(
                                job.company_name
                            )
                        )
                        and (
                            not existing.location
                            or not job.location
                            or normalized_key(
                                existing.location
                            )
                            == normalized_key(
                                job.location
                            )
                        )
                    )
                ),
                None,
            )

            if duplicate is not None:
                continue

            job.job_id = (
                f"JOB-{len(result) + 1:04d}"
            )

            for evidence in job.evidence:
                evidence.job_id = (
                    job.job_id
                )

            result.append(
                job
            )

        return result