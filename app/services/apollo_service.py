from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv




# ============================================================================
# Environment
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(
    dotenv_path=ENV_FILE,
    override=True,
)


# ============================================================================
# Configuration
# ============================================================================

LOGGER = logging.getLogger(__name__)

APOLLO_BASE_URL = "https://api.apollo.io/api/v1"


# ============================================================================
# Revenue-based contact search tiers
# ============================================================================

# Revenue routing thresholds are expressed in INR crore.
TIER_1_MAX_REVENUE_CRORE = 250.0
TIER_2_MAX_REVENUE_CRORE = 1000.0

# Hard limit requested for Apollo person enrichment / Excel output.
MAX_CONTACTS_PER_COMPANY = 3


# ---------------------------------------------------------------------------
# TIER 1
# Revenue below INR 250 Crore
# Priority order is important. Each tuple is one title priority and can contain
# aliases that Apollo may return for the same business role.
# ---------------------------------------------------------------------------

TIER_1_TITLE_GROUPS: list[tuple[str, ...]] = [
    ("Managing Director", "MD"),
    ("Chief Executive Officer", "CEO"),
    ("Chief Financial Officer", "CFO"),
    ("Chief Operating Officer", "COO"),
    ("Chief Technology Officer", "CTO"),
    ("Head of IT", "IT Head"),
    ("Finance Head", "Head of Finance"),
    ("ERP Manager", "Business Applications Manager"),
    ("Operations Head", "Plant Head"),
    ("EDP Manager",),
    ("IT Controller",),
]


# ---------------------------------------------------------------------------
# TIER 2
# Revenue INR 250 Crore to INR 1,000 Crore
# ---------------------------------------------------------------------------

TIER_2_TITLE_GROUPS: list[tuple[str, ...]] = [
    ("Chief Information Officer", "CIO"),
    ("Chief Financial Officer", "CFO"),
    ("Chief Executive Officer", "CEO"),
    ("Chief Technology Officer", "CTO"),
    ("Head of IT", "IT Head"),
    ("Head of ERP", "ERP Manager"),
    ("IT Director", "Director of IT"),
    ("Finance Director", "Director of Finance", "Head of Finance"),
    ("Head of Digital Transformation", "Digital Transformation Head"),
    ("Chief Operating Officer", "COO", "Operations Director"),
    ("IT Controller",),
]


# ---------------------------------------------------------------------------
# TIER 3
# Revenue above INR 1,000 Crore
# ---------------------------------------------------------------------------

TIER_3_TITLE_GROUPS: list[tuple[str, ...]] = [
    ("Chief Information Officer", "CIO"),
    ("Chief Financial Officer", "CFO"),
    ("Chief Executive Officer", "CEO"),
    ("Chief Technology Officer", "CTO"),
    ("Chief Digital Officer", "CDO"),
    (
        "VP Enterprise Applications",
        "Vice President Enterprise Applications",
        "Head of Enterprise Applications",
    ),
    ("ERP Director", "Director of ERP", "Head of ERP"),
    (
        "VP Information Technology",
        "Vice President Information Technology",
        "VP IT",
        "IT Director",
        "Director of IT",
    ),
    (
        "Head of Digital Transformation",
        "Digital Transformation Director",
        "Director of Digital Transformation",
    ),
    (
        "Finance Transformation Director",
        "Director of Finance Transformation",
        "Head of Finance Transformation",
    ),
]


# Flat title lists are used for the Apollo People Search request and scoring.
# The grouped constants above remain the source of truth for priority ordering.
TIER_1_TITLES = [
    title
    for group in TIER_1_TITLE_GROUPS
    for title in group
]
TIER_2_TITLES = [
    title
    for group in TIER_2_TITLE_GROUPS
    for title in group
]
TIER_3_TITLES = [
    title
    for group in TIER_3_TITLE_GROUPS
    for title in group
]


# ============================================================================
# Validation
# ============================================================================

FREE_EMAIL_DOMAINS = {
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

INVALID_COMPANY_VALUES = {
    "",
    "unknown",
    "n/a",
    "na",
    "not available",
    "remote",
    "hybrid",
    "on-site",
    "onsite",
    "hiring",
    "linkedin",
    "company",
    "job",
    "jobs",
}


# ============================================================================
# Models
# ============================================================================

@dataclass(slots=True)
class ApolloCompany:
    matched: bool = False
    match_status: str = "NOT_SEARCHED"

    apollo_id: str = ""
    name: str = ""
    domain: str = ""
    website_url: str = ""
    linkedin_url: str = ""

    industry: str = ""
    employee_count: int | None = None
    annual_revenue: float | None = None
    founded_year: int | None = None

    phone: str = ""
    city: str = ""
    state: str = ""
    country: str = ""

    raw_data: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class ApolloOfficial:
    matched: bool = False

    apollo_person_id: str = ""

    full_name: str = ""
    first_name: str = ""
    last_name: str = ""

    title: str = ""
    seniority: str = ""

    work_email: str = ""
    email_status: str = ""

    linkedin_url: str = ""

    city: str = ""
    state: str = ""
    country: str = ""

    organization_name: str = ""

    raw_data: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class ApolloEnrichmentResult:
    company: ApolloCompany

    officials: list[ApolloOfficial] = field(
        default_factory=list
    )

    error: str = ""


@dataclass(slots=True)
class ApolloCreditUsage:
    """Per-company Apollo API credit estimate for logging.

    Current workflow assumptions:
    - Organization Enrichment: estimated 1 credit per successful call.
    - People API Search: 0 credits.
    - People Enrichment: estimated 1 credit when a person record is returned,
      because phone reveal and waterfall enrichment are disabled.

    Apollo plan/legacy-plan billing can differ, so this is logged as an
    estimate rather than an authoritative billing total.
    """

    organization_enrichment_calls: int = 0
    organization_enrichment_credits_estimated: int = 0

    people_search_calls: int = 0
    people_search_credits_estimated: int = 0

    people_enrichment_calls: int = 0
    people_enrichment_matches: int = 0
    people_enrichment_credits_estimated: int = 0

    @property
    def total_estimated_credits(self) -> int:
        return (
            self.organization_enrichment_credits_estimated
            + self.people_search_credits_estimated
            + self.people_enrichment_credits_estimated
        )


class ApolloApiError(RuntimeError):
    """Raised when an Apollo API request fails."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: str = "",
    ) -> None:
        super().__init__(message)

        self.status_code = status_code
        self.response_body = response_body


# ============================================================================
# Service
# ============================================================================

class ApolloService:
    def __init__(
        self,
        api_key: str | None = None,
        max_contacts_per_company: int | None = MAX_CONTACTS_PER_COMPANY,
        timeout_seconds: float | None = None,
    ) -> None:
        self.api_key = (
            api_key
            or os.getenv(
                "APOLLO_API_KEY",
                "",
            )
        ).strip()

        if not self.api_key:
            raise ValueError(
                (
                    "APOLLO_API_KEY is missing. "
                    f"Expected .env file: {ENV_FILE}"
                )
            )


        # Keep this limit fixed at three even if an older caller passes a
        # different value. This prevents accidental enrichment of many people.
        self.max_contacts_per_company = MAX_CONTACTS_PER_COMPANY

        if max_contacts_per_company not in {
            None,
            MAX_CONTACTS_PER_COMPANY,
        }:
            LOGGER.info(
                "Ignoring requested Apollo contact limit=%s; fixed limit=%s",
                max_contacts_per_company,
                MAX_CONTACTS_PER_COMPANY,
            )

        configured_timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else self._env_float(
                "APOLLO_TIMEOUT_SECONDS",
                30.0,
            )
        )

        self.client = httpx.Client(
            base_url=APOLLO_BASE_URL,
            timeout=httpx.Timeout(
                configured_timeout
            ),
            headers={
                "accept": "application/json",
                "content-type": "application/json",
                "x-api-key": self.api_key,
            },
            follow_redirects=True,
        )

        # Total estimated Apollo credits for this processing run / Excel.
        # Per-company credit details are intentionally not logged.
        self.total_estimated_credits = 0
        self._credit_total_logged = False

        LOGGER.info(
            (
                "Apollo service initialized "
                "env=%s key_length=%s key_ending=%s "
                "contact_limit=%s"
            ),
            ENV_FILE,
            len(self.api_key),
            (
                self.api_key[-4:]
                if len(self.api_key) >= 4
                else "INVALID"
            ),
            self.max_contacts_per_company,
        )

    def __enter__(
        self,
    ) -> ApolloService:
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.close()

    def close(
        self,
    ) -> None:
        # Print one simple credit total for the complete Excel/process run.
        if not self._credit_total_logged:
            LOGGER.info(
                "TOTAL APOLLO CREDITS USED FOR THIS EXCEL = %s (ESTIMATED)",
                self.total_estimated_credits,
            )
            self._credit_total_logged = True

        self.client.close()

    # ========================================================================
    # Complete flow
    # ========================================================================

    def enrich_company_and_find_talent(
        self,
        company_name: str,
        known_email: str = "",
        known_domain: str = "",
        company_linkedin_url: str = "",
    ) -> ApolloEnrichmentResult:
        cleaned_company = self.clean_company_name(
            company_name
        )

        # Do not call Apollo at all when the extracted company name is empty
        # or was cleaned into an invalid/blank company value.
        # This also means no Apollo credits are counted for this row.
        if not cleaned_company:
            return ApolloEnrichmentResult(
                company=ApolloCompany(
                    match_status="MISSING_COMPANY_NAME",
                ),
                officials=[],
                error=(
                    "Company name is empty. "
                    "Apollo enrichment was skipped."
                ),
            )

        domain = (
            self.normalize_domain(
                known_domain
            )
            or self.domain_from_email(
                known_email
            )
        )

        credit_usage = ApolloCreditUsage()

        def finish(
            result: ApolloEnrichmentResult,
        ) -> ApolloEnrichmentResult:
            # Add this company's estimated credits to the one total for
            # the complete Excel/process run. No per-company credit log.
            self.total_estimated_credits += (
                credit_usage.total_estimated_credits
            )
            return result

        if (
            not cleaned_company
            and not domain
        ):
            return finish(
                ApolloEnrichmentResult(
                    company=ApolloCompany(
                        match_status=(
                            "MISSING_COMPANY_IDENTIFIER"
                        )
                    ),
                    error=(
                        "Company name and company domain "
                        "are both missing."
                    ),
                )
            )

        # --------------------------------------------------------------------
        # Step 1: Organization enrichment
        # --------------------------------------------------------------------

        try:
            company = self.enrich_company(
                company_name=cleaned_company,
                domain=domain,
                linkedin_url=(
                    company_linkedin_url
                ),
                credit_usage=credit_usage,
            )

        except ApolloApiError as exc:
            LOGGER.warning(
                (
                    "Apollo company enrichment failed "
                    "company=%s status=%s error=%s"
                ),
                cleaned_company,
                exc.status_code,
                exc,
            )

            return finish(
                ApolloEnrichmentResult(
                    company=ApolloCompany(
                        name=cleaned_company,
                        domain=domain,
                        match_status="API_ERROR",
                    ),
                    error=(
                        "Company enrichment failed: "
                        f"{exc}"
                    ),
                )
            )

        except httpx.HTTPError as exc:
            LOGGER.warning(
                (
                    "Apollo company connection failed "
                    "company=%s error=%s"
                ),
                cleaned_company,
                exc,
            )

            return finish(
                ApolloEnrichmentResult(
                    company=ApolloCompany(
                        name=cleaned_company,
                        domain=domain,
                        match_status=(
                            "CONNECTION_ERROR"
                        ),
                    ),
                    error=(
                        "Company enrichment connection "
                        f"error: {exc}"
                    ),
                )
            )

        if not company.matched:
            return finish(
                ApolloEnrichmentResult(
                    company=company,
                    error=(
                        "Apollo did not find a matching company."
                    ),
                )
            )

        # --------------------------------------------------------------------
        # Step 2: Find revenue-tier officials
        # --------------------------------------------------------------------

        try:
            officials = self.search_talent_officials(
                company=company,
                credit_usage=credit_usage,
            )

        except ApolloApiError as exc:
            LOGGER.warning(
                (
                    "Apollo company matched but people "
                    "search failed company=%s "
                    "status=%s error=%s"
                ),
                company.name,
                exc.status_code,
                exc,
            )

            return finish(
                ApolloEnrichmentResult(
                    company=company,
                    officials=[],
                    error=(
                        "Company matched successfully, "
                        "but people search failed: "
                        f"{exc}"
                    ),
                )
            )

        except httpx.HTTPError as exc:
            LOGGER.warning(
                (
                    "Apollo company matched but people "
                    "search connection failed "
                    "company=%s error=%s"
                ),
                company.name,
                exc,
            )

            return finish(
                ApolloEnrichmentResult(
                    company=company,
                    officials=[],
                    error=(
                        "Company matched successfully, "
                        "but people search connection "
                        f"failed: {exc}"
                    ),
                )
            )

        if not officials:
            return finish(
                ApolloEnrichmentResult(
                    company=company,
                    officials=[],
                    error=(
                        "Company matched successfully, "
                        "but Apollo did not return a "
                        "contactable official for the "
                        "selected revenue tier."
                    ),
                )
            )

        return finish(
            ApolloEnrichmentResult(
                company=company,
                officials=officials,
                error="",
            )
        )

    # ========================================================================
    # Organization enrichment
    # ========================================================================

    def enrich_company(
        self,
        company_name: str = "",
        domain: str = "",
        linkedin_url: str = "",
        website_url: str = "",
        credit_usage: ApolloCreditUsage | None = None,
    ) -> ApolloCompany:
        params: dict[str, str] = {}

        company_name = self.clean_company_name(
            company_name
        )

        domain = self.normalize_domain(
            domain
        )

        if company_name:
            params["name"] = company_name

        if domain:
            params["domain"] = domain

        if linkedin_url:
            params["linkedin_url"] = (
                linkedin_url.strip()
            )

        if website_url:
            params["website"] = (
                website_url.strip()
            )

        if not params:
            return ApolloCompany(
                match_status=(
                    "MISSING_COMPANY_IDENTIFIER"
                )
            )

        LOGGER.info(
            (
                "Calling Apollo organization "
                "enrichment name=%s domain=%s"
            ),
            company_name,
            domain,
        )

        if credit_usage is not None:
            credit_usage.organization_enrichment_calls += 1

        response = self.client.get(
            "/organizations/enrich",
            params=params,
        )

        data = self._read_response(
            response=response,
            operation=(
                "organization enrichment"
            ),
        )

        # Organization Enrichment is documented as a credit-consuming
        # endpoint. We count one estimated credit after a successful API
        # response. The final authoritative total remains Apollo billing.
        if credit_usage is not None:
            credit_usage.organization_enrichment_credits_estimated += 1

        organization = (
            data.get("organization")
            or data.get("account")
            or {}
        )

        if (
            not isinstance(
                organization,
                dict,
            )
            or not organization
        ):
            return ApolloCompany(
                name=company_name,
                domain=domain,
                match_status="NO_MATCH",
                raw_data=data,
            )

        primary_address = (
            organization.get(
                "primary_address"
            )
            or {}
        )

        if not isinstance(
            primary_address,
            dict,
        ):
            primary_address = {}

        result = ApolloCompany(
            matched=True,
            match_status="MATCHED",

            apollo_id=str(
                organization.get("id")
                or organization.get(
                    "organization_id"
                )
                or ""
            ),

            name=str(
                organization.get("name")
                or company_name
            ).strip(),

            domain=self.normalize_domain(
                str(
                    organization.get(
                        "primary_domain"
                    )
                    or organization.get(
                        "domain"
                    )
                    or domain
                )
            ),

            website_url=str(
                organization.get(
                    "website_url"
                )
                or organization.get(
                    "website"
                )
                or ""
            ).strip(),

            linkedin_url=str(
                organization.get(
                    "linkedin_url"
                )
                or ""
            ).strip(),

            industry=str(
                organization.get(
                    "industry"
                )
                or ""
            ).strip(),

            employee_count=self._safe_int(
                organization.get(
                    "estimated_num_employees"
                )
                or organization.get(
                    "num_employees"
                )
            ),

            annual_revenue=self._safe_float(
    organization.get(
        "annual_revenue"
    )
    or organization.get(
        "organization_revenue"
    )
),

            founded_year=self._safe_int(
                organization.get(
                    "founded_year"
                )
            ),

            phone=self._extract_company_phone(
                organization
            ),

            city=str(
                organization.get("city")
                or primary_address.get(
                    "city"
                )
                or ""
            ).strip(),

            state=str(
                organization.get("state")
                or primary_address.get(
                    "state"
                )
                or ""
            ).strip(),

            country=str(
                organization.get("country")
                or primary_address.get(
                    "country"
                )
                or ""
            ).strip(),

            raw_data=organization,
        )

        LOGGER.info(
            (
                "Apollo matched company "
                "input=%s matched=%s "
                "organization_id=%s domain=%s"
            ),
            company_name,
            result.name,
            result.apollo_id,
            result.domain,
        )

        return result

    # ========================================================================
    # Contact search
    # ========================================================================
    def _revenue_to_crore(
        self,
        company: ApolloCompany,
    ) -> float | None:
        """
        Convert Apollo revenue into INR crore.

        ``company.annual_revenue`` is kept exactly as returned by Apollo.
        APOLLO_REVENUE_TO_INR_RATE converts one raw revenue unit into INR.

        Examples:

        If Apollo revenue is already INR:
            APOLLO_REVENUE_TO_INR_RATE=1

        If Apollo revenue is USD:
            set APOLLO_REVENUE_TO_INR_RATE to the USD -> INR rate you
            want the application to use.

        1 crore INR = 10,000,000 INR.
        """

        revenue = company.annual_revenue

        if revenue is None:
            return None

        try:
            revenue = float(revenue)
        except (TypeError, ValueError):
            return None

        if revenue <= 0:
            return None

        conversion_rate = self._env_float(
            "APOLLO_REVENUE_TO_INR_RATE",
            1.0,
        )

        if conversion_rate <= 0:
            LOGGER.warning(
                (
                    "Invalid APOLLO_REVENUE_TO_INR_RATE=%s; "
                    "falling back to 1.0"
                ),
                conversion_rate,
            )
            conversion_rate = 1.0

        revenue_inr = revenue * conversion_rate
        revenue_crore = revenue_inr / 10_000_000

        return round(revenue_crore, 2)

    def _select_revenue_tiers(
        self,
        company: ApolloCompany,
    ) -> list[str]:
        """
        Select the employee-title tier from company revenue.

        Revenue < 250 crore:
            Tier 1

        Revenue 250-1000 crore inclusive:
            Tier 2

        Revenue > 1000 crore:
            Tier 3

        Missing, zero, or invalid revenue:
            Tier 1 + Tier 2 + Tier 3
        """

        revenue_crore = self._revenue_to_crore(company)

        if revenue_crore is None:
            LOGGER.info(
                (
                    "Apollo revenue unavailable "
                    "company=%s raw_revenue=%s "
                    "routing=tier_1+tier_2+tier_3"
                ),
                company.name,
                company.annual_revenue,
            )
            return [
                "tier_1",
                "tier_2",
                "tier_3",
            ]

        if revenue_crore < TIER_1_MAX_REVENUE_CRORE:
            LOGGER.info(
                (
                    "Apollo revenue routing "
                    "company=%s revenue_crore=%s "
                    "routing=tier_1"
                ),
                company.name,
                revenue_crore,
            )
            return ["tier_1"]

        if revenue_crore <= TIER_2_MAX_REVENUE_CRORE:
            LOGGER.info(
                (
                    "Apollo revenue routing "
                    "company=%s revenue_crore=%s "
                    "routing=tier_2"
                ),
                company.name,
                revenue_crore,
            )
            return ["tier_2"]

        LOGGER.info(
            (
                "Apollo revenue routing "
                "company=%s revenue_crore=%s "
                "routing=tier_3"
            ),
            company.name,
            revenue_crore,
        )
        return ["tier_3"]

    def search_talent_officials(
        self,
        company: ApolloCompany,
        credit_usage: ApolloCreditUsage | None = None,
    ) -> list[ApolloOfficial]:
        """
        Return at most three Apollo officials for a company.

        Selection rules:
        - Route the company to the revenue tier first.
        - Search Apollo once per selected tier using all configured title aliases.
        - Respect the title priority order defined in *_TITLE_GROUPS.
        - Select at most one person from each priority row, then move to the next.
        - Stop after three candidate people have been selected.
        - Perform at most three People Enrichment calls for the company.

        This keeps the result capped at three people and prevents the previous
        behaviour of enriching every candidate returned by Apollo.
        """

        selected_tiers = self._select_revenue_tiers(company)
        contact_limit = self.max_contacts_per_company

        tier_config: dict[
            str,
            tuple[list[str], list[tuple[str, ...]]],
        ] = {
            "tier_1": (
                TIER_1_TITLES,
                TIER_1_TITLE_GROUPS,
            ),
            "tier_2": (
                TIER_2_TITLES,
                TIER_2_TITLE_GROUPS,
            ),
            "tier_3": (
                TIER_3_TITLES,
                TIER_3_TITLE_GROUPS,
            ),
        }

        people_by_tier: dict[str, list[dict[str, Any]]] = {}

        # Search once for each selected tier. Normally only one tier is selected.
        # Revenue-missing companies retain the existing fallback of checking all
        # three tiers, but person enrichment is still capped at three total.
        for tier_name in selected_tiers:
            titles, _ = tier_config[tier_name]

            people = self._search_people(
                company=company,
                titles=titles,
                per_page=50,
                credit_usage=credit_usage,
            )

            people_by_tier[tier_name] = people

            LOGGER.info(
                (
                    "Apollo revenue contact tier completed "
                    "company=%s tier=%s people=%s"
                ),
                company.name,
                tier_name,
                len(people),
            )

        # Pick candidates strictly by configured title priority. Only one person
        # is chosen from each priority group; then selection advances to the next
        # title group until three unique people have been collected.
        priority_candidates: list[dict[str, Any]] = []
        selected_candidate_ids: set[str] = set()

        for tier_name in selected_tiers:
            _, title_groups = tier_config[tier_name]
            people = people_by_tier.get(tier_name, [])

            for priority_number, aliases in enumerate(
                title_groups,
                start=1,
            ):
                if len(priority_candidates) >= contact_limit:
                    break

                matching_people: list[dict[str, Any]] = []

                for person in people:
                    person_id = str(
                        person.get("id")
                        or person.get("person_id")
                        or ""
                    ).strip()

                    if not person_id:
                        continue

                    if person_id in selected_candidate_ids:
                        continue

                    if not self._title_matches_priority_group(
                        returned_title=str(
                            person.get("title")
                            or ""
                        ),
                        aliases=aliases,
                    ):
                        continue

                    candidate = dict(person)
                    candidate["_search_tier"] = tier_name
                    candidate["_title_priority"] = priority_number
                    matching_people.append(candidate)

                if not matching_people:
                    continue

                # If Apollo returns multiple people for the same priority title,
                # choose the strongest candidate and then continue to the next
                # title row instead of consuming all three slots on one title.
                best_candidate = max(
                    matching_people,
                    key=self._person_score,
                )

                person_id = str(
                    best_candidate.get("id")
                    or best_candidate.get("person_id")
                    or ""
                ).strip()

                selected_candidate_ids.add(person_id)
                priority_candidates.append(best_candidate)

                LOGGER.info(
                    (
                        "Apollo priority candidate selected "
                        "company=%s tier=%s priority=%s aliases=%s "
                        "person_id=%s title=%s selected=%s/%s"
                    ),
                    company.name,
                    tier_name,
                    priority_number,
                    " / ".join(aliases),
                    person_id,
                    best_candidate.get("title", ""),
                    len(priority_candidates),
                    contact_limit,
                )

            if len(priority_candidates) >= contact_limit:
                break

        if not priority_candidates:
            LOGGER.info(
                (
                    "No Apollo contacts matched configured priority titles "
                    "company=%s selected_tiers=%s"
                ),
                company.name,
                selected_tiers,
            )
            return []

        officials: list[ApolloOfficial] = []
        seen_emails: set[str] = set()
        seen_linkedin_urls: set[str] = set()

        # Only the three selected priority candidates are enriched. Therefore a
        # company can consume at most three People Enrichment calls here.
        for attempt_number, person in enumerate(
            priority_candidates[:contact_limit],
            start=1,
        ):
            person_id = str(
                person.get("id")
                or person.get("person_id")
                or ""
            ).strip()

            LOGGER.info(
                (
                    "Enriching Apollo priority contact "
                    "company=%s person_id=%s title=%s tier=%s "
                    "priority=%s attempt=%s/%s"
                ),
                company.name,
                person_id,
                person.get("title", ""),
                person.get("_search_tier", ""),
                person.get("_title_priority", ""),
                attempt_number,
                contact_limit,
            )

            try:
                official = self.enrich_person(
                    person_id=person_id,
                    company_name=company.name,
                    credit_usage=credit_usage,
                )
            except ApolloApiError as exc:
                LOGGER.warning(
                    (
                        "Apollo person enrichment failed "
                        "company=%s person_id=%s status=%s error=%s"
                    ),
                    company.name,
                    person_id,
                    exc.status_code,
                    exc,
                )
                continue
            except httpx.HTTPError as exc:
                LOGGER.warning(
                    (
                        "Apollo person connection failed "
                        "company=%s person_id=%s error=%s"
                    ),
                    company.name,
                    person_id,
                    exc,
                )
                continue

            if not official.matched:
                continue

            email_key = official.work_email.strip().casefold()
            linkedin_key = official.linkedin_url.strip().casefold()

            if email_key and email_key in seen_emails:
                continue

            if linkedin_key and linkedin_key in seen_linkedin_urls:
                continue

            if not email_key and not linkedin_key:
                LOGGER.info(
                    (
                        "Skipping Apollo official without contact method "
                        "company=%s name=%s title=%s"
                    ),
                    company.name,
                    official.full_name,
                    official.title,
                )
                continue

            if email_key:
                seen_emails.add(email_key)

            if linkedin_key:
                seen_linkedin_urls.add(linkedin_key)

            officials.append(official)

            LOGGER.info(
                (
                    "Selected Apollo official "
                    "company=%s number=%s name=%s title=%s "
                    "email_available=%s tier=%s priority=%s"
                ),
                company.name,
                len(officials),
                official.full_name,
                official.title,
                bool(official.work_email),
                person.get("_search_tier", ""),
                person.get("_title_priority", ""),
            )

        LOGGER.info(
            (
                "Apollo priority official search completed "
                "company=%s raw_revenue=%s revenue_crore=%s "
                "selected_tiers=%s priority_candidates=%s "
                "enrichment_attempts=%s officials=%s limit=%s"
            ),
            company.name,
            company.annual_revenue,
            self._revenue_to_crore(company),
            selected_tiers,
            len(priority_candidates),
            min(len(priority_candidates), contact_limit),
            len(officials),
            contact_limit,
        )

        return officials

    @staticmethod
    def _title_matches_priority_group(
        returned_title: str,
        aliases: tuple[str, ...],
    ) -> bool:
        """Match an Apollo title against one configured priority group."""

        normalized_returned = re.sub(
            r"[^a-z0-9]+",
            " ",
            (returned_title or "").casefold(),
        ).strip()

        if not normalized_returned:
            return False

        padded_returned = f" {normalized_returned} "

        for alias in aliases:
            normalized_alias = re.sub(
                r"[^a-z0-9]+",
                " ",
                alias.casefold(),
            ).strip()

            if not normalized_alias:
                continue

            if normalized_returned == normalized_alias:
                return True

            # Whole phrase/token match. The padding avoids short aliases such as
            # MD, CIO, CFO, CTO and COO matching inside unrelated words.
            if f" {normalized_alias} " in padded_returned:
                return True

        return False

    # ========================================================================
    # People Search
    # ========================================================================

    def _search_people(
        self,
        company: ApolloCompany,
        titles: list[str],
        per_page: int,
        credit_usage: ApolloCreditUsage | None = None,
    ) -> list[dict[str, Any]]:
        """Search Apollo people by organization and title only."""

        params: list[tuple[str, str]] = [
            ("page", "1"),
            (
                "per_page",
                str(max(1, min(per_page, 100))),
            ),
            ("include_similar_titles", "true"),
        ]

        for title in titles:
            cleaned = title.strip()
            if cleaned:
                params.append(
                    ("person_titles[]", cleaned)
                )

        if company.apollo_id:
            params.append(
                (
                    "organization_ids[]",
                    company.apollo_id,
                )
            )
        elif company.domain:
            params.append(
                (
                    "q_organization_domains_list[]",
                    company.domain,
                )
            )
        else:
            LOGGER.warning(
                (
                    "Skipping Apollo people search: "
                    "company has no Apollo ID/domain "
                    "company=%s"
                ),
                company.name,
            )
            return []

        LOGGER.info(
            (
                "Searching Apollo officials "
                "company=%s organization_id=%s "
                "domain=%s title_count=%s"
            ),
            company.name,
            company.apollo_id,
            company.domain,
            len(titles),
        )

        if credit_usage is not None:
            credit_usage.people_search_calls += 1

        response = self.client.post(
            "/mixed_people/api_search",
            params=params,
            json={},
        )

        data = self._read_response(
            response=response,
            operation="people search",
        )

        people = data.get("people") or []

        if not isinstance(people, list):
            return []

        results = [
            person
            for person in people
            if isinstance(person, dict)
        ]

        LOGGER.info(
            (
                "Apollo people search returned "
                "company=%s people=%s"
            ),
            company.name,
            len(results),
        )

        return results

    # ========================================================================
    # People Enrichment
    # ========================================================================

    def enrich_person(
        self,
        person_id: str,
        company_name: str = "",
        credit_usage: ApolloCreditUsage | None = None,
    ) -> ApolloOfficial:
        person_id = (
            person_id
            or ""
        ).strip()

        if not person_id:
            return ApolloOfficial(
                matched=False
            )

        LOGGER.info(
            (
                "Calling Apollo people enrichment "
                "person_id=%s"
            ),
            person_id,
        )

        # Waterfall and phone enrichment are disabled.
        # With this configuration, a matched person is estimated at 1 credit
        # under Apollo's current standard credit rules; People Search itself
        # is 0 credits.
        if credit_usage is not None:
            credit_usage.people_enrichment_calls += 1

        response = self.client.post(
            "/people/match",
            params={
                "id": person_id,
                "reveal_personal_emails": (
                    "false"
                ),
                "reveal_phone_number": (
                    "false"
                ),
                "run_waterfall_email": (
                    "false"
                ),
                "run_waterfall_phone": (
                    "false"
                ),
            },
            json={},
        )

        data = self._read_response(
            response=response,
            operation="people enrichment",
        )

        person = (
            data.get("person")
            or data.get("contact")
            or {}
        )

        if (
            not isinstance(
                person,
                dict,
            )
            or not person
        ):
            return ApolloOfficial(
                matched=False,
                apollo_person_id=(
                    person_id
                ),
                raw_data=data,
            )

        if credit_usage is not None:
            credit_usage.people_enrichment_matches += 1
            credit_usage.people_enrichment_credits_estimated += 1

        organization = (
            person.get("organization")
            or {}
        )

        if not isinstance(
            organization,
            dict,
        ):
            organization = {}

        first_name = str(
            person.get("first_name")
            or ""
        ).strip()

        last_name = str(
            person.get("last_name")
            or person.get(
                "last_name_obfuscated"
            )
            or ""
        ).strip()

        full_name = str(
            person.get("name")
            or " ".join(
                part
                for part in (
                    first_name,
                    last_name,
                )
                if part
            )
        ).strip()

        work_email = (
            self._extract_work_email(
                person
            )
        )

        official = ApolloOfficial(
            matched=True,

            apollo_person_id=str(
                person.get("id")
                or person_id
            ),

            full_name=full_name,
            first_name=first_name,
            last_name=last_name,

            title=str(
                person.get("title")
                or ""
            ).strip(),

            seniority=str(
                person.get("seniority")
                or ""
            ).strip(),

            work_email=work_email,

            email_status=str(
                person.get(
                    "email_status"
                )
                or ""
            ).strip(),

            linkedin_url=str(
                person.get(
                    "linkedin_url"
                )
                or ""
            ).strip(),

            city=str(
                person.get("city")
                or ""
            ).strip(),

            state=str(
                person.get("state")
                or ""
            ).strip(),

            country=str(
                person.get("country")
                or ""
            ).strip(),

            organization_name=str(
                organization.get("name")
                or ""
            ).strip(),

            raw_data=person,
        )

        LOGGER.info(
            (
                "Apollo person enriched "
                "person_id=%s "
                "name=%s "
                "title=%s "
                "email_available=%s "
                "email_status=%s"
            ),
            person_id,
            official.full_name,
            official.title,
            bool(
                official.work_email
            ),
            official.email_status,
        )

        return official

    # ========================================================================
    # Ranking
    # ========================================================================

    @staticmethod
    def _person_score(
        person: dict[str, Any],
    ) -> int:
        """
        Rank people returned from the selected revenue tier.

        No seniority filter or seniority score is used.
        Work-email availability is the strongest preference.
        """

        title = str(
            person.get("title")
            or ""
        ).strip().casefold()

        tier = str(
            person.get("_search_tier")
            or ""
        ).strip().casefold()

        score = 0

        # Prefer candidates whose returned title closely matches one of the
        # configured titles for the selected revenue tier. Earlier configured
        # titles receive a slightly higher score.
        tier_titles = {
            "tier_1": TIER_1_TITLES,
            "tier_2": TIER_2_TITLES,
            "tier_3": TIER_3_TITLES,
        }.get(tier, [])

        normalized_title = re.sub(
            r"[^a-z0-9]+",
            " ",
            title,
        ).strip()

        for index, configured_title in enumerate(tier_titles):
            configured = re.sub(
                r"[^a-z0-9]+",
                " ",
                configured_title.casefold(),
            ).strip()

            if not configured:
                continue

            if normalized_title == configured:
                score += max(80 - index, 40)
                break

            if (
                configured in normalized_title
                or normalized_title in configured
            ):
                score += max(55 - index, 25)
                break

        if person.get("has_email") is True:
            score += 100

        email_status = str(
            person.get("email_status")
            or ""
        ).strip().casefold()

        if email_status == "verified":
            score += 50
        elif email_status in {
            "likely to engage",
            "likely_to_engage",
        }:
            score += 25

        if person.get("linkedin_url"):
            score += 10

        return score

    # ========================================================================
    # Company cache key
    # ========================================================================

    def build_company_cache_key(
        self,
        company_name: str,
        known_email: str = "",
    ) -> str:
        domain = self.domain_from_email(
            known_email
        )

        if domain:
            return f"domain:{domain}"

        company = self.clean_company_name(
            company_name
        ).casefold()

        company = re.sub(
            r"[^a-z0-9]+",
            "",
            company,
        )

        if company:
            return f"name:{company}"

        return ""

    # ========================================================================
    # HTTP response handling
    # ========================================================================

    @staticmethod
    def _read_response(
        response: httpx.Response,
        operation: str,
    ) -> dict[str, Any]:
        status_code = (
            response.status_code
        )

        response_body = (
            response.text
            or ""
        ).strip()

        if status_code == 401:
            raise ApolloApiError(
                (
                    "Apollo rejected the API key. "
                    f"Response: "
                    f"{response_body[:500]}"
                ),
                status_code=status_code,
                response_body=response_body,
            )

        if status_code == 403:
            raise ApolloApiError(
                (
                    f"Apollo denied {operation} "
                    "with HTTP 403. "
                    f"Response: "
                    f"{response_body[:500]}"
                ),
                status_code=status_code,
                response_body=response_body,
            )

        if status_code == 404:
            raise ApolloApiError(
                (
                    f"Apollo {operation} endpoint "
                    "returned HTTP 404. "
                    f"Response: "
                    f"{response_body[:500]}"
                ),
                status_code=status_code,
                response_body=response_body,
            )

        if status_code == 422:
            raise ApolloApiError(
                (
                    f"Apollo rejected {operation}. "
                    f"Response: "
                    f"{response_body[:500]}"
                ),
                status_code=status_code,
                response_body=response_body,
            )

        if status_code == 429:
            raise ApolloApiError(
                (
                    "Apollo API rate limit reached. "
                    f"Response: "
                    f"{response_body[:500]}"
                ),
                status_code=status_code,
                response_body=response_body,
            )

        if status_code >= 400:
            raise ApolloApiError(
                (
                    f"Apollo {operation} failed "
                    f"with HTTP {status_code}. "
                    f"Response: "
                    f"{response_body[:500]}"
                ),
                status_code=status_code,
                response_body=response_body,
            )

        try:
            data = response.json()

        except ValueError as exc:
            raise ApolloApiError(
                (
                    "Apollo returned invalid JSON "
                    f"for {operation}. "
                    f"Response: "
                    f"{response_body[:500]}"
                ),
                status_code=status_code,
                response_body=response_body,
            ) from exc

        if not isinstance(
            data,
            dict,
        ):
            raise ApolloApiError(
                (
                    "Apollo returned an unexpected "
                    f"response type for {operation}."
                ),
                status_code=status_code,
                response_body=response_body,
            )

        return data

    # ========================================================================
    # Text/domain helpers
    # ========================================================================

    @staticmethod
    def clean_company_name(
        value: str,
    ) -> str:
        value = re.sub(
            r"\s+",
            " ",
            value or "",
        ).strip(
            " -|,.;:"
        )

        if (
            value.casefold()
            in INVALID_COMPANY_VALUES
        ):
            return ""

        return value

    @staticmethod
    def domain_from_email(
        email: str,
    ) -> str:
        email = (
            email
            or ""
        ).strip().casefold()

        if "@" not in email:
            return ""

        domain = email.rsplit(
            "@",
            1,
        )[1]

        domain = (
            ApolloService.normalize_domain(
                domain
            )
        )

        if domain in FREE_EMAIL_DOMAINS:
            return ""

        return domain

    @staticmethod
    def normalize_domain(
        value: str,
    ) -> str:
        value = (
            value
            or ""
        ).strip().casefold()

        if not value:
            return ""

        if "://" in value:
            parsed = urlparse(
                value
            )

            value = parsed.netloc

        value = value.split(
            "/",
            1,
        )[0]

        value = value.removeprefix(
            "www."
        )

        value = value.lstrip("@")

        return value.strip(" .")

    # ========================================================================
    # Work email extraction
    # ========================================================================

    @staticmethod
    def _extract_work_email(
        person: dict[str, Any],
    ) -> str:
        email = str(
            person.get("email")
            or ""
        ).strip()

        if email:
            return email

        email_entries = (
            person.get(
                "email_addresses"
            )
            or person.get("emails")
            or []
        )

        if not isinstance(
            email_entries,
            list,
        ):
            return ""

        # Prefer explicitly work/business emails first.
        preferred: list[str] = []
        fallback: list[str] = []

        for item in email_entries:
            if isinstance(
                item,
                str,
            ):
                candidate = item.strip()

                if candidate:
                    fallback.append(
                        candidate
                    )

                continue

            if not isinstance(
                item,
                dict,
            ):
                continue

            candidate = str(
                item.get("email")
                or item.get("value")
                or ""
            ).strip()

            if not candidate:
                continue

            email_type = str(
                item.get("type")
                or ""
            ).strip().casefold()

            if email_type in {
                "work",
                "professional",
                "business",
            }:
                preferred.append(
                    candidate
                )

            elif not email_type:
                fallback.append(
                    candidate
                )

        if preferred:
            return preferred[0]

        if fallback:
            return fallback[0]

        return ""

    # ========================================================================
    # Company phone
    # ========================================================================

    @staticmethod
    def _extract_company_phone(
        organization: dict[str, Any],
    ) -> str:
        direct_phone = str(
            organization.get("phone")
            or ""
        ).strip()

        if direct_phone:
            return direct_phone

        primary_phone = (
            organization.get(
                "primary_phone"
            )
        )

        if isinstance(
            primary_phone,
            dict,
        ):
            value = str(
                primary_phone.get(
                    "sanitized_number"
                )
                or primary_phone.get(
                    "number"
                )
                or ""
            ).strip()

            if value:
                return value

        phone_numbers = (
            organization.get(
                "phone_numbers"
            )
        )

        if isinstance(
            phone_numbers,
            list,
        ):
            for phone in phone_numbers:
                if isinstance(
                    phone,
                    str,
                ):
                    value = phone.strip()

                    if value:
                        return value

                elif isinstance(
                    phone,
                    dict,
                ):
                    value = str(
                        phone.get(
                            "sanitized_number"
                        )
                        or phone.get(
                            "number"
                        )
                        or ""
                    ).strip()

                    if value:
                        return value

        return ""

    # ========================================================================
    # Safe conversions
    # ========================================================================

    @staticmethod
    def _safe_int(
        value: Any,
    ) -> int | None:
        try:
            if value in (
                None,
                "",
            ):
                return None

            if isinstance(
                value,
                str,
            ):
                value = (
                    value
                    .replace(",", "")
                    .strip()
                )

            return int(
                float(value)
            )

        except (
            TypeError,
            ValueError,
        ):
            return None

    @staticmethod
    def _safe_float(
        value: Any,
    ) -> float | None:
        try:
            if value in (
                None,
                "",
            ):
                return None

            if isinstance(
                value,
                str,
            ):
                value = (
                    value
                    .replace("$", "")
                    .replace(",", "")
                    .strip()
                )

            return float(value)

        except (
            TypeError,
            ValueError,
        ):
            return None

    # ========================================================================
    # Environment helpers
    # ========================================================================

    @staticmethod
    def _env_int(
        name: str,
        fallback: int,
    ) -> int:
        try:
            return int(
                os.getenv(
                    name,
                    str(fallback),
                )
            )

        except ValueError:
            return fallback

    @staticmethod
    def _env_float(
        name: str,
        fallback: float,
    ) -> float:
        try:
            return float(
                os.getenv(
                    name,
                    str(fallback),
                )
            )

        except ValueError:
            return fallback



