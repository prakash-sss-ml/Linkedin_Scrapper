from dotenv import load_dotenv

from app.services.apollo_service import (
    ApolloService,
)


load_dotenv()


def main() -> None:
    with ApolloService() as apollo:
        result = (
            apollo
            .enrich_company_and_find_talent(
                company_name="Microsoft",
                known_domain=(
                    "microsoft.com"
                ),
            )
        )

    print(
        "Error:",
        result.error,
    )

    print(
        "Match status:",
        result.company.match_status,
    )

    print(
        "Company:",
        result.company.name,
    )

    print(
        "Domain:",
        result.company.domain,
    )

    print(
        "Industry:",
        result.company.industry,
    )

    print(
        "Employees:",
        result.company.employee_count,
    )

    print(
        "Officials:",
        len(result.officials),
    )

    for official in result.officials:
        print("-" * 60)

        print(
            "Name:",
            official.full_name,
        )

        print(
            "Title:",
            official.title,
        )

        print(
            "Email:",
            official.work_email,
        )

        print(
            "Email status:",
            official.email_status,
        )

        print(
            "LinkedIn:",
            official.linkedin_url,
        )


if __name__ == "__main__":
    main()              
    