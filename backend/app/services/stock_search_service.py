
import json

from functools import lru_cache
from backend.app.core.config import PROJECT_ROOT


STOCK_MASTER_FILE = (
    PROJECT_ROOT / "backend" / "app" / "data" / "stock_master.json"
)


@lru_cache(maxsize=1)
def load_stock_master() -> list[dict]:
    """Load and cache the stock master dataset."""

    if not STOCK_MASTER_FILE.exists():
        raise FileNotFoundError("Stock master file not found.")

    with open(STOCK_MASTER_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data["stocks"]


def search_stocks(query: str, limit: int = 10) -> list[dict]:
    """Search stocks using company names or ticker symbols."""

    query = query.strip().casefold()

    if not query:
        return []

    stocks = load_stock_master()

    matches = []

    for stock in stocks:
        company_name = stock["company_name"].casefold()
        display_name = stock["display_name"].casefold()
        ticker = stock["internal_ticker"].casefold()

        if (
            query in display_name
            or query in company_name
            or query in ticker
        ):
            matches.append(stock)

    # Prioritize names that begin with the search query.
    matches.sort(
        key=lambda stock: (
            not stock["display_name"].casefold().startswith(query),
            stock["display_name"],
        )
    )

    return matches[:limit]
def resolve_stock_ticker(identifier: str) -> str:
    """
    Resolve a company name, display name, or ticker
    into the internal stock ticker.
    """

    identifier = identifier.strip().casefold()

    if not identifier:
        raise ValueError("Stock identifier cannot be empty.")

    stocks = load_stock_master()

    # First, look for an exact match.
    exact_matches = [
        stock
        for stock in stocks
        if (
            stock["display_name"].casefold() == identifier
            or stock["company_name"].casefold() == identifier
            or stock["internal_ticker"].casefold() == identifier
        )
    ]

    if len(exact_matches) == 1:
        return exact_matches[0]["internal_ticker"]

    if len(exact_matches) > 1:
        raise ValueError("Multiple stocks match this identifier.")

    # Then look for a partial match.
    partial_matches = [
        stock
        for stock in stocks
        if (
            identifier in stock["display_name"].casefold()
            or identifier in stock["company_name"].casefold()
            or identifier in stock["internal_ticker"].casefold()
        )
    ]

    if len(partial_matches) == 1:
        return partial_matches[0]["internal_ticker"]

    if len(partial_matches) > 1:
        raise ValueError(
            "Multiple stocks match this name. Please use a more specific name or ticker."
        )

    raise ValueError("Stock not found.")