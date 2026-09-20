import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}


def get_connection():
    missing = [k for k, v in DB_CONFIG.items() if not v]

    if missing:
        raise ValueError(
            f"Missing database environment variables: {', '.join(missing)}"
        )

    return psycopg2.connect(**DB_CONFIG)


def audit_table(cur, table_name, asset_type):
    query = f"""
        SELECT
            ticker,
            MIN(date) AS first_date,
            MAX(date) AS last_date,
            COUNT(*) AS total_rows
        FROM {table_name}
        GROUP BY ticker
        ORDER BY ticker;
    """

    cur.execute(query)
    rows = cur.fetchall()

    print()
    print("=" * 90)
    print(f"{asset_type.upper()} DATA AVAILABILITY")
    print("=" * 90)

    print(
        f"{'Ticker':<20}"
        f"{'First Date':<15}"
        f"{'Last Date':<15}"
        f"{'Rows':>10}"
    )

    print("-" * 90)

    for ticker, first_date, last_date, total_rows in rows:
        print(
            f"{ticker:<20}"
            f"{str(first_date):<15}"
            f"{str(last_date):<15}"
            f"{total_rows:>10}"
        )

    print("-" * 90)
    print(f"Total {asset_type.lower()}s: {len(rows)}")

    return rows


def horizon_summary(rows, asset_type):
    horizons = {
        "6M": 180,
        "1Y": 365,
        "2Y+": 730,
    }

    print()
    print("=" * 70)
    print(f"{asset_type.upper()} HORIZON AVAILABILITY")
    print("=" * 70)

    # We use the latest common date available in the table.
    latest_date = max(row[2] for row in rows)

    print(f"Reference date: {latest_date}")
    print()

    for name, days in horizons.items():
        required_start = latest_date.fromordinal(
            latest_date.toordinal() - days
        )

        eligible = [
            row for row in rows
            if row[1] <= required_start
        ]

        print(
            f"{name:<5} : "
            f"{len(eligible):>3} / {len(rows)} assets "
            f"have at least approximately {days} days of history"
        )


def main():
    conn = get_connection()

    try:
        cur = conn.cursor()

        stock_rows = audit_table(
            cur,
            "stock_prices",
            "Stocks"
        )

        etf_rows = audit_table(
            cur,
            "etf_prices",
            "ETFs"
        )

        horizon_summary(stock_rows, "Stocks")
        horizon_summary(etf_rows, "ETFs")

        cur.close()

    finally:
        conn.close()


if __name__ == "__main__":
    main()