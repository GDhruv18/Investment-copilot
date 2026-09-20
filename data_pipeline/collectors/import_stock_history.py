import os

import pandas as pd
import psycopg2
from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv()

RAW_DIR = "data/raw/stocks"

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
}


def validate_db_config():
    """Make sure required database credentials are available."""

    required = ["DB_NAME", "DB_USER", "DB_PASSWORD"]

    missing = [
        variable
        for variable in required
        if not os.getenv(variable)
    ]

    if missing:
        raise RuntimeError(
            f"Missing environment variables: {', '.join(missing)}"
        )


def normalize_columns(df):
    """Normalize Yahoo Finance column structure."""

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [
        str(column).strip().lower().replace(" ", "_")
        for column in df.columns
    ]

    return df


def import_stock_file(cursor, filepath, ticker):
    """Import one stock Parquet file into PostgreSQL."""

    print(f"Loading: {ticker}")

    df = pd.read_parquet(filepath)
    df = normalize_columns(df)

    # If date is stored as the index
    if "date" not in df.columns:
        df = df.reset_index()
        df = normalize_columns(df)

    required_columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        print(f"  ERROR: Missing columns: {missing}")
        return 0

    # Convert date
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # Remove rows with missing price values
    df = df.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
            "adj_close",
        ]
    )

    # Remove duplicate dates
    df = df.drop_duplicates(subset=["date"])

    rows_imported = 0

    for _, row in df.iterrows():

        volume = (
            int(row["volume"])
            if pd.notna(row["volume"])
            else None
        )

        cursor.execute(
            """
            INSERT INTO stock_prices
            (
                ticker,
                date,
                open,
                high,
                low,
                close,
                adj_close,
                volume
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)

            ON CONFLICT (ticker, date)
            DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                adj_close = EXCLUDED.adj_close,
                volume = EXCLUDED.volume;
            """,
            (
                ticker,
                row["date"],
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(row["adj_close"]),
                volume,
            ),
        )

        rows_imported += 1

    print(f"  Imported: {rows_imported} rows")

    return rows_imported


def main():

    validate_db_config()

    print("=" * 70)
    print("STOCK HISTORY IMPORT")
    print("=" * 70)

    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    total_rows = 0
    total_files = 0

    files = sorted(
        file
        for file in os.listdir(RAW_DIR)
        if file.endswith(".parquet")
    )

    print(f"Parquet files found: {len(files)}")
    print()

    try:

        for filename in files:

            # Example:
            # RELIANCE_NS.parquet → RELIANCE.NS
            ticker = (
                filename
                .replace(".parquet", "")
                .replace("_", ".")
            )

            filepath = os.path.join(
                RAW_DIR,
                filename
            )

            rows = import_stock_file(
                cursor,
                filepath,
                ticker
            )

            total_rows += rows
            total_files += 1

            # Commit after each file
            conn.commit()

        print()
        print("=" * 70)
        print("IMPORT COMPLETE")
        print("=" * 70)
        print(f"Files processed: {total_files}")
        print(f"Rows processed:  {total_rows}")

    except Exception as error:

        conn.rollback()

        print()
        print("=" * 70)
        print("IMPORT FAILED")
        print("=" * 70)
        print(error)

        raise

    finally:

        cursor.close()
        conn.close()


if __name__ == "__main__":
    main()