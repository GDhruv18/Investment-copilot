import json
from pathlib import Path

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data_pipeline"
    / "config"
    / "etfs_42_validated.json"
)


connection = psycopg2.connect(
    host="localhost",
    port=5432,
    database="investment_copilot",
    user="postgres",
    password=os.getenv("DB_PASSWORD")
)

cursor = connection.cursor()


cursor.execute(
    """
    SELECT ticker
    FROM etf_prices
    GROUP BY ticker
    HAVING COUNT(*) >= 500
       OR ticker = 'METAL.NS'
    ORDER BY ticker;
    """
)

etfs = [row[0] for row in cursor.fetchall()]


cursor.close()
connection.close()


print(f"Final ETF universe: {len(etfs)} ETFs")


if len(etfs) != 42:
    raise ValueError(
        f"Expected 42 ETFs, but found {len(etfs)}"
    )


with open(OUTPUT_FILE, "w") as file:
    json.dump(
        {"etfs": etfs},
        file,
        indent=4
    )


print(f"Saved to: {OUTPUT_FILE}")

print("\nFinal ETFs:")
for ticker in etfs:
    print(ticker)