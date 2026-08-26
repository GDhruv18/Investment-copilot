import requests
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()
import time


connection = psycopg2.connect(
    host="localhost",
    port=5432,
    database="investment_copilot",
    user="postgres",
    password=os.getenv("DB_PASSWORD")
)

cursor = connection.cursor()

cursor.execute("""
    SELECT scheme_code
    FROM mutual_funds
    WHERE category IS NULL
      AND plan = 'Direct Plan'
      AND LOWER(option) = 'growth'
    ORDER BY RANDOM()
    LIMIT 1500;
""")


schemes = cursor.fetchall()
print(f"Found {len(schemes)} mutual funds.")


for index, (scheme_code,) in enumerate(schemes, start=1):

    print(f"[{index}/{len(schemes)}] Updating {scheme_code}...")

    url = f"https://api.mfapi.in/mf/{scheme_code}"

    try:
        response = requests.get(url, timeout=15)

        if response.status_code != 200:
            print("Skipped:", response.status_code)
            continue

        data = response.json()
        meta = data.get("meta", {})

        fund_house = meta.get("fund_house")
        category = meta.get("scheme_category")

        cursor.execute(
            """
            UPDATE mutual_funds
            SET
                amc_name = %s,
                category = %s
            WHERE scheme_code = %s;
            """,
            (
                fund_house,
                category,
                scheme_code
            )
        )

        connection.commit()

    except requests.RequestException as error:
        print(f"Error: {error}")

    time.sleep(0.2)


cursor.close()
connection.close()

print("\nCategory update completed!")

