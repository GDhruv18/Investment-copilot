import requests
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()
from datetime import datetime
import time


connection = psycopg2.connect(
    host="localhost",
    port=5432,
    database="investment_copilot",
    user="postgres",
    password=os.getenv("DB_PASSWORD")
)

cursor = connection.cursor()


# Get all scheme codes from our database
cursor.execute("""
    SELECT scheme_code
    FROM investment_fund_universe;
""")
schemes = cursor.fetchall()

print(f"Found {len(schemes)} mutual funds.")


for index, (scheme_code,) in enumerate(schemes, start=1):

    print(f"[{index}/{len(schemes)}] Collecting NAV for {scheme_code}...")

    url = f"https://api.mfapi.in/mf/{scheme_code}"

    try:

        response = requests.get(url, timeout=15)

        if response.status_code != 200:
            print(f"Skipping {scheme_code}: HTTP {response.status_code}")
            continue

        data = response.json()

        for record in data.get("data", []):

            try:
                nav_date = datetime.strptime(
                    record["date"],
                    "%d-%m-%Y"
                ).date()

                nav = float(record["nav"])

                cursor.execute(
                    """
                    INSERT INTO mutual_fund_nav
                    (scheme_code, nav_date, nav)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (scheme_code, nav_date)
                    DO UPDATE SET nav = EXCLUDED.nav;
                    """,
                    (
                        scheme_code,
                        nav_date,
                        nav
                    )
                )

            except (ValueError, KeyError):
                continue

        connection.commit()

    except requests.RequestException as error:

        print(f"Error for {scheme_code}: {error}")
        continue

    # Avoid sending requests too quickly
    time.sleep(0.2)


cursor.close()
connection.close()

print("\nMutual fund NAV collection completed!")