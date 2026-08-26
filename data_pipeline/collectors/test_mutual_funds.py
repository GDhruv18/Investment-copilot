import requests
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()
from datetime import datetime


URL = "https://www.amfiindia.com/spages/NAVAll.txt"


response = requests.get(URL)
response.raise_for_status()

lines = response.text.splitlines()


connection = psycopg2.connect(
    host="localhost",
    port=5432,
    database="investment_copilot",
    user="postgres",
    password=os.getenv("DB_PASSWORD")
)

cursor = connection.cursor()


count = 0

for line in lines:

    parts = line.split(";")

    # A valid scheme row should have 9 fields
    if len(parts) != 8:
        continue

    scheme_code = parts[0].strip()
    isin_div_payout = parts[1].strip()
    isin_div_reinvestment = parts[2].strip()
    scheme_name = parts[3].strip()
    plan = parts[4].strip()
    option = parts[5].strip()
    nav = parts[6].strip()
    nav_date = parts[7].strip()

    # Skip non-data rows
    if not scheme_code.isdigit():
        continue

    if not scheme_name or not nav:
        continue

    try:
        nav_value = float(nav)
        date_value = datetime.strptime(
            nav_date,
            "%d-%b-%Y"
        ).date()
    except ValueError:
        continue

    # We store the basic scheme information
    cursor.execute(
        """
        INSERT INTO mutual_funds
        (
            scheme_code,
            scheme_name,
            inception_date
        )
        VALUES (%s, %s, %s)
        ON CONFLICT (scheme_code)
        DO UPDATE SET
            scheme_name = EXCLUDED.scheme_name;
        """,
        (
            scheme_code,
            scheme_name,
            None
        )
    )

    count += 1


connection.commit()

cursor.close()
connection.close()

print(f"Successfully processed {count} mutual fund records.")