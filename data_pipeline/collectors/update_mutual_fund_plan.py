import requests
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()


URL = "https://www.amfiindia.com/spages/NAVAll.txt"

response = requests.get(URL, timeout=30)
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

    if len(parts) != 8:
        continue

    scheme_code = parts[0].strip()

    if not scheme_code.isdigit():
        continue

    scheme_name = parts[3].strip()
    plan = parts[4].strip()
    option = parts[5].strip()

    if not scheme_name:
        continue

    cursor.execute(
        """
        UPDATE mutual_funds
        SET
            plan = %s,
            option = %s
        WHERE scheme_code = %s;
        """,
        (
            plan,
            option,
            scheme_code
        )
    )

    if cursor.rowcount > 0:
        count += 1


connection.commit()

cursor.close()
connection.close()

print(f"Updated plan/option for {count} mutual funds.")