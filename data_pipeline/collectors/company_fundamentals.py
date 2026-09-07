import yfinance as yf
import os
import json
import psycopg2
from dotenv import load_dotenv
from datetime import date


# Load environment variables
load_dotenv()


# --------------------------------------------------
# Load 100-stock universe
# --------------------------------------------------

config_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "config",
    "stocks_100.json"
)

with open(config_path, "r") as file:
    stock_config = json.load(file)

stocks = stock_config["stocks"]

print(f"Total stocks to process: {len(stocks)}")


# --------------------------------------------------
# Connect to PostgreSQL
# --------------------------------------------------

connection = psycopg2.connect(
    host="localhost",
    port=5432,
    database="investment_copilot",
    user="postgres",
    password=os.getenv("DB_PASSWORD")
)

cursor = connection.cursor()


# --------------------------------------------------
# Collect fundamentals
# --------------------------------------------------

success_count = 0
failed_stocks = []

report_date = date.today()


for ticker_symbol in stocks:

    print(f"\nCollecting fundamentals for {ticker_symbol}...")

    try:

        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info

        # ------------------------------------------
        # Fundamental metrics
        # ------------------------------------------

        revenue = info.get("totalRevenue")
        net_income = info.get("netIncomeToCommon")
        eps = info.get("trailingEps")

        pe_ratio = info.get("trailingPE")
        pb_ratio = info.get("priceToBook")

        roe = info.get("returnOnEquity")

        # ROCE is not currently collected from Yahoo Finance
        roce = None

        debt = info.get("totalDebt")
        debt_to_equity = info.get("debtToEquity")


        # ------------------------------------------
        # Save to PostgreSQL
        # ------------------------------------------

        cursor.execute(
            """
            INSERT INTO company_fundamentals
            (
                ticker,
                report_date,
                revenue,
                net_income,
                eps,
                pe_ratio,
                pb_ratio,
                roe,
                roce,
                debt,
                debt_to_equity
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticker, report_date)
            DO UPDATE SET
                revenue = EXCLUDED.revenue,
                net_income = EXCLUDED.net_income,
                eps = EXCLUDED.eps,
                pe_ratio = EXCLUDED.pe_ratio,
                pb_ratio = EXCLUDED.pb_ratio,
                roe = EXCLUDED.roe,
                roce = EXCLUDED.roce,
                debt = EXCLUDED.debt,
                debt_to_equity = EXCLUDED.debt_to_equity;
            """,
            (
                ticker_symbol,
                report_date,
                revenue,
                net_income,
                eps,
                pe_ratio,
                pb_ratio,
                roe,
                roce,
                debt,
                debt_to_equity
            )
        )

        connection.commit()

        success_count += 1

        print(f"Saved {ticker_symbol}")

    except Exception as e:

        connection.rollback()

        failed_stocks.append(
            (ticker_symbol, str(e))
        )

        print(f"FAILED {ticker_symbol}: {e}")


# --------------------------------------------------
# Close database connection
# --------------------------------------------------

cursor.close()
connection.close()


# --------------------------------------------------
# Final report
# --------------------------------------------------

print("\n" + "=" * 50)
print("FUNDAMENTALS COLLECTION COMPLETED")
print("=" * 50)

print(f"Total stocks : {len(stocks)}")
print(f"Successful   : {success_count}")
print(f"Failed       : {len(failed_stocks)}")


if failed_stocks:

    print("\nFailed stocks:")

    for ticker, error in failed_stocks:
        print(f"- {ticker}: {error}")
else:

    print("\nAll stocks were collected successfully!")