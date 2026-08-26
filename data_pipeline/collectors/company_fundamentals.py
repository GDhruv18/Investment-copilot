import yfinance as yf
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()
from datetime import date


stocks = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS"
]


connection = psycopg2.connect(
    host="localhost",
    port=5432,
    database="investment_copilot",
    user="postgres",
    password=os.getenv("DB_PASSWORD")
)

cursor = connection.cursor()


for ticker_symbol in stocks:

    print(f"Collecting fundamentals for {ticker_symbol}...")

    ticker = yf.Ticker(ticker_symbol)
    info = ticker.info

    report_date = date.today()

    revenue = info.get("totalRevenue")
    net_income = info.get("netIncomeToCommon")
    eps = info.get("trailingEps")

    pe_ratio = info.get("trailingPE")
    pb_ratio = info.get("priceToBook")

    roe = info.get("returnOnEquity")
    roce = None

    debt = info.get("totalDebt")
    debt_to_equity = info.get("debtToEquity")

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

    print(f"Saved {ticker_symbol}")


connection.commit()

cursor.close()
connection.close()

print("\nFundamentals collection completed!")