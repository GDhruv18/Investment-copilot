import yfinance as yf
import os

ticker = "IDFCFIRSTB.NS"

df = yf.Ticker(ticker).history(
    start="2014-01-01",
    end="2026-09-20",
    auto_adjust=False
)

if df.empty:
    raise RuntimeError("No data returned")

df = df[
    ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
].dropna()

df.index.name = "Date"

output = "data/raw/stocks/IDFCFIRSTB_NS.parquet"

df.to_parquet(output)

print("=" * 60)
print("IDFCFIRSTB.NS REPLACED")
print("=" * 60)
print(f"Rows:  {len(df)}")
print(f"First: {df.index.min().date()}")
print(f"Last:  {df.index.max().date()}")
print(f"File:  {output}")