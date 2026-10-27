from pathlib import Path
import json
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "data_pipeline/config/etfs_41_validated.json"
OUTPUT = ROOT / "data/staging/etfs_refresh"
START_DATE = "2014-01-01"

OUTPUT.mkdir(parents=True, exist_ok=True)

config = json.loads(CONFIG.read_text(encoding="utf-8"))
tickers = config["etfs"] if isinstance(config, dict) else config

if len(tickers) != 41:
    raise ValueError(f"Expected 41 validated ETFs, found {len(tickers)}")
if len(set(tickers)) != 41:
    raise ValueError("ETF configuration contains duplicate tickers")

print("ETF refresh staging")
print("Tickers:", len(tickers))
print("Output:", OUTPUT)

success, failed = 0, []
required = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

for i, ticker in enumerate(tickers, 1):
    print(f"[{i}/{len(tickers)}] {ticker}", flush=True)
    try:
        df = yf.download(
            ticker,
            start=START_DATE,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False
        )

        if df.empty:
            failed.append((ticker, "empty download"))
            print("  FAILED: empty download")
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        missing = [c for c in required if c not in df.columns]
        if missing:
            failed.append((ticker, f"missing columns {missing}"))
            print("  FAILED:", missing)
            continue

        df = df[required].copy()
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"
        df = df.dropna(subset=["Open", "High", "Low", "Close", "Adj Close"])
        df = df[~df.index.duplicated(keep="last")].sort_index()

        if df.empty:
            failed.append((ticker, "no valid rows"))
            continue

        name = ticker.replace(".", "_") + ".parquet"
        df.to_parquet(OUTPUT / name)
        success += 1
        print(f"  Saved {len(df):,} rows; latest={df.index.max().date()}")

    except Exception as exc:
        failed.append((ticker, repr(exc)))
        print("  FAILED:", repr(exc))

print("\n=== ETF REFRESH SUMMARY ===")
print("Configured:", len(tickers))
print("Successful:", success)
print("Failed:", len(failed))
for item in failed:
    print(" ", item)

if failed:
    raise SystemExit(1)
