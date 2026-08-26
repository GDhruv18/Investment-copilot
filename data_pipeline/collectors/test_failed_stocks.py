import yfinance as yf

stocks = [
    "MARICO.NS",
    "HAVELLS.NS",
    "AMBUJACEM.NS"
]

for ticker in stocks:

    print("\n" + "=" * 50)
    print("Testing:", ticker)

    try:
        data = yf.download(
            ticker,
            period="5y",
            interval="1d",
            auto_adjust=False,
            progress=False
        )

        print("Rows:", len(data))

        if not data.empty:
            print("VALID:", ticker)
            print("First date:", data.index[0])
            print("Last date:", data.index[-1])
        else:
            print("NO DATA")

    except Exception as error:
        print("ERROR:", error)