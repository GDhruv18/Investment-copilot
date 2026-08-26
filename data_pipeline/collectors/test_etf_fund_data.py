import yfinance as yf


ticker = "NIFTYBEES.NS"

print(f"Testing fund data for {ticker}...\n")

fund = yf.Ticker(ticker)

try:
    data = fund.funds_data

    print("Fund data object:")
    print(data)

    print("\nFund operations:")
    print(data.fund_operations)

    print("\nDescription:")
    print(data.description)

except Exception as error:
    print("ERROR:")
    print(error)