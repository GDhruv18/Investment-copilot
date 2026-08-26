import yfinance as yf

ticker = yf.Ticker("RELIANCE.NS")

info = ticker.info

print("Company:", info.get("longName"))
print("Sector:", info.get("sector"))
print("Industry:", info.get("industry"))
print("Market Cap:", info.get("marketCap"))
print("P/E Ratio:", info.get("trailingPE"))
print("P/B Ratio:", info.get("priceToBook"))
print("ROE:", info.get("returnOnEquity"))
print("Debt to Equity:", info.get("debtToEquity"))
print("Revenue:", info.get("totalRevenue"))
print("Net Income:", info.get("netIncomeToCommon"))
print("EPS:", info.get("trailingEps"))