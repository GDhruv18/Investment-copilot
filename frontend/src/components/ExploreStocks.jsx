
import { useEffect, useMemo, useState } from "react";
import {
  searchStocks,
  getStockDetails,
  getStockDailyData,
} from "../services/stockService";
import "./ExploreStocks.css";

const CURRENCY = new Intl.NumberFormat("en-IN", {
  style: "currency",
  currency: "INR",
  maximumFractionDigits: 2,
});

function formatLabel(key) {
  return key
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatValue(value) {
  if (value === null || value === undefined || value === "") {
    return "—";
  }

  if (typeof value === "number") {
    return Number.isFinite(value) ? value.toLocaleString("en-IN", {
      maximumFractionDigits: 2,
    }) : "—";
  }

  return String(value);
}

function DataGrid({ title, data }) {
  const entries = Object.entries(data || {}).filter(
    ([, value]) =>
      value !== null &&
      value !== undefined &&
      value !== "" &&
      typeof value !== "object"
  );

  return (
    <section className="stock-detail-card">
      <div className="detail-section-heading">
        <h2>{title}</h2>
        <span>{entries.length} metrics</span>
      </div>

      {entries.length === 0 ? (
        <p className="detail-unavailable">
          No {title.toLowerCase()} are currently available.
        </p>
      ) : (
        <div className="metrics-grid">
          {entries.map(([key, value]) => (
            <div className="metric-item" key={key}>
              <span>{formatLabel(key)}</span>
              <strong>{formatValue(value)}</strong>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function PriceChart({ history }) {
  const points = useMemo(() => {
    if (!Array.isArray(history)) return [];

    return history.filter(
      (item) =>
        Number.isFinite(Number(item.close)) &&
        item.date
    );
  }, [history]);

  if (points.length < 2) {
    return (
      <div className="chart-empty">
        Historical price data is not available for this period.
      </div>
    );
  }

  const width = 900;
  const height = 280;
  const padding = 30;

  const prices = points.map((item) => Number(item.close));
  const min = Math.min(...prices);
  const max = Math.max(...prices);
  const range = max - min || 1;

  const coordinates = prices.map((price, index) => {
    const x =
      padding +
      (index / (prices.length - 1)) * (width - padding * 2);

    const y =
      height -
      padding -
      ((price - min) / range) * (height - padding * 2);

    return { x, y };
  });

  const polyline = coordinates
    .map((point) => `${point.x},${point.y}`)
    .join(" ");

  const firstPrice = prices[0];
  const lastPrice = prices[prices.length - 1];
  const change = lastPrice - firstPrice;
  const changePercentage =
    firstPrice !== 0 ? (change / firstPrice) * 100 : 0;

  return (
    <div className="price-chart-container">
      <div className="chart-summary">
        <div>
          <span>Latest close</span>
          <strong>{CURRENCY.format(lastPrice)}</strong>
        </div>

        <div>
          <span>Period change</span>
          <strong className={change >= 0 ? "positive-value" : "negative-value"}>
            {change >= 0 ? "+" : ""}
            {changePercentage.toFixed(2)}%
          </strong>
        </div>

        <div>
          <span>Data points</span>
          <strong>{points.length}</strong>
        </div>
      </div>

      <div className="price-chart">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label="Historical stock closing price chart"
          preserveAspectRatio="none"
        >
          {[0, 1, 2, 3, 4].map((line) => {
            const y =
              padding +
              (line / 4) * (height - padding * 2);

            return (
              <line
                key={line}
                x1={padding}
                y1={y}
                x2={width - padding}
                y2={y}
                className="chart-grid-line"
              />
            );
          })}

          <polyline
            points={polyline}
            fill="none"
            className="chart-price-line"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </div>

      <div className="chart-axis-labels">
        <span>{points[0].date}</span>
        <span>{points[points.length - 1].date}</span>
      </div>

      <p className="chart-source">
        Source: Yahoo Finance · Daily closing prices
      </p>
    </div>
  );
}

function StockDetails({ stock, onBack }) {
  const [details, setDetails] = useState(null);
  const [history, setHistory] = useState([]);
  const [period, setPeriod] = useState("1y");
  const [loading, setLoading] = useState(true);
  const [chartLoading, setChartLoading] = useState(false);
  const [error, setError] = useState("");
  const [chartError, setChartError] = useState("");

  useEffect(() => {
    let active = true;

    async function loadDetails() {
      setLoading(true);
      setError("");

      try {
        const data = await getStockDetails(stock.internal_ticker, period);

        if (!active) return;

        setDetails(data);
        setHistory(data.historical_data?.history || []);
      } catch (err) {
        if (active) {
          setError(err.message || "Unable to load stock details.");
        }
      } finally {
        if (active) setLoading(false);
      }
    }

    loadDetails();

    return () => {
      active = false;
    };
  }, [stock.internal_ticker, period]);

  const company = details?.company || stock;
  const latest = details?.market_data?.latest;

  const latestPrice = latest?.close;
  const latestDate = latest?.timestamp
    ? new Date(latest.timestamp).toLocaleString("en-IN")
    : "Not available";

  const handlePeriodChange = async (nextPeriod) => {
    setPeriod(nextPeriod);
  };

  if (loading && !details) {
    return (
      <div className="stock-detail-page">
        <button className="back-button" onClick={onBack}>
          ← Back to stock directory
        </button>
        <div className="stock-message">
          <div className="stock-spinner"></div>
          <p>Loading stock information...</p>
        </div>
      </div>
    );
  }

  if (error && !details) {
    return (
      <div className="stock-detail-page">
        <button className="back-button" onClick={onBack}>
          ← Back to stock directory
        </button>
        <div className="stock-message stock-error">
          <h3>Unable to load stock details</h3>
          <p>{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="stock-detail-page">
      <button className="back-button" onClick={onBack}>
        ← Back to stock directory
      </button>

      <section className="stock-profile-header">
        <div>
          <p className="page-eyebrow">STOCK PROFILE</p>
          <h1>{company.display_name || company.company_name}</h1>
          <div className="stock-profile-meta">
            <span>{company.internal_ticker}</span>
            <span>{company.exchange || "Exchange not available"}</span>
            {company.yahoo_ticker && <span>{company.yahoo_ticker}</span>}
          </div>
        </div>

        <div className="profile-price">
          <span>Latest available price</span>
          <strong>
            {Number.isFinite(Number(latestPrice))
              ? CURRENCY.format(Number(latestPrice))
              : "—"}
          </strong>
          <small>{latestDate}</small>
        </div>
      </section>

      {details?.market_data?.available === false && (
        <div className="detail-notice">
          {details.market_data.message || "Latest market data is unavailable."}
        </div>
      )}

      <section className="stock-detail-card">
        <div className="detail-section-heading chart-heading">
          <div>
            <h2>Historical price performance</h2>
            <p>Daily OHLCV market history</p>
          </div>

          <div className="period-selector">
            {[
              ["6mo", "6M"],
              ["1y", "1Y"],
              ["2y", "2Y"],
            ].map(([value, label]) => (
              <button
                key={value}
                className={period === value ? "selected" : ""}
                onClick={() => handlePeriodChange(value)}
                disabled={chartLoading || loading}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        {loading ? (
          <div className="chart-empty">Updating historical data...</div>
        ) : (
          <PriceChart history={history} />
        )}

        {details?.historical_data?.available === false && (
          <p className="detail-unavailable">
            {details.historical_data.message}
          </p>
        )}
      </section>

      <DataGrid
        title="Technical indicators"
        data={details?.technical_features}
      />

      <DataGrid
        title="Company fundamentals"
        data={details?.fundamentals}
      />

      <section className="stock-detail-card">
        <div className="detail-section-heading">
          <h2>Latest market information</h2>
        </div>

        {latest ? (
          <div className="metrics-grid">
            {Object.entries(latest)
              .filter(([, value]) => value !== null && value !== undefined)
              .map(([key, value]) => (
                <div className="metric-item" key={key}>
                  <span>{formatLabel(key)}</span>
                  <strong>
                    {key === "open" ||
                    key === "high" ||
                    key === "low" ||
                    key === "close"
                      ? CURRENCY.format(Number(value))
                      : formatValue(value)}
                  </strong>
                </div>
              ))}
          </div>
        ) : (
          <p className="detail-unavailable">
            Latest market information is currently unavailable.
          </p>
        )}
      </section>
    </div>
  );
}

function ExploreStocks() {
  const [query, setQuery] = useState("");
  const [stocks, setStocks] = useState([]);
  const [selectedStock, setSelectedStock] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [hasSearched, setHasSearched] = useState(false);

  useEffect(() => {
    const trimmedQuery = query.trim();

    if (!trimmedQuery) {
      setStocks([]);
      setError("");
      setLoading(false);
      setHasSearched(false);
      return;
    }

    let active = true;

    const timer = setTimeout(async () => {
      setLoading(true);
      setError("");
      setHasSearched(true);

      try {
        const data = await searchStocks(trimmedQuery, 20);

        if (active) {
          setStocks(data.results || []);
        }
      } catch (err) {
        if (active) {
          setStocks([]);
          setError(err.message || "Unable to search stocks.");
        }
      } finally {
        if (active) setLoading(false);
      }
    }, 350);

    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [query]);

  const handleClear = () => {
    setQuery("");
    setStocks([]);
    setError("");
    setHasSearched(false);
  };

  if (selectedStock) {
    return (
      <StockDetails
        stock={selectedStock}
        onBack={() => setSelectedStock(null)}
      />
    );
  }

  return (
    <div className="explore-stocks-page">
      <div className="stocks-intro">
        <p className="page-eyebrow">MARKET RESEARCH</p>
        <h1>Explore Stocks</h1>
        <p>
          Discover listed companies and explore their market information,
          technical indicators, and historical performance.
        </p>
      </div>

      <section className="stock-search-section">
        <div className="stock-search-heading">
          <div>
            <h2>Find a company</h2>
            <p>Search using a company name or stock ticker.</p>
          </div>

          <span className="stock-count">
            {stocks.length} {stocks.length === 1 ? "result" : "results"}
          </span>
        </div>

        <div className="stock-search-box">
          <span className="search-icon">⌕</span>

          <input
            type="text"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search companies or tickers (e.g. Reliance, TCS)"
            aria-label="Search stocks"
          />

          {query && (
            <button
              type="button"
              className="clear-search-button"
              onClick={handleClear}
            >
              Clear
            </button>
          )}
        </div>

        <p className="search-hint">
          Try searching for Reliance, TCS, INFY or HDFCBANK.
        </p>
      </section>

      <section className="stock-results-section">
        <div className="results-heading">
          <div>
            <h2>Stock directory</h2>
            <p>Companies available in your research universe.</p>
          </div>
        </div>

        {loading && (
          <div className="stock-message">
            <div className="stock-spinner"></div>
            <p>Searching stocks...</p>
          </div>
        )}

        {!loading && error && (
          <div className="stock-message stock-error">
            <h3>Unable to load stocks</h3>
            <p>{error}</p>
          </div>
        )}

        {!loading && !error && !hasSearched && (
          <div className="stock-message">
            <span className="stock-message-icon">⌕</span>
            <h3>Search for a company</h3>
            <p>
              Enter a company name or ticker above to explore available stocks.
            </p>
          </div>
        )}

        {!loading && !error && hasSearched && stocks.length === 0 && (
          <div className="stock-message">
            <h3>No matching stocks found</h3>
            <p>Try a different company name or ticker symbol.</p>
          </div>
        )}

        {!loading && !error && stocks.length > 0 && (
          <div className="stocks-table-wrapper">
            <table className="stocks-table">
              <thead>
                <tr>
                  <th>COMPANY</th>
                  <th>COMPANY NAME</th>
                  <th>INTERNAL TICKER</th>
                  <th>ACTION</th>
                </tr>
              </thead>

              <tbody>
                {stocks.map((stock) => (
                  <tr key={stock.internal_ticker}>
                    <td>
                      <div className="stock-company-cell">
                        <span className="stock-avatar">
                          {(stock.display_name || stock.company_name || "?")
                            .charAt(0)
                            .toUpperCase()}
                        </span>
                        <strong>
                          {stock.display_name || stock.company_name}
                        </strong>
                      </div>
                    </td>

                    <td>{stock.company_name}</td>

                    <td>
                      <span className="ticker-label">
                        {stock.internal_ticker}
                      </span>
                    </td>

                    <td>
                      <button
                        type="button"
                        className="view-stock-button"
                        onClick={() => setSelectedStock(stock)}
                      >
                        View details →
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

export default ExploreStocks;
