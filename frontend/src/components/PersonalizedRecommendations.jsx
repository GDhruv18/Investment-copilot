
import { useState } from "react";
import { getPersonalizedRecommendations } from "../services/recommendationService";
import "./PersonalizedRecommendations.css";

function PersonalizedRecommendations() {
  const [investmentAmount, setInvestmentAmount] = useState("100000");
  const [riskProfile, setRiskProfile] = useState("moderate");
  const [horizon, setHorizon] = useState("1y");
  const [assetClasses, setAssetClasses] = useState([
    "stocks",
    "etfs",
    "mutual_funds",
  ]);

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const toggleAssetClass = (asset) => {
    setAssetClasses((previous) =>
      previous.includes(asset)
        ? previous.filter((item) => item !== asset)
        : [...previous, asset]
    );
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    setError("");
    setResult(null);

    if (assetClasses.length === 0) {
      setError("Please select at least one asset class.");
      return;
    }

    if (!investmentAmount || Number(investmentAmount) <= 0) {
      setError("Please enter a valid investment amount.");
      return;
    }

    setLoading(true);

    try {
      const data = await getPersonalizedRecommendations({
        investment_amount: Number(investmentAmount),
        risk_profile: riskProfile,
        horizon,
        asset_classes: assetClasses,
      });

      setResult(data);
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  const formatCurrency = (amount) =>
    new Intl.NumberFormat("en-IN", {
      style: "currency",
      currency: "INR",
      maximumFractionDigits: 2,
    }).format(amount);

  return (
    <div className="recommendation-page">
      <div className="recommendation-header">
        <div>
          <p className="page-eyebrow">INVESTMENT PLANNING</p>
          <h1>Personalized Recommendations</h1>
          <p>
            Build an investment portfolio aligned with your financial
            goals, risk appetite and investment horizon.
          </p>
        </div>
      </div>

      <form className="recommendation-form" onSubmit={handleSubmit}>
        <div className="form-section">
          <div className="form-section-heading">
            <span className="step-number">01</span>
            <div>
              <h2>Investment amount</h2>
              <p>How much would you like to invest?</p>
            </div>
          </div>

          <label className="field-label" htmlFor="investmentAmount">
            Total investment amount
          </label>

          <div className="amount-input-wrapper">
            <span>₹</span>
            <input
              id="investmentAmount"
              type="number"
              min="1"
              step="0.01"
              value={investmentAmount}
              onChange={(event) => setInvestmentAmount(event.target.value)}
              placeholder="Enter investment amount"
              required
            />
          </div>
        </div>

        <div className="form-divider" />

        <div className="form-section">
          <div className="form-section-heading">
            <span className="step-number">02</span>
            <div>
              <h2>Risk profile</h2>
              <p>Select the level of investment risk you are comfortable with.</p>
            </div>
          </div>

          <div className="option-grid">
            {[
              {
                value: "conservative",
                title: "Conservative",
                description: "Prioritize capital preservation and stability.",
              },
              {
                value: "moderate",
                title: "Moderate",
                description: "Balance potential returns with risk.",
              },
              {
                value: "aggressive",
                title: "Aggressive",
                description: "Accept higher risk for potential growth.",
              },
            ].map((option) => (
              <button
                type="button"
                key={option.value}
                className={`selection-card ${
                  riskProfile === option.value ? "selected" : ""
                }`}
                onClick={() => setRiskProfile(option.value)}
                aria-pressed={riskProfile === option.value}
              >
                <span className="selection-indicator" />
                <strong>{option.title}</strong>
                <span>{option.description}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="form-divider" />

        <div className="form-section">
          <div className="form-section-heading">
            <span className="step-number">03</span>
            <div>
              <h2>Investment horizon</h2>
              <p>How long do you plan to stay invested?</p>
            </div>
          </div>

          <div className="horizon-options">
            {[
              { value: "6m", label: "6 Months" },
              { value: "1y", label: "1 Year" },
              { value: "2y", label: "2 Years" },
            ].map((option) => (
              <button
                type="button"
                key={option.value}
                className={`horizon-button ${
                  horizon === option.value ? "selected" : ""
                }`}
                onClick={() => setHorizon(option.value)}
                aria-pressed={horizon === option.value}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>

        <div className="form-divider" />

        <div className="form-section">
          <div className="form-section-heading">
            <span className="step-number">04</span>
            <div>
              <h2>Asset classes</h2>
              <p>Select the investment categories to include.</p>
            </div>
          </div>

          <div className="asset-options">
            {[
              {
                value: "stocks",
                title: "Stocks",
                description: "Individual listed companies",
              },
              {
                value: "etfs",
                title: "ETFs",
                description: "Exchange-traded funds",
              },
              {
                value: "mutual_funds",
                title: "Mutual Funds",
                description: "Professionally managed funds",
              },
            ].map((asset) => (
              <label
                className={`asset-option ${
                  assetClasses.includes(asset.value) ? "selected" : ""
                }`}
                key={asset.value}
              >
                <input
                  type="checkbox"
                  checked={assetClasses.includes(asset.value)}
                  onChange={() => toggleAssetClass(asset.value)}
                />
                <span className="custom-checkbox" />
                <span className="asset-option-text">
                  <strong>{asset.title}</strong>
                  <small>{asset.description}</small>
                </span>
              </label>
            ))}
          </div>
        </div>

        {error && <div className="form-error">{error}</div>}

        <div className="form-actions">
          <p>
            Recommendations are generated using your selected investment
            preferences.
          </p>

          <button
            type="submit"
            className="generate-button"
            disabled={loading}
          >
            {loading ? "Generating portfolio..." : "Generate Portfolio"}
            {!loading && <span>→</span>}
          </button>
        </div>
      </form>

      {result && (
        <section className="portfolio-results">
          <div className="results-heading">
            <div>
              <p className="page-eyebrow">YOUR INVESTMENT PLAN</p>
              <h2>Recommended Portfolio</h2>
            </div>
            <span className="result-profile">
              {result.risk_profile} · {result.horizon}
            </span>
          </div>

          <div className="summary-grid">
            <div className="summary-card">
              <span>Investment amount</span>
              <strong>{formatCurrency(result.investment_amount)}</strong>
            </div>
            <div className="summary-card">
              <span>Allocated amount</span>
              <strong>{formatCurrency(result.allocated_amount)}</strong>
            </div>
            <div className="summary-card">
              <span>Unallocated amount</span>
              <strong>{formatCurrency(result.unallocated_amount)}</strong>
            </div>
          </div>

          <div className="allocation-bar">
            <div
              className="allocation-filled"
              style={{
                width: `${
                  (result.allocated_amount / result.investment_amount) * 100
                }%`,
              }}
            />
          </div>

          <div className="results-table-wrapper">
            <table className="results-table">
              <thead>
                <tr>
                  <th>Asset</th>
                  <th>Category</th>
                  <th>Score</th>
                  <th>Allocation</th>
                  <th>Amount</th>
                </tr>
              </thead>
              <tbody>
                {result.recommendations.map((item) => (
                  <tr key={`${item.asset_class}-${item.ticker}`}>
                    <td>
                      <strong>{item.asset_name}</strong>
                      <small>{item.ticker}</small>
                    </td>
                    <td>{item.asset_class.replace("_", " ")}</td>
                    <td>{Number(item.final_score).toFixed(3)}</td>
                    <td>{item.allocation_percentage.toFixed(2)}%</td>
                    <td>{formatCurrency(item.allocation_amount)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}

export default PersonalizedRecommendations;
