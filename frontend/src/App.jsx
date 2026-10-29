
import { useState } from "react";
import "./App.css";

import PersonalizedRecommendations from "./components/PersonalizedRecommendations";
import ExploreStocks from "./components/ExploreStocks";

function App() {
  const [activePage, setActivePage] = useState("dashboard");

  const navigateTo = (page) => {
    setActivePage(page);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const pageTitles = {
    dashboard: "Dashboard",
    recommendations: "Recommendations",
    stocks: "Explore Stocks",
    etfs: "Explore ETFs",
    "mutual-funds": "Mutual Funds",
  };

  const currentPageTitle = pageTitles[activePage] || "Dashboard";

  return (
    <div className="app">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">IC</div>
          <div>
            <h2>Investment</h2>
            <span>Copilot</span>
          </div>
        </div>

        <div className="nav-section">
          <p className="nav-heading">WORKSPACE</p>

          <button
            className={`nav-item ${
              activePage === "dashboard" ? "active" : ""
            }`}
            onClick={() => navigateTo("dashboard")}
          >
            <span>▦</span> Dashboard
          </button>

          <button
            className={`nav-item ${
              activePage === "recommendations" ? "active" : ""
            }`}
            onClick={() => navigateTo("recommendations")}
          >
            <span>◈</span> Recommendations
          </button>

          <button
            className={`nav-item ${
              activePage === "stocks" ? "active" : ""
            }`}
            onClick={() => navigateTo("stocks")}
          >
            <span>↗</span> Explore Stocks
          </button>

          <button
            className={`nav-item ${
              activePage === "etfs" ? "active" : ""
            }`}
            onClick={() => navigateTo("etfs")}
          >
            <span>◫</span> Explore ETFs
          </button>

          <button
            className={`nav-item ${
              activePage === "mutual-funds" ? "active" : ""
            }`}
            onClick={() => navigateTo("mutual-funds")}
          >
            <span>▤</span> Mutual Funds
          </button>
        </div>

        <div className="sidebar-footer">
          <div className="user-avatar">DS</div>
          <div>
            <strong>Investor</strong>
            <span>Personal Account</span>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="main-content">
        {/* Header */}
        <header className="topbar">
          <div>
            <span className="breadcrumb">
              Workspace / {currentPageTitle}
            </span>

            <h1>{currentPageTitle}</h1>
          </div>

          <div className="market-status">
            <span className="status-dot"></span>
            Market insights
          </div>
        </header>

        {/* Dashboard */}
        {activePage === "dashboard" && (
          <>
            <section className="welcome-section">
              <div>
                <p className="eyebrow">INVESTMENT OVERVIEW</p>

                <h2>Make informed investment decisions.</h2>

                <p>
                  Discover investment opportunities aligned with your
                  financial goals and risk appetite.
                </p>
              </div>

              <button
                className="primary-button"
                onClick={() => navigateTo("recommendations")}
              >
                Build my portfolio <span>→</span>
              </button>
            </section>

            <section className="section">
              <div className="section-heading">
                <div>
                  <h2>Investment workspace</h2>
                  <p>Tools to help you plan your investments.</p>
                </div>
              </div>

              <div className="dashboard-grid">
                <div className="dashboard-card">
                  <div className="card-icon">↗</div>

                  <h3>Personalized recommendations</h3>

                  <p>
                    Get investment suggestions based on your risk profile,
                    investment horizon and budget.
                  </p>

                  <button
                    className="card-link"
                    onClick={() => navigateTo("recommendations")}
                  >
                    Explore recommendations →
                  </button>
                </div>

                <div className="dashboard-card">
                  <div className="card-icon">◫</div>

                  <h3>Explore stocks</h3>

                  <p>
                    Analyse stocks using quantitative signals and
                    machine-learning insights.
                  </p>

                  <button
                    className="card-link"
                    onClick={() => navigateTo("stocks")}
                  >
                    Explore stocks →
                  </button>
                </div>

                <div className="dashboard-card">
                  <div className="card-icon">◈</div>

                  <h3>Explore ETFs</h3>

                  <p>
                    Discover exchange-traded funds and compare available
                    investment opportunities.
                  </p>

                  <button
                    className="card-link"
                    onClick={() => navigateTo("etfs")}
                  >
                    Explore ETFs →
                  </button>
                </div>

                <div className="dashboard-card">
                  <div className="card-icon">▤</div>

                  <h3>Explore mutual funds</h3>

                  <p>
                    Discover mutual fund schemes and compare investment
                    opportunities.
                  </p>

                  <button
                    className="card-link"
                    onClick={() => navigateTo("mutual-funds")}
                  >
                    Explore mutual funds →
                  </button>
                </div>
              </div>
            </section>
          </>
        )}

        {/* Personalized recommendations */}
        {activePage === "recommendations" && (
          <PersonalizedRecommendations />
        )}

        {/* Explore stocks */}
        {activePage === "stocks" && <ExploreStocks />}

        {/* ETF placeholder */}
        {activePage === "etfs" && (
          <section className="placeholder-page">
            <p className="page-eyebrow">COMING SOON</p>

            <h2>Explore ETFs</h2>

            <p>
              This section is being developed. The ETF research and
              analysis tools will be available here.
            </p>

            <button
              className="primary-button"
              onClick={() => navigateTo("dashboard")}
            >
              Back to Dashboard
            </button>
          </section>
        )}

        {/* Mutual funds placeholder */}
        {activePage === "mutual-funds" && (
          <section className="placeholder-page">
            <p className="page-eyebrow">COMING SOON</p>

            <h2>Explore Mutual Funds</h2>

            <p>
              This section is being developed. The mutual fund research
              and analysis tools will be available here.
            </p>

            <button
              className="primary-button"
              onClick={() => navigateTo("dashboard")}
            >
              Back to Dashboard
            </button>
          </section>
        )}

        {/* Shared footer */}
        <footer className="footer">
          Investment Copilot · Data-driven investment research
        </footer>
      </main>
    </div>
  );
}

export default App;
