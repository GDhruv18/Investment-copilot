
const API_BASE_URL = "/api/v1/stocks";

async function apiRequest(url) {
  try {
    const response = await fetch(url);

    const data = await response.json();

    if (!response.ok) {
      const message =
        typeof data.detail === "string"
          ? data.detail
          : "Unable to retrieve stock information.";

      throw new Error(message);
    }

    return data;
  } catch (error) {
    if (error instanceof TypeError) {
      throw new Error(
        "Unable to connect to the backend. Check whether the server is running."
      );
    }

    throw error;
  }
}

export async function searchStocks(query, limit = 10) {
  const params = new URLSearchParams({
    q: query,
    limit: String(limit),
  });

  return apiRequest(`${API_BASE_URL}/search?${params}`);
}

export async function getStockDetails(ticker, period = "1y") {
  return apiRequest(
    `${API_BASE_URL}/${encodeURIComponent(ticker)}/details?period=${period}`
  );
}

export async function getStockDailyData(ticker, period = "1y") {
  return apiRequest(
    `${API_BASE_URL}/${encodeURIComponent(ticker)}/daily?period=${period}`
  );
}

export async function getStockIntradayData(
  ticker,
  period = "1d",
  interval = "5m"
) {
  const params = new URLSearchParams({
    period,
    interval,
  });

  return apiRequest(
    `${API_BASE_URL}/${encodeURIComponent(ticker)}/intraday?${params}`
  );
}
