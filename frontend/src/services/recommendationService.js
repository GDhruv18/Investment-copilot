
const API_BASE_URL = "/api/v1";

export async function getPersonalizedRecommendations(requestData) {
  try {
    const response = await fetch(
      `${API_BASE_URL}/recommendations/personalized`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestData),
      }
    );

    const data = await response.json();

    if (!response.ok) {
      const errorMessage =
        typeof data.detail === "string"
          ? data.detail
          : "Unable to generate recommendations.";

      throw new Error(errorMessage);
    }

    return data;
  } catch (error) {
    if (error instanceof TypeError) {
      throw new Error(
        "Unable to connect to the server. Please check whether the backend is running."
      );
    }

    throw error;
  }
}
