from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path.cwd()

SIGNAL_DIR = ROOT / "data" / "ml" / "current_signals"
FEATURE_DIR = ROOT / "data" / "features"
OUT_DIR = ROOT / "data" / "ml" / "recommendation_engine"

OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SIGNAL_AGE_DAYS = 14
MAX_FEATURE_GAP_DAYS = 14

# Initial research weights. These must be validated through backtesting.
PROFILE_WEIGHTS = {
    "conservative": {"ml": 0.45, "risk": 0.40, "trend": 0.15},
    "moderate": {"ml": 0.50, "risk": 0.30, "trend": 0.20},
    "aggressive": {"ml": 0.55, "risk": 0.20, "trend": 0.25},
}

# Risk component weights.
PROFILE_RISK_COMPONENT_WEIGHTS = {
    "conservative": {
        "volatility_quality": 0.65,
        "drawdown_quality": 0.35,
    },
    "moderate": {
        "volatility_quality": 0.50,
        "drawdown_quality": 0.50,
    },
    "aggressive": {
        "volatility_quality": 0.35,
        "drawdown_quality": 0.65,
    },
}

FEATURE_FILES = {
    "stocks": FEATURE_DIR / "stock_features.parquet",
    "etfs": FEATURE_DIR / "etf_features.parquet",
    "mutual_funds": FEATURE_DIR / "mutual_fund_features.parquet",
}

PROFILES = {
    "conservative": {
        "allowed_assets": {"etfs", "mutual_funds"},
        "max_weak_probability": 0.30,
        "min_attractive_probability": 0.30,
        "description": "Lower-risk screening; excludes individual stocks.",
    },
    "moderate": {
        "allowed_assets": {"etfs", "mutual_funds", "stocks"},
        "max_weak_probability": 0.35,
        "min_attractive_probability": 0.30,
        "description": "Balanced screening across asset classes.",
    },
    "aggressive": {
        "allowed_assets": {"etfs", "mutual_funds", "stocks"},
        "max_weak_probability": 0.40,
        "min_attractive_probability": 0.25,
        "description": "Broader screening; accepts higher model-indicated risk.",
    },
}


# ============================================================
# 1. LOAD LATEST MODEL SIGNALS
# ============================================================

def load_latest_signals():

    files = sorted(SIGNAL_DIR.glob("model_signals_*.parquet"))

    if not files:
        raise FileNotFoundError("No model signal files found.")

    signals = pd.read_parquet(files[-1]).copy()

    required = {
        "asset_class",
        "ticker",
        "feature_date",
        "horizon",
        "prediction",
        "prob_weak",
        "prob_neutral",
        "prob_attractive",
    }

    missing = required - set(signals.columns)

    if missing:
        raise ValueError(f"Missing signal columns: {sorted(missing)}")

    signals["ticker"] = signals["ticker"].astype(str)

    signals["asset_class"] = (
        signals["asset_class"].astype(str).str.lower()
    )

    signals["horizon"] = (
        signals["horizon"].astype(str).str.lower()
    )

    signals["feature_date"] = pd.to_datetime(
        signals["feature_date"],
        errors="coerce",
    ).dt.tz_localize(None).dt.normalize()

    probability_columns = [
        "prob_weak",
        "prob_neutral",
        "prob_attractive",
    ]

    for col in probability_columns:
        signals[col] = pd.to_numeric(
            signals[col],
            errors="coerce",
        )

    signals = signals.dropna(
        subset=[
            "ticker",
            "asset_class",
            "horizon",
            "feature_date",
        ] + probability_columns
    ).copy()

    if not signals[probability_columns].apply(
        lambda col: col.between(0, 1).all()
    ).all():
        raise ValueError("Invalid probability values detected.")

    probability_sum = signals[probability_columns].sum(axis=1)

    if not np.allclose(probability_sum, 1.0, atol=1e-4):
        raise ValueError("Model probabilities do not sum to one.")

    duplicate_key = [
        "asset_class",
        "ticker",
        "horizon",
    ]

    if signals.duplicated(duplicate_key).any():
        raise ValueError("Duplicate asset/horizon signal records found.")

    signals["signal_score"] = (
        signals["prob_attractive"] - signals["prob_weak"]
    )
    # Verify mutual-fund 2Y schemes belong to the model's
    # actual training universe.
    mf_train_path = (
    ROOT / "data" / "ml" / "mutual_funds" / "mutual_fund_2y_train.parquet"
    )

    if not mf_train_path.exists():
        raise FileNotFoundError(
            f"Mutual-fund training dataset not found: {mf_train_path}"
        )

    mf_train = pd.read_parquet(mf_train_path)

    trained_mf_schemes = set(
        mf_train["ticker"].astype(str).unique()
    )

    signals["model_training_universe_member"] = True

    mf_2y_mask = (
        signals["asset_class"].eq("mutual_funds")
        & signals["horizon"].eq("2y")
    )

    signals.loc[mf_2y_mask, "model_training_universe_member"] = (
        signals.loc[mf_2y_mask, "ticker"].isin(trained_mf_schemes)
    )

    print(
        "Mutual-fund 2Y signals outside training universe:",
        int((~signals.loc[mf_2y_mask, "model_training_universe_member"]).sum())
    )

    signals["ml_score"] = (
        (signals["signal_score"] + 1.0) / 2.0
    ).clip(0, 1)

    today = (
        pd.Timestamp.now(tz="UTC")
        .tz_localize(None)
        .normalize()
    )

    signals["signal_age_days"] = (
        today - signals["feature_date"]
    ).dt.days

    signals["freshness_status"] = np.where(
        signals["signal_age_days"].between(
            0,
            MAX_SIGNAL_AGE_DAYS,
        ),
        "RECENT_SIGNAL",
        "STALE_OR_FUTURE_SIGNAL",
    )

    return signals, files[-1]


# ============================================================
# 2. LOAD HISTORICAL FEATURE DATA
# ============================================================

def load_feature_data(asset_class):

    path = FEATURE_FILES[asset_class]

    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {path}"
        )

    features = pd.read_parquet(path).copy()

    required = {"ticker", "date"}

    missing = required - set(features.columns)

    if missing:
        raise ValueError(
            f"{asset_class}: missing feature columns {sorted(missing)}"
        )

    features["ticker"] = features["ticker"].astype(str)

    features["date"] = pd.to_datetime(
        features["date"],
        errors="coerce",
    ).dt.tz_localize(None).dt.normalize()

    features = features.dropna(
        subset=["ticker", "date"]
    ).copy()

    if features.duplicated(["ticker", "date"]).any():
        raise ValueError(
            f"{asset_class}: duplicate ticker/date feature records."
        )

    return features.sort_values(["date", "ticker"])


# ============================================================
# 3. ATTACH FEATURES TO MODEL SIGNALS
# ============================================================

def attach_features(signals):

    """
    Attach the most recent feature record on or before each
    signal date. This prevents future feature observations
    from leaking into a signal.
    """

    outputs = []

    for asset_class in [
        "stocks",
        "etfs",
        "mutual_funds",
    ]:

        left = signals[
            signals["asset_class"] == asset_class
        ].copy()

        if left.empty:
            continue

        right = load_feature_data(asset_class)

        left = left.sort_values(
            ["feature_date", "ticker"]
        )

        left["feature_date"] = pd.to_datetime(
            left["feature_date"]
        ).astype("datetime64[ns]")

        right["date"] = pd.to_datetime(
            right["date"]
        ).astype("datetime64[ns]")

        merged = pd.merge_asof(
            left,
            right,
            left_on="feature_date",
            right_on="date",
            by="ticker",
            direction="backward",
            tolerance=pd.Timedelta(
                days=MAX_FEATURE_GAP_DAYS
            ),
            suffixes=("", "_feature"),
        )

        merged = merged.rename(
            columns={"date": "feature_data_date"}
        )

        merged["feature_gap_days"] = (
            merged["feature_date"]
            - merged["feature_data_date"]
        ).dt.days

        outputs.append(merged)

    if not outputs:
        raise ValueError(
            "No supported asset classes found in signals."
        )

    return pd.concat(
        outputs,
        ignore_index=True,
    )


# ============================================================
# 4. PROFILE-AWARE RISK SCORE
# ============================================================

def calculate_risk_score(df):

    """
    Higher volatility percentile = worse.
    Higher drawdown percentile = better.
    Higher NAV range percentile = worse for mutual funds.

    Profile-specific weights change how strongly volatility
    and drawdown influence the risk score.
    """

    df = df.copy()

    volatility_quality = 1.0 - pd.to_numeric(
        df["volatility_60d_pct"],
        errors="coerce",
    )

    drawdown_quality = pd.to_numeric(
        df["drawdown_120d_pct"],
        errors="coerce",
    )

    nav_range_quality = 1.0 - pd.to_numeric(
        df["nav_range_60d_pct"],
        errors="coerce",
    )

    df["risk_score"] = np.nan

    for profile, weights in PROFILE_RISK_COMPONENT_WEIGHTS.items():

        profile_mask = df["risk_profile"].eq(profile)

        non_mf = (
            profile_mask
            & df["asset_class"].isin(["stocks", "etfs"])
        )

        df.loc[non_mf, "risk_score"] = (
            weights["volatility_quality"]
            * volatility_quality.loc[non_mf]
            +
            weights["drawdown_quality"]
            * drawdown_quality.loc[non_mf]
        )

        mf = (
            profile_mask
            & df["asset_class"].eq("mutual_funds")
        )

        profile_risk_quality = (
            weights["volatility_quality"]
            * volatility_quality.loc[mf]
            +
            weights["drawdown_quality"]
            * drawdown_quality.loc[mf]
        )

        df.loc[mf, "risk_score"] = (
            (2.0 / 3.0) * profile_risk_quality
            +
            (1.0 / 3.0) * nav_range_quality.loc[mf]
        )

    return df


# ============================================================
# 5. TREND SCORE
# ============================================================

def calculate_trend_score(df):

    """
    Trend score uses three cross-sectional components:

    1. Historical return percentile.
    2. Price/NAV relative to 50-day moving average.
    3. Price/NAV relative to 200-day moving average.

    All rankings are calculated within:
    risk_profile + horizon + asset_class + feature_date.

    Missing components are not imputed.
    """

    df = df.copy()

    df["trend_score"] = np.nan

    group_columns = [
        "risk_profile",
        "horizon",
        "asset_class",
        "feature_date",
    ]

    horizon_aliases_6m = {
        "6m",
        "6mo",
        "6_month",
        "6_months",
        "6-month",
    }

    for group_key, group in df.groupby(
        group_columns,
        dropna=False,
        sort=False,
    ):

        profile, horizon, asset_class, feature_date = group_key

        horizon_key = str(horizon).strip().lower()

        return_col = (
            "return_120d_pct"
            if horizon_key in horizon_aliases_6m
            else "return_252d_pct"
        )

        if asset_class == "mutual_funds":

            ma_columns = [
                "nav_vs_sma_50",
                "nav_vs_sma_200",
            ]

        else:

            ma_columns = [
                "price_vs_sma_50",
                "price_vs_sma_200",
            ]

        required = [return_col] + ma_columns

        missing = [
            col for col in required
            if col not in group.columns
        ]

        if missing:
            continue

        ranked_components = []

        for column in required:

            values = pd.to_numeric(
                group[column],
                errors="coerce",
            )

            ranked_components.append(
                values.rank(
                    method="average",
                    pct=True,
                    ascending=True,
                )
            )

        component_frame = pd.concat(
            ranked_components,
            axis=1,
        )

        group_trend = component_frame.mean(
            axis=1,
            skipna=False,
        )

        df.loc[group.index, "trend_score"] = group_trend

    return df


# ============================================================
# 6. PROFILE-AWARE FINAL SCORE
# ============================================================

def calculate_final_score(df):

    df = calculate_risk_score(df)

    df = calculate_trend_score(df)

    df = df.copy()

    df["final_score"] = np.nan

    for profile, weights in PROFILE_WEIGHTS.items():

        if not np.isclose(
            sum(weights.values()),
            1.0,
        ):
            raise ValueError(
                f"PROFILE_WEIGHTS for {profile} must sum to 1.0."
            )

        mask = df["risk_profile"].eq(profile)

        df.loc[mask, "final_score"] = (
            weights["ml"] * df.loc[mask, "ml_score"]
            +
            weights["risk"] * df.loc[mask, "risk_score"]
            +
            weights["trend"] * df.loc[mask, "trend_score"]
        )

    return df


# ============================================================
# 7. SCREEN INVESTOR PROFILES
# ============================================================

def screen_profiles(signals):

    outputs = []

    for profile_name, policy in PROFILES.items():

        df = signals[
            signals["asset_class"].isin(
                policy["allowed_assets"]
            )
        ].copy()

        df["risk_profile"] = profile_name

        df["profile_policy"] = policy["description"]

        df["passes_probability_filter"] = (
            (
                df["prob_weak"]
                <= policy["max_weak_probability"]
            )
            &
            (
                df["prob_attractive"]
                >= policy["min_attractive_probability"]
            )
        )

        outputs.append(df)

    return pd.concat(
        outputs,
        ignore_index=True,
    )


# ============================================================
# 8. ASSIGN CANDIDATE STATUS AND COVERAGE
# ============================================================


def assign_status(df):

    df = df.copy()

    # Safely convert NAV-related features to numeric values.
    return_60d = pd.to_numeric(df["return_60d"], errors="coerce")
    return_120d = pd.to_numeric(df["return_120d"], errors="coerce")
    return_252d = pd.to_numeric(df["return_252d"], errors="coerce")

    volatility_60d = pd.to_numeric(
        df["volatility_60d"], errors="coerce"
    )
    volatility_120d = pd.to_numeric(
        df["volatility_120d"], errors="coerce"
    )

    # Detect suspiciously flat mutual-fund NAV histories.
    flat_nav_history = (
        df["asset_class"].eq("mutual_funds")
        & (return_60d.abs() < 1e-8)
        & (return_120d.abs() < 1e-8)
        & (return_252d.abs() < 1e-8)
        & (volatility_60d < 1e-6)
        & (volatility_120d < 1e-6)
    )

    conditions = [
        df["feature_data_date"].isna(),

        df["feature_gap_days"].isna(),

        df["feature_gap_days"] < 0,

        df["signal_age_days"] < 0,

        df["signal_age_days"] > MAX_SIGNAL_AGE_DAYS,

        (
            df["risk_score"].isna()
            | df["trend_score"].isna()
        ),

        ~df["model_training_universe_member"],

        flat_nav_history,

        ~df["passes_probability_filter"],

        df["prediction"]
        .astype(str)
        .str.lower()
        .eq("weak"),
    ]

    labels = [
        "REVIEW_MISSING_FEATURES",
        "REVIEW_MISSING_FEATURES",
        "REVIEW_FUTURE_FEATURE_DATE",
        "REVIEW_FUTURE_SIGNAL",
        "REVIEW_STALE_SIGNAL",
        "REVIEW_INCOMPLETE_RISK_OR_TREND",
        "REVIEW_OUTSIDE_MODEL_TRAINING_UNIVERSE",
        "REVIEW_FLAT_NAV_HISTORY",
        "NOT_SHORTLISTED_PROBABILITY_FILTER",
        "REVIEW_WEAK_SIGNAL",
    ]

    if len(conditions) != len(labels):
        raise ValueError(
            f"Status condition/label mismatch: "
            f"{len(conditions)} conditions, {len(labels)} labels."
        )

    df["candidate_status"] = np.select(
        conditions,
        labels,
        default="RECOMMENDATION_CANDIDATE",
    )

    # --------------------------------------------------------
    # Prediction type: separate model classification from
    # recommendation eligibility.
    # --------------------------------------------------------

    prediction = (
        df["prediction"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    eligible = df["candidate_status"].eq(
        "RECOMMENDATION_CANDIDATE"
    )

    df["candidate_signal_type"] = np.select(
        [
            prediction.eq("attractive") & eligible,
            prediction.eq("neutral") & eligible,
            prediction.eq("attractive") & ~eligible,
            prediction.eq("neutral") & ~eligible,
            prediction.eq("weak"),
        ],
        [
            "MODEL_ATTRACTIVE_CANDIDATE",
            "MODEL_NEUTRAL_SCREENING_PASS",
            "MODEL_ATTRACTIVE_NOT_ELIGIBLE",
            "MODEL_NEUTRAL_NOT_ELIGIBLE",
            "MODEL_WEAK",
        ],
        default="MODEL_CLASS_UNRECOGNIZED",
    )

    # --------------------------------------------------------
    # Prediction universe and candidate pool coverage.
    # --------------------------------------------------------

    group_keys = [
        "risk_profile",
        "asset_class",
        "horizon",
    ]

    df["prediction_universe_count"] = (
        df.groupby(group_keys)["ticker"]
        .transform("count")
    )

    df["candidate_pool_size"] = (
        eligible.groupby(
            [df[key] for key in group_keys]
        )
        .transform("sum")
        .astype(int)
    )

    universe_size = df["prediction_universe_count"]

    df["coverage_status"] = np.select(
        [
            universe_size < 5,
            universe_size < 15,
            universe_size < 30,
        ],
        [
            "VERY_LOW_COVERAGE",
            "LIMITED_COVERAGE",
            "MODERATE_COVERAGE",
        ],
        default="GOOD_COVERAGE",
    )

    # --------------------------------------------------------
    # Rank only valid recommendation candidates.
    # --------------------------------------------------------

    df["rank_within_group"] = pd.Series(
        pd.NA,
        index=df.index,
        dtype="Int64",
    )

    eligible_scores = (
        df.loc[eligible]
        .groupby(group_keys)["final_score"]
        .rank(
            method="first",
            ascending=False,
        )
    )

    df.loc[
        eligible,
        "rank_within_group",
    ] = eligible_scores.astype("Int64")

    return df


# ============================================================
# 9. GENERATE EXPLANATIONS
# ============================================================

def create_explanations(df):

    df = df.copy()

    explanations = []

    for _, row in df.iterrows():

        parts = [
            (
                f"ML score={row['ml_score']:.3f}"
                if pd.notna(row["ml_score"])
                else "ML score unavailable"
            ),

            (
                f"Risk score={row['risk_score']:.3f}"
                if pd.notna(row["risk_score"])
                else "Risk score unavailable"
            ),

            (
                f"Trend score={row['trend_score']:.3f}"
                if pd.notna(row["trend_score"])
                else "Trend score unavailable"
            ),

            (
                f"Final score={row['final_score']:.3f}"
                if pd.notna(row["final_score"])
                else "Final score unavailable"
            ),

            f"Prediction type={row['candidate_signal_type']}",

            f"Coverage={row['coverage_status']}",

            f"Candidate pool size={row['candidate_pool_size']}",

            f"Status={row['candidate_status']}",
        ]

        explanations.append(
            "; ".join(parts)
        )

    df["reason"] = explanations

    return df


# ============================================================
# 10. MAIN EXECUTION
# ============================================================

def main():

    print("=" * 70)
    print("INVESTMENT RECOMMENDATION ENGINE")
    print("=" * 70)

    signals, source_file = load_latest_signals()

    print("Loaded signals:", len(signals))
    print("Source:", source_file.name)

    enriched = attach_features(signals)

    print("Feature integration completed.")

    screened = screen_profiles(enriched)

    print("Profile-screened rows:", len(screened))

    scored = calculate_final_score(screened)

    result = assign_status(scored)

    result = create_explanations(result)

    result["status"] = (
        "RESEARCH_CANDIDATE_NOT_PERSONALIZED_FINANCIAL_ADVICE"
    )

    result = result.sort_values(
        [
            "risk_profile",
            "horizon",
            "asset_class",
            "candidate_status",
            "final_score",
        ],
        ascending=[
            True,
            True,
            True,
            True,
            False,
        ],
        na_position="last",
    )

    # --------------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------------

    parquet_path = (
        OUT_DIR / "final_recommendations.parquet"
    )

    csv_path = (
        OUT_DIR / "final_recommendations.csv"
    )

    explanation_path = (
        OUT_DIR / "recommendation_explanations.csv"
    )

    summary_path = (
        OUT_DIR / "recommendation_engine_summary.json"
    )

    result.to_parquet(
        parquet_path,
        index=False,
    )

    result.to_csv(
        csv_path,
        index=False,
    )

    explanation_columns = [
        "risk_profile",
        "asset_class",
        "ticker",
        "horizon",
        "prediction",
        "prob_attractive",
        "prob_weak",
        "ml_score",
        "risk_score",
        "trend_score",
        "final_score",
        "rank_within_group",
        "candidate_signal_type",
        "prediction_universe_count",
        "candidate_pool_size",
        "coverage_status",
        "candidate_status",
        "reason",
    ]

    result[explanation_columns].to_csv(
        explanation_path,
        index=False,
    )

    # --------------------------------------------------------
    # SUMMARY JSON
    # --------------------------------------------------------

    summary = {
        "source_file": str(
            source_file.relative_to(ROOT)
        ),

        "rows_written": int(len(result)),

        "profile_weights": PROFILE_WEIGHTS,

        "profile_risk_component_weights":
            PROFILE_RISK_COMPONENT_WEIGHTS,

        "profiles": {},

        "feature_coverage": {},

        "limitations": [
            "Scores are research rankings, not expected returns.",
            "Weights require historical out-of-sample validation.",
            "Profile rules are initial screening policies, not personal suitability assessments.",
            "Fundamental scoring remains disabled pending approval.",
            "No portfolio allocation, liquidity assessment or tax analysis is included.",
            "Feature gaps and warm-up NaNs are not imputed.",
            "ETF 1Y and 2Y coverage is limited by available valid model predictions.",
        ],
    }

    for asset_class in FEATURE_FILES:

        part = result[
            result["asset_class"] == asset_class
        ]

        summary["feature_coverage"][asset_class] = {
            "rows": int(len(part)),

            "matched_feature_rows": int(
                part["feature_data_date"].notna().sum()
            ),

            "missing_feature_rows": int(
                part["feature_data_date"].isna().sum()
            ),
        }

    for profile in PROFILES:

        part = result[
            result["risk_profile"] == profile
        ]

        summary["profiles"][profile] = {

            "rows": int(len(part)),

            "recommendation_candidates": int(
                (
                    part["candidate_status"]
                    == "RECOMMENDATION_CANDIDATE"
                ).sum()
            ),

            "by_horizon": (
                part.groupby(
                    [
                        "horizon",
                        "candidate_status",
                    ]
                )
                .size()
                .unstack(fill_value=0)
                .astype(int)
                .to_dict("index")
            ),

            "by_asset_and_horizon": (
                part.groupby(
                    [
                        "asset_class",
                        "horizon",
                        "candidate_status",
                    ],
                    dropna=False,
                )
                .size()
                .reset_index(name="count")
                .to_dict(orient="records")
            ),
        }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # TERMINAL REPORT
    # --------------------------------------------------------

    print("\nRecommendation engine completed.")

    print("\nStatus counts:")

    print(
        result.groupby(
            [
                "risk_profile",
                "candidate_status",
            ]
        )
        .size()
        .to_string()
    )

    print("\nCandidate signal types:")

    print(
        result[
            result["candidate_status"]
            == "RECOMMENDATION_CANDIDATE"
        ]
        .groupby(
            [
                "risk_profile",
                "candidate_signal_type",
            ]
        )
        .size()
        .to_string()
    )

    print("\nCoverage summary:")

    print(
        result.groupby(
            [
                "risk_profile",
                "asset_class",
                "horizon",
                "coverage_status",
            ]
        )
        .size()
        .to_string()
    )

    print("\nSaved:")

    print(parquet_path)
    print(csv_path)
    print(explanation_path)
    print(summary_path)


if __name__ == "__main__":
    main()