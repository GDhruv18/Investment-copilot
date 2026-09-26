from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "targets"
    / "mutual_fund_forward_outcomes_clean.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "targets"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "mutual_fund_6m_1y_targets.parquet"
)

EXPECTED_ROWS = 206_639
EXPECTED_SCHEMES = 100

MIN_COHORT = 10


# ============================================================
# LOAD
# ============================================================

def load_data():

    print("=" * 70)
    print("MUTUAL FUND 6M / 1Y TARGET GENERATION")
    print("=" * 70)

    print("\nInput:")
    print(INPUT_FILE)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_parquet(
        INPUT_FILE
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

    print(
        f"Schemes: "
        f"{df['scheme_code'].nunique()}"
    )

    df["scheme_code"] = (
        df["scheme_code"]
        .astype(str)
    )

    df["date"] = (
        pd.to_datetime(df["date"])
        .dt.normalize()
    )

    return df


# ============================================================
# SOURCE VALIDATION
# ============================================================

def validate_source(df):

    print("\n" + "=" * 70)
    print("SOURCE VALIDATION")
    print("=" * 70)

    if len(df) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS:,} rows, "
            f"found {len(df):,}."
        )

    scheme_count = (
        df["scheme_code"].nunique()
    )

    print(
        f"Scheme count: {scheme_count}"
    )

    if scheme_count != EXPECTED_SCHEMES:
        raise RuntimeError(
            f"Expected {EXPECTED_SCHEMES} schemes, "
            f"found {scheme_count}."
        )

    duplicate_count = df.duplicated(
        subset=[
            "scheme_code",
            "date"
        ]
    ).sum()

    print(
        f"Duplicate scheme/date rows: "
        f"{duplicate_count}"
    )

    if duplicate_count != 0:
        raise RuntimeError(
            "Duplicate scheme/date rows detected."
        )

    # --------------------------------------------------------
    # Raw returns must exist where outcomes are valid
    # --------------------------------------------------------

    invalid_6m = (
        df["forward_return_6m"].notna()
        &
        ~np.isfinite(
            df["forward_return_6m"]
        )
    ).sum()

    invalid_1y = (
        df["forward_return_1y"].notna()
        &
        ~np.isfinite(
            df["forward_return_1y"]
        )
    ).sum()

    print(
        f"Invalid 6M returns: {invalid_6m}"
    )

    print(
        f"Invalid 1Y returns: {invalid_1y}"
    )

    if invalid_6m != 0 or invalid_1y != 0:
        raise RuntimeError(
            "Invalid forward returns detected."
        )

    print(
        "Source validation: PASS"
    )


# ============================================================
# CROSS-SECTIONAL RANKING
# ============================================================

def generate_targets(df):

    print("\n" + "=" * 70)
    print("GENERATING CROSS-SECTIONAL 6M / 1Y TARGETS")
    print("=" * 70)

    result = df.copy()

    # --------------------------------------------------------
    # Initialize
    # --------------------------------------------------------

    result["score_6m"] = np.nan
    result["label_6m"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64"
    )

    result["score_1y"] = np.nan
    result["label_1y"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64"
    )

    # --------------------------------------------------------
    # Helper
    # --------------------------------------------------------

    def assign_labels(
        values
    ):

        scores = values.rank(
            pct=True,
            method="average"
        )

        labels = pd.Series(
            pd.NA,
            index=values.index,
            dtype="Int64"
        )

        labels.loc[
            scores <= 0.30
        ] = 0

        labels.loc[
            (scores > 0.30)
            &
            (scores <= 0.70)
        ] = 1

        labels.loc[
            scores > 0.70
        ] = 2

        return scores, labels

    # --------------------------------------------------------
    # 6 MONTH
    # --------------------------------------------------------

    valid_6m = (
        result["forward_return_6m"].notna()
    )

    cohort_6m = (
        result.loc[
            valid_6m
        ]
        .groupby("date")["scheme_code"]
        .nunique()
    )

    eligible_dates_6m = cohort_6m[
        cohort_6m >= MIN_COHORT
    ].index

    eligible_mask_6m = (
        valid_6m
        &
        result["date"].isin(
            eligible_dates_6m
        )
    )

    for date_value, group in result.loc[
        eligible_mask_6m
    ].groupby("date"):

        scores, labels = assign_labels(
            group["forward_return_6m"]
        )

        result.loc[
            group.index,
            "score_6m"
        ] = scores

        result.loc[
            group.index,
            "label_6m"
        ] = labels

    # --------------------------------------------------------
    # 1 YEAR
    # --------------------------------------------------------

    valid_1y = (
        result["forward_return_1y"].notna()
    )

    cohort_1y = (
        result.loc[
            valid_1y
        ]
        .groupby("date")["scheme_code"]
        .nunique()
    )

    eligible_dates_1y = cohort_1y[
        cohort_1y >= MIN_COHORT
    ].index

    eligible_mask_1y = (
        valid_1y
        &
        result["date"].isin(
            eligible_dates_1y
        )
    )

    for date_value, group in result.loc[
        eligible_mask_1y
    ].groupby("date"):

        scores, labels = assign_labels(
            group["forward_return_1y"]
        )

        result.loc[
            group.index,
            "score_1y"
        ] = scores

        result.loc[
            group.index,
            "label_1y"
        ] = labels

    return result, cohort_6m, cohort_1y


# ============================================================
# VALIDATE TARGETS
# ============================================================

def validate_targets(
    source,
    result,
    cohort_6m,
    cohort_1y
):

    print("\n" + "=" * 70)
    print("TARGET VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Row preservation
    # --------------------------------------------------------

    if len(source) != len(result):
        raise RuntimeError(
            "Row count changed."
        )

    print(
        f"Row count preserved: "
        f"{len(result):,}"
    )

    # --------------------------------------------------------
    # Universe
    # --------------------------------------------------------

    if (
        result["scheme_code"].nunique()
        != EXPECTED_SCHEMES
    ):
        raise RuntimeError(
            "Scheme universe changed."
        )

    print(
        f"Schemes preserved: "
        f"{result['scheme_code'].nunique()}"
    )

    # --------------------------------------------------------
    # Keys
    # --------------------------------------------------------

    duplicate_count = result.duplicated(
        subset=[
            "scheme_code",
            "date"
        ]
    ).sum()

    print(
        f"Duplicate scheme/date rows: "
        f"{duplicate_count}"
    )

    if duplicate_count != 0:
        raise RuntimeError(
            "Duplicate keys detected."
        )

    # --------------------------------------------------------
    # Returns preserved
    # --------------------------------------------------------

    source_6m = source[
        "forward_return_6m"
    ].to_numpy()

    result_6m = result[
        "forward_return_6m"
    ].to_numpy()

    source_1y = source[
        "forward_return_1y"
    ].to_numpy()

    result_1y = result[
        "forward_return_1y"
    ].to_numpy()

    if not np.allclose(
        source_6m,
        result_6m,
        equal_nan=True
    ):
        raise RuntimeError(
            "6M forward returns changed."
        )

    if not np.allclose(
        source_1y,
        result_1y,
        equal_nan=True
    ):
        raise RuntimeError(
            "1Y forward returns changed."
        )

    print(
        "Raw forward returns preserved: PASS"
    )

    # --------------------------------------------------------
    # Score ranges
    # --------------------------------------------------------

    valid_score_6m = (
        result["score_6m"].notna()
    )

    valid_score_1y = (
        result["score_1y"].notna()
    )

    invalid_score_6m = (
        valid_score_6m
        &
        (
            (result["score_6m"] < 0)
            |
            (result["score_6m"] > 1)
        )
    ).sum()

    invalid_score_1y = (
        valid_score_1y
        &
        (
            (result["score_1y"] < 0)
            |
            (result["score_1y"] > 1)
        )
    ).sum()

    print(
        f"Invalid 6M scores: "
        f"{invalid_score_6m}"
    )

    print(
        f"Invalid 1Y scores: "
        f"{invalid_score_1y}"
    )

    if (
        invalid_score_6m != 0
        or invalid_score_1y != 0
    ):
        raise RuntimeError(
            "Invalid scores detected."
        )

    # --------------------------------------------------------
    # Labels must exist exactly where scores exist
    # --------------------------------------------------------

    score_without_label_6m = (
        result["score_6m"].notna()
        &
        result["label_6m"].isna()
    ).sum()

    score_without_label_1y = (
        result["score_1y"].notna()
        &
        result["label_1y"].isna()
    ).sum()

    label_without_score_6m = (
        result["score_6m"].isna()
        &
        result["label_6m"].notna()
    ).sum()

    label_without_score_1y = (
        result["score_1y"].isna()
        &
        result["label_1y"].notna()
    ).sum()

    print(
        f"6M score without label: "
        f"{score_without_label_6m}"
    )

    print(
        f"1Y score without label: "
        f"{score_without_label_1y}"
    )

    print(
        f"6M label without score: "
        f"{label_without_score_6m}"
    )

    print(
        f"1Y label without score: "
        f"{label_without_score_1y}"
    )

    if any(
        value != 0
        for value in [
            score_without_label_6m,
            score_without_label_1y,
            label_without_score_6m,
            label_without_score_1y,
        ]
    ):
        raise RuntimeError(
            "Score/label consistency failure."
        )

    # --------------------------------------------------------
    # Labels without valid returns
    # --------------------------------------------------------

    label_without_return_6m = (
        result["label_6m"].notna()
        &
        result["forward_return_6m"].isna()
    ).sum()

    label_without_return_1y = (
        result["label_1y"].notna()
        &
        result["forward_return_1y"].isna()
    ).sum()

    print(
        f"6M labels without return: "
        f"{label_without_return_6m}"
    )

    print(
        f"1Y labels without return: "
        f"{label_without_return_1y}"
    )

    if (
        label_without_return_6m != 0
        or label_without_return_1y != 0
    ):
        raise RuntimeError(
            "Labels exist without valid returns."
        )

    # --------------------------------------------------------
    # Validate labels from scores
    # --------------------------------------------------------

    def expected_label(score):

        if score <= 0.30:
            return 0

        if score <= 0.70:
            return 1

        return 2

    for column_score, column_label in [
        ("score_6m", "label_6m"),
        ("score_1y", "label_1y"),
    ]:

        mask = result[column_score].notna()

        expected = result.loc[
            mask,
            column_score
        ].apply(
            expected_label
        )

        actual = result.loc[
            mask,
            column_label
        ].astype(int)

        boundary_errors = (
            expected.to_numpy()
            != actual.to_numpy()
        ).sum()

        print(
            f"{column_label} boundary errors: "
            f"{boundary_errors}"
        )

        if boundary_errors != 0:
            raise RuntimeError(
                f"Invalid labels in {column_label}."
            )

    # --------------------------------------------------------
    # Cohort validation
    # --------------------------------------------------------

    labeled_6m = result[
        "label_6m"
    ].notna()

    labeled_1y = result[
        "label_1y"
    ].notna()

    actual_cohort_6m = (
        result.loc[
            labeled_6m
        ]
        .groupby("date")["scheme_code"]
        .nunique()
    )

    actual_cohort_1y = (
        result.loc[
            labeled_1y
        ]
        .groupby("date")["scheme_code"]
        .nunique()
    )

    if (
        actual_cohort_6m.min()
        < MIN_COHORT
    ):
        raise RuntimeError(
            "6M labeled cohort below minimum."
        )

    if (
        actual_cohort_1y.min()
        < MIN_COHORT
    ):
        raise RuntimeError(
            "1Y labeled cohort below minimum."
        )

    # --------------------------------------------------------
    # Label distributions
    # --------------------------------------------------------

    print("\n6M label distribution:")

    counts_6m = (
        result.loc[
            labeled_6m,
            "label_6m"
        ]
        .astype(int)
        .value_counts()
        .sort_index()
    )

    print(
        counts_6m.to_string()
    )

    print(
        "\n6M label percentages:"
    )

    print(
        (
            counts_6m
            / counts_6m.sum()
            * 100
        ).round(2).to_string()
    )

    print("\n1Y label distribution:")

    counts_1y = (
        result.loc[
            labeled_1y,
            "label_1y"
        ]
        .astype(int)
        .value_counts()
        .sort_index()
    )

    print(
        counts_1y.to_string()
    )

    print(
        "\n1Y label percentages:"
    )

    print(
        (
            counts_1y
            / counts_1y.sum()
            * 100
        ).round(2).to_string()
    )

    # --------------------------------------------------------
    # Per-date label balance
    # --------------------------------------------------------

    def print_date_balance(
        label_column,
        name
    ):

        labeled = result[
            result[label_column].notna()
        ].copy()

        balance = (
            labeled
            .assign(
                label=labeled[
                    label_column
                ].astype(int)
            )
            .groupby("date")["label"]
            .value_counts(
                normalize=True
            )
            .unstack(
                fill_value=0
            )
            * 100
        )

        for label_value in [
            0,
            1,
            2,
        ]:

            if label_value in balance.columns:

                print(
                    f"{name} label {label_value} "
                    f"range: "
                    f"{balance[label_value].min():.2f}% → "
                    f"{balance[label_value].max():.2f}%"
                )

    print("\nPer-date label balance:")

    print_date_balance(
        "label_6m",
        "6M"
    )

    print_date_balance(
        "label_1y",
        "1Y"
    )

    print(
        "\nTarget validation: PASS"
    )


# ============================================================
# SAVE
# ============================================================

def save_output(result):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("MUTUAL FUND TARGETS SAVED")
    print("=" * 70)

    print(
        f"File: {OUTPUT_FILE}"
    )

    print(
        f"Rows: {len(result):,}"
    )

    print(
        f"Schemes: "
        f"{result['scheme_code'].nunique()}"
    )


# ============================================================
# RELOAD VALIDATION
# ============================================================

def reload_validation():

    print("\n" + "=" * 70)
    print("RELOAD VALIDATION")
    print("=" * 70)

    reloaded = pd.read_parquet(
        OUTPUT_FILE
    )

    print(
        f"Reloaded rows: "
        f"{len(reloaded):,}"
    )

    print(
        f"Reloaded schemes: "
        f"{reloaded['scheme_code'].nunique()}"
    )

    if len(reloaded) != EXPECTED_ROWS:
        raise RuntimeError(
            "Unexpected row count after reload."
        )

    if (
        reloaded["scheme_code"].nunique()
        != EXPECTED_SCHEMES
    ):
        raise RuntimeError(
            "Unexpected scheme count after reload."
        )

    print(
        "Reload validation: PASS"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\nIMPORTANT:"
    )

    print(
        "Targets are generated from the CLEAN "
        "mutual-fund forward outcomes."
    )

    print(
        "Ranking is cross-sectional among mutual funds only."
    )

    print(
        "Minimum cohort size: 10."
    )

    print(
        "Labels: bottom 30% = Weak, "
        "middle 40% = Neutral, "
        "top 30% = Attractive."
    )

    print(
        "No absolute-return threshold is used."
    )

    print(
        "Raw forward returns are preserved."
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_source(
        df
    )

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    result, cohort_6m, cohort_1y = (
        generate_targets(df)
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_targets(
        df,
        result,
        cohort_6m,
        cohort_1y
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_output(
        result
    )

    # --------------------------------------------------------
    # Reload
    # --------------------------------------------------------

    reload_validation()

    print("\n" + "=" * 70)
    print("FINAL STATUS: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()