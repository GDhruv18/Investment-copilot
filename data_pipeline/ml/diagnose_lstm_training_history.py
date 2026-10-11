from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "lstm_results"
)

ASSETS = [
    "stocks",
    "etfs",
    "mutual_funds",
]

HORIZONS = [
    "6m",
    "1y",
    "2y",
]


def main():

    rows = []

    for asset in ASSETS:

        for horizon in HORIZONS:

            history_path = (
                RESULTS_DIR
                / asset
                / horizon
                / "training_history.csv"
            )

            if not history_path.exists():

                print(
                    f"WARNING: Missing file:\n"
                    f"{history_path}"
                )

                continue

            df = pd.read_csv(
                history_path
            )

            if df.empty:
                print(
                    f"WARNING: Empty history:\n"
                    f"{history_path}"
                )
                continue

            # Best epoch according to the same
            # criterion used by EarlyStopping:
            # minimum validation loss.
            best_index = df[
                "val_loss"
            ].idxmin()

            best = df.loc[
                best_index
            ]

            rows.append(
                {
                    "asset": asset,
                    "horizon": horizon,
                    "best_epoch": int(
                        best["epoch"]
                    ),
                    "train_loss": float(
                        best["loss"]
                    ),
                    "val_loss": float(
                        best["val_loss"]
                    ),
                    "loss_gap": float(
                        best["val_loss"]
                        - best["loss"]
                    ),
                    "train_accuracy": float(
                        best["accuracy"]
                    ),
                    "val_accuracy": float(
                        best["val_accuracy"]
                    ),
                    "accuracy_gap": float(
                        best["accuracy"]
                        - best["val_accuracy"]
                    ),
                    "epochs_trained": len(df),
                }
            )

    if not rows:

        raise RuntimeError(
            "No training_history.csv files "
            "were found."
        )

    result = pd.DataFrame(
        rows
    )

    print()
    print("=" * 100)
    print("LSTM TRAINING DYNAMICS DIAGNOSTIC")
    print("=" * 100)

    print()

    print(
        result.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print("=" * 100)
    print("INTERPRETATION COLUMNS")
    print("=" * 100)

    print(
        "loss_gap     = validation loss - training loss"
    )

    print(
        "accuracy_gap = training accuracy - validation accuracy"
    )

    print(
        "best_epoch   = epoch with minimum validation loss"
    )

    print()


if __name__ == "__main__":
    main()