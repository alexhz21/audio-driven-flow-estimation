import os
import warnings

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Settings
PROJECT_DIR = r"C:\school\project\everything everything"
KEY_FILE = os.path.join(PROJECT_DIR, "everything_key.xlsx")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "01_dataset_summary_outputs")

AUDIO_COL = "audio_path"
CSV_COL = "csv_path"
DEFAULT_SCALE_HZ = 8.0
URINE_DENSITY = 1.02
SMOOTH_SEC = 0.75


def resolve_path(value):
    """Return an absolute project path."""
    path = str(value).strip()
    return path if os.path.isabs(path) else os.path.join(PROJECT_DIR, path)


def find_column(df, choices):
    """Find the first matching column."""
    for name in choices:
        if name in df.columns:
            return name
    return None


def audio_duration(audio_path):
    """Read audio duration without loading the full signal."""
    return float(librosa.get_duration(path=audio_path))


def scale_results(csv_path):
    """Calculate duration, Qmax, and volume."""
    if csv_path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(csv_path)
    else:
        df = pd.read_csv(csv_path)

    weight_col = find_column(df, ["Weight", "weight", "grams", "g", "mass"])
    time_col = find_column(
        df, ["DateTime", "datetime", "time", "Time", "seconds", "Seconds", "t"]
    )

    if weight_col is None:
        raise ValueError("Weight column not found")

    weight = pd.to_numeric(df[weight_col], errors="coerce")
    weight = weight.interpolate().bfill().ffill().to_numpy(dtype=float)

    if time_col and "date" in time_col.lower():
        dates = pd.to_datetime(df[time_col], dayfirst=True, errors="coerce")
        time = (dates - dates.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    elif time_col:
        time = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
        time = time - time[0]
    else:
        time = np.arange(len(weight), dtype=float) / DEFAULT_SCALE_HZ

    valid_dt = np.diff(time)
    valid_dt = valid_dt[np.isfinite(valid_dt) & (valid_dt > 0)]
    dt = float(np.median(valid_dt)) if len(valid_dt) else 1.0 / DEFAULT_SCALE_HZ

    window = max(1, int(round(SMOOTH_SEC / dt)))
    smooth_weight = (
        pd.Series(weight)
        .rolling(window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )

    flow = np.gradient(smooth_weight, dt) / URINE_DENSITY
    flow = (
        pd.Series(flow)
        .rolling(window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )
    flow = np.maximum(flow, 0.0)

    return {
        "scale_duration_s": float(time[-1] - time[0]),
        "qmax_ml_s": float(np.max(flow)),
        "voided_volume_ml": float(np.trapezoid(flow, time)),
    }


def save_histogram(data, column, title, xlabel, filename):
    """Save and show one histogram."""
    values = data[column].dropna()
    if values.empty:
        return

    plt.figure(figsize=(7, 5))
    plt.hist(values, bins="auto", edgecolor="black", alpha=0.8)
    plt.axvline(values.mean(), color="red", linestyle="--", label="Mean")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of recordings")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=250)
    plt.show()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n=== DATASET SUMMARY ===")
    print(f"Reading: {KEY_FILE}")

    key = pd.read_excel(KEY_FILE)
    required = [AUDIO_COL, CSV_COL]
    missing = [column for column in required if column not in key.columns]
    if missing:
        raise ValueError(f"Missing Excel columns: {missing}")

    rows = []
    total = len(key)

    for index, row in key.iterrows():
        audio_path = resolve_path(row[AUDIO_COL])
        csv_path = resolve_path(row[CSV_COL])

        result = {
            "excel_row": index + 2,
            "audio_path": audio_path,
            "csv_path": csv_path,
            "status": "ok",
        }

        split_col = find_column(key, ["split", "set", "dataset", "group"])
        if split_col:
            result["split"] = row[split_col]

        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio not found: {audio_path}")
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Scale file not found: {csv_path}")

            result["audio_duration_s"] = audio_duration(audio_path)
            result.update(scale_results(csv_path))

            print(
                f"[{index + 1}/{total}] OK | "
                f"Qmax={result['qmax_ml_s']:.2f} mL/s | "
                f"{os.path.basename(audio_path)}"
            )
        except Exception as error:
            result["status"] = f"error: {error}"
            print(f"[{index + 1}/{total}] ERROR | {error}")

        rows.append(result)

    details = pd.DataFrame(rows)
    valid = details[details["status"] == "ok"].copy()

    numeric_columns = [
        "audio_duration_s",
        "scale_duration_s",
        "qmax_ml_s",
        "voided_volume_ml",
    ]
    summary = valid[numeric_columns].agg(["count", "mean", "std", "min", "max"]).T
    summary.index.name = "measurement"

    details_path = os.path.join(OUTPUT_DIR, "dataset_details.xlsx")
    summary_path = os.path.join(OUTPUT_DIR, "dataset_summary.xlsx")
    details.to_excel(details_path, index=False)
    summary.to_excel(summary_path)

    print("\n=== TOTALS ===")
    print(f"Excel rows:       {len(details)}")
    print(f"Usable rows:      {len(valid)}")
    print(f"Failed rows:      {len(details) - len(valid)}")

    if "split" in valid.columns:
        print("\nRecordings by split:")
        print(valid["split"].value_counts(dropna=False).to_string())

    print("\n=== NUMERIC SUMMARY ===")
    print(summary.round(3).to_string())

    save_histogram(
        valid,
        "qmax_ml_s",
        "Qmax distribution",
        "Qmax (mL/s)",
        "qmax_distribution.png",
    )
    save_histogram(
        valid,
        "audio_duration_s",
        "Recording-duration distribution",
        "Duration (s)",
        "duration_distribution.png",
    )
    save_histogram(
        valid,
        "voided_volume_ml",
        "Voided-volume distribution",
        "Volume (mL)",
        "volume_distribution.png",
    )

    print("\n  SAVED FILES ")
    print(details_path)
    print(summary_path)
    print(f"Plots: {OUTPUT_DIR}")
    print("\nDone.")


if __name__ == "__main__":
    main()
