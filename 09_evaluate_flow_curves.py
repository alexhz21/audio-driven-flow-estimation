import glob
import os
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# =============================================================================
# SETTINGS
# =============================================================================

PROJECT_DIR = r"C:\school\project\everything everything"

CURVE_DIR = os.path.join(
    PROJECT_DIR,
    "08_flow_curve_outputs",
    "curve_data",
)
OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "09_flow_method_comparison_outputs",
)

RESULTS_XLSX = os.path.join(OUTPUT_DIR, "flow_method_comparison.xlsx")
PER_RECORDING_CSV = os.path.join(OUTPUT_DIR, "per_recording_metrics.csv")
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "method_summary.csv")
METRIC_PLOT = os.path.join(OUTPUT_DIR, "flow_method_metrics.png")
QMAX_PLOT = os.path.join(OUTPUT_DIR, "qmax_method_comparison.png")

TIME_COLUMN = "time_s"
REFERENCE_COLUMN = "actual_flow_ml_s"

METHOD_COLUMNS = {
    "Gradient Boosting": "gradient_boosting_flow_ml_s",
    "Calibration equation": "calibration_equation_flow_ml_s",
}

# Backward-compatible name used by older Code 8 versions.
OLD_GRADIENT_BOOSTING_COLUMN = "predicted_flow_ml_s"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def rmse(actual, predicted):
    """Return root-mean-square error."""
    return float(np.sqrt(mean_squared_error(actual, predicted)))


def safe_r2(actual, predicted):
    """Return R² when it is defined."""
    # R² is undefined (or meaningless) when there's only one point or the
    # reference doesn't vary at all, so just return NaN instead of crashing
    if len(actual) < 2 or np.allclose(actual, actual[0]):
        return np.nan
    return float(r2_score(actual, predicted))


def safe_correlation(actual, predicted):
    """Return Pearson correlation when both signals vary."""
    if (
        len(actual) < 2
        or np.allclose(actual, actual[0])
        or np.allclose(predicted, predicted[0])
    ):
        return np.nan
    return float(np.corrcoef(actual, predicted)[0, 1])


def normalized_shape_rmse(actual, predicted):
    """Compare curve shape after scaling each curve to a peak of one."""
    # normalizing both curves to peak = 1 before computing RMSE isolates
    # how well the *shape* matches, independent of the magnitude/Qmax error
    actual_peak = float(np.max(actual)) if len(actual) else 0.0
    predicted_peak = float(np.max(predicted)) if len(predicted) else 0.0

    if actual_peak <= 0 or predicted_peak <= 0:
        return np.nan

    actual_normalized = actual / actual_peak
    predicted_normalized = predicted / predicted_peak
    return rmse(actual_normalized, predicted_normalized)


def clean_recording_name(curve_path):
    """Remove the Code 8 numeric file prefix and curve suffix."""
    name = os.path.splitext(os.path.basename(curve_path))[0]
    name = re.sub(r"^\d+_", "", name)
    name = re.sub(r"_curves$", "", name)
    return name


def determine_flow_type(recording_name):
    """Convert recording names into the flow-pattern labels used in plots."""
    name = recording_name.lower()
    name = name.replace("rec_data", "normal")
    name = re.sub(r"[\s_-]*\d+$", "", name)
    name = name.replace("_data", "")
    name = name.replace("_", " ").replace("-", " ")
    name = " ".join(name.split()).strip()
    return name.capitalize() if name else "Unknown"


def save_csv_safely(data, output_path):
    """Save a CSV, using a timestamped name when the normal file is open."""
    try:
        data.to_csv(output_path, index=False)
        return output_path
    except PermissionError:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
        base, extension = os.path.splitext(output_path)
        alternative = f"{base}_{timestamp}{extension}"
        data.to_csv(alternative, index=False)
        print(f"CSV file was open. Saved instead as: {alternative}")
        return alternative


def safe_excel_path(output_path):
    """Return an available Excel path."""
    try:
        with open(output_path, "a+b"):
            pass
        return output_path
    except PermissionError:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
        base, extension = os.path.splitext(output_path)
        alternative = f"{base}_{timestamp}{extension}"
        print(f"Excel file was open. Saving instead as: {alternative}")
        return alternative


# =============================================================================
# METRIC CALCULATION
# =============================================================================

def prepare_curve_table(curve_path):
    """Read one Code 8 curve file and retain valid shared samples."""
    data = pd.read_csv(curve_path)

    # Accept curve files created before the column name was made explicit.
    if (
        METHOD_COLUMNS["Gradient Boosting"] not in data.columns
        and OLD_GRADIENT_BOOSTING_COLUMN in data.columns
    ):
        data = data.rename(
            columns={
                OLD_GRADIENT_BOOSTING_COLUMN:
                    METHOD_COLUMNS["Gradient Boosting"]
            }
        )

    required_columns = [
        TIME_COLUMN,
        REFERENCE_COLUMN,
        *METHOD_COLUMNS.values(),
    ]
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for column in required_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    valid = np.ones(len(data), dtype=bool)
    for column in required_columns:
        valid &= np.isfinite(data[column].to_numpy(dtype=float))

    data = data.loc[valid, required_columns].reset_index(drop=True)
    if len(data) < 5:
        raise ValueError("Too few valid shared curve samples")

    return data


def calculate_recording_metrics(curve_path):
    """Calculate both methods' errors for one recording."""
    data = prepare_curve_table(curve_path)
    time = data[TIME_COLUMN].to_numpy(dtype=float)
    actual = data[REFERENCE_COLUMN].to_numpy(dtype=float)

    recording_name = clean_recording_name(curve_path)
    flow_type = determine_flow_type(recording_name)
    actual_qmax = float(np.max(actual))
    actual_volume = float(np.trapz(actual, time))

    rows = []
    for method_name, method_column in METHOD_COLUMNS.items():
        predicted = data[method_column].to_numpy(dtype=float)
        errors = predicted - actual
        predicted_qmax = float(np.max(predicted))
        predicted_volume = float(np.trapz(predicted, time))

        rows.append(
            {
                "recording": recording_name,
                "flow_type": flow_type,
                "curve_file": curve_path,
                "method": method_name,
                "sample_count": len(data),
                "duration_s": float(time[-1] - time[0]),
                "rmse_ml_s": rmse(actual, predicted),
                "mae_ml_s": float(mean_absolute_error(actual, predicted)),
                "r2": safe_r2(actual, predicted),
                "correlation_r": safe_correlation(actual, predicted),
                "mean_error_ml_s": float(np.mean(errors)),
                "maximum_absolute_error_ml_s": float(np.max(np.abs(errors))),
                "shape_rmse_normalized": normalized_shape_rmse(
                    actual,
                    predicted,
                ),
                "reference_qmax_ml_s": actual_qmax,
                "predicted_qmax_ml_s": predicted_qmax,
                "qmax_error_ml_s": predicted_qmax - actual_qmax,
                "absolute_qmax_error_ml_s": abs(predicted_qmax - actual_qmax),
                "reference_volume_ml": actual_volume,
                "predicted_volume_ml": predicted_volume,
                "volume_error_ml": predicted_volume - actual_volume,
                "absolute_volume_error_ml": abs(
                    predicted_volume - actual_volume
                ),
            }
        )

    return rows


def create_method_summary(per_recording):
    """Summarize the distribution of recording-level metrics."""
    metric_columns = [
        "rmse_ml_s",
        "mae_ml_s",
        "r2",
        "correlation_r",
        "mean_error_ml_s",
        "maximum_absolute_error_ml_s",
        "shape_rmse_normalized",
        "absolute_qmax_error_ml_s",
        "absolute_volume_error_ml",
    ]

    # averaging per-recording metrics (rather than pooling all samples first)
    # gives every recording equal weight, regardless of how long it is
    rows = []
    for method_name, method_data in per_recording.groupby("method"):
        row = {
            "method": method_name,
            "recordings": int(method_data["recording"].nunique()),
        }
        for column in metric_columns:
            values = pd.to_numeric(method_data[column], errors="coerce")
            row[f"mean_{column}"] = float(values.mean())
            row[f"sd_{column}"] = float(values.std(ddof=1))
            row[f"median_{column}"] = float(values.median())
        rows.append(row)

    return pd.DataFrame(rows).sort_values("mean_rmse_ml_s").reset_index(drop=True)


def create_pooled_summary(curve_files):
    """Calculate metrics after combining all valid time samples."""
    actual_parts = []
    prediction_parts = {method: [] for method in METHOD_COLUMNS}

    for curve_path in curve_files:
        try:
            data = prepare_curve_table(curve_path)
        except Exception:
            continue

        actual_parts.append(data[REFERENCE_COLUMN].to_numpy(dtype=float))
        for method_name, column in METHOD_COLUMNS.items():
            prediction_parts[method_name].append(
                data[column].to_numpy(dtype=float)
            )

    if not actual_parts:
        return pd.DataFrame()

    actual = np.concatenate(actual_parts)
    rows = []
    for method_name, parts in prediction_parts.items():
        predicted = np.concatenate(parts)
        rows.append(
            {
                "method": method_name,
                "total_time_samples": len(actual),
                "pooled_rmse_ml_s": rmse(actual, predicted),
                "pooled_mae_ml_s": float(
                    mean_absolute_error(actual, predicted)
                ),
                "pooled_r2": safe_r2(actual, predicted),
                "pooled_correlation_r": safe_correlation(actual, predicted),
                "pooled_mean_error_ml_s": float(np.mean(predicted - actual)),
            }
        )

    return pd.DataFrame(rows).sort_values("pooled_rmse_ml_s").reset_index(
        drop=True
    )


# =============================================================================
# FIGURES
# =============================================================================

def save_metric_plot(per_recording):
    """Create recording-level boxplots for the main comparison metrics."""
    methods = list(METHOD_COLUMNS.keys())
    colors = ["#1f77b4", "#ff7f0e"]
    specifications = [
        ("rmse_ml_s", "Curve RMSE (mL/s)"),
        ("mae_ml_s", "Curve MAE (mL/s)"),
        ("absolute_qmax_error_ml_s", "Absolute Qmax error (mL/s)"),
        ("shape_rmse_normalized", "Normalized shape RMSE"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)

    for axis, (column, title) in zip(axes.ravel(), specifications):
        values = [
            per_recording.loc[
                per_recording["method"] == method,
                column,
            ].dropna().to_numpy()
            for method in methods
        ]
        boxes = axis.boxplot(values, labels=methods, patch_artist=True)
        for box, color in zip(boxes["boxes"], colors):
            box.set_facecolor(color)
            box.set_alpha(0.65)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=10)

    fig.suptitle(
        "Flow-Curve Reconstruction Method Comparison",
        fontsize=16,
    )
    plt.savefig(METRIC_PLOT, dpi=220)
    plt.close()


def save_qmax_plot(per_recording):
    """Compare Qmax obtained from each reconstructed curve."""
    methods = list(METHOD_COLUMNS.keys())
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)

    all_values = pd.concat(
        [
            per_recording["reference_qmax_ml_s"],
            per_recording["predicted_qmax_ml_s"],
        ],
        ignore_index=True,
    )
    low = float(all_values.min())
    high = float(all_values.max())

    for axis, method in zip(axes, methods):
        subset = per_recording[per_recording["method"] == method]
        axis.scatter(
            subset["reference_qmax_ml_s"],
            subset["predicted_qmax_ml_s"],
            alpha=0.75,
        )
        axis.plot([low, high], [low, high], "k--", label="Perfect agreement")
        axis.set_title(method)
        axis.set_xlabel("Reference curve Qmax (mL/s)")
        axis.set_ylabel("Reconstructed curve Qmax (mL/s)")
        axis.grid(alpha=0.25)
        axis.legend()

    fig.suptitle("Qmax Obtained From Reconstructed Flow Curves", fontsize=16)
    plt.savefig(QMAX_PLOT, dpi=220)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n=== FLOW RECONSTRUCTION METHOD COMPARISON ===")
    print(f"Reading curve files from: {CURVE_DIR}")

    curve_files = sorted(glob.glob(os.path.join(CURVE_DIR, "*_curves.csv")))
    if not curve_files:
        raise FileNotFoundError(
            f"No *_curves.csv files were found in:\n{CURVE_DIR}\n"
            "Run the updated Code 8 first."
        )

    rows = []
    status_rows = []

    for index, curve_path in enumerate(curve_files, start=1):
        try:
            rows.extend(calculate_recording_metrics(curve_path))
            status_rows.append(
                {
                    "curve_file": curve_path,
                    "status": "ok",
                }
            )
            print(f"[{index}/{len(curve_files)}] Processed | {curve_path}")
        except Exception as error:
            status_rows.append(
                {
                    "curve_file": curve_path,
                    "status": f"error: {error}",
                }
            )
            print(f"[{index}/{len(curve_files)}] ERROR | {error}")

    if not rows:
        raise ValueError("No curve files could be evaluated")

    per_recording = pd.DataFrame(rows)
    method_summary = create_method_summary(per_recording)
    pooled_summary = create_pooled_summary(curve_files)
    status_table = pd.DataFrame(status_rows)

    saved_per_recording_csv = save_csv_safely(
        per_recording,
        PER_RECORDING_CSV,
    )
    saved_summary_csv = save_csv_safely(method_summary, SUMMARY_CSV)

    excel_path = safe_excel_path(RESULTS_XLSX)
    with pd.ExcelWriter(excel_path) as writer:
        method_summary.to_excel(writer, sheet_name="method_summary", index=False)
        pooled_summary.to_excel(writer, sheet_name="pooled_summary", index=False)
        per_recording.to_excel(
            writer,
            sheet_name="per_recording_metrics",
            index=False,
        )
        status_table.to_excel(writer, sheet_name="file_status", index=False)

    save_metric_plot(per_recording)
    save_qmax_plot(per_recording)

    successful_files = int((status_table["status"] == "ok").sum())
    failed_files = len(status_table) - successful_files
    best_method = method_summary.iloc[0]["method"]

    print("\n=== SUMMARY ===")
    print(
        method_summary[
            [
                "method",
                "recordings",
                "mean_rmse_ml_s",
                "mean_mae_ml_s",
                "mean_r2",
                "mean_correlation_r",
                "mean_absolute_qmax_error_ml_s",
                "mean_shape_rmse_normalized",
            ]
        ].round(4).to_string(index=False)
    )
    print(f"\nLowest mean curve RMSE: {best_method}")
    print(f"Successful curve files: {successful_files}")
    print(f"Failed curve files:     {failed_files}")
    print(f"Excel results:          {excel_path}")
    print(f"Per-recording CSV:      {saved_per_recording_csv}")
    print(f"Summary CSV:            {saved_summary_csv}")
    print(f"Metric plot:            {METRIC_PLOT}")
    print(f"Qmax plot:              {QMAX_PLOT}")
    print("Done.")


if __name__ == "__main__":
    main()
