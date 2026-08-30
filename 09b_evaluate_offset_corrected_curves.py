"""
Code 9b - evaluate raw and offset-corrected flow reconstructions.

Run Code 8 and Code 8b first. This script reads the corrected curve CSV files
created by Code 8b and compares three methods with the scale reference:

1. Gradient Boosting reconstruction
2. Original calibration-equation reconstruction
3. Offset-corrected calibration-equation reconstruction

The script calculates metrics separately for every recording and then reports
their mean, standard deviation, and median. It also calculates pooled metrics
after combining all valid time samples.

Important: the offset-corrected method is post-hoc because the same validation
recordings were used to estimate and evaluate the average offset.
"""

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
    "08b_offset_corrected_outputs",
    "corrected_curve_data",
)
OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "09b_offset_corrected_evaluation_outputs",
)

RESULTS_XLSX = os.path.join(OUTPUT_DIR, "offset_corrected_comparison.xlsx")
PER_RECORDING_CSV = os.path.join(OUTPUT_DIR, "per_recording_metrics.csv")
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "method_summary.csv")
POOLED_CSV = os.path.join(OUTPUT_DIR, "pooled_summary.csv")
METRIC_PLOT = os.path.join(OUTPUT_DIR, "offset_corrected_method_metrics.png")
QMAX_PLOT = os.path.join(OUTPUT_DIR, "offset_corrected_qmax_comparison.png")
TEXT_SUMMARY = os.path.join(OUTPUT_DIR, "offset_correction_summary.txt")

TIME_COLUMN = "time_s"
REFERENCE_COLUMN = "actual_flow_ml_s"

METHOD_COLUMNS = {
    "Gradient Boosting": "gradient_boosting_flow_ml_s",
    "Calibration equation": "calibration_equation_flow_ml_s",
    "Offset-corrected calibration": (
        "offset_corrected_calibration_flow_ml_s"
    ),
}

OLD_GRADIENT_BOOSTING_COLUMN = "predicted_flow_ml_s"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# METRIC HELPERS
# =============================================================================

def rmse(actual, predicted):
    """Return root-mean-square error."""
    return float(np.sqrt(mean_squared_error(actual, predicted)))


def safe_r2(actual, predicted):
    """Return R-squared when it is defined."""
    if len(actual) < 2 or np.allclose(actual, actual[0]):
        return np.nan
    return float(r2_score(actual, predicted))


def safe_correlation(actual, predicted):
    """Return Pearson correlation when both curves vary."""
    if (
        len(actual) < 2
        or np.allclose(actual, actual[0])
        or np.allclose(predicted, predicted[0])
    ):
        return np.nan
    return float(np.corrcoef(actual, predicted)[0, 1])


def normalized_shape_rmse(actual, predicted):
    """Compare curve shape after scaling each curve to a peak of one."""
    actual_peak = float(np.max(actual)) if len(actual) else 0.0
    predicted_peak = float(np.max(predicted)) if len(predicted) else 0.0

    if actual_peak <= 0 or predicted_peak <= 0:
        return np.nan

    return rmse(actual / actual_peak, predicted / predicted_peak)


def clean_recording_name(curve_path):
    """Remove the Code 8 numeric prefix and curve suffix."""
    name = os.path.splitext(os.path.basename(curve_path))[0]
    name = re.sub(r"^\d+_", "", name)
    name = re.sub(r"_curves$", "", name)
    return name


def determine_flow_type(recording_name):
    """Convert a recording name into a readable flow-pattern label."""
    name = recording_name.lower().replace("rec_data", "normal")
    name = re.sub(r"[\s_-]*\d+$", "", name)
    name = name.replace("_data", "")
    name = name.replace("_", " ").replace("-", " ")
    name = " ".join(name.split()).strip()
    return name.capitalize() if name else "Unknown"


def prepare_curve_table(curve_path):
    """Read one corrected curve file and retain valid shared samples."""
    data = pd.read_csv(curve_path)

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

    required = [TIME_COLUMN, REFERENCE_COLUMN, *METHOD_COLUMNS.values()]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for column in required:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    valid = np.ones(len(data), dtype=bool)
    for column in required:
        valid &= np.isfinite(data[column].to_numpy(dtype=float))

    data = data.loc[valid, required].reset_index(drop=True)
    if len(data) < 5:
        raise ValueError("Too few valid shared curve samples")

    return data


# =============================================================================
# RECORDING-LEVEL AND POOLED METRICS
# =============================================================================

def calculate_recording_metrics(curve_path):
    """Calculate all three methods' metrics for one recording."""
    data = prepare_curve_table(curve_path)
    time = data[TIME_COLUMN].to_numpy(dtype=float)
    actual = data[REFERENCE_COLUMN].to_numpy(dtype=float)

    recording_name = clean_recording_name(curve_path)
    flow_type = determine_flow_type(recording_name)
    reference_qmax = float(np.max(actual))
    reference_volume = float(np.trapz(actual, time))

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
                "maximum_absolute_error_ml_s": float(
                    np.max(np.abs(errors))
                ),
                "shape_rmse_normalized": normalized_shape_rmse(
                    actual,
                    predicted,
                ),
                "reference_qmax_ml_s": reference_qmax,
                "predicted_qmax_ml_s": predicted_qmax,
                "qmax_error_ml_s": predicted_qmax - reference_qmax,
                "absolute_qmax_error_ml_s": abs(
                    predicted_qmax - reference_qmax
                ),
                "reference_volume_ml": reference_volume,
                "predicted_volume_ml": predicted_volume,
                "volume_error_ml": predicted_volume - reference_volume,
                "absolute_volume_error_ml": abs(
                    predicted_volume - reference_volume
                ),
            }
        )

    return rows


def create_method_summary(per_recording):
    """Average recording-level metrics so every recording has equal weight."""
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

    return pd.DataFrame(rows).sort_values("mean_rmse_ml_s").reset_index(
        drop=True
    )


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
                "pooled_mean_error_ml_s": float(
                    np.mean(predicted - actual)
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("pooled_rmse_ml_s").reset_index(
        drop=True
    )


# =============================================================================
# FIGURES
# =============================================================================

def save_metric_plot(per_recording):
    """Create boxplots for the main recording-level metrics."""
    methods = list(METHOD_COLUMNS.keys())
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    specifications = [
        ("rmse_ml_s", "Curve RMSE (mL/s)"),
        ("mae_ml_s", "Curve MAE (mL/s)"),
        ("absolute_qmax_error_ml_s", "Absolute Qmax error (mL/s)"),
        ("shape_rmse_normalized", "Normalized shape RMSE"),
    ]

    figure, axes = plt.subplots(2, 2, figsize=(15, 10))

    for axis, (column, title) in zip(axes.ravel(), specifications):
        values = [
            per_recording.loc[
                per_recording["method"] == method,
                column,
            ].dropna().to_numpy()
            for method in methods
        ]
        boxes = axis.boxplot(values, labels=methods, patch_artist=True)
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", labelrotation=12)

    figure.suptitle(
        "Raw and Offset-Corrected Flow-Reconstruction Metrics",
        fontsize=17,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    figure.savefig(METRIC_PLOT, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_qmax_plot(per_recording):
    """Create a three-panel Qmax agreement plot."""
    methods = list(METHOD_COLUMNS.keys())
    figure, axes = plt.subplots(1, 3, figsize=(19, 6))

    maximum_value = float(
        max(
            per_recording["reference_qmax_ml_s"].max(),
            per_recording["predicted_qmax_ml_s"].max(),
        )
    )
    upper_limit = max(1.0, 1.05 * maximum_value)

    for axis, method_name in zip(axes, methods):
        subset = per_recording[per_recording["method"] == method_name]
        reference = subset["reference_qmax_ml_s"].to_numpy(dtype=float)
        predicted = subset["predicted_qmax_ml_s"].to_numpy(dtype=float)

        axis.scatter(reference, predicted, alpha=0.8)
        axis.plot(
            [0, upper_limit],
            [0, upper_limit],
            "k--",
            label="Perfect agreement",
        )
        axis.set_xlim(0, upper_limit)
        axis.set_ylim(0, upper_limit)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Reference curve Qmax (mL/s)")
        axis.set_ylabel("Reconstructed curve Qmax (mL/s)")
        axis.set_title(method_name)
        axis.grid(alpha=0.25)
        axis.legend()

    figure.suptitle(
        "Qmax Comparison Before and After Offset Correction",
        fontsize=17,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.95])
    figure.savefig(QMAX_PLOT, dpi=300, bbox_inches="tight")
    plt.close(figure)


# =============================================================================
# SAVING AND MAIN
# =============================================================================

def save_text_summary(method_summary, pooled_summary, failed_files):
    """Write a compact human-readable summary."""
    with open(TEXT_SUMMARY, "w", encoding="utf-8") as file:
        file.write("CODE 9B - OFFSET-CORRECTED CURVE EVALUATION\n\n")
        file.write(
            "The offset-corrected method is post-hoc: the same validation "
            "recordings estimated and evaluated the offset.\n\n"
        )
        file.write("MEAN RECORDING-LEVEL METRICS\n")
        for _, row in method_summary.iterrows():
            file.write(
                f"{row['method']}: "
                f"RMSE={row['mean_rmse_ml_s']:.4f} mL/s, "
                f"MAE={row['mean_mae_ml_s']:.4f} mL/s, "
                f"correlation={row['mean_correlation_r']:.4f}, "
                "absolute Qmax error="
                f"{row['mean_absolute_qmax_error_ml_s']:.4f} mL/s, "
                "normalized shape RMSE="
                f"{row['mean_shape_rmse_normalized']:.4f}\n"
            )

        file.write("\nPOOLED METRICS\n")
        for _, row in pooled_summary.iterrows():
            file.write(
                f"{row['method']}: "
                f"RMSE={row['pooled_rmse_ml_s']:.4f} mL/s, "
                f"MAE={row['pooled_mae_ml_s']:.4f} mL/s, "
                f"correlation={row['pooled_correlation_r']:.4f}\n"
            )

        file.write(f"\nFailed curve files: {failed_files}\n")


def main():
    curve_files = sorted(glob.glob(os.path.join(CURVE_DIR, "*_curves.csv")))
    if not curve_files:
        raise FileNotFoundError(
            "No corrected curve files were found in:\n"
            f"{CURVE_DIR}\n"
            "Run Code 8 and Code 8b before running Code 9b."
        )

    print("\n=== CODE 9B: OFFSET-CORRECTED CURVE EVALUATION ===")
    print(f"Curve files found: {len(curve_files)}")

    all_rows = []
    failed_files = 0

    for index, curve_path in enumerate(curve_files, start=1):
        try:
            all_rows.extend(calculate_recording_metrics(curve_path))
            print(f"[{index}/{len(curve_files)}] Processed: {curve_path}")
        except Exception as error:
            failed_files += 1
            print(f"[{index}/{len(curve_files)}] ERROR: {curve_path}")
            print(f"    {error}")

    if not all_rows:
        raise ValueError("No valid corrected curve files could be evaluated")

    per_recording = pd.DataFrame(all_rows)
    method_summary = create_method_summary(per_recording)
    pooled_summary = create_pooled_summary(curve_files)

    per_recording.to_csv(PER_RECORDING_CSV, index=False)
    method_summary.to_csv(SUMMARY_CSV, index=False)
    pooled_summary.to_csv(POOLED_CSV, index=False)

    with pd.ExcelWriter(RESULTS_XLSX) as writer:
        per_recording.to_excel(
            writer,
            sheet_name="Per recording",
            index=False,
        )
        method_summary.to_excel(
            writer,
            sheet_name="Recording-level summary",
            index=False,
        )
        pooled_summary.to_excel(
            writer,
            sheet_name="Pooled summary",
            index=False,
        )

    save_metric_plot(per_recording)
    save_qmax_plot(per_recording)
    save_text_summary(method_summary, pooled_summary, failed_files)

    display_columns = [
        "method",
        "recordings",
        "mean_rmse_ml_s",
        "mean_mae_ml_s",
        "mean_correlation_r",
        "mean_absolute_qmax_error_ml_s",
        "mean_shape_rmse_normalized",
    ]

    print("\n=== MEAN RECORDING-LEVEL RESULTS ===")
    print(method_summary[display_columns].to_string(index=False))
    print("\n=== POOLED RESULTS ===")
    print(pooled_summary.to_string(index=False))
    print("\n=== FILE SUMMARY ===")
    print(f"Successful curve files: {len(curve_files) - failed_files}")
    print(f"Failed curve files:     {failed_files}")
    print(f"Excel results:          {RESULTS_XLSX}")
    print(f"Per-recording CSV:      {PER_RECORDING_CSV}")
    print(f"Summary CSV:            {SUMMARY_CSV}")
    print(f"Pooled CSV:             {POOLED_CSV}")
    print(f"Metric plot:            {METRIC_PLOT}")
    print(f"Qmax plot:              {QMAX_PLOT}")
    print(
        "\nImportant: the offset-corrected results are post-hoc and "
        "are not an independent validation."
    )


if __name__ == "__main__":
    main()
