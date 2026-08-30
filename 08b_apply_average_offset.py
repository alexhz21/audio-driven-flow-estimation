"""
Code 8b - apply an average-offset correction to calibration-equation curves.

This script reads the curve CSV files created by Code 8. It calculates the
mean calibration Qmax error across all valid validation recordings:

    offset = mean(calibration Qmax - reference Qmax)

The same offset is subtracted from every point of every calibration curve,
and negative corrected values are set to zero. Gradient Boosting and reference
curves are not changed.

Outputs:
    08b_offset_corrected_outputs/
        corrected_curve_data/*.csv
        average_offset_summary.csv
        average_offset.txt
        Figure_10_offset_corrected_flow_overview.png
        Figure_11_offset_corrected_qmax_comparison.png

Important: the offset is estimated and evaluated on the same 53 validation
recordings. The corrected results are therefore a post-hoc analysis, not an
independent validation of a new method.
"""

import glob
import os
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =============================================================================
# SETTINGS
# =============================================================================

PROJECT_DIR = r"C:\school\project\everything everything"

INPUT_CURVE_DIR = os.path.join(
    PROJECT_DIR,
    "08_flow_curve_outputs",
    "curve_data",
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "08b_offset_corrected_outputs",
)
CORRECTED_CURVE_DIR = os.path.join(OUTPUT_DIR, "corrected_curve_data")

FIGURE_10_PATH = os.path.join(
    OUTPUT_DIR,
    "Figure_10_offset_corrected_flow_overview.png",
)
FIGURE_11_PATH = os.path.join(
    OUTPUT_DIR,
    "Figure_11_offset_corrected_qmax_comparison.png",
)
SUMMARY_CSV_PATH = os.path.join(OUTPUT_DIR, "average_offset_summary.csv")
OFFSET_TXT_PATH = os.path.join(OUTPUT_DIR, "average_offset.txt")

TIME_COLUMN = "time_s"
REFERENCE_COLUMN = "actual_flow_ml_s"
GRADIENT_BOOSTING_COLUMN = "gradient_boosting_flow_ml_s"
CALIBRATION_COLUMN = "calibration_equation_flow_ml_s"
CORRECTED_CALIBRATION_COLUMN = "offset_corrected_calibration_flow_ml_s"

# Accept curve files created by an older Code 8 version.
OLD_GRADIENT_BOOSTING_COLUMN = "predicted_flow_ml_s"

OVERVIEW_COUNT = 9
OVERVIEW_RANDOM_SEED = 42


# =============================================================================
# HELPERS
# =============================================================================

def clean_recording_name(curve_path):
    """Return a readable recording name from a Code 8 curve filename."""
    name = os.path.splitext(os.path.basename(curve_path))[0]
    name = re.sub(r"^\d+_", "", name)
    name = re.sub(r"_curves$", "", name)
    return name


def make_flow_title(recording_name):
    """Convert the recording name into the flow-pattern title."""
    name = str(recording_name).strip().lower()
    name = name.replace("rec_data", "normal")
    name = re.sub(r"[\s_-]*\d+$", "", name)
    name = name.replace("_data", "")
    name = name.replace("_", " ").replace("-", " ")
    name = " ".join(name.split()).strip()

    if name.endswith(" flow"):
        name = name[:-5].strip()

    flow_type = name.capitalize() if name else "Normal"
    return f"{flow_type} - flow"


def read_curve(curve_path):
    """Read one Code 8 curve file and keep valid shared samples."""
    data = pd.read_csv(curve_path)

    # rename the old column name if this curve file predates the rename in Code 8
    if (
        GRADIENT_BOOSTING_COLUMN not in data.columns
        and OLD_GRADIENT_BOOSTING_COLUMN in data.columns
    ):
        data = data.rename(
            columns={OLD_GRADIENT_BOOSTING_COLUMN: GRADIENT_BOOSTING_COLUMN}
        )

    required = [
        TIME_COLUMN,
        REFERENCE_COLUMN,
        GRADIENT_BOOSTING_COLUMN,
        CALIBRATION_COLUMN,
    ]
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
        raise ValueError("Too few valid shared samples")

    return data


def load_all_curves(curve_files):
    """Load all valid Code 8 curves and collect their raw Qmax values."""
    records = []

    for index, curve_path in enumerate(curve_files, start=1):
        try:
            data = read_curve(curve_path)
            reference_qmax = float(data[REFERENCE_COLUMN].max())
            gradient_boosting_qmax = float(
                data[GRADIENT_BOOSTING_COLUMN].max()
            )
            calibration_qmax = float(data[CALIBRATION_COLUMN].max())

            records.append(
                {
                    "recording": clean_recording_name(curve_path),
                    "source_path": curve_path,
                    "data": data,
                    "reference_qmax_ml_s": reference_qmax,
                    "gradient_boosting_qmax_ml_s": gradient_boosting_qmax,
                    "raw_calibration_qmax_ml_s": calibration_qmax,
                    "raw_calibration_qmax_error_ml_s": (
                        calibration_qmax - reference_qmax
                    ),
                }
            )
            print(f"[{index}/{len(curve_files)}] Loaded: {curve_path}")
        except Exception as error:
            print(f"[{index}/{len(curve_files)}] SKIPPED: {curve_path}")
            print(f"    {error}")

    return records


def apply_offset(records, average_offset):
    """Subtract the mean Qmax error from every calibration curve."""
    for record in records:
        data = record["data"].copy()
        corrected = np.maximum(
            data[CALIBRATION_COLUMN].to_numpy(dtype=float) - average_offset,
            0.0,
        )
        data[CORRECTED_CALIBRATION_COLUMN] = corrected
        record["data"] = data
        record["corrected_calibration_qmax_ml_s"] = float(np.max(corrected))
        record["corrected_calibration_qmax_error_ml_s"] = (
            record["corrected_calibration_qmax_ml_s"]
            - record["reference_qmax_ml_s"]
        )


def save_corrected_curves(records):
    """Save corrected curve data without overwriting Code 8 files."""
    os.makedirs(CORRECTED_CURVE_DIR, exist_ok=True)

    for record in records:
        source_name = os.path.basename(record["source_path"])
        output_path = os.path.join(CORRECTED_CURVE_DIR, source_name)
        record["data"].to_csv(output_path, index=False)
        record["corrected_path"] = output_path


def save_summary(records, average_offset):
    """Save recording-level Qmax values and the calculated offset."""
    rows = []

    for record in records:
        rows.append(
            {
                "recording": record["recording"],
                "reference_qmax_ml_s": record["reference_qmax_ml_s"],
                "gradient_boosting_qmax_ml_s": (
                    record["gradient_boosting_qmax_ml_s"]
                ),
                "raw_calibration_qmax_ml_s": (
                    record["raw_calibration_qmax_ml_s"]
                ),
                "raw_calibration_qmax_error_ml_s": (
                    record["raw_calibration_qmax_error_ml_s"]
                ),
                "average_offset_subtracted_ml_s": average_offset,
                "corrected_calibration_qmax_ml_s": (
                    record["corrected_calibration_qmax_ml_s"]
                ),
                "corrected_calibration_qmax_error_ml_s": (
                    record["corrected_calibration_qmax_error_ml_s"]
                ),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_CSV_PATH, index=False)

    raw_mae = float(
        np.mean(np.abs(summary["raw_calibration_qmax_error_ml_s"]))
    )
    corrected_mae = float(
        np.mean(np.abs(summary["corrected_calibration_qmax_error_ml_s"]))
    )

    with open(OFFSET_TXT_PATH, "w", encoding="utf-8") as file:
        file.write(f"Valid recordings: {len(summary)}\n")
        file.write(
            "Average calibration Qmax offset "
            f"(calibration - reference): {average_offset:.6f} mL/s\n"
        )
        file.write(f"Raw calibration Qmax MAE: {raw_mae:.6f} mL/s\n")
        file.write(
            "Offset-corrected calibration Qmax MAE: "
            f"{corrected_mae:.6f} mL/s\n"
        )
        file.write(
            "Warning: the offset was estimated and evaluated on the same "
            "validation recordings.\n"
        )

    return summary, raw_mae, corrected_mae


# =============================================================================
# FIGURES
# =============================================================================

def save_figure_10(records):
    """Create a nine-recording overview with offset-corrected curves."""
    count = min(OVERVIEW_COUNT, len(records))
    random_generator = np.random.default_rng(OVERVIEW_RANDOM_SEED)
    selected_indices = random_generator.choice(
        len(records),
        size=count,
        replace=False,
    )

    figure, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.ravel()

    for axis_index, record_index in enumerate(selected_indices):
        record = records[int(record_index)]
        data = record["data"]
        axis = axes[axis_index]

        axis.plot(
            data[TIME_COLUMN],
            data[GRADIENT_BOOSTING_COLUMN],
            linewidth=2.0,
            label="Gradient Boosting",
        )
        axis.plot(
            data[TIME_COLUMN],
            data[CORRECTED_CALIBRATION_COLUMN],
            linestyle=":",
            linewidth=2.0,
            label="Offset-corrected calibration",
        )
        axis.plot(
            data[TIME_COLUMN],
            data[REFERENCE_COLUMN],
            linestyle="--",
            linewidth=1.8,
            label="Scale reference",
        )

        axis.set_title(make_flow_title(record["recording"]))
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Flow rate (mL/s)")
        axis.grid(alpha=0.25)

    for axis_index in range(count, len(axes)):
        axes[axis_index].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.965),
    )
    figure.suptitle(
        "Offset-Corrected Flow-Curve Reconstruction",
        fontsize=17,
        y=0.995,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.94])
    figure.savefig(FIGURE_10_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_figure_11(summary):
    """Compare reconstructed Qmax after applying the average offset."""
    reference = summary["reference_qmax_ml_s"].to_numpy(dtype=float)
    gradient_boosting = summary[
        "gradient_boosting_qmax_ml_s"
    ].to_numpy(dtype=float)
    corrected_calibration = summary[
        "corrected_calibration_qmax_ml_s"
    ].to_numpy(dtype=float)

    maximum_value = float(
        np.max(np.concatenate([reference, gradient_boosting, corrected_calibration]))
    )
    upper_limit = max(1.0, 1.05 * maximum_value)

    figure, axes = plt.subplots(1, 2, figsize=(14, 6))

    plot_data = [
        ("Gradient Boosting", gradient_boosting),
        ("Offset-corrected calibration equation", corrected_calibration),
    ]

    for axis, (title, predicted) in zip(axes, plot_data):
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
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend()

    figure.suptitle("Qmax Obtained From Reconstructed Flow Curves", fontsize=17)
    figure.tight_layout(rect=[0, 0, 1, 0.95])
    figure.savefig(FIGURE_11_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    curve_files = sorted(glob.glob(os.path.join(INPUT_CURVE_DIR, "*_curves.csv")))
    if not curve_files:
        raise FileNotFoundError(
            "No Code 8 curve CSV files were found in:\n"
            f"{INPUT_CURVE_DIR}\n"
            "Run Code 8 before running Code 8b."
        )

    print("\n=== CODE 8B: AVERAGE-OFFSET CORRECTION ===")
    print(f"Input curve files: {len(curve_files)}")

    records = load_all_curves(curve_files)
    if not records:
        raise ValueError("No valid Code 8 curve files could be processed")

    # this is the single number subtracted from every calibration curve below
    average_offset = float(
        np.mean(
            [record["raw_calibration_qmax_error_ml_s"] for record in records]
        )
    )

    apply_offset(records, average_offset)
    save_corrected_curves(records)
    summary, raw_mae, corrected_mae = save_summary(records, average_offset)
    save_figure_10(records)
    save_figure_11(summary)

    print("\n=== OFFSET SUMMARY ===")
    print(f"Valid recordings: {len(records)}")
    print(f"Average Qmax offset: {average_offset:.4f} mL/s")
    print(f"Raw calibration Qmax MAE: {raw_mae:.4f} mL/s")
    print(f"Corrected calibration Qmax MAE: {corrected_mae:.4f} mL/s")
    print("\nSaved:")
    print(f"Corrected curves: {CORRECTED_CURVE_DIR}")
    print(f"Summary:          {SUMMARY_CSV_PATH}")
    print(f"Figure 10:        {FIGURE_10_PATH}")
    print(f"Figure 11:        {FIGURE_11_PATH}")
    print(
        "\nImportant: this is a post-hoc correction because the same "
        "recordings were used to estimate and evaluate the offset."
    )


if __name__ == "__main__":
    main()
