import os
import re
import warnings

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =============================================================================
# SETTINGS
# =============================================================================

PROJECT_DIR = r"C:\school\project\everything everything"

GRADIENT_BOOSTING_RESULTS = os.path.join(
    PROJECT_DIR,
    "07_gradient_boosting_validation_outputs",
    "validation_predictions.xlsx",
)

OUTPUT_DIR = os.path.join(PROJECT_DIR, "08_flow_curve_outputs")
CURVE_DIR = os.path.join(OUTPUT_DIR, "curve_data")
PLOT_DIR = os.path.join(OUTPUT_DIR, "curve_plots")

AUDIO_COL = "audio_path"
CSV_COL = "csv_path"
QMAX_COL = "predicted_qmax_ml_s"

URINE_DENSITY = 1.02
SCALE_WINDOW = 7
AUDIO_SMOOTH_SEC = 1.0
ONSET_SUSTAIN_SEC = 0.3

# Power-law calibration developed in Section 3.2:
# Q = CALIBRATION_A * E ** CALIBRATION_B
CALIBRATION_A = 38.336
CALIBRATION_B = 0.390654
CALIBRATION_TARGET_SR = 22050
CALIBRATION_ENERGY_WINDOW_SEC = 0.5
CALIBRATION_NOISE_FRACTION = 0.05

# Number of curves shown in the combined overview.
OVERVIEW_COUNT = 9

# A fixed seed makes the randomly selected recordings reproducible.
# Change the number to obtain a different random selection.
# Set to None to select different recordings every time the code runs.
OVERVIEW_RANDOM_SEED = 42


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def resolve_path(value):
    """Return an absolute project path."""
    path = str(value).strip()
    return path if os.path.isabs(path) else os.path.join(PROJECT_DIR, path)


def make_flow_title(audio_name):
    """Convert the recording filename into a clean flow-type title."""
    name = str(audio_name).strip().lower()

    # The recordings named "rec_data" represent normal flow.
    name = name.replace("rec_data", "normal")

    # Remove a general trailing "_data" label from other flow types.
    if name.endswith("_data"):
        name = name[:-5]

    # Remove recording identifiers such as _93, -19, or " 104".
    # Recordings of the same flow type therefore receive the same title.
    name = re.sub(r"[\s_-]*\d+$", "", name).strip()

    name = name.replace("_", " ").replace("-", " ")
    name = " ".join(name.split()).strip()

    # Avoid creating titles such as "Normal flow - flow".
    if name.endswith(" flow"):
        name = name[:-5].strip()

    flow_type = name.capitalize() if name else "Normal"
    return f"{flow_type} - flow"


def moving_average(values, window):
    """Smooth a signal with a moving-average filter."""
    window = max(1, int(window))
    return np.convolve(values, np.ones(window) / window, mode="same")


def first_sustained(mask, length):
    """Find the first sustained True region."""
    run = 0

    for index, value in enumerate(mask):
        run = run + 1 if value else 0
        if run >= length:
            return index - length + 1

    return 0


def align_onset(time, signal, sampling_rate, fraction):
    """Align a signal to the beginning of its sustained activity."""
    peak = float(np.nanmax(signal)) if len(signal) else 0.0
    threshold = max(1e-12, fraction * peak)

    onset_index = first_sustained(
        np.isfinite(signal) & (signal > threshold),
        max(1, int(round(ONSET_SUSTAIN_SEC * sampling_rate))),
    )

    return time[onset_index:] - time[onset_index], signal[onset_index:]


# =============================================================================
# FLOW-CURVE CALCULATION
# =============================================================================

def predicted_flows_from_audio(audio_path, predicted_qmax):
    """Create Gradient Boosting and calibration-equation flow curves."""
    signal, sampling_rate = librosa.load(
        audio_path,
        sr=CALIBRATION_TARGET_SR,
        mono=True,
    )

    if len(signal) == 0:
        raise ValueError("Empty audio file")

    hop_length = 512
    frame_length = 2048
    rms = librosa.feature.rms(
        y=signal,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]
    time = librosa.frames_to_time(
        np.arange(len(rms)),
        sr=sampling_rate,
        hop_length=hop_length,
    )
    frame_rate = sampling_rate / hop_length

    rms = np.nan_to_num(rms, nan=0.0)
    smooth_window = max(1, int(round(AUDIO_SMOOTH_SEC * frame_rate)))

    # Apply the moving average twice to produce a smoother model-based curve.
    rms = moving_average(rms, smooth_window)
    rms = moving_average(rms, smooth_window)

    # Use the same normalized mel-power definition used when the calibration
    # equation was developed. A 0.5-second moving window approximates the
    # energy window used for the power-law calibration feature.
    calibration_signal = signal.copy()
    signal_peak = float(np.max(np.abs(calibration_signal)))
    if signal_peak > 0:
        calibration_signal = calibration_signal / signal_peak

    mel_spectrogram = librosa.feature.melspectrogram(
        y=calibration_signal,
        sr=sampling_rate,
        n_fft=frame_length,
        hop_length=hop_length,
        n_mels=128,
        power=2.0,
    )
    mean_mel_power = np.mean(mel_spectrogram, axis=0)
    calibration_window = max(
        1,
        int(round(CALIBRATION_ENERGY_WINDOW_SEC * frame_rate)),
    )
    mean_mel_power = moving_average(mean_mel_power, calibration_window)
    mean_mel_power = np.maximum(
        np.nan_to_num(mean_mel_power, nan=0.0),
        0.0,
    )

    # Very low energy is treated as background so the reconstructed curve can
    # return to zero after the flow event.
    maximum_mel_power = float(np.max(mean_mel_power))
    if maximum_mel_power > 0:
        mean_mel_power[
            mean_mel_power
            < CALIBRATION_NOISE_FRACTION * maximum_mel_power
        ] = 0.0

    calibration_flow = np.where(
        mean_mel_power > 0,
        CALIBRATION_A * np.power(mean_mel_power, CALIBRATION_B),
        0.0,
    )
    calibration_flow = moving_average(calibration_flow, smooth_window)

    # RMS and mel power use the same frame and hop settings, but clip all
    # arrays to the same length as a safeguard.
    common_length = min(len(time), len(rms), len(calibration_flow))
    time = time[:common_length]
    rms = rms[:common_length]
    calibration_flow = calibration_flow[:common_length]

    rms_peak = float(np.max(rms)) if len(rms) else 0.0
    onset_index = first_sustained(
        np.isfinite(rms) & (rms > max(1e-12, 0.10 * rms_peak)),
        max(1, int(round(ONSET_SUSTAIN_SEC * frame_rate))),
    )
    time = time[onset_index:] - time[onset_index]
    rms = rms[onset_index:]
    calibration_flow = calibration_flow[onset_index:]

    peak = float(np.max(rms))
    if peak > 0:
        predicted_flow = predicted_qmax * rms / peak
    else:
        predicted_flow = np.zeros_like(rms)

    return time, predicted_flow, calibration_flow


def actual_flow_from_scale(csv_path):
    """Convert the scale-weight recording into a reference flow curve."""
    data = pd.read_csv(csv_path)

    if "Weight" not in data.columns or "DateTime" not in data.columns:
        raise ValueError("Scale CSV needs Weight and DateTime columns")

    weight = pd.to_numeric(data["Weight"], errors="coerce")
    dates = pd.to_datetime(data["DateTime"], dayfirst=True, errors="coerce")

    valid = weight.notna() & dates.notna()
    weight = weight[valid].reset_index(drop=True)
    dates = dates[valid].reset_index(drop=True)

    if len(weight) < 10:
        raise ValueError("Too few scale samples")

    time = (dates - dates.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    valid_time_differences = np.diff(time)
    valid_time_differences = valid_time_differences[
        valid_time_differences > 0
    ]

    if len(valid_time_differences) == 0:
        raise ValueError("Scale timestamps are invalid")

    time_step = float(np.median(valid_time_differences))
    sampling_rate = 1.0 / time_step

    smooth_weight = (
        weight.rolling(
            SCALE_WINDOW,
            center=True,
            min_periods=1,
        )
        .mean()
        .to_numpy()
    )

    flow = np.gradient(smooth_weight, time_step) / URINE_DENSITY
    flow = moving_average(flow, SCALE_WINDOW)
    flow = np.maximum(np.nan_to_num(flow, nan=0.0), 0.0)
    time, flow = align_onset(time, flow, sampling_rate, 0.03)

    return time, flow


# =============================================================================
# SAVING INDIVIDUAL CURVES AND FIGURES
# =============================================================================

def save_common_curve(
    index,
    audio_name,
    predicted_time,
    predicted_flow,
    calibration_flow,
    actual_time,
    actual_flow,
):
    """Interpolate all three curves onto one shared time grid and save them."""
    end_time = min(float(predicted_time[-1]), float(actual_time[-1]))

    if end_time <= 0:
        raise ValueError("Curves have no overlapping duration")

    predicted_time_step = np.median(np.diff(predicted_time))
    actual_time_step = np.median(np.diff(actual_time))
    common_time_step = max(
        float(predicted_time_step),
        float(actual_time_step),
    )

    common_time = np.arange(0, end_time, common_time_step)
    predicted = np.interp(
        common_time,
        predicted_time,
        predicted_flow,
    )
    calibration = np.interp(
        common_time,
        predicted_time,
        calibration_flow,
    )
    actual = np.interp(
        common_time,
        actual_time,
        actual_flow,
    )

    filename = f"{index:03d}_{audio_name}_curves.csv"
    output_path = os.path.join(CURVE_DIR, filename)

    pd.DataFrame(
        {
            "time_s": common_time,
            "gradient_boosting_flow_ml_s": predicted,
            "calibration_equation_flow_ml_s": calibration,
            "actual_flow_ml_s": actual,
        }
    ).to_csv(output_path, index=False)

    return output_path, common_time, predicted, calibration, actual


def save_plot(
    index,
    audio_name,
    time,
    predicted,
    calibration,
    actual,
    predicted_qmax,
):
    """Save Gradient Boosting, calibration, and reference flow curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(
        time,
        predicted,
        linewidth=2,
        label="Gradient Boosting prediction",
    )
    plt.plot(
        time,
        calibration,
        linestyle=":",
        linewidth=2,
        label="Calibration-equation prediction",
    )
    plt.plot(
        time,
        actual,
        linestyle="--",
        linewidth=1.7,
        label="Reference from scale",
    )
    plt.xlabel("Time after onset (s)")
    plt.ylabel("Flow rate (mL/s)")
    flow_title = make_flow_title(audio_name)
    plt.title(
        f"{flow_title}\n"
        f"Predicted Qmax = {predicted_qmax:.2f} mL/s"
    )
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(
        PLOT_DIR,
        f"{index:03d}_{audio_name}.png",
    )
    plt.savefig(output_path, dpi=220)
    plt.close()

    return output_path


# =============================================================================
# RANDOM OVERVIEW FIGURE
# =============================================================================

def show_overview(items):
    """Display nine randomly selected successful comparisons."""
    if not items:
        return

    number_to_show = min(OVERVIEW_COUNT, len(items))
    random_generator = np.random.default_rng(OVERVIEW_RANDOM_SEED)
    selected_indices = random_generator.choice(
        len(items),
        size=number_to_show,
        replace=False,
    )
    shown = [items[index] for index in selected_indices]

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(18, 13),
        constrained_layout=True,
    )

    for axis, item in zip(axes.ravel(), shown):
        axis.plot(
            item["time"],
            item["predicted"],
            label="Gradient Boosting prediction",
        )
        axis.plot(
            item["time"],
            item["calibration"],
            linestyle=":",
            label="Calibration equation",
        )
        axis.plot(
            item["time"],
            item["actual"],
            linestyle="--",
            label="Scale reference",
        )
        axis.set_title(item["title"], fontsize=10, pad=9)
        axis.set_xlabel("Time (s)", labelpad=5)
        axis.set_ylabel("Flow rate (mL/s)", labelpad=5)
        axis.grid(alpha=0.2)

    # Hide unused panels if fewer than nine curves were created.
    for axis in axes.ravel()[len(shown):]:
        axis.axis("off")

    axes.ravel()[0].legend()
    fig.suptitle(
        f"Flow-curve overview: {len(shown)} randomly selected recordings",
        fontsize=15,
        y=1.02,
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        "flow_curve_overview_random.png",
    )
    plt.savefig(output_path, dpi=220)
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CURVE_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    prediction_file = GRADIENT_BOOSTING_RESULTS
    model_name = "Gradient Boosting"

    print("\n=== FLOW-CURVE RECONSTRUCTION ===")
    print(f"Model:   {model_name}")
    print(f"Reading: {prediction_file}")

    data = pd.read_excel(prediction_file)
    required_columns = [AUDIO_COL, CSV_COL, QMAX_COL]
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    result_rows = []
    overview_items = []

    for index, row in data.iterrows():
        audio_path = resolve_path(row[AUDIO_COL])
        csv_path = resolve_path(row[CSV_COL])
        audio_name = os.path.splitext(os.path.basename(audio_path))[0]

        try:
            predicted_qmax = float(row[QMAX_COL])
            if not np.isfinite(predicted_qmax):
                raise ValueError("Predicted Qmax is missing")

            predicted_time, predicted_flow, calibration_flow = (
                predicted_flows_from_audio(
                    audio_path,
                    predicted_qmax,
                )
            )
            actual_time, actual_flow = actual_flow_from_scale(csv_path)

            (
                curve_path,
                common_time,
                predicted,
                calibration,
                actual,
            ) = save_common_curve(
                index + 1,
                audio_name,
                predicted_time,
                predicted_flow,
                calibration_flow,
                actual_time,
                actual_flow,
            )
            plot_path = save_plot(
                index + 1,
                audio_name,
                common_time,
                predicted,
                calibration,
                actual,
                predicted_qmax,
            )

            result_rows.append(
                {
                    "excel_row": index + 2,
                    "audio_path": audio_path,
                    "csv_path": csv_path,
                    "predicted_qmax_ml_s": predicted_qmax,
                    "curve_csv": curve_path,
                    "plot_file": plot_path,
                    "status": "ok",
                }
            )

            # Store every successful curve. Nine are selected randomly later.
            overview_items.append(
                {
                    "time": common_time,
                    "predicted": predicted,
                    "calibration": calibration,
                    "actual": actual,
                    "title": make_flow_title(audio_name),
                }
            )

            print(f"[{index + 1}/{len(data)}] Saved | {audio_name}")

        except Exception as error:
            result_rows.append(
                {
                    "excel_row": index + 2,
                    "audio_path": audio_path,
                    "csv_path": csv_path,
                    "status": f"error: {error}",
                }
            )
            print(f"[{index + 1}/{len(data)}] ERROR | {error}")

    results = pd.DataFrame(result_rows)
    results_path = os.path.join(
        OUTPUT_DIR,
        "flow_curve_results.xlsx",
    )
    results.to_excel(results_path, index=False)

    successful = int((results["status"] == "ok").sum())

    print("\n=== RESULTS ===")
    print(f"Total recordings: {len(results)}")
    print(f"Curves created:   {successful}")
    print(f"Failed:           {len(results) - successful}")
    print(f"Curve data:       {CURVE_DIR}")
    print(f"Plots:            {PLOT_DIR}")
    print(f"Status table:     {results_path}")

    show_overview(overview_items)
    print("Done.")


if __name__ == "__main__":
    main()