import os
import warnings

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")

# Settings
PROJECT_DIR = r"C:\school\project\everything everything"
KEY_FILE = os.path.join(PROJECT_DIR, "calibration_key.xlsx")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "03_pump_calibration_outputs")

AUDIO_COL = "audio_path"
CSV_COL = "csv_path"
N_STEPS = 10
TARGET_SR = 22050
URINE_DENSITY = 1.02
SCALE_SMOOTH_SEC = 0.6
IGNORE_INITIAL_SEC = 0.5
MIN_STEP_SEC = 1.0
PEAK_DISTANCE_SEC = 0.8
PEAK_PROMINENCE = 0.25


def resolve_path(value):
    """Return an absolute project path."""
    path = str(value).strip()
    return path if os.path.isabs(path) else os.path.join(PROJECT_DIR, path)


def moving_average(values, fs, seconds):
    """Smooth a signal."""
    window = max(1, int(round(fs * seconds)))
    if window == 1:
        return values.copy()
    return np.convolve(values, np.ones(window) / window, mode="same")


def first_sustained(mask, length):
    """Find the first sustained True region."""
    run = 0
    for index, value in enumerate(mask):
        run = run + 1 if value else 0
        if run >= length:
            return index - length + 1
    return None


def onset_time(time, signal, fs, fraction, seconds=0.3):
    """Find signal onset."""
    peak = float(np.nanmax(np.abs(signal)))
    threshold = max(1e-9, fraction * peak)
    index = first_sustained(
        np.isfinite(signal) & (np.abs(signal) > threshold),
        max(1, int(round(seconds * fs))),
    )
    if index is None:
        index = int(np.nanargmax(np.abs(signal)))
    return float(time[index])


def load_scale(csv_path):
    """Convert scale weight to flow."""
    df = pd.read_csv(csv_path)
    if "Weight" not in df.columns or "DateTime" not in df.columns:
        raise ValueError("Scale CSV needs Weight and DateTime columns")

    dates = pd.to_datetime(df["DateTime"], dayfirst=True, errors="coerce")
    time = (dates - dates.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    weight = pd.to_numeric(df["Weight"], errors="coerce")
    weight = weight.interpolate().bfill().ffill()

    valid_dt = np.diff(time)
    valid_dt = valid_dt[np.isfinite(valid_dt) & (valid_dt > 0)]
    dt = float(np.median(valid_dt))
    fs = 1.0 / dt

    smooth_weight = weight.rolling(7, center=True, min_periods=1).mean().to_numpy()
    flow = np.gradient(smooth_weight, dt) / URINE_DENSITY
    return time, flow, fs


def load_audio(audio_path):
    """Calculate smoothed mel energy."""
    y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    if len(y) == 0:
        raise ValueError("Empty audio file")

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128, power=2.0
    )
    energy = np.log10(np.mean(mel, axis=0) + 1e-12)
    time = librosa.frames_to_time(np.arange(len(energy)), sr=sr, hop_length=512)
    fs = 1.0 / float(np.median(np.diff(time)))
    energy = moving_average(energy, fs, 0.25)
    return time - time[0], energy, fs


def truncate_shutdown(time, flow, fs):
    """Remove the final pump shutdown."""
    if len(flow) < 10:
        return time, flow

    maximum = int(np.nanargmax(flow))
    derivative = np.gradient(flow)
    positive_peak = max(float(np.nanmax(derivative)), 1e-9)
    negative = derivative < -0.05 * positive_peak
    run = max(1, int(round(fs)))
    index = first_sustained(negative[maximum:], run)

    if index is None:
        return time, flow

    cut = max(2, maximum + index)
    return time[:cut], flow[:cut]


def force_ten_segments(boundaries, length, fs):
    """Return exactly ten step segments."""
    minimum = max(1, int(round(MIN_STEP_SEC * fs)))
    boundaries = sorted(set([max(0, min(length, int(x))) for x in boundaries]))

    cleaned = [boundaries[0]]
    for boundary in boundaries[1:]:
        if boundary - cleaned[-1] >= minimum:
            cleaned.append(boundary)
    if cleaned[-1] != boundaries[-1]:
        cleaned.append(boundaries[-1])
    boundaries = cleaned

    target = N_STEPS + 1
    if len(boundaries) > target:
        inner = boundaries[1:-1]
        selected = np.linspace(0, len(inner) - 1, target - 2).round().astype(int)
        boundaries = [boundaries[0]] + [inner[i] for i in selected] + [boundaries[-1]]

    while len(boundaries) < target:
        lengths = np.diff(boundaries)
        longest = int(np.argmax(lengths))
        left, right = boundaries[longest], boundaries[longest + 1]
        if right - left < 2 * minimum:
            break
        boundaries.insert(longest + 1, left + (right - left) // 2)

    if len(boundaries) != target:
        boundaries = np.linspace(boundaries[0], boundaries[-1], target).round().astype(int).tolist()
    return boundaries


def find_steps(flow, fs, start_index):
    """Detect the ten scale steps."""
    derivative = np.abs(np.gradient(flow))
    distance = max(1, int(round(PEAK_DISTANCE_SEC * fs)))
    prominence = PEAK_PROMINENCE * float(np.nanmax(derivative))
    peaks, _ = find_peaks(derivative, distance=distance, prominence=prominence)
    peaks = [int(peak) for peak in peaks if peak > start_index]
    return force_ten_segments([start_index] + peaks + [len(flow)], len(flow), fs)


def step_statistics(time, signal, boundaries):
    """Calculate step means and standard deviations."""
    rows = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        values = signal[left:right]
        values = values[np.isfinite(values)]
        start = float(time[left])
        end = float(time[max(left, right - 1)])
        rows.append(
            {
                "mean": float(np.mean(values)) if len(values) else np.nan,
                "sd": float(np.std(values)) if len(values) else np.nan,
                "start": start,
                "end": end,
                "mid": 0.5 * (start + end),
            }
        )
    return rows


def raw_scale_plot(raw_runs):
    """Plot raw scale runs and their mean."""
    end = min(run[0][-1] for run in raw_runs)
    fs = np.median([1.0 / np.median(np.diff(run[0])) for run in raw_runs])
    grid = np.arange(0, end, 1.0 / fs)
    matrix = np.vstack([np.interp(grid, time, flow) for time, flow in raw_runs])

    plt.figure(figsize=(10, 4.5))
    for index, (time, flow) in enumerate(raw_runs, start=1):
        plt.plot(time, flow, alpha=0.55, linewidth=1, label=f"Run {index}")
    plt.plot(grid, np.mean(matrix, axis=0), color="black", linewidth=3, label="Mean")
    plt.xlabel("Time (s)")
    plt.ylabel("Flow (mL/s)")
    plt.title("Raw scale flow: runs and mean")
    plt.grid(alpha=0.25)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "scale_raw_runs_plus_mean.png"), dpi=220)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n=== PUMP CALIBRATION ===")
    print(f"Reading: {KEY_FILE}")

    key = pd.read_excel(KEY_FILE).dropna(subset=[AUDIO_COL, CSV_COL])
    scale_runs, audio_runs, time_runs, raw_runs, labels = [], [], [], [], []

    for index, row in key.iterrows():
        audio_path = resolve_path(row[AUDIO_COL])
        csv_path = resolve_path(row[CSV_COL])
        label = f"Run {len(labels) + 1}"

        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(audio_path)
            if not os.path.exists(csv_path):
                raise FileNotFoundError(csv_path)

            scale_time, flow, scale_fs = load_scale(csv_path)
            raw_runs.append((scale_time - scale_time[0], flow))
            smooth_flow = moving_average(flow, scale_fs, SCALE_SMOOTH_SEC)

            scale_onset = onset_time(scale_time, smooth_flow, scale_fs, 0.03)
            cut = int(np.searchsorted(scale_time, scale_onset))
            aligned_time = scale_time[cut:] - scale_time[cut]
            aligned_flow = smooth_flow[cut:]
            aligned_time, aligned_flow = truncate_shutdown(aligned_time, aligned_flow, scale_fs)

            start = int(np.searchsorted(aligned_time, IGNORE_INITIAL_SEC))
            boundaries = find_steps(aligned_flow, scale_fs, start)
            scale_stats = step_statistics(aligned_time, aligned_flow, boundaries)

            audio_time, energy, audio_fs = load_audio(audio_path)
            audio_onset = onset_time(audio_time, energy, audio_fs, 0.10, 0.2)
            audio_time = audio_time - audio_onset

            audio_stats = []
            for step in scale_stats:
                mask = (audio_time >= step["start"]) & (audio_time <= step["end"])
                values = energy[mask]
                audio_stats.append(
                    {
                        "mean": float(np.mean(values)) if len(values) else np.nan,
                        "sd": float(np.std(values)) if len(values) else np.nan,
                    }
                )

            scale_runs.append([step["mean"] for step in scale_stats])
            audio_runs.append([step["mean"] for step in audio_stats])
            time_runs.append([step["mid"] for step in scale_stats])
            labels.append(label)
            print(f"[{index + 1}/{len(key)}] OK | {label} | 10 steps")

        except Exception as error:
            print(f"[{index + 1}/{len(key)}] ERROR | {error}")

    if not labels:
        raise ValueError("No valid calibration runs")

    raw_scale_plot(raw_runs)

    scale_matrix = np.asarray(scale_runs, dtype=float)
    audio_matrix = np.asarray(audio_runs, dtype=float)
    time_matrix = np.asarray(time_runs, dtype=float)

    scale_mean = np.nanmean(scale_matrix, axis=0)
    scale_sd = np.nanstd(scale_matrix, axis=0)
    audio_mean = np.nanmean(audio_matrix, axis=0)
    audio_sd = np.nanstd(audio_matrix, axis=0)
    time_mean = np.nanmean(time_matrix, axis=0)
    steps = np.arange(1, N_STEPS + 1)

    plt.figure(figsize=(8, 5))
    for run in range(len(labels)):
        plt.scatter(audio_matrix[run], scale_matrix[run], alpha=0.25)
    plt.errorbar(audio_mean, scale_mean, xerr=audio_sd, yerr=scale_sd, fmt="o", capsize=4)
    for step, x, y in zip(steps, audio_mean, scale_mean):
        plt.annotate(str(step), (x, y), xytext=(5, 3), textcoords="offset points")
    plt.xlabel("Audio energy (log mean mel power)")
    plt.ylabel("Flow (mL/s)")
    plt.title("Flow versus audio energy")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "flow_vs_audio_energy_errorbars.png"), dpi=220)

    plt.figure(figsize=(10, 4.5))
    for run in range(len(labels)):
        plt.plot(time_matrix[run], scale_matrix[run], alpha=0.35)
    plt.plot(time_mean, scale_mean, linewidth=2.5, label="Mean")
    plt.fill_between(time_mean, scale_mean - scale_sd, scale_mean + scale_sd, alpha=0.2, label="±1 SD")
    plt.xlabel("Time (s)")
    plt.ylabel("Flow (mL/s)")
    plt.title("Step flow versus time")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "scale_steps_time_runs_mean_sd.png"), dpi=220)

    plt.figure(figsize=(8, 4.5))
    plt.errorbar(steps, scale_mean, yerr=scale_sd, fmt="o-", capsize=4)
    plt.xlabel("Step")
    plt.ylabel("Flow (mL/s)")
    plt.title("Scale flow per step")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "scale_step_errorbars.png"), dpi=220)

    plt.figure(figsize=(8, 4.5))
    plt.errorbar(steps, audio_mean, yerr=audio_sd, fmt="o-", capsize=4)
    plt.xlabel("Step")
    plt.ylabel("Audio energy")
    plt.title("Audio energy per step")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "audio_step_errorbars.png"), dpi=220)

    summary = pd.DataFrame(
        {
            "step": steps,
            "mean_time_s": time_mean,
            "flow_mean_ml_s": scale_mean,
            "flow_sd_ml_s": scale_sd,
            "audio_mean_energy": audio_mean,
            "audio_sd_energy": audio_sd,
        }
    )
    summary.to_csv(os.path.join(OUTPUT_DIR, "step_summary.csv"), index=False)
    summary.to_excel(os.path.join(OUTPUT_DIR, "step_summary.xlsx"), index=False)

    print("\n=== STEP RESULTS ===")
    print(summary.round(4).to_string(index=False))
    print(f"\nValid runs: {len(labels)}")
    print(f"Saved to: {OUTPUT_DIR}")
    print("Close the plot windows to finish.")
    plt.show()
    print("Done.")


if __name__ == "__main__":
    main()
