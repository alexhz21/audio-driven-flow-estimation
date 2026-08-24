import os
import warnings

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydub import AudioSegment
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")

# Settings
PROJECT_DIR = r"C:\school\project\everything everything"
KEY_FILE = os.path.join(PROJECT_DIR, "everything_key.xlsx")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "05_feature_analysis_outputs")
WAV_DIR = os.path.join(OUTPUT_DIR, "wav_files")
PLOT_DIR = os.path.join(OUTPUT_DIR, "feature_plots")

AUDIO_COL = "audio_path"
CSV_COL = "csv_path"
TARGET_SR = 22050
DEFAULT_SCALE_HZ = 8.0
SELECTED_FEATURES = [
    "max_audio_energy",
    "std_audio_energy",
    "flow_eq_from_mel_emax_window",
]

# Updated power-law calibration from Section 3.2:
# Q = 38.336 * E^0.390654
EQ_A = 38.336
EQ_B = 0.390654


# Friendly titles and x-axis labels for the report figures.
FEATURE_LABELS = {
    "avg_freq_above_80pct_energy": (
        "Qmax vs. Mean Frequency in the High-Energy Region",
        "Mean frequency above 80% of peak energy (Hz)",
    ),
    "dominant_freq_at_peak_energy": (
        "Qmax vs. Dominant Frequency at Peak Energy",
        "Dominant frequency at peak acoustic energy (Hz)",
    ),
    "duration_s_audio": (
        "Qmax vs. Recording Duration",
        "Audio recording duration (s)",
    ),
    "energy_slope": (
        "Qmax vs. Acoustic-Energy Slope",
        "Acoustic-energy slope (a.u./s)",
    ),
    "flow_eq_from_mel_emax_window": (
        "Qmax vs. Calibration-Based Flow Estimate",
        "Calibration-based flow estimate (mL/s)",
    ),
    "log_energy_slope": (
        "Qmax vs. Log Acoustic-Energy Slope",
        "Log acoustic-energy slope (1/s)",
    ),
    "max_audio_energy": (
        "Qmax vs. Maximum Acoustic Energy",
        "Maximum acoustic energy (a.u.)",
    ),
    "max_dominant_freq": (
        "Qmax vs. Maximum Dominant Frequency",
        "Maximum dominant frequency (Hz)",
    ),
    "max_freq_at_max_energy": (
        "Qmax vs. Peak-Energy Frequency",
        "Frequency at maximum acoustic energy (Hz)",
    ),
    "max_log_audio_energy": (
        "Qmax vs. Maximum Log Acoustic Energy",
        "Maximum log acoustic energy (a.u.)",
    ),
    "mean_audio_energy": (
        "Qmax vs. Mean Acoustic Energy",
        "Mean acoustic energy (a.u.)",
    ),
    "mean_dominant_freq": (
        "Qmax vs. Mean Dominant Frequency",
        "Mean dominant frequency (Hz)",
    ),
    "mean_log_audio_energy": (
        "Qmax vs. Mean Log Acoustic Energy",
        "Mean log acoustic energy (a.u.)",
    ),
    "mean_mel_power_emax_window_unlogged": (
        "Qmax vs. Acoustic Energy Near Peak Energy",
        "Mean mel power near peak energy (a.u.)",
    ),
    "std_audio_energy": (
        "Qmax vs. Variation in Acoustic Energy",
        "Standard deviation of acoustic energy (a.u.)",
    ),
    "std_dominant_freq": (
        "Qmax vs. Variation in Dominant Frequency",
        "Standard deviation of dominant frequency (Hz)",
    ),
    "std_log_audio_energy": (
        "Qmax vs. Variation in Log Acoustic Energy",
        "Standard deviation of log acoustic energy (a.u.)",
    ),
    "time_of_max_energy_s": (
        "Qmax vs. Time of Peak Acoustic Energy",
        "Time of peak acoustic energy (s)",
    ),
}


def resolve_path(value):
    """Return an absolute project path."""
    path = str(value).strip()
    return path if os.path.isabs(path) else os.path.join(PROJECT_DIR, path)


def first_column(df, choices):
    """Find the first matching column."""
    for name in choices:
        if name in df.columns:
            return name
    return None


def convert_to_wav(audio_path):
    """Convert non-WAV audio to mono WAV."""
    if audio_path.lower().endswith(".wav"):
        return audio_path

    os.makedirs(WAV_DIR, exist_ok=True)
    name = os.path.splitext(os.path.basename(audio_path))[0] + ".wav"
    output_path = os.path.join(WAV_DIR, name)

    if not os.path.exists(output_path):
        (
            AudioSegment.from_file(audio_path)
            .set_channels(1)
            .export(output_path, format="wav")
        )

    return output_path


def compute_qmax(csv_path):
    """Calculate reference Qmax from scale data."""
    if csv_path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(csv_path)
    else:
        df = pd.read_csv(csv_path)

    weight_col = first_column(df, ["Weight", "weight", "grams", "g", "mass"])
    time_col = first_column(df, ["time", "Time", "seconds", "Seconds", "t"])
    date_col = first_column(df, ["DateTime", "datetime"])

    if weight_col is None:
        raise ValueError("Weight column not found")

    weight = pd.to_numeric(df[weight_col], errors="coerce").to_numpy(dtype=float)

    if date_col:
        dates = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
        time = (dates - dates.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    elif time_col:
        time = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
        time = time - time[0]
    else:
        time = np.arange(len(weight), dtype=float) / DEFAULT_SCALE_HZ

    valid = np.isfinite(weight) & np.isfinite(time)
    weight = weight[valid]
    time = time[valid]

    if len(weight) < 10:
        raise ValueError("Too few scale samples")

    if np.any(np.diff(time) <= 0):
        time = np.arange(len(weight), dtype=float) / DEFAULT_SCALE_HZ

    fs = 1.0 / float(np.median(np.diff(time)))
    window = max(5, int(round(0.75 * fs)))
    window += 1 - window % 2

    if window >= len(weight):
        window = len(weight) - 1 if len(weight) % 2 == 0 else len(weight)

    smooth_weight = savgol_filter(weight, window, 2) if window >= 5 else weight
    flow = np.maximum(np.gradient(smooth_weight, time) / 1.02, 0)
    smooth_flow = savgol_filter(flow, window, 2) if window >= 5 else flow

    return float(np.max(np.maximum(smooth_flow, 0)))


def extract_features(audio_path):
    """Extract audio-only features."""
    y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)

    if len(y) == 0:
        raise ValueError("Empty audio file")

    trimmed, _ = librosa.effects.trim(y, top_db=25)
    signal = trimmed if len(trimmed) >= int(0.5 * sr) else y

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak

    n_fft = 2048
    hop = 512

    rms = librosa.feature.rms(
        y=signal,
        frame_length=n_fft,
        hop_length=hop,
    )[0]
    log_rms = np.log(rms + 1e-8)
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop,
        n_mels=128,
        power=2.0,
    )
    mel_freqs = librosa.mel_frequencies(n_mels=128, fmin=0, fmax=sr / 2)
    energy_by_frequency = np.sum(mel, axis=1)
    max_bin = int(np.argmax(energy_by_frequency))
    high_energy = energy_by_frequency >= 0.8 * np.max(energy_by_frequency)

    spectrum = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop))
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    dominant = frequencies[np.argmax(spectrum, axis=0)]

    peak_frame = int(np.argmax(rms))
    peak_frame = min(peak_frame, spectrum.shape[1] - 1)

    # 0.5-second window centered on the maximum-energy frame.
    half_window = int(round(0.25 * sr / hop))
    left = max(0, peak_frame - half_window)
    right = min(mel.shape[1], peak_frame + half_window + 1)
    mean_mel_power = max(float(np.mean(mel[:, left:right])), 1e-12)

    return {
        "duration_s_audio": float(len(y) / sr),
        "max_freq_at_max_energy": float(mel_freqs[max_bin]),
        "avg_freq_above_80pct_energy": float(np.mean(mel_freqs[high_energy])),
        "max_audio_energy": float(np.max(rms)),
        "mean_audio_energy": float(np.mean(rms)),
        "std_audio_energy": float(np.std(rms)),
        "max_log_audio_energy": float(np.max(log_rms)),
        "mean_log_audio_energy": float(np.mean(log_rms)),
        "std_log_audio_energy": float(np.std(log_rms)),
        "energy_slope": float(np.polyfit(times, rms, 1)[0]),
        "log_energy_slope": float(np.polyfit(times, log_rms, 1)[0]),
        "time_of_max_energy_s": float(times[peak_frame]),
        "dominant_freq_at_peak_energy": float(
            frequencies[np.argmax(spectrum[:, peak_frame])]
        ),
        "mean_dominant_freq": float(np.mean(dominant)),
        "std_dominant_freq": float(np.std(dominant)),
        "max_dominant_freq": float(np.max(dominant)),
        "mean_mel_power_emax_window_unlogged": mean_mel_power,
        "flow_eq_from_mel_emax_window": float(EQ_A * mean_mel_power ** EQ_B),
    }


def linear_result(x, y):
    """Fit one feature to Qmax."""
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 3 or np.ptp(x) == 0:
        return None

    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    total = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - np.sum((y - predicted) ** 2) / total if total > 0 else np.nan
    correlation = np.corrcoef(x, y)[0, 1]

    return x, y, float(slope), float(intercept), float(r2), float(correlation)


def friendly_labels(feature):
    """Return a readable graph title and x-axis label."""
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]

    readable = feature.replace("_", " ").title()
    return f"Qmax vs. {readable}", readable


def draw_feature_plot(axis, feature, result):
    """Draw a report-ready feature plot."""
    x, y, slope, intercept, r2, correlation = result
    line_x = np.linspace(float(x.min()), float(x.max()), 250)
    title, xlabel = friendly_labels(feature)

    axis.scatter(
        x,
        y,
        s=55,
        alpha=0.75,
        color="#377eb8",
        edgecolors="#1f77b4",
        linewidths=0.8,
    )
    axis.plot(
        line_x,
        slope * line_x + intercept,
        color="#1f77b4",
        linewidth=2.5,
    )
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Reference Qmax (mL/s)")
    axis.set_title(
        f"{title}\n$R^2$ = {r2:.3f} | r = {correlation:.3f}"
    )
    axis.grid(alpha=0.25)


def create_feature_plot(feature, result):
    """Save one full-size feature plot."""
    fig, axis = plt.subplots(figsize=(9, 6.5))
    draw_feature_plot(axis, feature, result)
    fig.tight_layout()
    fig.savefig(
        os.path.join(PLOT_DIR, f"Qmax_vs_{feature}.png"),
        dpi=300,
    )
    plt.close(fig)


def create_selected_features_figure(data):
    """Save one figure containing the three requested audio features."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))

    for axis, feature in zip(axes, SELECTED_FEATURES):
        result = linear_result(
            data[feature].to_numpy(dtype=float),
            data["qmax_ml_s"].to_numpy(dtype=float),
        )
        draw_feature_plot(axis, feature, result)

    fig.suptitle(
        "Selected Audio Features and Reference Qmax",
        fontsize=18,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(
        os.path.join(OUTPUT_DIR, "selected_three_features.png"),
        dpi=300,
    )
    plt.close(fig)


def save_excel_safely(dataframe, output_path):
    """Save Excel output without crashing if the original file is open."""
    try:
        dataframe.to_excel(output_path, index=False)
        return output_path
    except PermissionError:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
        base, extension = os.path.splitext(output_path)
        fallback = f"{base}_{timestamp}{extension}"
        dataframe.to_excel(fallback, index=False)
        print(f"Output file was open. Saved instead as: {fallback}")
        return fallback


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(WAV_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("\n=== FEATURE ANALYSIS ===")
    print(f"Reading: {KEY_FILE}")

    key = pd.read_excel(KEY_FILE)
    if AUDIO_COL not in key.columns or CSV_COL not in key.columns:
        raise ValueError("Excel needs audio_path and csv_path columns")

    rows = []

    for index, row in key.iterrows():
        audio_path = resolve_path(row[AUDIO_COL])
        csv_path = resolve_path(row[CSV_COL])

        try:
            wav_path = convert_to_wav(audio_path)
            qmax = compute_qmax(csv_path)
            features = extract_features(wav_path)

            rows.append(
                {
                    "excel_row": index + 2,
                    "audio_path": audio_path,
                    "csv_path": csv_path,
                    "qmax_ml_s": qmax,
                    **features,
                }
            )
            print(f"[{index + 1}/{len(key)}] OK | Qmax={qmax:.2f} mL/s")

        except Exception as error:
            print(f"[{index + 1}/{len(key)}] ERROR | {error}")

    data = pd.DataFrame(rows)
    if len(data) < 3:
        raise ValueError("Too few valid recordings")

    # Code 5 creates this combined feature/Qmax table.
    data_xlsx = os.path.join(OUTPUT_DIR, "audio_features_and_qmax.xlsx")
    data_csv = os.path.join(OUTPUT_DIR, "audio_features_and_qmax.csv")
    saved_data_xlsx = save_excel_safely(data, data_xlsx)
    data.to_csv(data_csv, index=False)

    excluded = {"excel_row", "audio_path", "csv_path", "qmax_ml_s"}
    feature_columns = [column for column in data.columns if column not in excluded]
    summary_rows = []

    for feature in feature_columns:
        result = linear_result(
            data[feature].to_numpy(dtype=float),
            data["qmax_ml_s"].to_numpy(dtype=float),
        )
        if result is None:
            continue

        _, _, slope, intercept, r2, correlation = result
        summary_rows.append(
            {
                "feature": feature,
                "r2": r2,
                "correlation_r": correlation,
                "absolute_r": abs(correlation),
                "slope": slope,
                "intercept": intercept,
            }
        )

    # Ranking by absolute r includes strong positive and negative relationships.
    complete_summary = (
        pd.DataFrame(summary_rows)
        .sort_values("absolute_r", ascending=False)
        .reset_index(drop=True)
    )
    complete_summary.insert(
        0,
        "rank",
        np.arange(1, len(complete_summary) + 1),
    )
    # Save the complete ranking for all extracted features.
    ranking_xlsx = os.path.join(OUTPUT_DIR, "feature_ranking.xlsx")
    ranking_csv = os.path.join(OUTPUT_DIR, "feature_ranking.csv")
    saved_ranking_xlsx = save_excel_safely(complete_summary, ranking_xlsx)
    complete_summary.to_csv(ranking_csv, index=False)

    # Create a separate output graph for every feature.
    for feature in complete_summary["feature"]:
        result = linear_result(
            data[feature].to_numpy(dtype=float),
            data["qmax_ml_s"].to_numpy(dtype=float),
        )
        create_feature_plot(feature, result)

    # Create one three-panel graph containing exactly the requested features.
    missing_selected = [
        feature for feature in SELECTED_FEATURES if feature not in data.columns
    ]
    if missing_selected:
        raise KeyError(
            "Missing requested features: " + ", ".join(missing_selected)
        )
    create_selected_features_figure(data)

    print("\n=== COMPLETE FEATURE RANKING ===")
    print(
        complete_summary[
            ["rank", "feature", "r2", "correlation_r"]
        ].round(4).to_string(index=False)
    )
    print(f"\nValid recordings: {len(data)}")
    print(f"Saved feature table: {saved_data_xlsx}")
    print(f"Saved feature CSV: {data_csv}")
    print(f"Saved ranking: {saved_ranking_xlsx}")
    print(f"Saved ranking CSV: {ranking_csv}")
    print(f"Saved individual plots: {PLOT_DIR}")
    print(
        "Saved combined figure: "
        f"{os.path.join(OUTPUT_DIR, 'selected_three_features.png')}"
    )
    print("Done.")


if __name__ == "__main__":
    main()