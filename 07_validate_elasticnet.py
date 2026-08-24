import os
import joblib
import pandas as pd
from pydub import AudioSegment
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET_SR = 22050

EQ_A = 38.053
EQ_B = 0.386282
EMAX_WINDOW_SEC = 0.5
EMAX_HALF_WINDOW_SEC = EMAX_WINDOW_SEC / 2.0

PROJECT_DIR = r"C:\school\project\everything everything"
MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "06_elasticnet_training_outputs",
    "elasticnet_audio_only_model.joblib",
)
INPUT_EXCEL = os.path.join(PROJECT_DIR, "everything_val_key.xlsx")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "07_elasticnet_validation_outputs")
OUTPUT_EXCEL = os.path.join(OUTPUT_DIR, "validation_predictions.xlsx")
METRICS_EXCEL = os.path.join(OUTPUT_DIR, "validation_metrics.xlsx")
TEMP_WAV_DIR = os.path.join(OUTPUT_DIR, "wav_files")

DEFAULT_SCALE_HZ = 8.0
URINE_DENSITY = 1.02
WEIGHT_SMOOTH_SEC = 0.75
FLOW_SMOOTH_SEC = 0.75

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_WAV_DIR, exist_ok=True)


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


def convert_audio_to_wav(audio_path, out_wav_path):
    ext = os.path.splitext(audio_path)[1].lower()
    if ext == ".wav":
        return audio_path

    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1)
    audio.export(out_wav_path, format="wav")
    return out_wav_path


def extract_audio_features(audio_path, sr=TARGET_SR):
    y, sr = librosa.load(audio_path, sr=sr, mono=True)

    if len(y) == 0:
        raise ValueError(f"Empty audio file: {audio_path}")

    yt, _ = librosa.effects.trim(y, top_db=25)
    y_use = yt if len(yt) >= int(0.5 * sr) else y

    peak = np.max(np.abs(y_use))
    if peak > 0:
        y_use = y_use / peak

    duration_s_audio = float(len(y) / sr)

    n_fft = 2048
    hop_length = 512

    rms = librosa.feature.rms(y=y_use, frame_length=n_fft, hop_length=hop_length)[0]
    log_rms = np.log(rms + 1e-8)

    frame_times = librosa.frames_to_time(
        np.arange(len(rms)),
        sr=sr,
        hop_length=hop_length
    )

    mel_S = librosa.feature.melspectrogram(
        y=y_use,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=128,
        power=2.0
    )
    mel_freqs = librosa.mel_frequencies(n_mels=128, fmin=0, fmax=sr / 2)

    mel_energy_per_bin = np.sum(mel_S, axis=1)

    idx_max_energy_bin = int(np.argmax(mel_energy_per_bin))
    max_freq_at_max_energy = float(mel_freqs[idx_max_energy_bin])

    threshold_80 = 0.80 * np.max(mel_energy_per_bin)
    mask_80 = mel_energy_per_bin >= threshold_80
    if np.any(mask_80):
        avg_freq_above_80pct_energy = float(np.mean(mel_freqs[mask_80]))
    else:
        avg_freq_above_80pct_energy = np.nan

    S = np.abs(librosa.stft(y_use, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    if S.shape[1] > 0:
        dominant_freq_per_frame = freqs[np.argmax(S, axis=0)]
        mean_dominant_freq = float(np.mean(dominant_freq_per_frame))
        std_dominant_freq = float(np.std(dominant_freq_per_frame))
        max_dominant_freq = float(np.max(dominant_freq_per_frame))
    else:
        dominant_freq_per_frame = np.array([])
        mean_dominant_freq = np.nan
        std_dominant_freq = np.nan
        max_dominant_freq = np.nan

    if len(rms) > 0:
        peak_frame = int(np.argmax(rms))
        time_of_max_energy_s = float(frame_times[peak_frame]) if peak_frame < len(frame_times) else np.nan
    else:
        peak_frame = 0
        time_of_max_energy_s = np.nan

    if S.shape[1] > 0:
        peak_frame_clamped = min(peak_frame, S.shape[1] - 1)
        dominant_freq_at_peak_energy = float(freqs[np.argmax(S[:, peak_frame_clamped])])
    else:
        dominant_freq_at_peak_energy = np.nan

    frames_per_second = sr / hop_length
    half_window_frames = int(round(EMAX_HALF_WINDOW_SEC * frames_per_second))

    if mel_S.shape[1] > 0:
        peak_frame_mel = min(peak_frame, mel_S.shape[1] - 1)
        start_frame = max(0, peak_frame_mel - half_window_frames)
        end_frame = min(mel_S.shape[1], peak_frame_mel + half_window_frames + 1)

        mel_window = mel_S[:, start_frame:end_frame]

        if mel_window.size > 0:
            mean_mel_power_emax_window_unlogged = float(np.mean(mel_window))
            mean_mel_power_emax_window_unlogged = max(mean_mel_power_emax_window_unlogged, 1e-12)
        else:
            mean_mel_power_emax_window_unlogged = np.nan
    else:
        mean_mel_power_emax_window_unlogged = np.nan

    if np.isfinite(mean_mel_power_emax_window_unlogged):
        flow_eq_from_mel_emax_window = float(
            EQ_A * (mean_mel_power_emax_window_unlogged ** EQ_B)
        )
    else:
        flow_eq_from_mel_emax_window = np.nan

    if len(rms) >= 2:
        energy_slope = float(np.polyfit(frame_times, rms, 1)[0])
    else:
        energy_slope = np.nan

    if len(log_rms) >= 2:
        log_energy_slope = float(np.polyfit(frame_times, log_rms, 1)[0])
    else:
        log_energy_slope = np.nan

    max_audio_energy = float(np.max(rms)) if len(rms) > 0 else np.nan
    mean_audio_energy = float(np.mean(rms)) if len(rms) > 0 else np.nan
    std_audio_energy = float(np.std(rms)) if len(rms) > 0 else np.nan

    max_log_audio_energy = float(np.max(log_rms)) if len(log_rms) > 0 else np.nan
    mean_log_audio_energy = float(np.mean(log_rms)) if len(log_rms) > 0 else np.nan
    std_log_audio_energy = float(np.std(log_rms)) if len(log_rms) > 0 else np.nan

    return {
        "duration_s_audio": duration_s_audio,
        "max_freq_at_max_energy": max_freq_at_max_energy,
        "avg_freq_above_80pct_energy": avg_freq_above_80pct_energy,
        "max_audio_energy": max_audio_energy,
        "mean_audio_energy": mean_audio_energy,
        "std_audio_energy": std_audio_energy,
        "max_log_audio_energy": max_log_audio_energy,
        "mean_log_audio_energy": mean_log_audio_energy,
        "std_log_audio_energy": std_log_audio_energy,
        "energy_slope": energy_slope,
        "log_energy_slope": log_energy_slope,
        "time_of_max_energy_s": time_of_max_energy_s,
        "dominant_freq_at_peak_energy": dominant_freq_at_peak_energy,
        "mean_dominant_freq": mean_dominant_freq,
        "std_dominant_freq": std_dominant_freq,
        "max_dominant_freq": max_dominant_freq,
        "mean_mel_power_emax_window_unlogged": mean_mel_power_emax_window_unlogged,
        "flow_eq_from_mel_emax_window": flow_eq_from_mel_emax_window,
    }


def compute_qmax(csv_path):
    """Calculate true Qmax from scale data."""
    if csv_path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(csv_path)
    else:
        df = pd.read_csv(csv_path)

    weight_col = first_column(df, ["Weight", "weight", "grams", "g", "mass"])
    time_col = first_column(df, ["time", "Time", "seconds", "Seconds", "t"])
    if weight_col is None:
        raise ValueError("Weight column not found")

    weight = pd.to_numeric(df[weight_col], errors="coerce").to_numpy(dtype=float)

    if time_col:
        time = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
        time = time - time[0]
    else:
        time = np.arange(len(weight), dtype=float) / DEFAULT_SCALE_HZ

    valid = np.isfinite(weight) & np.isfinite(time)
    weight, time = weight[valid], time[valid]
    if len(weight) < 10:
        raise ValueError("Too few scale samples")
    if np.any(np.diff(time) <= 0):
        time = np.arange(len(weight), dtype=float) / DEFAULT_SCALE_HZ

    fs = 1.0 / float(np.median(np.diff(time)))
    weight_window = max(5, int(round(fs * WEIGHT_SMOOTH_SEC)))
    weight_window += 1 - weight_window % 2
    if weight_window >= len(weight):
        weight_window = len(weight) - 1 if len(weight) % 2 == 0 else len(weight)

    smooth_weight = (
        savgol_filter(weight, weight_window, 2)
        if weight_window >= 5 else weight
    )
    flow = np.maximum(np.gradient(smooth_weight, time) / URINE_DENSITY, 0)

    flow_window = max(5, int(round(fs * FLOW_SMOOTH_SEC)))
    flow_window += 1 - flow_window % 2
    if flow_window >= len(flow):
        flow_window = len(flow) - 1 if len(flow) % 2 == 0 else len(flow)
    smooth_flow = (
        savgol_filter(flow, flow_window, 2)
        if flow_window >= 5 else flow
    )
    return float(np.max(np.maximum(smooth_flow, 0)))


def show_results(results, metrics):
    """Show validation plots."""
    actual = results["actual_qmax_ml_s"].to_numpy()
    predicted = results["predicted_qmax_ml_s"].to_numpy()
    error = predicted - actual
    low = min(float(actual.min()), float(predicted.min()))
    high = max(float(actual.max()), float(predicted.max()))

    plt.figure(figsize=(6.5, 6))
    plt.scatter(actual, predicted, alpha=0.8)
    plt.plot([low, high], [low, high], "k--", label="Perfect prediction")
    plt.xlabel("Actual Qmax (mL/s)")
    plt.ylabel("Predicted Qmax (mL/s)")
    plt.title(
        f"Independent ElasticNet validation\n"
        f"RMSE={metrics['rmse_ml_s']:.3f} | MAE={metrics['mae_ml_s']:.3f} | "
        f"R²={metrics['r2']:.3f}"
    )
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"), dpi=220)

    plt.figure(figsize=(7, 5))
    plt.scatter(predicted, error, alpha=0.8)
    plt.axhline(0, color="black", linestyle="--")
    plt.xlabel("Predicted Qmax (mL/s)")
    plt.ylabel("Error: predicted - actual (mL/s)")
    plt.title("Validation residuals")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "validation_residuals.png"), dpi=220)

    plt.figure(figsize=(7, 5))
    plt.hist(error, bins="auto", edgecolor="black", alpha=0.8)
    plt.axvline(0, color="black", linestyle="--")
    plt.axvline(np.mean(error), color="red", linestyle="--", label="Mean error")
    plt.xlabel("Prediction error (mL/s)")
    plt.ylabel("Number of recordings")
    plt.title("Validation error distribution")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "validation_error_distribution.png"), dpi=220)


def main():
    print("\n=== ELASTICNET VALIDATION ===")
    print(f"Model: {MODEL_PATH}")
    print(f"Key:   {INPUT_EXCEL}")

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    df = pd.read_excel(INPUT_EXCEL)

    if "audio_path" not in df.columns or "csv_path" not in df.columns:
        raise ValueError("Input Excel needs audio_path and csv_path columns")

    rows = []

    for i, row in df.iterrows():
        audio_path = resolve_path(row["audio_path"])
        csv_path = resolve_path(row["csv_path"])

        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Scale file not found: {csv_path}")

            base = os.path.splitext(os.path.basename(audio_path))[0]
            temp_wav_path = os.path.join(TEMP_WAV_DIR, base + ".wav")

            wav_path = convert_audio_to_wav(audio_path, temp_wav_path)
            feat = extract_audio_features(wav_path)

            X_new = pd.DataFrame([feat]).reindex(columns=feature_columns)
            pred_qmax = float(model.predict(X_new)[0])
            actual_qmax = compute_qmax(csv_path)
            error = pred_qmax - actual_qmax

            rows.append({
                "excel_row": i + 2,
                "audio_path": audio_path,
                "csv_path": csv_path,
                "actual_qmax_ml_s": actual_qmax,
                "predicted_qmax_ml_s": pred_qmax,
                "error_ml_s": error,
                "absolute_error_ml_s": abs(error),
                "status": "ok",
            })
            print(
                f"[{i + 1}/{len(df)}] OK | actual={actual_qmax:.2f} | "
                f"predicted={pred_qmax:.2f} | error={error:+.2f} mL/s"
            )

        except Exception as e:
            rows.append({
                "excel_row": i + 2,
                "audio_path": audio_path,
                "csv_path": csv_path,
                "status": f"error: {e}",
            })
            print(f"[{i + 1}/{len(df)}] ERROR | {e}")

    all_results = pd.DataFrame(rows)
    all_results.to_excel(OUTPUT_EXCEL, index=False)
    valid = all_results.dropna(
        subset=["actual_qmax_ml_s", "predicted_qmax_ml_s"]
    ).copy()
    if len(valid) < 2:
        raise ValueError("Too few valid validation recordings")

    actual = valid["actual_qmax_ml_s"].to_numpy()
    predicted = valid["predicted_qmax_ml_s"].to_numpy()
    error = predicted - actual
    metrics = {
        "n": len(valid),
        "rmse_ml_s": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae_ml_s": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
        "mean_error_ml_s": float(np.mean(error)),
        "error_sd_ml_s": float(np.std(error, ddof=1)),
        "maximum_absolute_error_ml_s": float(np.max(np.abs(error))),
    }
    pd.DataFrame([metrics]).to_excel(METRICS_EXCEL, index=False)

    print("\n=== VALIDATION RESULTS ===")
    print(f"Valid recordings: {metrics['n']}")
    print(f"RMSE:             {metrics['rmse_ml_s']:.4f} mL/s")
    print(f"MAE:              {metrics['mae_ml_s']:.4f} mL/s")
    print(f"R²:               {metrics['r2']:.4f}")
    print(f"Mean error:       {metrics['mean_error_ml_s']:+.4f} mL/s")
    print(f"Error SD:         {metrics['error_sd_ml_s']:.4f} mL/s")

    show_results(valid, metrics)

    print("\n=== SAVED FILES ===")
    print(f"Predictions: {OUTPUT_EXCEL}")
    print(f"Metrics:     {METRICS_EXCEL}")
    print(f"Plots:       {OUTPUT_DIR}")
    print("Close the plot windows to finish.")
    plt.show()
    print("Done.")


if __name__ == "__main__":
    main()
