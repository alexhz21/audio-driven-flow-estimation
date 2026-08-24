import os
import warnings

import joblib
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydub import AudioSegment
from scipy.signal import savgol_filter
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")


# =============================================================================
# SETTINGS
# =============================================================================

PROJECT_DIR = r"C:\school\project\everything everything"
INDEX_FILE = os.path.join(PROJECT_DIR, "everything_train_key.xlsx")
WORK_DIR = os.path.join(PROJECT_DIR, "06_gradient_boosting_training_outputs")
WAV_DIR = os.path.join(WORK_DIR, "wav_files")

DATASET_OUT = os.path.join(WORK_DIR, "training_dataset_audio_only.csv")
PREDICTIONS_OUT = os.path.join(WORK_DIR, "loocv_predictions_audio_only.csv")
MODEL_OUT = os.path.join(WORK_DIR, "gradient_boosting_audio_only_model.joblib")
IMPORTANCE_PNG = os.path.join(WORK_DIR, "feature_importance_audio_only.png")
IMPORTANCE_XLSX = os.path.join(WORK_DIR, "gradient_boosting_feature_importance.xlsx")
LOOCV_PLOT = os.path.join(WORK_DIR, "loocv_actual_vs_predicted.png")
RESIDUAL_PLOT = os.path.join(WORK_DIR, "loocv_residuals.png")
METRICS_OUT = os.path.join(WORK_DIR, "loocv_metrics.xlsx")

TARGET_SR = 22050
DEFAULT_SCALE_HZ = 8.0
URINE_DENSITY = 1.02
MIN_USABLE_ROWS = 20
RANDOM_STATE = 42

TIME_COL_CANDIDATES = ["time", "Time", "seconds", "Seconds", "t"]
WEIGHT_COL_CANDIDATES = ["weight", "Weight", "grams", "g", "mass"]

WEIGHT_SMOOTH_SEC = 0.75
FLOW_SMOOTH_SEC = 0.75

# Updated power-law calibration used in Section 3.2.
EQ_A = 38.336
EQ_B = 0.390654
EMAX_WINDOW_SEC = 0.5
EMAX_HALF_WINDOW_SEC = EMAX_WINDOW_SEC / 2.0

os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(WAV_DIR, exist_ok=True)


# =============================================================================
# DATA PREPARATION
# =============================================================================

def read_index_file(path):
    extension = os.path.splitext(path)[1].lower()

    if extension in [".xlsx", ".xls"]:
        data = pd.read_excel(path)
    elif extension == ".csv":
        data = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported index file: {path}")

    required = {"audio_path", "csv_path"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return data.dropna(subset=["audio_path", "csv_path"]).reset_index(drop=True)


def find_first_existing_column(data, candidates):
    for column in candidates:
        if column in data.columns:
            return column
    return None


def resolve_path(value):
    path = str(value).strip()
    return path if os.path.isabs(path) else os.path.join(PROJECT_DIR, path)


def convert_audio_to_wav(audio_path, wav_dir):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    extension = os.path.splitext(audio_path)[1].lower()
    base = os.path.splitext(os.path.basename(audio_path))[0]
    wav_path = os.path.join(wav_dir, base + ".wav")

    if extension == ".wav":
        return audio_path

    if not os.path.exists(wav_path):
        audio = AudioSegment.from_file(audio_path).set_channels(1)
        audio.export(wav_path, format="wav")

    return wav_path


def extract_audio_features(audio_path, sr=TARGET_SR):
    """Extract exactly the same audio-only features used by ElasticNet."""
    signal, sr = librosa.load(audio_path, sr=sr, mono=True)
    if len(signal) == 0:
        raise ValueError(f"Empty audio file: {audio_path}")

    trimmed, _ = librosa.effects.trim(signal, top_db=25)
    signal_used = trimmed if len(trimmed) >= int(0.5 * sr) else signal

    peak = np.max(np.abs(signal_used))
    if peak > 0:
        signal_used = signal_used / peak

    duration_s_audio = float(len(signal) / sr)
    n_fft = 2048
    hop_length = 512

    rms = librosa.feature.rms(
        y=signal_used,
        frame_length=n_fft,
        hop_length=hop_length,
    )[0]
    log_rms = np.log(rms + 1e-8)
    frame_times = librosa.frames_to_time(
        np.arange(len(rms)), sr=sr, hop_length=hop_length
    )

    mel_spectrogram = librosa.feature.melspectrogram(
        y=signal_used,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=128,
        power=2.0,
    )
    mel_frequencies = librosa.mel_frequencies(
        n_mels=128, fmin=0, fmax=sr / 2
    )
    energy_by_frequency = np.sum(mel_spectrogram, axis=1)
    maximum_frequency_bin = int(np.argmax(energy_by_frequency))
    max_freq_at_max_energy = float(mel_frequencies[maximum_frequency_bin])

    threshold = 0.8 * np.max(energy_by_frequency)
    high_energy_mask = energy_by_frequency >= threshold
    avg_freq_above_80pct_energy = (
        float(np.mean(mel_frequencies[high_energy_mask]))
        if np.any(high_energy_mask)
        else np.nan
    )

    spectrum = np.abs(
        librosa.stft(signal_used, n_fft=n_fft, hop_length=hop_length)
    )
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    if spectrum.shape[1] > 0:
        dominant_frequency = frequencies[np.argmax(spectrum, axis=0)]
        mean_dominant_freq = float(np.mean(dominant_frequency))
        std_dominant_freq = float(np.std(dominant_frequency))
        max_dominant_freq = float(np.max(dominant_frequency))
    else:
        mean_dominant_freq = np.nan
        std_dominant_freq = np.nan
        max_dominant_freq = np.nan

    peak_frame = int(np.argmax(rms)) if len(rms) else 0
    time_of_max_energy_s = (
        float(frame_times[peak_frame])
        if peak_frame < len(frame_times)
        else np.nan
    )

    if spectrum.shape[1] > 0:
        peak_frame_clamped = min(peak_frame, spectrum.shape[1] - 1)
        dominant_freq_at_peak_energy = float(
            frequencies[np.argmax(spectrum[:, peak_frame_clamped])]
        )
    else:
        dominant_freq_at_peak_energy = np.nan

    frames_per_second = sr / hop_length
    half_window_frames = int(round(EMAX_HALF_WINDOW_SEC * frames_per_second))

    if mel_spectrogram.shape[1] > 0:
        peak_frame_mel = min(peak_frame, mel_spectrogram.shape[1] - 1)
        start_frame = max(0, peak_frame_mel - half_window_frames)
        end_frame = min(
            mel_spectrogram.shape[1], peak_frame_mel + half_window_frames + 1
        )
        mel_window = mel_spectrogram[:, start_frame:end_frame]
        mean_mel_power = (
            max(float(np.mean(mel_window)), 1e-12)
            if mel_window.size
            else np.nan
        )
    else:
        mean_mel_power = np.nan

    flow_equation_value = (
        float(EQ_A * mean_mel_power ** EQ_B)
        if np.isfinite(mean_mel_power)
        else np.nan
    )

    return {
        "duration_s_audio": duration_s_audio,
        "max_freq_at_max_energy": max_freq_at_max_energy,
        "avg_freq_above_80pct_energy": avg_freq_above_80pct_energy,
        "max_audio_energy": float(np.max(rms)) if len(rms) else np.nan,
        "mean_audio_energy": float(np.mean(rms)) if len(rms) else np.nan,
        "std_audio_energy": float(np.std(rms)) if len(rms) else np.nan,
        "max_log_audio_energy": float(np.max(log_rms)) if len(log_rms) else np.nan,
        "mean_log_audio_energy": float(np.mean(log_rms)) if len(log_rms) else np.nan,
        "std_log_audio_energy": float(np.std(log_rms)) if len(log_rms) else np.nan,
        "energy_slope": (
            float(np.polyfit(frame_times, rms, 1)[0]) if len(rms) >= 2 else np.nan
        ),
        "log_energy_slope": (
            float(np.polyfit(frame_times, log_rms, 1)[0])
            if len(log_rms) >= 2
            else np.nan
        ),
        "time_of_max_energy_s": time_of_max_energy_s,
        "dominant_freq_at_peak_energy": dominant_freq_at_peak_energy,
        "mean_dominant_freq": mean_dominant_freq,
        "std_dominant_freq": std_dominant_freq,
        "max_dominant_freq": max_dominant_freq,
        "mean_mel_power_emax_window_unlogged": mean_mel_power,
        "flow_eq_from_mel_emax_window": flow_equation_value,
    }


def compute_qmax_from_csv(csv_path):
    """Calculate the reference Qmax. It is used only as the target."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Scale file not found: {csv_path}")

    extension = os.path.splitext(csv_path)[1].lower()
    if extension == ".csv":
        data = pd.read_csv(csv_path)
    elif extension in [".xlsx", ".xls"]:
        data = pd.read_excel(csv_path)
    else:
        raise ValueError(f"Unsupported scale file: {csv_path}")

    time_column = find_first_existing_column(data, TIME_COL_CANDIDATES)
    weight_column = find_first_existing_column(data, WEIGHT_COL_CANDIDATES)
    if weight_column is None:
        raise ValueError(f"No weight column found in {csv_path}")

    weight = pd.to_numeric(data[weight_column], errors="coerce").to_numpy(float)
    if time_column is not None:
        time = pd.to_numeric(data[time_column], errors="coerce").to_numpy(float)
        valid = np.isfinite(weight) & np.isfinite(time)
        weight, time = weight[valid], time[valid]
        if len(weight) < 5:
            raise ValueError(f"Too few valid samples in {csv_path}")
        if np.any(np.diff(time) <= 0):
            time = np.arange(len(weight)) / DEFAULT_SCALE_HZ
    else:
        weight = weight[np.isfinite(weight)]
        time = np.arange(len(weight)) / DEFAULT_SCALE_HZ

    if len(weight) < 5:
        raise ValueError(f"Too few samples in {csv_path}")

    sampling_rate = (
        1.0 / np.median(np.diff(time)) if len(time) > 1 else DEFAULT_SCALE_HZ
    )
    weight_window = max(5, int(round(sampling_rate * WEIGHT_SMOOTH_SEC)))
    if weight_window % 2 == 0:
        weight_window += 1
    if weight_window >= len(weight):
        weight_window = len(weight) - 1 if len(weight) % 2 == 0 else len(weight)

    smooth_weight = (
        savgol_filter(weight, weight_window, 2)
        if weight_window >= 5 and weight_window < len(weight)
        else weight.copy()
    )
    flow = np.maximum(np.gradient(smooth_weight, time) / URINE_DENSITY, 0)

    flow_window = max(5, int(round(sampling_rate * FLOW_SMOOTH_SEC)))
    if flow_window % 2 == 0:
        flow_window += 1
    if flow_window >= len(flow):
        flow_window = len(flow) - 1 if len(flow) % 2 == 0 else len(flow)

    smooth_flow = (
        savgol_filter(flow, flow_window, 2)
        if flow_window >= 5 and flow_window < len(flow)
        else flow.copy()
    )
    return float(np.max(np.maximum(smooth_flow, 0)))


def build_dataset(index_data):
    rows = []

    for index, row in index_data.iterrows():
        audio_path = resolve_path(row["audio_path"])
        csv_path = resolve_path(row["csv_path"])
        print(f"[{index + 1}/{len(index_data)}] {audio_path}")

        try:
            wav_path = convert_audio_to_wav(audio_path, WAV_DIR)
            features = extract_audio_features(wav_path)
            reference_qmax = compute_qmax_from_csv(csv_path)
            rows.append(
                {
                    "audio_path": audio_path,
                    "wav_path": wav_path,
                    "csv_path": csv_path,
                    "qmax_ml_s": reference_qmax,
                    **features,
                }
            )
        except Exception as error:
            print(f"  ERROR: {error}")

    return pd.DataFrame(rows)


# =============================================================================
# GRADIENT BOOSTING MODEL
# =============================================================================

def get_gradient_boosting_model(feature_columns):
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                feature_columns,
            )
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        [
            ("prep", preprocessor),
            (
                "model",
                GradientBoostingRegressor(
                    loss="huber",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    # A compact grid keeps nested LOOCV computationally manageable.
    parameter_grid = {
        "model__n_estimators": [150, 250],
        "model__learning_rate": [0.03, 0.07],
        "model__max_depth": [2, 3],
        "model__min_samples_leaf": [2],
        "model__subsample": [0.8],
    }
    return pipeline, parameter_grid


def evaluate_loocv(features, target, pipeline, parameter_grid):
    leave_one_out = LeaveOneOut()
    true_values = []
    predictions = []
    best_parameters = []

    for fold, (train_indices, test_indices) in enumerate(
        leave_one_out.split(features), start=1
    ):
        x_train = features.iloc[train_indices]
        x_test = features.iloc[test_indices]
        y_train = target.iloc[train_indices]
        y_test = target.iloc[test_indices]

        inner_cv = KFold(n_splits=min(5, len(x_train)), shuffle=True, random_state=42)
        search = GridSearchCV(
            clone(pipeline),
            parameter_grid,
            scoring="neg_root_mean_squared_error",
            cv=inner_cv,
            n_jobs=-1,
        )
        search.fit(x_train, y_train)
        prediction = float(search.best_estimator_.predict(x_test)[0])

        true_values.append(float(y_test.iloc[0]))
        predictions.append(float(prediction))
        best_parameters.append(search.best_params_)

        if fold % 10 == 0 or fold == len(features):
            print(f"  fold {fold}/{len(features)}")

    true_values = np.asarray(true_values)
    predictions = np.asarray(predictions)
    summary = {
        "model": "Gradient Boosting",
        "rmse": float(np.sqrt(mean_squared_error(true_values, predictions))),
        "mae": float(mean_absolute_error(true_values, predictions)),
        "r2": float(r2_score(true_values, predictions)),
    }
    prediction_table = pd.DataFrame(
        {
            "y_true_qmax": true_values,
            "y_pred_qmax": predictions,
            "abs_error": np.abs(true_values - predictions),
        }
    )
    return summary, prediction_table, best_parameters


def fit_final_model(features, target, pipeline, parameter_grid):
    inner_cv = KFold(n_splits=min(5, len(features)), shuffle=True, random_state=42)
    search = GridSearchCV(
        clone(pipeline),
        parameter_grid,
        scoring="neg_root_mean_squared_error",
        cv=inner_cv,
        n_jobs=-1,
    )
    search.fit(features, target)
    return search.best_estimator_, search.best_params_


def save_feature_importance(artifact):
    model = artifact["model"].named_steps["model"]
    names = artifact["feature_columns"]
    importance = pd.DataFrame(
        {"feature": names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance.to_excel(IMPORTANCE_XLSX, index=False)

    top = importance.head(12).sort_values("importance")
    plt.figure(figsize=(10, 6))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Feature importance")
    plt.title("Top Gradient Boosting Feature Importances (Audio Only)")
    plt.tight_layout()
    plt.savefig(IMPORTANCE_PNG, dpi=220)
    plt.close()


def plot_loocv_results(predictions, summary):
    actual = predictions["y_true_qmax"].to_numpy()
    predicted = predictions["y_pred_qmax"].to_numpy()
    residual = predicted - actual
    low = min(float(actual.min()), float(predicted.min()))
    high = max(float(actual.max()), float(predicted.max()))

    plt.figure(figsize=(6.5, 6))
    plt.scatter(actual, predicted, alpha=0.8)
    plt.plot([low, high], [low, high], "k--", label="Perfect prediction")
    plt.xlabel("Reference Qmax (mL/s)")
    plt.ylabel("LOOCV-predicted Qmax (mL/s)")
    plt.title(
        "Gradient Boosting: Nested LOOCV\n"
        f"RMSE={summary['rmse']:.3f} | MAE={summary['mae']:.3f} | "
        f"R²={summary['r2']:.3f}"
    )
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOOCV_PLOT, dpi=220)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(predicted, residual, alpha=0.8)
    plt.axhline(0, color="black", linestyle="--")
    plt.xlabel("Predicted Qmax (mL/s)")
    plt.ylabel("Residual: predicted - reference (mL/s)")
    plt.title("Gradient Boosting LOOCV Residuals")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESIDUAL_PLOT, dpi=220)
    plt.close()


def main():
    index_data = read_index_file(INDEX_FILE)
    print(f"Found {len(index_data)} training file pairs")
    dataset = build_dataset(index_data)
    print(f"Usable training rows: {len(dataset)}")

    if len(dataset) < MIN_USABLE_ROWS:
        raise ValueError(f"Too few usable rows: {len(dataset)}")

    dataset.to_csv(DATASET_OUT, index=False)
    non_features = ["audio_path", "wav_path", "csv_path", "qmax_ml_s"]
    feature_columns = [column for column in dataset.columns if column not in non_features]
    features = dataset[feature_columns].copy()
    target = dataset["qmax_ml_s"].copy()

    pipeline, parameter_grid = get_gradient_boosting_model(feature_columns)
    print("\nEvaluating standard Gradient Boosting with nested LOOCV...")
    summary, predictions, _ = evaluate_loocv(
        features, target, pipeline, parameter_grid
    )

    print("\n=== TRAINING LOOCV RESULTS ===")
    print(f"RMSE: {summary['rmse']:.4f} mL/s")
    print(f"MAE:  {summary['mae']:.4f} mL/s")
    print(f"R²:   {summary['r2']:.4f}")
    pd.DataFrame([summary]).to_excel(METRICS_OUT, index=False)

    predictions.insert(0, "audio_path", dataset["audio_path"].to_numpy())
    predictions.insert(1, "csv_path", dataset["csv_path"].to_numpy())
    predictions["error"] = predictions["y_pred_qmax"] - predictions["y_true_qmax"]
    predictions.to_csv(PREDICTIONS_OUT, index=False)

    final_model, best_parameters = fit_final_model(
        features, target, pipeline, parameter_grid
    )
    artifact = {
        "model": final_model,
        "feature_columns": feature_columns,
        "best_model_name": "Gradient Boosting",
        "best_params": best_parameters,
        "summary": summary,
        "calibration_equation": {"a": EQ_A, "b": EQ_B},
    }
    joblib.dump(artifact, MODEL_OUT)
    save_feature_importance(artifact)
    plot_loocv_results(predictions, summary)

    print(f"Best hyperparameters: {best_parameters}")
    print(f"Saved model: {MODEL_OUT}")
    print(f"Saved outputs: {WORK_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()