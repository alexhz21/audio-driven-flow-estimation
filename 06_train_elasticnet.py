import os
import warnings
import numpy as np
import pandas as pd
import joblib
import librosa
import matplotlib.pyplot as plt

from pydub import AudioSegment
from scipy.signal import savgol_filter
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import LeaveOneOut, GridSearchCV, KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


PROJECT_DIR = r"C:\school\project\everything everything"
INDEX_FILE = os.path.join(PROJECT_DIR, "everything_key.xlsx")
WORK_DIR = os.path.join(PROJECT_DIR, "06_elasticnet_training_outputs")
WAV_DIR = os.path.join(WORK_DIR, "wav_files")
DATASET_OUT = os.path.join(WORK_DIR, "training_dataset_audio_only.csv")
PREDICTIONS_OUT = os.path.join(WORK_DIR, "loocv_predictions_audio_only.csv")
MODEL_OUT = os.path.join(WORK_DIR, "elasticnet_audio_only_model.joblib")
FEATURE_IMPORTANCE_PNG = os.path.join(WORK_DIR, "feature_importance_audio_only.png")
LOOCV_PLOT = os.path.join(WORK_DIR, "loocv_actual_vs_predicted.png")
RESIDUAL_PLOT = os.path.join(WORK_DIR, "loocv_residuals.png")
METRICS_OUT = os.path.join(WORK_DIR, "loocv_metrics.xlsx")
COEFFICIENTS_OUT = os.path.join(WORK_DIR, "elasticnet_coefficients.xlsx")

TARGET_SR = 22050
DEFAULT_SCALE_HZ = 8.0
URINE_DENSITY = 1.02
MIN_USABLE_ROWS = 20

TIME_COL_CANDIDATES = ["time", "Time", "seconds", "Seconds", "t"]
WEIGHT_COL_CANDIDATES = ["weight", "Weight", "grams", "g", "mass"]

WEIGHT_SMOOTH_SEC = 0.75
FLOW_SMOOTH_SEC = 0.75

EQ_A = 38.053
EQ_B = 0.386282
EMAX_WINDOW_SEC = 0.5
EMAX_HALF_WINDOW_SEC = EMAX_WINDOW_SEC / 2.0

os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(WAV_DIR, exist_ok=True)


def read_index_file(path):
    ext = os.path.splitext(path)[1].lower()

    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported index file: {path}")

    required = {"audio_path", "csv_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=["audio_path", "csv_path"]).reset_index(drop=True)
    return df


def find_first_existing_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def resolve_path(value):
    """Return an absolute project path."""
    path = str(value).strip()
    return path if os.path.isabs(path) else os.path.join(PROJECT_DIR, path)


def convert_audio_to_wav(audio_path, wav_dir):
    """
    Converts non-wav audio to wav if needed.
    Returns a wav path that librosa can read reliably.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    ext = os.path.splitext(audio_path)[1].lower()
    base = os.path.splitext(os.path.basename(audio_path))[0]
    wav_path = os.path.join(wav_dir, base + ".wav")

    if ext == ".wav":
        return audio_path

    if not os.path.exists(wav_path):
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1)
        audio.export(wav_path, format="wav")

    return wav_path


def extract_audio_features(audio_path, sr=TARGET_SR):
    """
    Audio-only features.
    No target leakage. No CSV-derived quantities are used as inputs.
    Includes equation-based feature:
        flow_eq_from_mel_emax_window = a * (x ** b)
    where x is mean mel power (unlogged) in a 0.5 s window around e_max.
    """
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


def compute_qmax_from_csv(csv_path):
    """
    Reads CSV and computes qmax_ml_s.
    IMPORTANT: This is used only to create the target y, not as an input feature.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    ext = os.path.splitext(csv_path)[1].lower()
    if ext == ".csv":
        wdf = pd.read_csv(csv_path)
    elif ext in [".xlsx", ".xls"]:
        wdf = pd.read_excel(csv_path)
    else:
        raise ValueError(f"Unsupported CSV file: {csv_path}")

    time_col = find_first_existing_column(wdf, TIME_COL_CANDIDATES)
    weight_col = find_first_existing_column(wdf, WEIGHT_COL_CANDIDATES)

    if weight_col is None:
        raise ValueError(f"No weight column found in {csv_path}")

    weight = pd.to_numeric(wdf[weight_col], errors="coerce").to_numpy(dtype=float)

    if time_col is not None:
        time = pd.to_numeric(wdf[time_col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(weight) & np.isfinite(time)
        weight = weight[valid]
        time = time[valid]

        if len(weight) < 5:
            raise ValueError(f"Too few valid samples in {csv_path}")

        dt = np.diff(time)
        if np.any(dt <= 0):
            time = np.arange(len(weight)) / DEFAULT_SCALE_HZ
    else:
        weight = weight[np.isfinite(weight)]
        time = np.arange(len(weight)) / DEFAULT_SCALE_HZ

    if len(weight) < 5:
        raise ValueError(f"Too few samples in {csv_path}")

    est_hz = 1.0 / np.median(np.diff(time)) if len(time) > 1 else DEFAULT_SCALE_HZ

    weight_window = max(5, int(round(est_hz * WEIGHT_SMOOTH_SEC)))
    if weight_window % 2 == 0:
        weight_window += 1
    if weight_window >= len(weight):
        weight_window = len(weight) - 1 if len(weight) % 2 == 0 else len(weight)

    if weight_window >= 5 and weight_window < len(weight):
        weight_smooth = savgol_filter(weight, window_length=weight_window, polyorder=2)
    else:
        weight_smooth = weight.copy()

    flow = np.gradient(weight_smooth, time) / URINE_DENSITY
    flow = np.where(np.isfinite(flow), flow, 0.0)
    flow = np.maximum(flow, 0.0)

    flow_window = max(5, int(round(est_hz * FLOW_SMOOTH_SEC)))
    if flow_window % 2 == 0:
        flow_window += 1
    if flow_window >= len(flow):
        flow_window = len(flow) - 1 if len(flow) % 2 == 0 else len(flow)

    if flow_window >= 5 and flow_window < len(flow):
        flow_smooth = savgol_filter(flow, window_length=flow_window, polyorder=2)
    else:
        flow_smooth = flow.copy()

    flow_smooth = np.maximum(flow_smooth, 0.0)

    qmax_ml_s = float(np.max(flow_smooth))
    return qmax_ml_s


def build_dataset(index_df):
    rows = []

    for i, row in index_df.iterrows():
        audio_path = resolve_path(row["audio_path"])
        csv_path = resolve_path(row["csv_path"])

        print(f"[{i + 1}/{len(index_df)}]")
        print(f"  audio : {audio_path}")
        print(f"  csv   : {csv_path}")

        try:
            wav_path = convert_audio_to_wav(audio_path, WAV_DIR)
            feature_dict = extract_audio_features(wav_path)
            qmax_ml_s = compute_qmax_from_csv(csv_path)

            out_row = {
                "audio_path": audio_path,
                "wav_path": wav_path,
                "csv_path": csv_path,
                "qmax_ml_s": qmax_ml_s,
            }
            out_row.update(feature_dict)
            rows.append(out_row)

        except Exception as e:
            print(f"  ERROR: {e}")

    return pd.DataFrame(rows)


def get_elasticnet_model(feature_cols):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), feature_cols)
        ],
        remainder="drop"
    )

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", ElasticNet(max_iter=100000, random_state=42))
    ])

    param_grid = {
        "model__alpha": [0.01, 0.1, 1.0],
        "model__l1_ratio": [0.3, 0.5, 0.7]
    }

    return pipe, param_grid


def evaluate_loocv(X, y, pipe, param_grid):
    loo = LeaveOneOut()

    y_true_all = []
    y_pred_all = []
    best_params_each_fold = []

    for fold_idx, (train_idx, test_idx) in enumerate(loo.split(X), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        inner_splits = min(5, len(X_train))
        inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=42)

        grid = GridSearchCV(
            estimator=clone(pipe),
            param_grid=param_grid,
            scoring="neg_root_mean_squared_error",
            cv=inner_cv,
            n_jobs=-1
        )

        grid.fit(X_train, y_train)
        pred = grid.best_estimator_.predict(X_test)[0]

        y_true_all.append(float(y_test.iloc[0]))
        y_pred_all.append(float(pred))
        best_params_each_fold.append(grid.best_params_)

        if fold_idx % 10 == 0 or fold_idx == len(X):
            print(f"  fold {fold_idx}/{len(X)}")

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    rmse = float(np.sqrt(mean_squared_error(y_true_all, y_pred_all)))
    mae = float(mean_absolute_error(y_true_all, y_pred_all))
    r2 = float(r2_score(y_true_all, y_pred_all))

    pred_df = pd.DataFrame({
        "y_true_qmax": y_true_all,
        "y_pred_qmax": y_pred_all,
        "abs_error": np.abs(y_true_all - y_pred_all)
    })

    summary = {
        "model": "ElasticNet",
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }

    return summary, pred_df, best_params_each_fold


def fit_final_model(X, y, pipe, param_grid):
    inner_splits = min(5, len(X))
    inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=clone(pipe),
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=inner_cv,
        n_jobs=-1
    )
    grid.fit(X, y)

    return grid.best_estimator_, grid.best_params_


def print_and_save_feature_importance(artifact, out_png_path):
    model = artifact["model"]
    feature_names = artifact["feature_columns"]

    coefs = model.named_steps["model"].coef_
    importance = list(zip(feature_names, coefs))
    importance_sorted = sorted(importance, key=lambda x: abs(x[1]), reverse=True)

    print("\n=== FEATURE COEFFICIENTS ===\n")
    for name, coef in importance_sorted:
        print(f"{name:<45s} {coef:+.4f}")

    non_zero = [(f, c) for f, c in importance if abs(c) > 1e-4]
    print(f"\nUsed features: {len(non_zero)} / {len(feature_names)}")

    top_items = importance_sorted[:12]
    names = [x[0] for x in top_items]
    values = [x[1] for x in top_items]

    plt.figure(figsize=(10, 6))
    plt.barh(names, values)
    plt.gca().invert_yaxis()
    plt.title("Top ElasticNet Feature Coefficients (Audio Only)")
    plt.xlabel("Coefficient")
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=200)

    coefficient_df = pd.DataFrame(
        importance_sorted, columns=["feature", "coefficient"]
    )
    coefficient_df["absolute_coefficient"] = coefficient_df["coefficient"].abs()
    coefficient_df.to_excel(COEFFICIENTS_OUT, index=False)

    print(f"Saved feature importance plot to: {out_png_path}")


def plot_loocv_results(pred_df, summary):
    """Show LOOCV predictions and residuals."""
    actual = pred_df["y_true_qmax"].to_numpy()
    predicted = pred_df["y_pred_qmax"].to_numpy()
    residual = predicted - actual
    low = min(float(actual.min()), float(predicted.min()))
    high = max(float(actual.max()), float(predicted.max()))

    plt.figure(figsize=(6.5, 6))
    plt.scatter(actual, predicted, alpha=0.8)
    plt.plot([low, high], [low, high], "k--", label="Perfect prediction")
    plt.xlabel("Actual Qmax (mL/s)")
    plt.ylabel("LOOCV-predicted Qmax (mL/s)")
    plt.title(
        f"ElasticNet nested LOOCV\n"
        f"RMSE={summary['rmse']:.3f} | MAE={summary['mae']:.3f} | R²={summary['r2']:.3f}"
    )
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOOCV_PLOT, dpi=220)

    plt.figure(figsize=(7, 5))
    plt.scatter(predicted, residual, alpha=0.8)
    plt.axhline(0, color="black", linestyle="--")
    plt.xlabel("Predicted Qmax (mL/s)")
    plt.ylabel("Residual: predicted - actual (mL/s)")
    plt.title("ElasticNet LOOCV residuals")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESIDUAL_PLOT, dpi=220)


def main():
    index_df = read_index_file(INDEX_FILE)
    print(f"Found {len(index_df)} file pairs")

    dataset_df = build_dataset(index_df)
    print(f"\nUsable rows: {len(dataset_df)}")

    if len(dataset_df) < MIN_USABLE_ROWS:
        raise ValueError(f"Too few usable rows: {len(dataset_df)}")

    if os.path.exists(DATASET_OUT):
        try:
            os.remove(DATASET_OUT)
        except OSError:
            pass

    dataset_df.to_csv(DATASET_OUT, index=False)
    print(f"Saved dataset to: {DATASET_OUT}")

    target_col = "qmax_ml_s"

    non_feature_cols = [
        "audio_path",
        "wav_path",
        "csv_path",
        "qmax_ml_s",
    ]

    feature_cols = [c for c in dataset_df.columns if c not in non_feature_cols]

    print("\nAudio features used in the model:")
    for c in feature_cols:
        print(f" - {c}")

    X = dataset_df[feature_cols].copy()
    y = dataset_df[target_col].copy()

    pipe, param_grid = get_elasticnet_model(feature_cols)

    print("\nEvaluating ElasticNet with LOOCV...")
    summary, pred_df, best_params_each_fold = evaluate_loocv(X, y, pipe, param_grid)

    print("\n=== RESULTS ===")
    print(f"RMSE: {summary['rmse']:.4f} mL/s")
    print(f"MAE:  {summary['mae']:.4f} mL/s")
    print(f"R²:   {summary['r2']:.4f}")

    pd.DataFrame([summary]).to_excel(METRICS_OUT, index=False)

    pred_df.insert(0, "audio_path", dataset_df["audio_path"].to_numpy())
    pred_df.insert(1, "csv_path", dataset_df["csv_path"].to_numpy())
    pred_df["error"] = pred_df["y_pred_qmax"] - pred_df["y_true_qmax"]

    if os.path.exists(PREDICTIONS_OUT):
        try:
            os.remove(PREDICTIONS_OUT)
        except OSError:
            pass

    pred_df.to_csv(PREDICTIONS_OUT, index=False)
    print(f"Saved LOOCV predictions to: {PREDICTIONS_OUT}")

    final_model, best_params = fit_final_model(X, y, pipe, param_grid)

    artifact = {
        "model": final_model,
        "feature_columns": feature_cols,
        "best_model_name": "ElasticNet",
        "best_params": best_params,
        "summary": summary
    }

    joblib.dump(artifact, MODEL_OUT)
    print(f"Saved trained model to: {MODEL_OUT}")
    print(f"Best hyperparameters: {best_params}")

    print_and_save_feature_importance(artifact, FEATURE_IMPORTANCE_PNG)

    plot_loocv_results(pred_df, summary)

    print("\n=== SAVED FILES ===")
    print(f"Model:        {MODEL_OUT}")
    print(f"Predictions:  {PREDICTIONS_OUT}")
    print(f"Metrics:      {METRICS_OUT}")
    print(f"Coefficients: {COEFFICIENTS_OUT}")
    print(f"Plots:        {WORK_DIR}")
    print("Close the plot windows to finish.")
    plt.show()
    print("Done.")


if __name__ == "__main__":
    main()
