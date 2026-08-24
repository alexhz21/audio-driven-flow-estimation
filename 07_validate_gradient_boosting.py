import importlib.util
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =============================================================================
# SETTINGS
# =============================================================================

PROJECT_DIR = r"C:\school\project\everything everything"
TRAIN_SCRIPT = os.path.join(PROJECT_DIR, "06_train_gradient_boosting.py")
MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "06_gradient_boosting_training_outputs",
    "gradient_boosting_audio_only_model.joblib",
)
INPUT_EXCEL = os.path.join(PROJECT_DIR, "everything_val_key.xlsx")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "07_gradient_boosting_validation_outputs")
TEMP_WAV_DIR = os.path.join(OUTPUT_DIR, "wav_files")

OUTPUT_EXCEL = os.path.join(OUTPUT_DIR, "validation_predictions.xlsx")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "validation_predictions.csv")
METRICS_EXCEL = os.path.join(OUTPUT_DIR, "validation_metrics.xlsx")
ACTUAL_PREDICTED_PLOT = os.path.join(OUTPUT_DIR, "actual_vs_predicted.png")
RESIDUAL_PLOT = os.path.join(OUTPUT_DIR, "validation_residuals.png")
ERROR_DISTRIBUTION_PLOT = os.path.join(
    OUTPUT_DIR, "validation_error_distribution.png"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_WAV_DIR, exist_ok=True)


def load_training_functions():
    """
    Load feature extraction and Qmax calculation directly from Code 6.
    This guarantees that training and validation use identical processing.
    """
    if not os.path.exists(TRAIN_SCRIPT):
        raise FileNotFoundError(
            f"Training script not found: {TRAIN_SCRIPT}\n"
            "Place 06_train_gradient_boosting.py in the project folder."
        )

    specification = importlib.util.spec_from_file_location(
        "gradient_boosting_training", TRAIN_SCRIPT
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def convert_for_validation(training_module, audio_path):
    """Convert a validation recording to WAV when necessary."""
    extension = os.path.splitext(audio_path)[1].lower()
    if extension == ".wav":
        return audio_path

    base = os.path.splitext(os.path.basename(audio_path))[0]
    output_path = os.path.join(TEMP_WAV_DIR, base + ".wav")
    return training_module.convert_audio_to_wav(audio_path, TEMP_WAV_DIR)


def create_validation_plots(results, metrics):
    actual = results["actual_qmax_ml_s"].to_numpy()
    predicted = results["predicted_qmax_ml_s"].to_numpy()
    error = predicted - actual
    low = min(float(actual.min()), float(predicted.min()))
    high = max(float(actual.max()), float(predicted.max()))

    plt.figure(figsize=(6.5, 6))
    plt.scatter(actual, predicted, alpha=0.8)
    plt.plot([low, high], [low, high], "k--", label="Perfect prediction")
    plt.xlabel("Reference Qmax (mL/s)")
    plt.ylabel("Predicted Qmax (mL/s)")
    plt.title(
        "Independent Gradient Boosting Validation\n"
        f"RMSE={metrics['rmse_ml_s']:.3f} | "
        f"MAE={metrics['mae_ml_s']:.3f} | "
        f"R²={metrics['r2']:.3f}"
    )
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ACTUAL_PREDICTED_PLOT, dpi=220)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(predicted, error, alpha=0.8)
    plt.axhline(0, color="black", linestyle="--")
    plt.xlabel("Predicted Qmax (mL/s)")
    plt.ylabel("Error: predicted - reference (mL/s)")
    plt.title("Gradient Boosting Validation Residuals")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESIDUAL_PLOT, dpi=220)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.hist(error, bins="auto", edgecolor="black", alpha=0.8)
    plt.axvline(0, color="black", linestyle="--", label="Zero error")
    plt.axvline(
        np.mean(error),
        color="red",
        linestyle="--",
        label="Mean error",
    )
    plt.xlabel("Prediction error (mL/s)")
    plt.ylabel("Number of recordings")
    plt.title("Gradient Boosting Validation Error Distribution")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ERROR_DISTRIBUTION_PLOT, dpi=220)
    plt.close()


def save_excel_safely(data, output_path):
    try:
        data.to_excel(output_path, index=False)
        return output_path
    except PermissionError:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
        base, extension = os.path.splitext(output_path)
        alternative = f"{base}_{timestamp}{extension}"
        data.to_excel(alternative, index=False)
        print(f"Output was open. Saved instead as: {alternative}")
        return alternative


def main():
    print("\n=== STANDARD GRADIENT BOOSTING VALIDATION ===")
    print(f"Model: {MODEL_PATH}")
    print(f"Key:   {INPUT_EXCEL}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Run 06_train_gradient_boosting.py first."
        )

    training_module = load_training_functions()
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    validation_key = pd.read_excel(INPUT_EXCEL)
    required = {"audio_path", "csv_path"}
    missing = required - set(validation_key.columns)
    if missing:
        raise ValueError(f"Validation Excel is missing columns: {missing}")

    rows = []

    for index, row in validation_key.iterrows():
        audio_path = training_module.resolve_path(row["audio_path"])
        csv_path = training_module.resolve_path(row["csv_path"])

        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Scale file not found: {csv_path}")

            wav_path = convert_for_validation(training_module, audio_path)
            features = training_module.extract_audio_features(wav_path)
            feature_table = pd.DataFrame([features]).reindex(
                columns=feature_columns
            )
            predicted_qmax = float(model.predict(feature_table)[0])
            actual_qmax = training_module.compute_qmax_from_csv(csv_path)
            error = predicted_qmax - actual_qmax

            rows.append(
                {
                    "excel_row": index + 2,
                    "audio_path": audio_path,
                    "csv_path": csv_path,
                    "actual_qmax_ml_s": actual_qmax,
                    "predicted_qmax_ml_s": predicted_qmax,
                    "error_ml_s": error,
                    "absolute_error_ml_s": abs(error),
                    "status": "ok",
                }
            )
            print(
                f"[{index + 1}/{len(validation_key)}] OK | "
                f"actual={actual_qmax:.2f} | predicted={predicted_qmax:.2f} | "
                f"error={error:+.2f} mL/s"
            )

        except Exception as error:
            rows.append(
                {
                    "excel_row": index + 2,
                    "audio_path": audio_path,
                    "csv_path": csv_path,
                    "status": f"error: {error}",
                }
            )
            print(f"[{index + 1}/{len(validation_key)}] ERROR | {error}")

    all_results = pd.DataFrame(rows)
    saved_predictions = save_excel_safely(all_results, OUTPUT_EXCEL)
    all_results.to_csv(OUTPUT_CSV, index=False)

    valid = all_results.dropna(
        subset=["actual_qmax_ml_s", "predicted_qmax_ml_s"]
    ).copy()
    if len(valid) < 2:
        raise ValueError("Too few valid validation recordings")

    actual = valid["actual_qmax_ml_s"].to_numpy()
    predicted = valid["predicted_qmax_ml_s"].to_numpy()
    errors = predicted - actual

    metrics = {
        "model": "Gradient Boosting",
        "n": len(valid),
        "rmse_ml_s": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae_ml_s": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
        "mean_error_ml_s": float(np.mean(errors)),
        "error_sd_ml_s": float(np.std(errors, ddof=1)),
        "maximum_absolute_error_ml_s": float(np.max(np.abs(errors))),
    }
    saved_metrics = save_excel_safely(pd.DataFrame([metrics]), METRICS_EXCEL)
    create_validation_plots(valid, metrics)

    print("\n=== VALIDATION RESULTS ===")
    print(f"Valid recordings: {metrics['n']}")
    print(f"RMSE:             {metrics['rmse_ml_s']:.4f} mL/s")
    print(f"MAE:              {metrics['mae_ml_s']:.4f} mL/s")
    print(f"R²:               {metrics['r2']:.4f}")
    print(f"Mean error:       {metrics['mean_error_ml_s']:+.4f} mL/s")
    print(f"Error SD:         {metrics['error_sd_ml_s']:.4f} mL/s")
    print(f"Predictions:      {saved_predictions}")
    print(f"Metrics:          {saved_metrics}")
    print(f"Plots:            {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()