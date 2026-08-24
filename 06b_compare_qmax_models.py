"""
Compare four Qmax prediction models using one fixed training/validation split.

Models:
1. Ridge regression
2. ElasticNet
3. Random forest
4. Gradient boosting

Input:
    05_feature_analysis_outputs/audio_features_and_qmax.xlsx

Main outputs:
    model_comparison.xlsx / .csv
    validation_predictions.xlsx / .csv
    predicted_vs_reference_models.png
    validation_metrics_comparison.png
    best_qmax_model.joblib
"""

import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# =============================================================================
# SETTINGS
# =============================================================================

PROJECT_DIR = r"C:\school\project\everything everything"
INPUT_FILE = os.path.join(
    PROJECT_DIR,
    "05_feature_analysis_outputs",
    "audio_features_and_qmax.xlsx",
)
OUTPUT_DIR = os.path.join(PROJECT_DIR, "06_model_comparison_outputs")

TARGET_COLUMN = "qmax_ml_s"
VALIDATION_SIZE = 53
RANDOM_STATE = 42


# Only audio-derived inputs are included. Scale paths, Qmax, and row numbers
# are excluded from the prediction features.
FEATURE_COLUMNS = [
    "duration_s_audio",
    "max_freq_at_max_energy",
    "avg_freq_above_80pct_energy",
    "max_audio_energy",
    "mean_audio_energy",
    "std_audio_energy",
    "max_log_audio_energy",
    "mean_log_audio_energy",
    "std_log_audio_energy",
    "energy_slope",
    "log_energy_slope",
    "time_of_max_energy_s",
    "dominant_freq_at_peak_energy",
    "mean_dominant_freq",
    "std_dominant_freq",
    "max_dominant_freq",
    "mean_mel_power_emax_window_unlogged",
    "flow_eq_from_mel_emax_window",
]


MODEL_COLORS = {
    "Ridge": "#4C78A8",
    "ElasticNet": "#F58518",
    "Random Forest": "#54A24B",
    "Gradient Boosting": "#E45756",
}


# =============================================================================
# HELPERS
# =============================================================================

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_excel_writer(output_path):
    """Return a writable Excel path even if the normal output is open."""
    try:
        with open(output_path, "a+b"):
            pass
        return output_path
    except PermissionError:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
        base, extension = os.path.splitext(output_path)
        fallback = f"{base}_{timestamp}{extension}"
        print(f"Excel file is open; saving instead as: {fallback}")
        return fallback


def save_csv_safely(data, output_path):
    """Save a CSV, using a timestamped name if the normal file is open."""
    try:
        data.to_csv(output_path, index=False)
        return output_path
    except PermissionError:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
        base, extension = os.path.splitext(output_path)
        fallback = f"{base}_{timestamp}{extension}"
        data.to_csv(fallback, index=False)
        print(f"CSV file is open; saved instead as: {fallback}")
        return fallback


def build_models():
    """Create four models. All tuning is performed using training data only."""
    ridge = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                RidgeCV(alphas=np.logspace(-4, 4, 100)),
            ),
        ]
    )

    elastic_net = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                ElasticNetCV(
                    l1_ratio=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
                    alphas=np.logspace(-4, 2, 120),
                    cv=5,
                    max_iter=100000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    random_forest = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=500,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    gradient_boosting = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                GradientBoostingRegressor(
                    n_estimators=250,
                    learning_rate=0.03,
                    max_depth=2,
                    min_samples_leaf=3,
                    loss="huber",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return {
        "Ridge": ridge,
        "ElasticNet": elastic_net,
        "Random Forest": random_forest,
        "Gradient Boosting": gradient_boosting,
    }


def evaluate_model(name, model, x_train, y_train, x_validation, y_validation):
    """Fit one model and calculate its training and validation metrics."""
    print(f"Training: {name}")
    model.fit(x_train, y_train)

    train_prediction = model.predict(x_train)
    validation_prediction = model.predict(x_validation)

    metrics = {
        "model": name,
        "train_rmse_ml_s": rmse(y_train, train_prediction),
        "validation_rmse_ml_s": rmse(y_validation, validation_prediction),
        "train_mae_ml_s": float(mean_absolute_error(y_train, train_prediction)),
        "validation_mae_ml_s": float(
            mean_absolute_error(y_validation, validation_prediction)
        ),
        "train_r2": float(r2_score(y_train, train_prediction)),
        "validation_r2": float(r2_score(y_validation, validation_prediction)),
        "validation_bias_ml_s": float(
            np.mean(validation_prediction - y_validation.to_numpy())
        ),
    }

    return model, train_prediction, validation_prediction, metrics


def extract_model_details(name, fitted_model, feature_names):
    """Extract coefficients or feature importances for interpretation."""
    estimator = fitted_model.named_steps["model"]

    if hasattr(estimator, "coef_"):
        values = np.asarray(estimator.coef_, dtype=float)
        value_name = "standardized_coefficient"
    elif hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
        value_name = "feature_importance"
    else:
        return pd.DataFrame()

    details = pd.DataFrame(
        {
            "model": name,
            "feature": feature_names,
            "value_type": value_name,
            "value": values,
            "absolute_value": np.abs(values),
        }
    )
    return details.sort_values("absolute_value", ascending=False)


def extract_model_parameters(name, fitted_model):
    """Save the most useful fitted settings for each model."""
    estimator = fitted_model.named_steps["model"]
    rows = []

    if name == "Ridge":
        rows.append({"model": name, "parameter": "alpha", "value": estimator.alpha_})

    elif name == "ElasticNet":
        rows.extend(
            [
                {"model": name, "parameter": "alpha", "value": estimator.alpha_},
                {
                    "model": name,
                    "parameter": "l1_ratio",
                    "value": estimator.l1_ratio_,
                },
            ]
        )

    elif name == "Random Forest":
        rows.extend(
            [
                {
                    "model": name,
                    "parameter": "n_estimators",
                    "value": estimator.n_estimators,
                },
                {
                    "model": name,
                    "parameter": "min_samples_leaf",
                    "value": estimator.min_samples_leaf,
                },
                {
                    "model": name,
                    "parameter": "max_features",
                    "value": estimator.max_features,
                },
            ]
        )

    elif name == "Gradient Boosting":
        rows.extend(
            [
                {
                    "model": name,
                    "parameter": "n_estimators",
                    "value": estimator.n_estimators,
                },
                {
                    "model": name,
                    "parameter": "learning_rate",
                    "value": estimator.learning_rate,
                },
                {
                    "model": name,
                    "parameter": "max_depth",
                    "value": estimator.max_depth,
                },
            ]
        )

    return rows


def plot_predictions(results, y_validation, output_path):
    """Create a 2x2 predicted-versus-reference validation figure."""
    all_predictions = np.concatenate(
        [result["validation_prediction"] for result in results.values()]
    )
    lower = float(min(y_validation.min(), all_predictions.min()))
    upper = float(max(y_validation.max(), all_predictions.max()))
    margin = 0.05 * (upper - lower)
    lower -= margin
    upper += margin

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    for axis, (name, result) in zip(axes.ravel(), results.items()):
        prediction = result["validation_prediction"]
        metrics = result["metrics"]

        axis.scatter(
            y_validation,
            prediction,
            s=58,
            alpha=0.78,
            color=MODEL_COLORS[name],
            edgecolors="white",
            linewidths=0.5,
        )
        axis.plot([lower, upper], [lower, upper], "k--", linewidth=1.7)
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Reference Qmax (mL/s)")
        axis.set_ylabel("Predicted Qmax (mL/s)")
        axis.set_title(
            f"{name}\n"
            f"Validation RMSE = {metrics['validation_rmse_ml_s']:.2f} mL/s | "
            f"$R^2$ = {metrics['validation_r2']:.3f}"
        )
        axis.grid(alpha=0.25)

    fig.suptitle("Validation-Set Qmax Predictions", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_metric_comparison(comparison, output_path):
    """Compare validation RMSE, MAE, and R2 across the four models."""
    ordered = comparison.sort_values("validation_rmse_ml_s").reset_index(drop=True)
    colors = [MODEL_COLORS[name] for name in ordered["model"]]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    axes[0].bar(ordered["model"], ordered["validation_rmse_ml_s"], color=colors)
    axes[0].set_ylabel("RMSE (mL/s)")
    axes[0].set_title("Validation RMSE")

    axes[1].bar(ordered["model"], ordered["validation_mae_ml_s"], color=colors)
    axes[1].set_ylabel("MAE (mL/s)")
    axes[1].set_title("Validation MAE")

    axes[2].bar(ordered["model"], ordered["validation_r2"], color=colors)
    axes[2].axhline(0, color="black", linewidth=1)
    axes[2].set_ylabel("R²")
    axes[2].set_title("Validation R²")

    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)

    fig.suptitle("Comparison of Qmax Prediction Models", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n=== QMAX MODEL COMPARISON ===")
    print(f"Reading: {INPUT_FILE}")

    data = pd.read_excel(INPUT_FILE)

    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    available_features = [
        feature for feature in FEATURE_COLUMNS if feature in data.columns
    ]
    missing_features = [
        feature for feature in FEATURE_COLUMNS if feature not in data.columns
    ]

    if missing_features:
        print("Missing features will be skipped:")
        for feature in missing_features:
            print(f"  - {feature}")

    if len(available_features) < 2:
        raise ValueError("Too few audio features are available for modeling")

    working = data.copy()
    working[TARGET_COLUMN] = pd.to_numeric(
        working[TARGET_COLUMN], errors="coerce"
    )
    working = working.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

    if len(working) <= VALIDATION_SIZE:
        raise ValueError(
            f"The dataset must contain more than {VALIDATION_SIZE} valid rows"
        )

    x = working[available_features].apply(pd.to_numeric, errors="coerce")
    y = working[TARGET_COLUMN]
    row_indices = np.arange(len(working))

    # All four models use exactly the same fixed split.
    train_indices, validation_indices = train_test_split(
        row_indices,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    x_train = x.iloc[train_indices].copy()
    x_validation = x.iloc[validation_indices].copy()
    y_train = y.iloc[train_indices].copy()
    y_validation = y.iloc[validation_indices].copy()

    print(f"Training recordings: {len(train_indices)}")
    print(f"Validation recordings: {len(validation_indices)}")
    print(f"Features used: {len(available_features)}")

    models = build_models()
    results = {}
    metric_rows = []
    parameter_rows = []
    detail_tables = []

    for name, model in models.items():
        fitted, train_prediction, validation_prediction, metrics = evaluate_model(
            name,
            model,
            x_train,
            y_train,
            x_validation,
            y_validation,
        )

        results[name] = {
            "model": fitted,
            "train_prediction": train_prediction,
            "validation_prediction": validation_prediction,
            "metrics": metrics,
        }
        metric_rows.append(metrics)
        parameter_rows.extend(extract_model_parameters(name, fitted))
        details = extract_model_details(name, fitted, available_features)
        if not details.empty:
            detail_tables.append(details)

    comparison = pd.DataFrame(metric_rows).sort_values(
        "validation_rmse_ml_s"
    ).reset_index(drop=True)
    comparison.insert(0, "rank", np.arange(1, len(comparison) + 1))

    # Store the exact split so every result is reproducible.
    split_table = working[["excel_row", "audio_path", "csv_path", TARGET_COLUMN]].copy()
    split_table["dataset_split"] = "training"
    split_table.loc[validation_indices, "dataset_split"] = "validation"

    validation_output = working.loc[
        validation_indices,
        ["excel_row", "audio_path", "csv_path", TARGET_COLUMN],
    ].copy()
    validation_output = validation_output.rename(
        columns={TARGET_COLUMN: "reference_qmax_ml_s"}
    )

    for name, result in results.items():
        safe_name = name.lower().replace(" ", "_")
        prediction = result["validation_prediction"]
        validation_output[f"predicted_qmax_{safe_name}_ml_s"] = prediction
        validation_output[f"error_{safe_name}_ml_s"] = (
            prediction - validation_output["reference_qmax_ml_s"].to_numpy()
        )

    model_details = (
        pd.concat(detail_tables, ignore_index=True)
        if detail_tables
        else pd.DataFrame()
    )
    model_parameters = pd.DataFrame(parameter_rows)

    comparison_csv = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    predictions_csv = os.path.join(OUTPUT_DIR, "validation_predictions.csv")
    split_csv = os.path.join(OUTPUT_DIR, "train_validation_split.csv")
    comparison_csv = save_csv_safely(comparison, comparison_csv)
    predictions_csv = save_csv_safely(validation_output, predictions_csv)
    split_csv = save_csv_safely(split_table, split_csv)

    comparison_xlsx = safe_excel_writer(
        os.path.join(OUTPUT_DIR, "model_comparison.xlsx")
    )
    with pd.ExcelWriter(comparison_xlsx) as writer:
        comparison.to_excel(writer, sheet_name="model_comparison", index=False)
        validation_output.to_excel(
            writer, sheet_name="validation_predictions", index=False
        )
        split_table.to_excel(writer, sheet_name="data_split", index=False)
        model_parameters.to_excel(
            writer, sheet_name="model_parameters", index=False
        )
        if not model_details.empty:
            model_details.to_excel(
                writer, sheet_name="coefficients_importance", index=False
            )

    prediction_plot = os.path.join(
        OUTPUT_DIR, "predicted_vs_reference_models.png"
    )
    metric_plot = os.path.join(
        OUTPUT_DIR, "validation_metrics_comparison.png"
    )
    plot_predictions(results, y_validation, prediction_plot)
    plot_metric_comparison(comparison, metric_plot)

    best_name = comparison.iloc[0]["model"]
    best_model = results[best_name]["model"]
    model_path = os.path.join(OUTPUT_DIR, "best_qmax_model.joblib")
    joblib.dump(
        {
            "model_name": best_name,
            "model": best_model,
            "feature_columns": available_features,
            "target_column": TARGET_COLUMN,
            "random_state": RANDOM_STATE,
            "validation_size": VALIDATION_SIZE,
        },
        model_path,
    )

    print("\n=== MODEL COMPARISON ===")
    print(
        comparison[
            [
                "rank",
                "model",
                "train_rmse_ml_s",
                "validation_rmse_ml_s",
                "train_r2",
                "validation_r2",
            ]
        ].round(4).to_string(index=False)
    )
    print(f"\nBest validation model: {best_name}")
    print(f"Saved results: {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
