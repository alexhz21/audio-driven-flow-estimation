import importlib.util
import json
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split

warnings.filterwarnings("ignore")


# =============================================================================
# SETTINGS
# =============================================================================

PROJECT_DIR = r"C:\school\project\everything everything"
TRAINING_CODE = os.path.join(PROJECT_DIR, "06_train_gradient_boosting.py")
INDEX_FILE = os.path.join(PROJECT_DIR, "everything_key.xlsx")

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "06c_four_random_gradient_boosting_outputs",
)
WAV_DIR = os.path.join(OUTPUT_DIR, "wav_files")
DATASET_FILE = os.path.join(OUTPUT_DIR, "complete_feature_dataset.csv")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "four_split_metrics_summary.xlsx")
COMBINED_PLOT = os.path.join(OUTPUT_DIR, "four_validation_comparison.png")

NUMBER_OF_RUNS = 4
TRAIN_SIZE = 115
VALIDATION_SIZE = 53

# Different fixed seeds create different random splits while keeping the
# experiment reproducible if the code is run again.
SPLIT_SEEDS = [1,2, 3, 4]

# If the feature dataset already exists, reuse it instead of processing every
# audio file again. Set this to False if feature-extraction settings changed.
REUSE_CACHED_DATASET = True

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(WAV_DIR, exist_ok=True)


# =============================================================================
# LOAD THE FEATURE-EXTRACTION FUNCTIONS FROM CODE 6
# =============================================================================

def load_training_code():
    """Load Code 6 so all runs use the same feature-processing functions."""
    if not os.path.exists(TRAINING_CODE):
        raise FileNotFoundError(
            f"Training code not found: {TRAINING_CODE}\n"
            "Place 06_train_gradient_boosting.py in the project folder."
        )

    specification = importlib.util.spec_from_file_location(
        "gradient_boosting_training",
        TRAINING_CODE,
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


# =============================================================================
# DATASET PREPARATION
# =============================================================================

def build_complete_dataset(training_module):
    """Extract features once from every pair listed in everything_key.xlsx."""
    if REUSE_CACHED_DATASET and os.path.exists(DATASET_FILE):
        print(f"Using cached feature dataset: {DATASET_FILE}")
        dataset = pd.read_csv(DATASET_FILE)
    else:
        index_data = training_module.read_index_file(INDEX_FILE)
        print(f"Found {len(index_data)} paired recordings in everything_key.xlsx")

        rows = []
        for index, row in index_data.iterrows():
            audio_path = training_module.resolve_path(row["audio_path"])
            csv_path = training_module.resolve_path(row["csv_path"])
            print(f"[{index + 1}/{len(index_data)}] {audio_path}")

            try:
                wav_path = training_module.convert_audio_to_wav(
                    audio_path,
                    WAV_DIR,
                )
                features = training_module.extract_audio_features(wav_path)
                reference_qmax = training_module.compute_qmax_from_csv(csv_path)

                rows.append(
                    {
                        "source_row": index + 2,
                        "audio_path": audio_path,
                        "wav_path": wav_path,
                        "csv_path": csv_path,
                        "qmax_ml_s": reference_qmax,
                        **features,
                    }
                )
            except Exception as error:
                print(f"  ERROR: {error}")

        dataset = pd.DataFrame(rows)
        dataset.to_csv(DATASET_FILE, index=False)

    # this run only makes sense with exactly TRAIN_SIZE + VALIDATION_SIZE usable
    # recordings, so fail loudly rather than silently running on a different count
    expected_total = TRAIN_SIZE + VALIDATION_SIZE
    if len(dataset) != expected_total:
        raise ValueError(
            f"Expected exactly {expected_total} usable recordings for a "
            f"{TRAIN_SIZE}/{VALIDATION_SIZE} split, but found {len(dataset)}.\n"
            "Check everything_key.xlsx and the feature-extraction error messages."
        )

    return dataset.reset_index(drop=True)


# =============================================================================
# MODEL TRAINING AND VALIDATION
# =============================================================================

def calculate_metrics(actual, predicted):
    """Calculate validation performance metrics."""
    errors = predicted - actual
    return {
        "rmse_ml_s": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae_ml_s": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
        "mean_error_ml_s": float(np.mean(errors)),
        "error_sd_ml_s": float(np.std(errors, ddof=1)),
        "maximum_absolute_error_ml_s": float(np.max(np.abs(errors))),
    }


def fit_one_run(
    run_number,
    split_seed,
    dataset,
    feature_columns,
    training_module,
):
    """Create one random 115/53 split, train a model, and validate it."""
    run_dir = os.path.join(OUTPUT_DIR, f"run_{run_number}")
    os.makedirs(run_dir, exist_ok=True)

    all_indices = np.arange(len(dataset))
    train_indices, validation_indices = train_test_split(
        all_indices,
        train_size=TRAIN_SIZE,
        test_size=VALIDATION_SIZE,
        random_state=split_seed,
        shuffle=True,
    )

    train_data = dataset.iloc[train_indices].reset_index(drop=True)
    validation_data = dataset.iloc[validation_indices].reset_index(drop=True)

    x_train = train_data[feature_columns].copy()
    y_train = train_data["qmax_ml_s"].copy()
    x_validation = validation_data[feature_columns].copy()
    y_validation = validation_data["qmax_ml_s"].to_numpy(dtype=float)

    pipeline, parameter_grid = training_module.get_gradient_boosting_model(
        feature_columns
    )
    pipeline = clone(pipeline)
    # each run gets its own random_state so the four runs aren't just copies
    # of each other with a different data split
    pipeline.set_params(model__random_state=split_seed)

    inner_cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=split_seed,
    )
    search = GridSearchCV(
        pipeline,
        parameter_grid,
        scoring="neg_root_mean_squared_error",
        cv=inner_cv,
        n_jobs=-1,
    )
    search.fit(x_train, y_train)

    fitted_model = search.best_estimator_
    predicted = fitted_model.predict(x_validation).astype(float)
    metrics = calculate_metrics(y_validation, predicted)

    model_file = os.path.join(run_dir, "gradient_boosting_model.joblib")
    artifact = {
        "model": fitted_model,
        "feature_columns": feature_columns,
        "model_name": "Gradient Boosting",
        "run_number": run_number,
        "split_seed": split_seed,
        "train_size": TRAIN_SIZE,
        "validation_size": VALIDATION_SIZE,
        "best_params": search.best_params_,
        "validation_metrics": metrics,
    }
    joblib.dump(artifact, model_file)

    membership = dataset[
        ["source_row", "audio_path", "csv_path", "qmax_ml_s"]
    ].copy()
    membership["set"] = ""
    membership.loc[train_indices, "set"] = "train"
    membership.loc[validation_indices, "set"] = "validation"
    membership.to_excel(
        os.path.join(run_dir, "split_membership.xlsx"),
        index=False,
    )

    predictions = validation_data[
        ["source_row", "audio_path", "csv_path", "qmax_ml_s"]
    ].copy()
    predictions = predictions.rename(columns={"qmax_ml_s": "actual_qmax_ml_s"})
    predictions["predicted_qmax_ml_s"] = predicted
    predictions["error_ml_s"] = predicted - y_validation
    predictions["absolute_error_ml_s"] = np.abs(predicted - y_validation)
    predictions.to_excel(
        os.path.join(run_dir, "validation_predictions.xlsx"),
        index=False,
    )
    predictions.to_csv(
        os.path.join(run_dir, "validation_predictions.csv"),
        index=False,
    )

    pd.DataFrame(
        [
            {
                "run": run_number,
                "split_seed": split_seed,
                "train_n": TRAIN_SIZE,
                "validation_n": VALIDATION_SIZE,
                **metrics,
            }
        ]
    ).to_excel(os.path.join(run_dir, "validation_metrics.xlsx"), index=False)

    with open(
        os.path.join(run_dir, "best_hyperparameters.json"),
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(search.best_params_, output_file, indent=2)

    save_run_plots(
        run_number,
        run_dir,
        y_validation,
        predicted,
        metrics,
        fitted_model,
        feature_columns,
    )

    result_row = {
        "run": run_number,
        "split_seed": split_seed,
        "train_n": TRAIN_SIZE,
        "validation_n": VALIDATION_SIZE,
        **metrics,
    }

    plot_item = {
        "run": run_number,
        "actual": y_validation,
        "predicted": predicted,
        "metrics": metrics,
    }
    return result_row, plot_item


# =============================================================================
# PLOTS
# =============================================================================

def save_run_plots(
    run_number,
    run_dir,
    actual,
    predicted,
    metrics,
    fitted_model,
    feature_columns,
):
    """Save prediction, residual, and feature-importance figures for one run."""
    low = min(float(np.min(actual)), float(np.min(predicted)))
    high = max(float(np.max(actual)), float(np.max(predicted)))
    residuals = predicted - actual

    plt.figure(figsize=(6.5, 6))
    plt.scatter(actual, predicted, alpha=0.8)
    plt.plot([low, high], [low, high], "k--", label="Perfect prediction")
    plt.xlabel("Reference Qmax (mL/s)")
    plt.ylabel("Predicted Qmax (mL/s)")
    plt.title(
        f"Gradient Boosting Validation - Run {run_number}\n"
        f"RMSE={metrics['rmse_ml_s']:.3f} | "
        f"MAE={metrics['mae_ml_s']:.3f} | "
        f"R²={metrics['r2']:.3f}"
    )
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "actual_vs_predicted.png"), dpi=220)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(predicted, residuals, alpha=0.8)
    plt.axhline(0, color="black", linestyle="--")
    plt.xlabel("Predicted Qmax (mL/s)")
    plt.ylabel("Error: predicted - reference (mL/s)")
    plt.title(f"Gradient Boosting Validation Residuals - Run {run_number}")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "validation_residuals.png"), dpi=220)
    plt.close()

    regressor = fitted_model.named_steps["model"]
    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": regressor.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_excel(
        os.path.join(run_dir, "feature_importance.xlsx"),
        index=False,
    )

    top = importance.head(12).sort_values("importance")
    plt.figure(figsize=(10, 6))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Feature importance")
    plt.title(f"Gradient Boosting Feature Importance - Run {run_number}")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "feature_importance.png"), dpi=220)
    plt.close()


def save_combined_plot(plot_items):
    """Save all four independent validation plots in one figure."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)

    for axis, item in zip(axes.ravel(), plot_items):
        actual = item["actual"]
        predicted = item["predicted"]
        metrics = item["metrics"]
        low = min(float(np.min(actual)), float(np.min(predicted)))
        high = max(float(np.max(actual)), float(np.max(predicted)))

        axis.scatter(actual, predicted, alpha=0.75)
        axis.plot([low, high], [low, high], "k--")
        axis.set_xlabel("Reference Qmax (mL/s)")
        axis.set_ylabel("Predicted Qmax (mL/s)")
        axis.set_title(
            f"Run {item['run']} | RMSE={metrics['rmse_ml_s']:.3f} | "
            f"R²={metrics['r2']:.3f}"
        )
        axis.grid(alpha=0.25)

    fig.suptitle(
        "Gradient Boosting Performance Across Four Random 115/53 Splits",
        fontsize=15,
    )
    plt.savefig(COMBINED_PLOT, dpi=220)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n=== FOUR RANDOM GRADIENT BOOSTING SPLITS ===")
    print(f"Index file:      {INDEX_FILE}")
    print(f"Train size:      {TRAIN_SIZE}")
    print(f"Validation size: {VALIDATION_SIZE}")

    training_module = load_training_code()
    dataset = build_complete_dataset(training_module)

    non_features = [
        "source_row",
        "audio_path",
        "wav_path",
        "csv_path",
        "qmax_ml_s",
    ]
    feature_columns = [
        column for column in dataset.columns if column not in non_features
    ]

    summary_rows = []
    plot_items = []

    for run_number, split_seed in enumerate(SPLIT_SEEDS, start=1):
        print(
            f"\n--- RUN {run_number}/{NUMBER_OF_RUNS} "
            f"| split seed {split_seed} ---"
        )
        result_row, plot_item = fit_one_run(
            run_number,
            split_seed,
            dataset,
            feature_columns,
            training_module,
        )
        summary_rows.append(result_row)
        plot_items.append(plot_item)

        print(f"RMSE: {result_row['rmse_ml_s']:.4f} mL/s")
        print(f"MAE:  {result_row['mae_ml_s']:.4f} mL/s")
        print(f"R²:   {result_row['r2']:.4f}")

    summary = pd.DataFrame(summary_rows)
    metric_columns = [
        "rmse_ml_s",
        "mae_ml_s",
        "r2",
        "mean_error_ml_s",
        "error_sd_ml_s",
        "maximum_absolute_error_ml_s",
    ]

    mean_row = {
        "run": "Mean",
        "split_seed": "",
        "train_n": TRAIN_SIZE,
        "validation_n": VALIDATION_SIZE,
    }
    standard_deviation_row = {
        "run": "SD",
        "split_seed": "",
        "train_n": "",
        "validation_n": "",
    }
    for column in metric_columns:
        mean_row[column] = float(summary[column].mean())
        standard_deviation_row[column] = float(summary[column].std(ddof=1))

    complete_summary = pd.concat(
        [
            summary,
            pd.DataFrame([mean_row, standard_deviation_row]),
        ],
        ignore_index=True,
    )
    complete_summary.to_excel(SUMMARY_FILE, index=False)
    save_combined_plot(plot_items)

    print("\n=== FOUR-RUN SUMMARY ===")
    print(summary.to_string(index=False))
    print("\nMean metrics:")
    print(pd.Series(mean_row)[metric_columns].to_string())
    print(f"\nSaved summary: {SUMMARY_FILE}")
    print(f"Saved comparison plot: {COMBINED_PLOT}")
    print(f"All run folders: {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
