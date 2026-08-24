from datetime import datetime
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut


# ============================================================
# SETTINGS
# ============================================================

PROJECT_DIR = Path(r"C:\school\project\everything everything")

INPUT_FILE = (
    PROJECT_DIR
    / "03_pump_calibration_outputs"
    / "step_summary.xlsx"
)

OUTPUT_DIR = PROJECT_DIR / "04_equation_fitting_outputs"

X_COLUMN = "audio_mean_energy"
Y_COLUMN = "flow_mean_ml_s"

# Audio energy is stored in log10 form in step_summary.xlsx.
ENERGY_IS_LOG10 = True

FIGURE_DPI = 220


# ============================================================
# MODELS
# ============================================================

def linear_model(x, a, b):
    return a * x + b


def quadratic_model(x, a, b, c):
    return a * x**2 + b * x + c


def cubic_model(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d


def logarithmic_model(x, a, b):
    return a * np.log(x) + b


def exponential_model(x, a, b):
    return a * np.exp(b * x)


def power_model(x, a, b):
    return a * x**b


MODELS = {
    "Linear": {
        "function": linear_model,
        "initial_guess": None,
    },
    "Quadratic": {
        "function": quadratic_model,
        "initial_guess": None,
    },
    "Cubic": {
        "function": cubic_model,
        "initial_guess": None,
    },
    "Logarithmic": {
        "function": logarithmic_model,
        "initial_guess": [1.0, 1.0],
    },
    "Exponential": {
        "function": exponential_model,
        "initial_guess": [1.0, 1.0],
    },
    "Power": {
        "function": power_model,
        "initial_guess": [20.0, 0.5],
    },
}


# ============================================================
# SAFE OUTPUT FUNCTIONS
# ============================================================

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_excel_safely(dataframe, output_path):
    """
    Save an Excel file.

    If the original file is open in Excel, save a new timestamped
    version instead of stopping the program.
    """

    try:
        dataframe.to_excel(output_path, index=False)
        print(f"Saved: {output_path.name}")
        return output_path

    except PermissionError:
        alternative_path = output_path.with_name(
            f"{output_path.stem}_updated_{timestamp()}"
            f"{output_path.suffix}"
        )

        dataframe.to_excel(alternative_path, index=False)

        print(
            f"WARNING: {output_path.name} is open or locked.\n"
            f"Saved instead as: {alternative_path.name}"
        )

        return alternative_path


def save_csv_safely(dataframe, output_path):
    """Save a CSV, using a timestamped filename if necessary."""

    try:
        dataframe.to_csv(output_path, index=False)
        print(f"Saved: {output_path.name}")
        return output_path

    except PermissionError:
        alternative_path = output_path.with_name(
            f"{output_path.stem}_updated_{timestamp()}"
            f"{output_path.suffix}"
        )

        dataframe.to_csv(alternative_path, index=False)

        print(
            f"WARNING: {output_path.name} is open or locked.\n"
            f"Saved instead as: {alternative_path.name}"
        )

        return alternative_path


# ============================================================
# FITTING FUNCTIONS
# ============================================================

def fit_model(model_name, x, y):
    """Fit one candidate model."""

    if model_name == "Linear":
        parameters = np.polyfit(x, y, 1)

    elif model_name == "Quadratic":
        parameters = np.polyfit(x, y, 2)

    elif model_name == "Cubic":
        parameters = np.polyfit(x, y, 3)

    else:
        model_function = MODELS[model_name]["function"]
        initial_guess = MODELS[model_name]["initial_guess"]

        parameters, _ = curve_fit(
            model_function,
            x,
            y,
            p0=initial_guess,
            maxfev=100000,
        )

    return np.asarray(parameters, dtype=float)


def predict_model(model_name, parameters, x):
    """Calculate model predictions."""

    model_function = MODELS[model_name]["function"]
    return model_function(x, *parameters)


def calculate_loocv_rmse(model_name, x, y):
    """Calculate leave-one-out cross-validation RMSE."""

    loo = LeaveOneOut()

    true_values = []
    predicted_values = []

    for train_indices, test_indices in loo.split(x):
        x_train = x[train_indices]
        y_train = y[train_indices]

        x_test = x[test_indices]
        y_test = y[test_indices]

        try:
            parameters = fit_model(
                model_name,
                x_train,
                y_train,
            )

            prediction = predict_model(
                model_name,
                parameters,
                x_test,
            )

            predicted_value = float(prediction[0])

            if not np.isfinite(predicted_value):
                return np.nan

            true_values.append(float(y_test[0]))
            predicted_values.append(predicted_value)

        except Exception:
            return np.nan

    return float(
        np.sqrt(
            mean_squared_error(
                true_values,
                predicted_values,
            )
        )
    )


# ============================================================
# EQUATION TEXT
# ============================================================

def format_equation(model_name, parameters):
    """Create readable equations using E for energy and Q for flow."""

    if model_name == "Linear":
        a, b = parameters
        return f"Q = {a:.4f}E {b:+.4f}"

    if model_name == "Quadratic":
        a, b, c = parameters
        return (
            f"Q = {a:.4f}E² "
            f"{b:+.4f}E "
            f"{c:+.4f}"
        )

    if model_name == "Cubic":
        a, b, c, d = parameters
        return (
            f"Q = {a:.4f}E³ "
            f"{b:+.4f}E² "
            f"{c:+.4f}E "
            f"{d:+.4f}"
        )

    if model_name == "Logarithmic":
        a, b = parameters
        return f"Q = {a:.4f}ln(E) {b:+.4f}"

    if model_name == "Exponential":
        a, b = parameters
        return f"Q = {a:.4f}exp({b:.4f}E)"

    if model_name == "Power":
        a, b = parameters
        return f"Q = {a:.3f}E^{b:.6f}"

    return ""


def graph_title(model_name):
    if model_name == "Power":
        return "Power-Law Calibration of Acoustic Energy to Flow Rate"

    return f"{model_name} Calibration of Acoustic Energy to Flow Rate"


# ============================================================
# GRAPH FUNCTIONS
# ============================================================

def save_individual_plot(
    model_name,
    x,
    y,
    x_curve,
    y_curve,
    equation,
    r2,
):
    """Save the graph for one fitted model."""

    plt.figure(figsize=(8.5, 6.2))

    plt.scatter(
        x,
        y,
        s=55,
        color="#4C78A8",
        label="Calibration points",
        zorder=3,
    )

    plt.plot(
        x_curve,
        y_curve,
        linewidth=2.5,
        color="#2F5F9F",
        label=f"{model_name} fit",
    )

    plt.xlabel(
        "Mean acoustic energy (mel power, a.u.)",
        fontsize=11,
    )

    plt.ylabel(
        "Mean reference flow rate (mL/s)",
        fontsize=11,
    )

    # RMSE is not displayed on the graph.
    plt.title(
        f"{graph_title(model_name)}\n"
        f"{equation}   |   $R^2$ = {r2:.3f}",
        fontsize=13,
    )

    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output_file = (
        OUTPUT_DIR
        / f"{model_name.lower()}_fit.png"
    )

    try:
        plt.savefig(
            output_file,
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )

        print(f"Saved: {output_file.name}")

    except PermissionError:
        alternative_file = OUTPUT_DIR / (
            f"{model_name.lower()}_fit_updated_"
            f"{timestamp()}.png"
        )

        plt.savefig(
            alternative_file,
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )

        print(
            f"WARNING: {output_file.name} is locked.\n"
            f"Saved instead as: {alternative_file.name}"
        )

    plt.close()


def save_comparison_plot(x, y, fitted_models):
    """Save a graph containing all fitted equations."""

    plt.figure(figsize=(9.5, 6.7))

    plt.scatter(
        x,
        y,
        s=60,
        color="black",
        label="Calibration points",
        zorder=5,
    )

    for result in fitted_models:
        plt.plot(
            result["x_curve"],
            result["y_curve"],
            linewidth=2,
            label=result["model"],
        )

    plt.xlabel(
        "Mean acoustic energy (mel power, a.u.)",
        fontsize=11,
    )

    plt.ylabel(
        "Mean reference flow rate (mL/s)",
        fontsize=11,
    )

    plt.title(
        "Comparison of Acoustic-Energy Calibration Models",
        fontsize=13,
    )

    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output_file = OUTPUT_DIR / "all_fits_comparison.png"

    try:
        plt.savefig(
            output_file,
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )

        print(f"Saved: {output_file.name}")

    except PermissionError:
        alternative_file = OUTPUT_DIR / (
            f"all_fits_comparison_updated_{timestamp()}.png"
        )

        plt.savefig(
            alternative_file,
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )

        print(
            f"WARNING: {output_file.name} is locked.\n"
            f"Saved instead as: {alternative_file.name}"
        )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    warnings.filterwarnings("ignore")

    print("\n=== EQUATION FITTING ===")
    print(f"Reading: {INPUT_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_excel(INPUT_FILE)

    if X_COLUMN not in df.columns:
        raise KeyError(
            f"Column '{X_COLUMN}' was not found.\n"
            f"Available columns:\n{df.columns.tolist()}"
        )

    if Y_COLUMN not in df.columns:
        raise KeyError(
            f"Column '{Y_COLUMN}' was not found.\n"
            f"Available columns:\n{df.columns.tolist()}"
        )

    data = df[
        [X_COLUMN, Y_COLUMN]
    ].copy()

    data[X_COLUMN] = pd.to_numeric(
        data[X_COLUMN],
        errors="coerce",
    )

    data[Y_COLUMN] = pd.to_numeric(
        data[Y_COLUMN],
        errors="coerce",
    )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if ENERGY_IS_LOG10:
        data[X_COLUMN] = 10.0 ** data[X_COLUMN]

    data = data[
        (data[X_COLUMN] > 0)
        & (data[Y_COLUMN] > 0)
    ].copy()

    data = data.sort_values(
        X_COLUMN
    ).reset_index(drop=True)

    if len(data) < 5:
        raise ValueError(
            "At least five valid calibration points are required."
        )

    x = data[X_COLUMN].to_numpy(dtype=float)
    y = data[Y_COLUMN].to_numpy(dtype=float)

    print(f"Valid calibration points: {len(data)}")

    points_used = pd.DataFrame({
        "mean_acoustic_energy_au": x,
        "mean_reference_flow_ml_s": y,
    })

    save_excel_safely(
        points_used,
        OUTPUT_DIR / "points_used.xlsx",
    )

    save_csv_safely(
        points_used,
        OUTPUT_DIR / "points_used.csv",
    )

    x_curve = np.linspace(
        x.min(),
        x.max(),
        500,
    )

    summary_rows = []
    fitted_models = []

    for model_name in MODELS:
        print(f"Fitting: {model_name}")

        try:
            parameters = fit_model(
                model_name,
                x,
                y,
            )

            fitted_y = predict_model(
                model_name,
                parameters,
                x,
            )

            y_curve = predict_model(
                model_name,
                parameters,
                x_curve,
            )

            full_fit_rmse = float(
                np.sqrt(
                    mean_squared_error(
                        y,
                        fitted_y,
                    )
                )
            )

            r2 = float(
                r2_score(
                    y,
                    fitted_y,
                )
            )

            loocv_rmse = calculate_loocv_rmse(
                model_name,
                x,
                y,
            )

            equation = format_equation(
                model_name,
                parameters,
            )

            summary_rows.append({
                "model": model_name,
                "loocv_rmse": loocv_rmse,
                "full_fit_rmse": full_fit_rmse,
                "r2": r2,
                "equation": equation,
            })

            fitted_models.append({
                "model": model_name,
                "x_curve": x_curve,
                "y_curve": y_curve,
            })

            save_individual_plot(
                model_name=model_name,
                x=x,
                y=y,
                x_curve=x_curve,
                y_curve=y_curve,
                equation=equation,
                r2=r2,
            )

        except Exception as error:
            print(f"  ERROR: {error}")

            summary_rows.append({
                "model": model_name,
                "loocv_rmse": np.nan,
                "full_fit_rmse": np.nan,
                "r2": np.nan,
                "equation": f"ERROR: {error}",
            })

    summary = pd.DataFrame(summary_rows)

    summary = summary.sort_values(
        "loocv_rmse",
        na_position="last",
    ).reset_index(drop=True)

    summary.insert(
        0,
        "rank",
        np.arange(1, len(summary) + 1),
    )

    save_excel_safely(
        summary,
        OUTPUT_DIR / "fit_summary.xlsx",
    )

    save_csv_safely(
        summary,
        OUTPUT_DIR / "fit_summary.csv",
    )

    save_comparison_plot(
        x,
        y,
        fitted_models,
    )

    print("\n=== MODEL RANKING ===")
    print(summary.to_string(index=False))

    print(f"\nSaved outputs to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()