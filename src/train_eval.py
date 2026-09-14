"""
src/train_eval.py - Task 4: Least-Squares Regression Training & Evaluation

Orchestrates the entire model training, hyperparameter exploration,
test evaluation, comparison between on-site sensors (Set A) and
public weather data (Set B), figure generation, and model weight serialization.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import os

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.regression import (
    hypothesis,
    cost,
    fit_normal,
    fit_batch_gd,
    fit_sgd,
    rmse,
)

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_and_preprocess():
    """
    Load hourly plant data and Open-Meteo weather data, merge on datetime,
    compute cyclical time features, and split strictly by date into Train and Test.

    Returns:
        train_df, test_df, full_df
    """
    hourly_path = DATA_DIR / "plant1_hourly.csv"
    om_path = DATA_DIR / "plant1_openmeteo.csv"

    if not hourly_path.exists():
        raise FileNotFoundError(f"Missing {hourly_path}. Run src/prepare.py first.")
    if not om_path.exists():
        raise FileNotFoundError(f"Missing {om_path}. Run src/fetch_weather.py first.")

    h_df = pd.read_csv(hourly_path)
    om_df = pd.read_csv(om_path)

    h_df["datetime"] = pd.to_datetime(h_df["datetime"])
    om_df["datetime"] = pd.to_datetime(om_df["datetime"])

    df = pd.merge(h_df, om_df, on="datetime", how="inner").sort_values("datetime").reset_index(drop=True)

    # Add cyclical time features: sin(2pi * h / 24) and cos(2pi * h / 24)
    hours = df["datetime"].dt.hour
    df["sin_hour"] = np.sin(2.0 * np.pi * hours / 24.0)
    df["cos_hour"] = np.cos(2.0 * np.pi * hours / 24.0)

    # Train / Test split by date (strictly chronological, NO shuffling)
    # Train: 15 May - 10 June (27 days = 648 rows)
    # Test:  11 June - 17 June (7 days = 168 rows)
    train_mask = df["datetime"] <= "2020-06-10 23:59:59"
    test_mask = df["datetime"] >= "2020-06-11 00:00:00"

    train_df = df[train_mask].copy().reset_index(drop=True)
    test_df = df[test_mask].copy().reset_index(drop=True)

    print(f"Dataset split summary:")
    print(f"  Training period: {train_df['datetime'].min()} to {train_df['datetime'].max()} ({len(train_df)} rows)")
    print(f"  Testing period:  {test_df['datetime'].min()} to {test_df['datetime'].max()} ({len(test_df)} rows)")
    print(f"  Peak hourly AC power in training set: {train_df['ac_power'].max():.2f} kW")

    return train_df, test_df, df


def prepare_design_matrices(train_df, test_df, feature_cols):
    """
    Scale features using training statistics ONLY, and append intercept column x_0 = 1.

    Args:
        train_df: Training DataFrame
        test_df: Testing DataFrame
        feature_cols: List of 5 feature column names

    Returns:
        X_train, y_train, X_test, y_test, scaler_stats
    """
    # 4.1 Feature scaling: compute mean and standard deviation on training set only
    mu = train_df[feature_cols].mean().values
    # Use population std (ddof=0) or sample std (ddof=1)
    sigma = train_df[feature_cols].std(ddof=0).values

    # Prevent division by zero
    sigma = np.where(sigma == 0, 1.0, sigma)

    # Scale training features
    X_tr_scaled = (train_df[feature_cols].values - mu) / sigma
    # Scale test features with training statistics (no data leakage)
    X_te_scaled = (test_df[feature_cols].values - mu) / sigma

    # 4.2 Build design matrix X with intercept column x_0 = 1
    X_train = np.c_[np.ones(len(X_tr_scaled)), X_tr_scaled]
    X_test = np.c_[np.ones(len(X_te_scaled)), X_te_scaled]

    y_train = train_df["ac_power"].values.astype(float)
    y_test = test_df["ac_power"].values.astype(float)

    scaler_stats = {
        "features": feature_cols,
        "mean": mu.tolist(),
        "std": sigma.tolist(),
    }

    return X_train, y_train, X_test, y_test, scaler_stats


def tune_and_plot_learning_rates(X_train_A, y_train, save_dir=FIG_DIR):
    """
    Task 4.7:
    - Batch GD: run 500 iterations with alpha in {1e-5, 1e-4, 1e-3} on Set A and plot J(theta) vs iter.
    - SGD: run 50 epochs with alpha in {1e-4, 1e-3, 1e-2} on Set A and plot J(theta) vs epoch.
    """
    print("\n" + "=" * 60)
    print("TASK 4.7: LEARNING RATE TUNING & LOSS CURVES")
    print("=" * 60)

    # Figure 3: Batch Gradient Descent Loss Curves
    bgd_alphas = [1e-5, 1e-4, 1e-3]
    bgd_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    bgd_histories = {}

    plt.figure(figsize=(8, 5), dpi=300)
    for alpha, color in zip(bgd_alphas, bgd_colors):
        theta, hist = fit_batch_gd(X_train_A, y_train, alpha=alpha, n_iters=500)
        bgd_histories[alpha] = hist
        plt.plot(range(1, 501), hist, label=f"α = {alpha:.0e} (Final J: {hist[-1]:,.1f})", color=color, linewidth=2)
        print(f"  Batch GD α={alpha:.0e}: Initial J = {hist[0]:,.1f} | Final J (iter 500) = {hist[-1]:,.1f}")

    plt.title("Task 4.7 / Figure 3: Batch Gradient Descent Cost J(θ) vs. Iteration\n(500 iterations on Feature Set A)", fontsize=11, fontweight="bold")
    plt.xlabel("Iteration", fontsize=10)
    plt.ylabel("Cost J(θ)", fontsize=10)
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    fig3_path = save_dir / "task4_7_batch_gd_loss.png"
    plt.savefig(str(fig3_path))
    plt.close()
    print(f"[OK] Saved Figure 3 to: {fig3_path}")

    # Figure 4: Stochastic Gradient Descent Loss Curves
    sgd_alphas = [1e-4, 1e-3, 1e-2]
    sgd_colors = ["#1f77b4", "#ff7f0e", "#d62728"]
    sgd_histories = {}

    plt.figure(figsize=(8, 5), dpi=300)
    for alpha, color in zip(sgd_alphas, sgd_colors):
        theta, hist = fit_sgd(X_train_A, y_train, alpha=alpha, n_epochs=50)
        sgd_histories[alpha] = hist
        plt.plot(range(1, 51), hist, label=f"α = {alpha:.0e} (Final J: {hist[-1]:,.1f})", color=color, linewidth=2)
        print(f"  SGD α={alpha:.0e}: Initial J = {hist[0]:,.1f} | Final J (epoch 50) = {hist[-1]:,.1f}")

    plt.title("Task 4.7 / Figure 4: Stochastic Gradient Descent Cost J(θ) vs. Epoch\n(50 epochs on Feature Set A)", fontsize=11, fontweight="bold")
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Cost J(θ)", fontsize=10)
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    fig4_path = save_dir / "task4_7_sgd_loss.png"
    plt.savefig(str(fig4_path))
    plt.close()
    print(f"[OK] Saved Figure 4 to: {fig4_path}")

    return bgd_histories, sgd_histories


def run_training_and_evaluation():
    """
    Execute full Task 4 workflow across Set A and Set B for all three solvers:
    Normal Equation, Batch Gradient Descent, Stochastic Gradient Descent.
    """
    train_df, test_df, full_df = load_and_preprocess()

    # Define Feature Sets
    features_A = ["irradiation", "module_temp", "ambient_temp", "sin_hour", "cos_hour"]
    features_B = ["sw_radiation", "temp_2m", "cloud_cover", "sin_hour", "cos_hour"]

    # Prepare Design Matrices
    X_tr_A, y_tr, X_te_A, y_te, scaler_A = prepare_design_matrices(train_df, test_df, features_A)
    X_tr_B, _, X_te_B, _, scaler_B = prepare_design_matrices(train_df, test_df, features_B)

    # Generate Tuning Curves (Figures 3 & 4)
    tune_and_plot_learning_rates(X_tr_A, y_tr)

    print("\n" + "=" * 60)
    print("TASK 4.8: MODEL FITTING ACROSS ALL SOLVERS (SET A & SET B)")
    print("=" * 60)

    # Daytime mask on test set: sensor irradiation > 0
    test_daytime_mask = test_df["irradiation"].values > 0
    print(f"Test set: {len(test_df)} total hours, {np.sum(test_daytime_mask)} daytime hours (irradiation > 0)")

    # 1. SET A SOLVERS
    # 1.1 Normal Equation
    theta_norm_A = fit_normal(X_tr_A, y_tr)

    # 1.2 Batch GD (with converged alpha and sufficient iterations to match normal equation to < 0.01)
    # Using alpha=0.2, n_iters=15,000 for perfect numerical convergence
    alpha_bgd_A = 0.2
    iters_bgd_A = 15000
    theta_bgd_A, _ = fit_batch_gd(X_tr_A, y_tr, alpha=alpha_bgd_A, n_iters=iters_bgd_A)

    # 1.3 SGD (alpha=0.01, 100 epochs)
    alpha_sgd_A = 0.01
    epochs_sgd_A = 100
    theta_sgd_A, _ = fit_sgd(X_tr_A, y_tr, alpha=alpha_sgd_A, n_epochs=epochs_sgd_A)

    # 2. SET B SOLVERS
    # 2.1 Normal Equation
    theta_norm_B = fit_normal(X_tr_B, y_tr)

    # 2.2 Batch GD
    alpha_bgd_B = 0.2
    iters_bgd_B = 15000
    theta_bgd_B, _ = fit_batch_gd(X_tr_B, y_tr, alpha=alpha_bgd_B, n_iters=iters_bgd_B)

    # 2.3 SGD
    alpha_sgd_B = 0.01
    epochs_sgd_B = 100
    theta_sgd_B, _ = fit_sgd(X_tr_B, y_tr, alpha=alpha_sgd_B, n_epochs=epochs_sgd_B)

    # Verification: Confirm that batch GD reaches the same theta as the normal equation
    max_diff_A = float(np.max(np.abs(theta_bgd_A - theta_norm_A)))
    max_diff_B = float(np.max(np.abs(theta_bgd_B - theta_norm_B)))
    print(f"\nConvergence check (Normal Eq vs. Batch GD):")
    print(f"  Set A max absolute difference: {max_diff_A:.6f} (< 0.01 threshold: {max_diff_A < 0.01})")
    print(f"  Set B max absolute difference: {max_diff_B:.6f} (< 0.01 threshold: {max_diff_B < 0.01})")

    # Evaluate RMSE for all 6 models (All hours vs Daytime hours)
    models = [
        ("Normal equation", "Set A", theta_norm_A, X_te_A, f"Normal Eq"),
        ("Batch GD", "Set A", theta_bgd_A, X_te_A, f"α = {alpha_bgd_A}, {iters_bgd_A} iters"),
        ("Stochastic GD", "Set A", theta_sgd_A, X_te_A, f"α = {alpha_sgd_A}, {epochs_sgd_A} epochs"),
        ("Normal equation", "Set B", theta_norm_B, X_te_B, f"Normal Eq"),
        ("Batch GD", "Set B", theta_bgd_B, X_te_B, f"α = {alpha_bgd_B}, {iters_bgd_B} iters"),
        ("Stochastic GD", "Set B", theta_sgd_B, X_te_B, f"α = {alpha_sgd_B}, {epochs_sgd_B} epochs"),
    ]

    rmse_results = []
    print("\n" + "=" * 60)
    print("TASK 4.10: TEST RMSE EVALUATION (ALL HOURS & DAYTIME ONLY)")
    print("=" * 60)
    print(f"{'Solver':<18} {'Features':<8} {'Config':<25} {'All Hours RMSE (kW)':<22} {'Daytime RMSE (kW)':<20}")
    print("-" * 95)

    for solver, feat_name, th, X_test_mat, config in models:
        rmse_all = rmse(X_test_mat, y_te, th, clip_negative=True)
        # Daytime test subset
        rmse_day = rmse(X_test_mat[test_daytime_mask], y_te[test_daytime_mask], th, clip_negative=True)
        rmse_results.append(
            {
                "solver": solver,
                "features": feat_name,
                "config": config,
                "rmse_all": rmse_all,
                "rmse_day": rmse_day,
            }
        )
        print(f"{solver:<18} {feat_name:<8} {config:<25} {rmse_all:<22.2f} {rmse_day:<20.2f}")

    # Peak power comparison
    peak_power = float(train_df["ac_power"].max())
    daytime_rmse_A = rmse_results[0]["rmse_day"]
    daytime_rmse_B = rmse_results[3]["rmse_day"]
    diff_day_rmse = daytime_rmse_B - daytime_rmse_A
    pct_peak_A = (daytime_rmse_A / peak_power) * 100
    pct_peak_B = (daytime_rmse_B / peak_power) * 100
    pct_diff_peak = (diff_day_rmse / peak_power) * 100

    print(f"\nPhysical Performance Assessment:")
    print(f"  Plant Peak Hourly AC Power: {peak_power:.2f} kW")
    print(f"  Set A (On-Site Sensors) Daytime RMSE: {daytime_rmse_A:.2f} kW ({pct_peak_A:.2f}% of peak)")
    print(f"  Set B (Public Weather)  Daytime RMSE: {daytime_rmse_B:.2f} kW ({pct_peak_B:.2f}% of peak)")
    print(f"  Accuracy loss: +{diff_day_rmse:.2f} kW (+{pct_diff_peak:.2f}% of plant peak capacity)")

    # Figure 5: Actual vs Predicted AC Power on Test Week (Normal Equation: Set A vs Set B)
    pred_A = np.maximum(0.0, hypothesis(X_te_A, theta_norm_A))
    pred_B = np.maximum(0.0, hypothesis(X_te_B, theta_norm_B))

    plt.figure(figsize=(13, 5.5), dpi=300)
    plt.plot(test_df["datetime"], y_te, label="Actual AC Power (kW)", color="black", linewidth=2.2, zorder=3)
    plt.plot(test_df["datetime"], pred_A, label=f"Predicted Set A (On-Site Sensors) - Daytime RMSE {daytime_rmse_A:.1f} kW", color="#1f77b4", linestyle="--", linewidth=1.8)
    plt.plot(test_df["datetime"], pred_B, label=f"Predicted Set B (Open-Meteo Weather) - Daytime RMSE {daytime_rmse_B:.1f} kW", color="#d62728", linestyle=":", linewidth=1.8)

    plt.title("Task 4 / Figure 5: Test Week Actual vs. Predicted AC Power (Normal Equation)\nJune 11, 2020 – June 17, 2020 (Asia/Kolkata)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Datetime", fontsize=10)
    plt.ylabel("AC Power Output (kW)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, loc="upper right")
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    fig5_path = FIG_DIR / "task4_predicted_vs_actual.png"
    plt.savefig(str(fig5_path))
    plt.close()
    print(f"[OK] Saved Figure 5 to: {fig5_path}")

    # Figure 6: Task 5.5 Residuals vs Hour of Day for Set A Normal Equation
    residuals_A = y_te - pred_A
    test_hours = test_df["datetime"].dt.hour.values

    plt.figure(figsize=(9, 5), dpi=300)
    plt.scatter(test_hours, residuals_A, alpha=0.6, color="#9467bd", edgecolors="none", s=35)
    # Add mean residual per hour
    res_df = pd.DataFrame({"hour": test_hours, "residual": residuals_A})
    mean_res_per_hour = res_df.groupby("hour")["residual"].mean()
    plt.plot(mean_res_per_hour.index, mean_res_per_hour.values, color="red", linewidth=2, marker="o", label="Mean Residual by Hour")
    plt.axhline(0, color="black", linestyle="--", linewidth=1.2)
    plt.title("Task 5.5 / Figure 6: Test Set Residuals (Actual - Predicted) vs. Hour of Day\n(Set A Normal Equation Model)", fontsize=11, fontweight="bold")
    plt.xlabel("Hour of Day (0–23)", fontsize=10)
    plt.ylabel("Residual (kW)", fontsize=10)
    plt.xticks(range(0, 24, 2))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, loc="upper left")
    plt.tight_layout()
    fig6_path = FIG_DIR / "task5_5_residuals_vs_hour.png"
    plt.savefig(str(fig6_path))
    plt.close()
    print(f"[OK] Saved Figure 6 to: {fig6_path}")

    # Serialize weights and scaler statistics for Task 6 Front End
    saved_weights = {
        "features_A": features_A,
        "scaler_A": scaler_A,
        "theta_norm_A": theta_norm_A.tolist(),
        "theta_bgd_A": theta_bgd_A.tolist(),
        "theta_sgd_A": theta_sgd_A.tolist(),
        "features_B": features_B,
        "scaler_B": scaler_B,
        "theta_norm_B": theta_norm_B.tolist(),
        "theta_bgd_B": theta_bgd_B.tolist(),
        "theta_sgd_B": theta_sgd_B.tolist(),
        "peak_hourly_power": peak_power,
    }

    weights_json_path = RESULTS_DIR / "saved_weights.json"
    with open(weights_json_path, "w", encoding="utf-8") as f:
        json.dump(saved_weights, f, indent=2)
    print(f"[OK] Saved model weights JSON to: {weights_json_path}")

    # Also save as .npz
    np.savez(
        RESULTS_DIR / "saved_weights.npz",
        theta_norm_A=theta_norm_A,
        theta_bgd_A=theta_bgd_A,
        theta_sgd_A=theta_sgd_A,
        theta_norm_B=theta_norm_B,
        theta_bgd_B=theta_bgd_B,
        theta_sgd_B=theta_sgd_B,
        mu_A=scaler_A["mean"],
        sigma_A=scaler_A["std"],
        mu_B=scaler_B["mean"],
        sigma_B=scaler_B["std"],
    )
    print(f"[OK] Saved model weights NPZ to: {RESULTS_DIR / 'saved_weights.npz'}")

    return {
        "rmse_results": rmse_results,
        "theta_A": {
            "normal": theta_norm_A,
            "bgd": theta_bgd_A,
            "sgd": theta_sgd_A,
            "max_diff": max_diff_A,
        },
        "theta_B": {
            "normal": theta_norm_B,
            "bgd": theta_bgd_B,
            "sgd": theta_sgd_B,
            "max_diff": max_diff_B,
        },
        "peak_power": peak_power,
        "scaler_A": scaler_A,
        "scaler_B": scaler_B,
    }


if __name__ == "__main__":
    results = run_training_and_evaluation()
