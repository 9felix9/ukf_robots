import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

LOG_DIR = "analyze/logging"
PLOT_DIR = "analyze/plots"

os.makedirs(PLOT_DIR, exist_ok=True)

# -------------------------------------------------------
# Helper: Compute RMSE
# -------------------------------------------------------
def compute_rmse(df):
    dx = df["est_x"] - df["gt_x"]
    dy = df["est_y"] - df["gt_y"]
    return np.sqrt(np.mean(dx*dx + dy*dy))


# -------------------------------------------------------
# Process each CSV file
# -------------------------------------------------------
csv_files = sorted(glob.glob(os.path.join(LOG_DIR, "*.csv")))

results = []

for csv_path in csv_files:
    df = pd.read_csv(csv_path)
    rmse = compute_rmse(df)
    name = os.path.basename(csv_path)

    print(f"[INFO] File {name}: RMSE = {rmse:.4f}")

    parts = name.replace(".csv", "").split("_")

    task = parts[0]  # "B1", "B2", or "B3"

    if task in ["B1", "B2"]:
        # B1_procXY0.050_procTH0.050_meas0.050.csv
        proc_xy = float(parts[1].replace("procXY", ""))
        proc_th = float(parts[2].replace("procTH", ""))
        meas = float(parts[3].replace("meas", ""))
        scenario = None

    elif task == "B3":
        # B3_baseline_procXY0.050_procTH0.050_meas0.010.csv
        scenario = parts[1]
        proc_xy = float(parts[2].replace("procXY", ""))
        proc_th = float(parts[3].replace("procTH", ""))
        meas = float(parts[4].replace("meas", ""))

    results.append({
        "task": task,
        "scenario": scenario,
        "process_noise_xy": proc_xy,
        "process_noise_theta": proc_th,
        "measurement_noise": meas,
        "rmse": rmse,
        "csv": name
    })

results_df = pd.DataFrame(results)
print("\nAggregated results:\n", results_df)



# -------------------------------------------------------
# Plot B1: RMSE vs Process Noise (xy)
# -------------------------------------------------------
b1 = results_df[results_df["task"] == "B1"]
if not b1.empty:
    plt.figure()
    plt.plot(b1["process_noise_xy"], b1["rmse"], marker="o")
    plt.xlabel("Process Noise XY")
    plt.ylabel("RMSE")
    plt.title("B1: RMSE vs Process Noise (XY)")
    plt.grid(True)
    plot_path = os.path.join(PLOT_DIR, "B1_rmse_vs_process_noise_xy.png")
    plt.savefig(plot_path)
    print(f"[PLOT SAVED] {plot_path}")



# -------------------------------------------------------
# Plot B2: RMSE vs Measurement Noise
# -------------------------------------------------------
b2 = results_df[results_df["task"] == "B2"]
if not b2.empty:
    plt.figure()
    plt.plot(b2["measurement_noise"], b2["rmse"], marker="o")
    plt.xlabel("Measurement Noise")
    plt.ylabel("RMSE")
    plt.title("B2: RMSE vs Measurement Noise")
    plt.grid(True)
    plot_path = os.path.join(PLOT_DIR, "B2_rmse_vs_measurement_noise.png")
    plt.savefig(plot_path)
    print(f"[PLOT SAVED] {plot_path}")

# -------------------------------------------------------
# Plot B3: RMSE for different scenarios
# -------------------------------------------------------
b3 = results_df[results_df["task"] == "B3"]
if not b3.empty:
    plt.figure()
    scenarios = b3["scenario"]
    rmses = b3["rmse"]
    plt.bar(scenarios, rmses)
    plt.xlabel("Scenario")
    plt.ylabel("RMSE")
    plt.title("B3: RMSE across Simulation Scenarios")
    plt.grid(True, axis="y")
    plot_path = os.path.join(PLOT_DIR, "B3_scenarios_rmse.png")
    plt.savefig(plot_path)
    print(f"[PLOT SAVED] {plot_path}")
