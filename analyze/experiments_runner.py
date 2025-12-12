#!/usr/bin/env python3
import os
import time
import subprocess
import pandas as pd

# ============================================================
# EXPERIMENT PARAMETER TABLES
# ============================================================

exp_setup_B1 = pd.DataFrame([
    {"process_noise_xy": 0.001, "process_noise_theta": 0.001, "measurement_noise_xy": 0.05},
    {"process_noise_xy": 0.01,  "process_noise_theta": 0.01,  "measurement_noise_xy": 0.05},
    {"process_noise_xy": 0.05,  "process_noise_theta": 0.05,  "measurement_noise_xy": 0.05},
    {"process_noise_xy": 0.1,   "process_noise_theta": 0.1,   "measurement_noise_xy": 0.05},
    {"process_noise_xy": 0.3,   "process_noise_theta": 0.3,   "measurement_noise_xy": 0.05},
])

exp_setup_B2 = pd.DataFrame([
    {"process_noise_xy": 0.05, "process_noise_theta": 0.05, "measurement_noise_xy": 0.001},
    {"process_noise_xy": 0.05, "process_noise_theta": 0.05, "measurement_noise_xy": 0.01},
    {"process_noise_xy": 0.05, "process_noise_theta": 0.05, "measurement_noise_xy": 0.05},
    {"process_noise_xy": 0.05, "process_noise_theta": 0.05, "measurement_noise_xy": 0.1},
    {"process_noise_xy": 0.05, "process_noise_theta": 0.05, "measurement_noise_xy": 0.3},
])

exp_setup_B3 = pd.DataFrame([
    {"scenario": "baseline",    "process_noise_xy": 0.05, "process_noise_theta": 0.05, "measurement_noise_xy": 0.01},
    {"scenario": "largeCurve",  "process_noise_xy": 0.05, "process_noise_theta": 0.05, "measurement_noise_xy": 0.01},
    {"scenario": "highNoise",   "process_noise_xy": 0.05, "process_noise_theta": 0.05, "measurement_noise_xy": 0.1},
    {"scenario": "smallRadius", "process_noise_xy": 0.05, "process_noise_theta": 0.05, "measurement_noise_xy": 0.01},
    {"scenario": "largeRadius", "process_noise_xy": 0.05, "process_noise_theta": 0.05, "measurement_noise_xy": 0.01},
])

# ============================================================
# PATHS
# ============================================================

WORKSPACE = os.path.expanduser("/home/felix/Schreibtisch/projects/ros2_homework_1")
SETUP_CMD = f"source {WORKSPACE}/install/setup.zsh"
LOG_DIR = "analyze/logging"
os.makedirs(LOG_DIR, exist_ok=True)

SIM_DURATION_SEC = 35.0  # 30s logging + buffer

# ============================================================
# RUN SINGLE EXPERIMENT
# ============================================================

def build_csv_name(task: str, params) -> str:
    if task in ["B1", "B2"]:
        return (
            f"{task}_"
            f"procXY{params['process_noise_xy']:.3f}_"
            f"procTH{params['process_noise_theta']:.3f}_"
            f"meas{params['measurement_noise_xy']:.3f}.csv"
        )
    return (
        f"{task}_{params['scenario']}_"
        f"procXY{params['process_noise_xy']:.3f}_"
        f"procTH{params['process_noise_theta']:.3f}_"
        f"meas{params['measurement_noise_xy']:.3f}.csv"
    )

def run_simulation(task: str, params, index: int):
    csv_name = build_csv_name(task, params)
    csv_path = os.path.join(LOG_DIR, csv_name)

    print("\n================================================")
    print(f"Running {task} experiment #{index}")
    print(f"CSV output: {csv_path}")
    print("================================================")

    # Start fake_robot
    fake_cmd = f"{SETUP_CMD} && ros2 launch fake_robot fake_robot.launch.py"
    fake_proc = subprocess.Popen(fake_cmd, shell=True, executable="/bin/zsh")

    time.sleep(1.5)

    # Start kalman_positioning (IMPORTANT: pass env vars here)
    ukf_cmd = f"{SETUP_CMD} && ros2 launch kalman_positioning positioning.launch.py"

    env = os.environ.copy()
    env["PROC_NOISE_XY"] = str(params["process_noise_xy"])
    env["PROC_NOISE_TH"] = str(params["process_noise_theta"])
    env["MEAS_NOISE_XY"] = str(params["measurement_noise_xy"])
    env["LOG_CSV"] = csv_path
    env["TASK_TYPE"] = task
    if task == "B3":
        env["SCENARIO"] = params["scenario"]

    ukf_proc = subprocess.Popen(ukf_cmd, shell=True, executable="/bin/zsh", env=env)

    time.sleep(SIM_DURATION_SEC)

    # stop both
    ukf_proc.terminate()
    fake_proc.terminate()

    print(f"Finished {task} run #{index}")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("===== STARTING ALL EXPERIMENTS =====")

    for i, row in exp_setup_B1.iterrows():
        run_simulation("B1", row, i)

    for i, row in exp_setup_B2.iterrows():
        run_simulation("B2", row, i)

    for i, row in exp_setup_B3.iterrows():
        run_simulation("B3", row, i)

    print("===== ALL EXPERIMENTS COMPLETED =====")
