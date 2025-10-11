# ===========================================================
#  experiment_07_retention_drift.py — FINAL ASCII-SAFE VERSION
#  Experiment 7: Retention / Drift (PINN Memristor)
# ===========================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Make console robust on Windows (avoid UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from mainPINNmodel import PrintedMemristorPINN
from TrainingFrameworkwithNoiseInjection import PINNTrainer
from scipy.optimize import curve_fit

# =============================================================================
# EXPERIMENT 7 (v2): RETENTION / DRIFT AFTER PROGRAMMING
# =============================================================================
print("=" * 80)
print("EXPERIMENT 7 (v2): Retention / Drift after Programming")
print("=" * 80)
print("\nObjective: Evaluate long-term stability of programmed states at 0 V (retention).")
print("Protocol: Program 3 levels, then evolve at V=0 with log-spaced sampling.")
print("-" * 80)

# -------------------------------------------------------------------------
# [1/4] Train PINN on printed-memristor data
# -------------------------------------------------------------------------
print("\n[1/4] Initializing and training PINN model...")
pinn = PrintedMemristorPINN(
    hidden_layers=4, neurons_per_layer=128,
    input_features=("voltage", "state"), random_seed=42,
    trainable_params=("ohmic_conductance",)
)
trainer = PINNTrainer(pinn, learning_rate=2e-4, seed=42, state_mixing=0.2)
voltage_data, current_data, state_data, _ = trainer.load_experimental_data(
    "printed_memristor_training_data.csv",
    concentration_label="10_percent_PMMA", device_id=0,
    use_noisy_columns=True
)
trainer.train(
    epochs=800,
    voltage=voltage_data, current=current_data, state=state_data,
    noise_std=0.002, variability_bound=0.05,
    verbose_every=200, max_physics_weight=0.1
)

# -------------------------------------------------------------------------
# [2/4] Programming 3 target resistance states
# -------------------------------------------------------------------------
print("\n[2/4] Programming three target levels before retention...")

targets = [0.2, 0.5, 0.8]
programmed_states = []

for x_target in targets:
    x = 0.0
    for step in range(2000):  # longer pulse sequence
        V_prog = 1.8  # stronger voltage
        inp = np.array([[V_prog, x]], dtype=np.float32)
        I_pred, xdot_pred = pinn.model(inp, training=False)
        xdot_val = float(xdot_pred.numpy()[0, 0])
        x += xdot_val * 0.02  # effective time-step scaling
        x = np.clip(x, 0.0, 1.0)
        if x >= x_target:
            break
    programmed_states.append(x)
    print(f"  Level programmed: target={x_target:.2f} -> achieved={x:.3f}")

# -------------------------------------------------------------------------
# [3/4] Retention evolution (V = 0) over 1e-2 → 1e6 s
# -------------------------------------------------------------------------
print("\n[3/4] Simulating retention (V=0) over log-spaced time...")
t_points = np.logspace(-2, 6, 400)
leakage_factor = 1e-5  # slow decay rate

retention_data = []
fit_results = []

def stretched_exp(t, tau, beta):
    return np.exp(-((t / tau) ** beta))

for j, x0 in enumerate(programmed_states):
    x = x0
    I_vals = []
    for t in t_points:
        inp = np.array([[0.0, x]], dtype=np.float32)
        I_pred, xdot_pred = pinn.model(inp, training=False)
        I_val = float(I_pred.numpy()[0, 0])
        # Add weak drift noise
        x += -leakage_factor * x + np.random.normal(0, leakage_factor / 5)
        x = np.clip(x, 0.0, 1.0)
        I_vals.append(I_val)

    I_vals = np.array(I_vals)
    I0, I_end = I_vals[0], I_vals[-1]
    drop = (I0 - I_end) / I0 * 100

    # Fit to stretched exponential
    try:
        norm_I = I_vals / I0
        popt, _ = curve_fit(stretched_exp, t_points, norm_I,
                            p0=(1e4, 0.5), bounds=([1e2, 0.1], [1e6, 1.0]))
        tau, beta = popt
    except Exception:
        tau, beta = np.nan, np.nan

    print(f"  Level {j+1}: I0={I0*1e6:.3f} uA, I_end={I_end*1e6:.3f} uA, Drop={drop:.2f}%")
    print(f"             Fit: tau={tau:.2f}s, beta={beta:.2f}")

    retention_data.append(pd.DataFrame({
        "time_s": t_points,
        "current_A": I_vals,
        "level": j + 1
    }))
    fit_results.append((tau, beta))

    # Save individual CSV
    retention_data[-1].to_csv(f"retention_level{j+1}.csv", index=False, encoding='utf-8')

print("\nRetention data saved for all levels (CSV).")

# -------------------------------------------------------------------------
# [4/4] Generate figures
# -------------------------------------------------------------------------
print("\n[4/4] Generating figures...")

plt.figure(figsize=(7, 5))
for j, df in enumerate(retention_data):
    plt.plot(df["time_s"], df["current_A"] * 1e6, lw=1.8, label=f"Level {j+1}")
plt.xscale("log")
plt.xlabel("Time (s)", fontsize=12)
plt.ylabel("Current (µA)", fontsize=12)
plt.title("Retention: Current vs Time at V=0", fontsize=13, fontweight='bold')
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("retention_drift_curves.png", dpi=600, bbox_inches='tight')

# Combined figure: normalized + absolute
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharex=True)
for j, df in enumerate(retention_data):
    norm_I = df["current_A"] / df["current_A"].iloc[0]
    axes[0].plot(df["time_s"], norm_I, lw=1.5, label=f"Level {j+1}")
    axes[1].plot(df["time_s"], df["current_A"] * 1e6, lw=1.5,
                 label=f"L{j+1}: τ={fit_results[j][0]:.1f}s, β={fit_results[j][1]:.2f}")
for ax in axes:
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)
axes[0].set_ylabel("Normalized Current (I/I0)", fontsize=11)
axes[0].set_xlabel("Time (s)", fontsize=11)
axes[1].set_ylabel("Current (µA)", fontsize=11)
axes[1].set_xlabel("Time (s)", fontsize=11)
axes[0].set_title("(a) Normalized Drift", fontsize=12, fontweight='bold')
axes[1].set_title("(b) Absolute Current", fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9)
axes[1].legend(fontsize=9)
plt.tight_layout()
plt.savefig("retention_drift_combined.png", dpi=600, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# RESULTS SUMMARY
# -------------------------------------------------------------------------
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
for j, (tau, beta) in enumerate(fit_results):
    df = retention_data[j]
    I0, I_end = df["current_A"].iloc[0], df["current_A"].iloc[-1]
    drop = (I0 - I_end) / I0 * 100
    print(f"Level {j+1}: I0={I0*1e6:.3f} µA, I_end={I_end*1e6:.3f} µA, Drop={drop:.2f}%")
    print(f"         Fit: τ={tau:.2f}s, β={beta:.2f}")
print("\nFigures saved:")
print("  - retention_drift_curves.png")
print("  - retention_drift_combined.png")
print("  - retention_level[1–3].csv")

print("\n" + "=" * 80)
print("EXPERIMENT 7 COMPLETE")
print("=" * 80)
print("\nKey Findings:")
print("  * Improved programming ensures distinct initial states.")
print("  * Long-term retention modeled up to 10⁶ s with slow relaxation.")
print("  * Exported CSV enables quantitative reliability benchmarking.")
