# ===========================================================
#  experiment12b_self_heating_bias_sweep.py
#  PINN-Based Electro-Thermal Retention / Stress Simulation
#  Bias sweep + ΔT vs Bias visualization (publication-ready)
# ===========================================================

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os, gc, sys

# Make console robust on Windows (avoid UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from mainPINNmodel import PrintedMemristorPINN
from TrainingFrameworkwithNoiseInjection import PINNTrainer


print("=" * 80)
print("EXPERIMENT 12b: Bias Sweep with Self-Heating–Coupled Retention Simulation")
print("=" * 80)
print("\nObjective: Quantify how stress bias accelerates thermal drift.")
print("Protocol: Sweep V = [0.05, 0.1, 0.2 V], monitor T rise and drift rate.")
print("-" * 80)

# ------------------------------------------------------------
# [1/4] Initialize and train PINN model
# ------------------------------------------------------------
print("\n[1/4] Initializing and training PINN model...")
pinn = PrintedMemristorPINN(
    hidden_layers=4,
    neurons_per_layer=128,
    input_features=("voltage", "state"),
    random_seed=42,
    trainable_params=("ohmic_conductance",),
)
trainer = PINNTrainer(pinn, learning_rate=2e-4, seed=42, state_mixing=0.2)
voltage_data, current_data, state_data, _ = trainer.load_experimental_data(
    "printed_memristor_training_data.csv",
    concentration_label="10_percent_PMMA",
    device_id=0,
    use_noisy_columns=True,
)
trainer.train(
    epochs=800,
    voltage=voltage_data,
    current=current_data,
    state=state_data,
    noise_std=0.002,
    variability_bound=0.05,
    verbose_every=200,
    max_physics_weight=0.1,
)

# ------------------------------------------------------------
# [2/4] Electro-thermal parameters
# ------------------------------------------------------------
print("\n[2/4] Setting up electro-thermal parameters...")
bias_values = [0.05, 0.1, 0.2]
R_th = 1e3
C_th = 1e-2
tau_th = R_th * C_th
Tamb = 300.0
t_points = np.logspace(-2, 4, 400)
dt_default = np.diff(np.concatenate([[0], t_points]))
print(f"  Thermal RC: R_th={R_th:.2e} K/W, C_th={C_th:.2e} J/K, tau_th≈{tau_th:.2f} s")

# ------------------------------------------------------------
# [3/4] Define stable simulation
# ------------------------------------------------------------
@tf.function
def pinn_step(V, x_state):
    I_pred, xdot_pred = pinn.model(tf.stack([V, x_state], axis=1), training=False)
    return I_pred, xdot_pred

def run_stable_simulation(V_stress):
    x = tf.Variable([[0.2]], dtype=tf.float32)
    T = float(Tamb)
    currents, temps, states = [], [], []

    for i, dt in enumerate(dt_default):
        try:
            V_tensor = tf.constant([[V_stress]], dtype=tf.float32)
            I_pred, xdot_pred = pinn_step(V_tensor, x)
            I_val = float(I_pred.numpy()[0, 0])
            xdot_val = float(xdot_pred.numpy()[0, 0])

            # Stable thermal update
            P = I_val * V_stress
            dT_dt = (R_th * P - (T - Tamb)) / tau_th
            dT_dt = np.clip(dT_dt, -10.0, 10.0)
            T += dT_dt * dt
            T = np.clip(T, Tamb - 10, Tamb + 1000)

            # State update
            x.assign_add(tf.constant([[xdot_val * dt]], dtype=tf.float32))
            x.assign(tf.clip_by_value(x, 0.0, 1.0))

            currents.append(I_val)
            temps.append(T)
            states.append(float(x.numpy()[0, 0]))

            if i % 100 == 0 or i == len(dt_default) - 1:
                print(f"  t={t_points[i]:8.1f}s | I={I_val*1e6:8.2f} µA | T={T:6.2f} K | x={float(x.numpy()[0,0]):.3f}")

            if not np.isfinite(T):
                print("  ⚠ Numerical instability detected — stopping.")
                break

        except Exception as e:
            print(f"  ⚠ Exception at step {i}: {e}")
            break

    return np.array(currents), np.array(temps), np.array(states)

# ------------------------------------------------------------
# [4/4] Run simulations and collect data
# ------------------------------------------------------------
print("\n[4/4] Running bias sweep simulations...")
results, delta_T = {}, []

for V_stress in bias_values:
    print(f"\n→ Simulating at {V_stress:.2f} V ...")
    I, T, X = run_stable_simulation(V_stress)
    Tmax = np.max(T)
    delta_T.append(Tmax - Tamb)
    results[V_stress] = {"I": I, "T": T, "X": X}
    np.savetxt(f"self_heating_{V_stress:.2f}V.csv",
               np.column_stack([t_points[:len(I)], I, T, X]),
               delimiter=",",
               header="Time_s,Current_A,Temperature_K,State_x",
               comments="")
    gc.collect()

# ------------------------------------------------------------
# Plot 1: Time evolution for all biases
# ------------------------------------------------------------
print("\nGenerating composite figure...")
fig, ax1 = plt.subplots(figsize=(7, 4))
colors = ["C0", "C1", "C2"]

for idx, (V, res) in enumerate(results.items()):
    ax1.plot(t_points[:len(res["I"])], np.array(res["I"]) * 1e6,
             color=colors[idx], lw=2, label=f"I (V={V:.2f} V)")

ax1.set_xscale("log")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Current (µA)")
ax1.legend(loc="upper left", frameon=False)

ax2 = ax1.twinx()
for idx, (V, res) in enumerate(results.items()):
    ax2.plot(t_points[:len(res["T"])], res["T"],
             color=colors[idx], ls="--", lw=1.8, alpha=0.7)
ax2.set_ylabel("Temperature (K)")

plt.title("Electro-Thermal Self-Heating Simulation (PINN Model)")
plt.tight_layout()
plt.savefig("self_heating_drift_combined.png", dpi=600, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------
# Plot 2: ΔT vs Bias (publication inset)
# ------------------------------------------------------------
plt.figure(figsize=(4, 3))
plt.plot(bias_values, delta_T, "o-", color="C3", lw=2, markersize=7)
plt.xlabel("Stress Bias (V)")
plt.ylabel("ΔT (K)")
plt.title("Temperature Rise vs Bias Voltage")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("self_heating_dT_vs_bias.png", dpi=600, bbox_inches="tight")
plt.close()

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
for V, dT in zip(bias_values, delta_T):
    print(f"Bias {V:.2f} V → Max ΔT = {dT:.2f} K")
print("\nFigures saved:")
print("  - self_heating_drift_combined.png (I/T vs time)")
print("  - self_heating_dT_vs_bias.png (ΔT vs bias)")
print("CSV files saved for each bias point.")
print("=" * 80)
print("\nKey Insights:")
print(" • Higher bias → larger power dissipation → greater ΔT.")
print(" • Demonstrates thermally accelerated degradation potential.")
print(" • Second figure (ΔT vs Bias) ideal for paper inset or summary plot.")
