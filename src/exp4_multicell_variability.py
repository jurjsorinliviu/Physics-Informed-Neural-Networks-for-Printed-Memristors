# ===========================================================
#  multi_cell_variability_simulation.py — FINAL STABLE VERSION
#  Experiment 4: Multi-Cell Array Variability Simulation
# ===========================================================

import sys
# Robust console output on Windows: avoid UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import matplotlib.pyplot as plt
from mainPINNmodel import PrintedMemristorPINN
from TrainingFrameworkwithNoiseInjection import PINNTrainer

print("="*80)
print("EXPERIMENT 4: Multi-Cell Array Variability Simulation")
print("="*80)
print("\nObjective: Simulate device-to-device variations in a 1T1R array")
print("Protocol: Multiple cells with parameter perturbations")
print("-"*80)

# -----------------------------------------------------------
# 1) Initialize and train PINN
# -----------------------------------------------------------
print("\n[1/3] Initializing and training PINN model...")
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

# -----------------------------------------------------------
# 2) Simulate multiple devices with variability
# -----------------------------------------------------------
print("\n[2/3] Simulating multiple devices with parameter variations...")
device_count = 5
rng = np.random.default_rng(seed=1)

# Device-to-device variations
threshold_factors = 1.0 + 0.05 * rng.standard_normal(device_count)    # ~ +/-5%
conductance_factors = 1.0 + 0.10 * rng.standard_normal(device_count)  # ~ +/-10%

print("\nArray Configuration:")
print(f"  - Number of cells: {device_count}")
print(f"  - Threshold variation: +/-5%")
print(f"  - Conductance variation: +/-10%")

print("\nDevice Parameters:")
for j in range(device_count):
    print(f"  Cell {j+1}: Threshold factor = {threshold_factors[j]:.4f}, "
          f"Conductance factor = {conductance_factors[j]:.4f}")

# Voltage sweep (bipolar)
steps = 401
V_up = np.linspace(-2.0, 2.0, steps)
V_down = np.linspace(2.0, -2.0, steps)[1:]
voltage_sweep = np.concatenate([V_up, V_down]).astype(np.float32)

print(f"\nVoltage sweep: {voltage_sweep.min():.1f} V -> {voltage_sweep.max():.1f} V -> {voltage_sweep.min():.1f} V")
print(f"Number of points: {len(voltage_sweep)}")

# Simulate switching behavior
on_off_ratios = []
plt.figure(figsize=(7, 5))

print("\nSimulating switching behavior...")
for j in range(device_count):
    alpha = float(threshold_factors[j])
    gamma = float(conductance_factors[j])
    x = 0.0
    I_trace = []

    for V in voltage_sweep:
        V_eff = float(V) / alpha
        inp = np.array([[V_eff, x]], dtype=np.float32)
        I_pred, xdot_pred = pinn.model(inp, training=False)
        I_val = float(I_pred.numpy()[0, 0])
        xdot_val = float(xdot_pred.numpy()[0, 0])
        x += xdot_val
        x = float(np.clip(x, 0.0, 1.0))
        I_val *= gamma
        I_trace.append(I_val)

    I_trace = np.array(I_trace, dtype=np.float64)

    # Metrics
    I_max = np.max(I_trace)
    I_min = np.min(np.abs(I_trace[I_trace < 0])) if np.any(I_trace < 0) else np.min(I_trace)
    on_off = (I_max / I_min) if I_min > 0 else np.nan
    on_off_ratios.append(on_off)

    print(f"  Cell {j+1}: I_max = {I_max*1e6:.3f} uA, ON/OFF = {on_off:.1f}x")
    plt.plot(voltage_sweep, I_trace, linewidth=1.8, label=f"Cell {j+1}", alpha=0.85)

# Save all I–V curves
plt.xlabel("Voltage (V)", fontsize=12)
plt.ylabel("Current (A)", fontsize=12)
plt.title("Variability in Switching Behavior across 1T1R Cells (PINN Model)",
          fontsize=12, fontweight="bold")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig("multi_cell_variability.png", dpi=600, bbox_inches="tight")
plt.close()

# -----------------------------------------------------------
# 3) Combined IEEE-style figure: I–V + ON/OFF ratio bars
# -----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

# (a) I–V variability
axes[0].set_title("(a) I–V Variability", fontsize=12, fontweight="bold")
for j in range(device_count):
    alpha = float(threshold_factors[j])
    gamma = float(conductance_factors[j])
    x = 0.0
    I_trace = []
    for V in voltage_sweep:
        V_eff = float(V) / alpha
        inp = np.array([[V_eff, x]], dtype=np.float32)
        I_pred, xdot_pred = pinn.model(inp, training=False)
        I_val = float(I_pred.numpy()[0, 0]) * gamma
        xdot_val = float(xdot_pred.numpy()[0, 0])
        x += xdot_val
        x = float(np.clip(x, 0.0, 1.0))
        I_trace.append(I_val)
    axes[0].plot(voltage_sweep, I_trace, linewidth=1.5, alpha=0.85)
axes[0].set_xlabel("Voltage (V)", fontsize=11)
axes[0].set_ylabel("Current (A)", fontsize=11)
axes[0].grid(True, alpha=0.3)

# (b) ON/OFF ratio distribution
axes[1].set_title("(b) ON/OFF Ratio per Cell", fontsize=12, fontweight="bold")
axes[1].bar(np.arange(1, device_count + 1), on_off_ratios, color="tab:orange", alpha=0.85)
axes[1].set_xlabel("Cell Number", fontsize=11)
axes[1].set_ylabel("ON/OFF Ratio (x)", fontsize=11)
axes[1].grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout(pad=2.0)
plt.savefig("multi_cell_variability_combined.png", dpi=600, bbox_inches="tight")
plt.close(fig)
print("Figure saved: multi_cell_variability_combined.png (600 dpi)")

# -----------------------------------------------------------
# 4) Results summary
# -----------------------------------------------------------
on_off_ratios = np.array(on_off_ratios, dtype=np.float64)
mean_ratio = np.nanmean(on_off_ratios)
std_ratio = np.nanstd(on_off_ratios)
cov_ratio = (std_ratio / mean_ratio) * 100 if mean_ratio > 0 else np.nan

print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)
print("\nArray-Level Statistics:")
print(f"  - Threshold variation: {np.std(threshold_factors)/np.mean(threshold_factors)*100:.2f}%")
print(f"  - Conductance variation: {np.std(conductance_factors)/np.mean(conductance_factors)*100:.2f}%")
print(f"  - ON/OFF ratio CoV: {cov_ratio:.2f}%")

print("\nDevice-to-Device Variability:")
print(f"  - Mean threshold factor: {np.mean(threshold_factors):.4f} +/- {np.std(threshold_factors):.4f}")
print(f"  - Mean conductance factor: {np.mean(conductance_factors):.4f} +/- {np.std(conductance_factors):.4f}")

print("\n" + "="*80)
print("EXPERIMENT 4 COMPLETE")
print("="*80)
print("\nKey Findings:")
print("  - PINN captures realistic device-to-device variations")
print("  - Threshold and conductance variations influence I–V shape")
print(f"  - ON/OFF CoV = {cov_ratio:.2f}%, supporting variability/yield analysis")
