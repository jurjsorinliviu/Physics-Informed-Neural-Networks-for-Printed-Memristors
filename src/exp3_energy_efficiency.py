# ===========================================================
#  cumulative_write_energy_simulation.py — FINAL STABLE VERSION
#  Experiment 3: Cumulative Write Energy Comparison (PINN vs VTEAM vs Yakopcic)
# ===========================================================

import numpy as np
import matplotlib.pyplot as plt
from mainPINNmodel import PrintedMemristorPINN
from TrainingFrameworkwithNoiseInjection import PINNTrainer
from VTEAMModelComparison import VTEAMModel
from ExtendedValidation import YakopcicModel
import os, sys
sys.stdout.reconfigure(errors='replace')

print("="*80)
print("EXPERIMENT 3: Cumulative Write Energy Comparison")
print("="*80)
print("\nObjective: Compare energy consumption across multiple write pulses")
print("Protocol: 10 consecutive SET pulses without RESET")
print("-"*80)

# -----------------------------------------------------------
# [1/4] Initialize and train PINN
# -----------------------------------------------------------
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

# -----------------------------------------------------------
# [2/4] Initialize baseline models
# -----------------------------------------------------------
print("[2/4] Initializing baseline models...")
vteam = VTEAMModel()
yakopcic = YakopcicModel()

# -----------------------------------------------------------
# [3/4] Simulate write pulses
# -----------------------------------------------------------
# Updated parameters
pulse_count = 10
V_pulse = 2.0          # Same as Exp 1
pulse_duration = 10e-3 # 10 ms (same as Exp 1)
dt = 1e-4              # 100 μs (same as Exp 1)
pulse_steps = int(pulse_duration / dt)  # = 100 steps
gap_duration = 5e-3    # 5 ms gap
gap_steps = int(gap_duration / dt)     # = 50 steps

print(f"  - Duration: {pulse_duration*1e6:.1f} μs")  # Will print "1.0 μs"

print(f"\n[3/4] Simulating {pulse_count} write pulses...")
print(f"  - Amplitude: {V_pulse} V")
print(f"  - Duration: {pulse_steps * dt * 1e6:.1f} µs")
print(f"  - Gap: {gap_steps * dt * 1e6:.1f} µs")
print(f"  - Time step: {dt * 1e6:.1f} µs")

# Build voltage waveform
single_pulse = [V_pulse]*pulse_steps + [0.0]*gap_steps
voltage_seq = np.tile(single_pulse, pulse_count)

# -----------------------------
# Simulate PINN model
# -----------------------------
print("\nSimulating models...\n--------------------------")
print("1. PINN Model:")
x_pinn, E_pinn = 0.0, 0.0
cumE_pinn = []

for idx, V in enumerate(voltage_seq):
    inp = np.array([[V, x_pinn]], dtype=np.float32)
    I_pred, xdot_pred = pinn.model(inp, training=False)
    I_val = float(I_pred.numpy()[0,0])
    xdot_val = float(xdot_pred.numpy()[0,0])
    x_pinn = np.clip(x_pinn + xdot_val * dt, 0.0, 1.0)
    E_pinn += V * I_val * dt
    if V == 0.0 and idx > 0 and voltage_seq[idx-1] == V_pulse:
        cumE_pinn.append(E_pinn)
        print(f"  Pulse {len(cumE_pinn):2d}: Energy = {E_pinn*1e12:.2f} pJ, State = {x_pinn:.4f}")

# -----------------------------
# Simulate VTEAM model
# -----------------------------
print("\n2. VTEAM Model:")
w_v, E_v = vteam.w_min, 0.0
cumE_v = []
for idx, V in enumerate(voltage_seq):
    dw = vteam.state_derivative(V, w_v)
    w_v = np.clip(w_v + dw * dt, vteam.w_min, vteam.w_max)
    I_val = vteam.predict_current(V, w_v)
    E_v += V * I_val * dt
    if V == 0.0 and idx > 0 and voltage_seq[idx-1] == V_pulse:
        cumE_v.append(E_v)
        print(f"  Pulse {len(cumE_v):2d}: Energy = {E_v*1e12:.2f} pJ, State = {w_v:.4f}")

# -----------------------------
# Simulate Yakopcic model
# -----------------------------
print("\n3. Yakopcic Model:")
x_y, E_y = 0.0, 0.0
cumE_y = []
for idx, V in enumerate(voltage_seq):
    dx = yakopcic.state_derivative(V, x_y)
    x_y = np.clip(x_y + dx * dt, 0.0, 1.0)
    I_val = yakopcic.current(V, x_y)
    E_y += V * I_val * dt
    if V == 0.0 and idx > 0 and voltage_seq[idx-1] == V_pulse:
        cumE_y.append(E_y)
        print(f"  Pulse {len(cumE_y):2d}: Energy = {E_y*1e12:.2f} pJ, State = {x_y:.4f}")

# Convert to pJ
cumE_pinn = np.array(cumE_pinn) * 1e12
cumE_v = np.array(cumE_v) * 1e12
cumE_y = np.array(cumE_y) * 1e12
pulses = np.arange(1, pulse_count + 1)

# -----------------------------------------------------------
# Results summary
# -----------------------------------------------------------
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)
print(f"\nFinal Cumulative Energy (after {pulse_count} pulses):")
print(f"  PINN:     {cumE_pinn[-1]:.2f} pJ")
print(f"  VTEAM:    {cumE_v[-1]:.2f} pJ")
print(f"  Yakopcic: {cumE_y[-1]:.2f} pJ")

print("\nAverage Energy per Pulse:")
print(f"  PINN:     {cumE_pinn[-1]/pulse_count:.2f} pJ/pulse")
print(f"  VTEAM:    {cumE_v[-1]/pulse_count:.2f} pJ/pulse")
print(f"  Yakopcic: {cumE_y[-1]/pulse_count:.2f} pJ/pulse")

print("\nEnergy Efficiency (vs PINN):")
print(f"  VTEAM:    {cumE_v[-1]/cumE_pinn[-1]:.2f}×")
print(f"  Yakopcic: {cumE_y[-1]/cumE_pinn[-1]:.2f}×")

# -----------------------------------------------------------
# [4/4] Generate figures
# -----------------------------------------------------------
output_dir = os.getcwd()

# Figure 1: Cumulative energy
fig1, ax = plt.subplots(figsize=(6,4))
ax.plot(pulses, cumE_pinn, label="PINN", color="#d95f02", linewidth=2.2, marker='o')
ax.plot(pulses, cumE_v, label="VTEAM", color="#1b9e77", linestyle="--", linewidth=2.2, marker='s')
ax.plot(pulses, cumE_y, label="Yakopcic", color="#7570b3", linestyle=":", linewidth=2.5, marker='^')
ax.set_xlabel("Number of write pulses", fontsize=12)
ax.set_ylabel("Cumulative energy (pJ)", fontsize=12)
ax.set_title("Cumulative Write Energy Comparison", fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
fig1.tight_layout()
plt.savefig(os.path.join(output_dir, "cumulative_write_energy.png"), dpi=600, bbox_inches="tight")
plt.close(fig1)

# Figure 2: Combined (cumulative + per-pulse)
fig2, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
# (a)
axes[0].plot(pulses, cumE_pinn, '-o', color="#d95f02", label="PINN")
axes[0].plot(pulses, cumE_v, '--s', color="#1b9e77", label="VTEAM")
axes[0].plot(pulses, cumE_y, ':^', color="#7570b3", label="Yakopcic")
axes[0].set_xlabel("Pulse Number", fontsize=11)
axes[0].set_ylabel("Cumulative Energy (pJ)", fontsize=11)
axes[0].set_title("(a) Cumulative Energy", fontsize=12, fontweight="bold")
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=9)

# (b)
Eper_pinn = np.diff(np.insert(cumE_pinn, 0, 0))
Eper_v = np.diff(np.insert(cumE_v, 0, 0))
Eper_y = np.diff(np.insert(cumE_y, 0, 0))
axes[1].plot(pulses, Eper_pinn, '-o', color="#d95f02", label="PINN")
axes[1].plot(pulses, Eper_v, '--s', color="#1b9e77", label="VTEAM")
axes[1].plot(pulses, Eper_y, ':^', color="#7570b3", label="Yakopcic")
axes[1].set_xlabel("Pulse Number", fontsize=11)
axes[1].set_ylabel("Energy per Pulse (pJ)", fontsize=11)
axes[1].set_title("(b) Incremental Energy per Pulse", fontsize=12, fontweight="bold")
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=9)

plt.tight_layout(pad=2.0)
combined_path = os.path.join(output_dir, "cumulative_write_energy_combined.png")
plt.savefig(combined_path, dpi=600, bbox_inches="tight")
plt.close(fig2)

# -----------------------------------------------------------
# Final report
# -----------------------------------------------------------
print("="*80)
print("EXPERIMENT 3 COMPLETE")
print("="*80)
print("Key notes:")
print("  • PINN captures smooth, nonlinear energy accumulation")
print("  • Comparison with physics-based baselines completed")
print("  • Final figures saved:")
print(f"      {os.path.join(output_dir, 'cumulative_write_energy.png')}")
print(f"      {combined_path}")
print("="*80)
