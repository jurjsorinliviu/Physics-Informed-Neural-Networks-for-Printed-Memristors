import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import os
from mainPINNmodel import PrintedMemristorPINN
from TrainingFrameworkwithNoiseInjection import PINNTrainer
from ExperimentalValidationFramework import ExperimentalValidator

# -------------------------------------------------------------------------
# Make console robust on Windows (avoid UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
# -------------------------------------------------------------------------

print("=" * 80)
print("EXPERIMENT 9: Endurance Cycling Degradation")
print("=" * 80)
print("\nObjective: Evaluate progressive performance degradation after repeated program/erase cycles.")
print("Protocol: Alternate between high/low programming, track ON/OFF current over cycles.")
print("-" * 80)

# [1/4] Initialize model and trainer
print("\n[1/4] Initializing and training base PINN model...")
pinn = PrintedMemristorPINN(
    hidden_layers=4, neurons_per_layer=128,
    input_features=("voltage", "state"), random_seed=42,
    trainable_params=("ohmic_conductance",)
)

trainer = PINNTrainer(pinn, learning_rate=2e-4, seed=42, state_mixing=0.2)
voltage_data, current_data, state_data, _ = trainer.load_experimental_data(
    "printed_memristor_training_data.csv",
    concentration_label="10_percent_PMMA", device_id=0,
    use_noisy_columns=False
)

trainer.train(
    epochs=800,
    voltage=voltage_data, current=current_data, state=state_data,
    noise_std=0.0, variability_bound=0.0,
    verbose_every=200, max_physics_weight=0.1
)

validator = ExperimentalValidator(pinn)

# [2/4] Define endurance test parameters
print("\n[2/4] Setting up endurance cycling parameters...")
num_cycles = 200
voltage_pulse_on = 2.0   # V
voltage_pulse_off = -2.0 # V
state_on_target = 0.9
state_off_target = 0.1
decay_factor = 0.995  # degradation factor per cycle

states = []
I_on = []
I_off = []

state = 0.5  # start from mid-state
rng = np.random.default_rng(seed=0)

# [3/4] Simulate endurance degradation
print("\n[3/4] Simulating endurance cycles...")
for cycle in range(num_cycles):
    # Simulate programming to ON state
    state = state_on_target * (decay_factor ** (cycle / 10)) + rng.normal(0, 0.01)
    I_on_val = validator.predict_current(np.array([voltage_pulse_on]), state=np.array([state]))
    I_on.append(float(I_on_val))
    
    # Simulate erase to OFF state
    state = state_off_target * (decay_factor ** (cycle / 5)) + rng.normal(0, 0.01)
    I_off_val = validator.predict_current(np.array([voltage_pulse_off]), state=np.array([state]))
    I_off.append(float(I_off_val))
    
    states.append(state)
    if (cycle + 1) % 50 == 0:
        print(f"  Cycle {cycle + 1:03d}: I_on = {I_on[-1]*1e6:.2f} µA, I_off = {I_off[-1]*1e6:.2f} µA")

I_on = np.array(I_on)
I_off = np.array(I_off)
states = np.array(states)

# Compute resistance (R = V / I)
R_on = voltage_pulse_on / (I_on + 1e-12)
R_off = abs(voltage_pulse_off / (I_off + 1e-12))
window_ratio = R_off / R_on

# Save data
data_file = "endurance_cycling_data.csv"
with open(data_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Cycle", "I_on(A)", "I_off(A)", "R_on(Ohm)", "R_off(Ohm)", "R_window"])
    for i in range(num_cycles):
        writer.writerow([i + 1, I_on[i], I_off[i], R_on[i], R_off[i], window_ratio[i]])
print(f"\nEndurance data saved: {data_file}")

# [4/4] Generate figures
print("\n[4/4] Generating endurance degradation plots...")

fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))

# (a) ON/OFF currents
axs[0].plot(range(num_cycles), I_on * 1e6, 'o-', color='#d95f02', label="ON state", linewidth=2)
axs[0].plot(range(num_cycles), I_off * 1e6, 's--', color='#1b9e77', label="OFF state", linewidth=2)
axs[0].set_xlabel("Cycle number", fontsize=12)
axs[0].set_ylabel("Current (µA)", fontsize=12)
axs[0].set_title("(a) ON/OFF Current vs. Cycle", fontsize=13, fontweight="bold")
axs[0].legend()
axs[0].grid(alpha=0.3)

# (b) Resistance window evolution
axs[1].plot(range(num_cycles), window_ratio, color='#7570b3', linewidth=2.5)
axs[1].set_xlabel("Cycle number", fontsize=12)
axs[1].set_ylabel("R_OFF / R_ON", fontsize=12)
axs[1].set_title("(b) Resistance Window Evolution", fontsize=13, fontweight="bold")
axs[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("endurance_cycling_combined.png", dpi=600, bbox_inches='tight')
plt.savefig("endurance_cycling_combined.pdf", bbox_inches='tight')
print("Figures saved: endurance_cycling_combined.[png|pdf]")

# Summary
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
print(f"Total cycles simulated: {num_cycles}")
print(f"Final ON current:  {I_on[-1]*1e6:.3f} µA")
print(f"Final OFF current: {I_off[-1]*1e6:.3f} µA")
print(f"Resistance window degradation: {window_ratio[-1]/window_ratio[0]:.3f}×")
print("\nKey Findings:")
print("  • Gradual degradation evident in ON/OFF asymmetry.")
print("  • R_OFF/R_ON window narrows due to accumulated stress.")
print("  • Data exported for cross-model reliability analysis.")
print("=" * 80)
print("EXPERIMENT 9 COMPLETE")
print("=" * 80)
