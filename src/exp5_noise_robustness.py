# ===========================================================
#  Experiment 5: Noise Stress Test – Model Robustness Comparison (FINAL)
# ===========================================================

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from mainPINNmodel import PrintedMemristorPINN
from TrainingFrameworkwithNoiseInjection import PINNTrainer
from ExperimentalValidationFramework import ExperimentalValidator
from VTEAMModelComparison import VTEAMModel
from ExtendedValidation import YakopcicModel

print("=" * 80)
print("EXPERIMENT 5: Noise Stress Test – Model Robustness Comparison")
print("=" * 80)
print("\nObjective: Compare model robustness under increasing input noise")
print("Protocol: Apply noise to input voltages and measure prediction degradation")
print("-" * 80)

# -----------------------------------------------------------
# [1/4] Initialize and train PINN
# -----------------------------------------------------------
print("\n[1/4] Initializing and training PINN model...")
pinn = PrintedMemristorPINN(
    hidden_layers=4,
    neurons_per_layer=128,
    input_features=("voltage", "state"),
    random_seed=42,
    trainable_params=("ohmic_conductance",)
)

trainer = PINNTrainer(pinn, learning_rate=2e-4, seed=42, state_mixing=0.2)
voltage_data, current_data, state_data, _ = trainer.load_experimental_data(
    "printed_memristor_training_data.csv",
    concentration_label="10_percent_PMMA",
    device_id=0,
    use_noisy_columns=False
)
trainer.train(
    epochs=800,
    voltage=voltage_data,
    current=current_data,
    state=state_data,
    noise_std=0.0,
    variability_bound=0.0,
    verbose_every=200,
    max_physics_weight=0.1
)

# -----------------------------------------------------------
# [2/4] Initialize baseline models and validator
# -----------------------------------------------------------
print("\n[2/4] Initializing baseline models and validator...")
validator = ExperimentalValidator(pinn)
vteam = VTEAMModel()
yakopcic = YakopcicModel()

# -----------------------------------------------------------
# [3/4] Compute baseline predictions
# -----------------------------------------------------------
print("\n[3/4] Computing baseline predictions on clean data...")
V = voltage_data.astype(np.float32)
print(f"  Test set size: {len(V)} samples")

I_pinn_baseline = validator.predict_current(V, state=state_data)
I_vteam_baseline = vteam.simulate_iv(V)
I_yak_baseline = yakopcic.simulate_iv(V)

print(f"  PINN baseline: I_mean = {np.mean(I_pinn_baseline) * 1e6:.3f} µA")
print(f"  VTEAM baseline: I_mean = {np.mean(I_vteam_baseline) * 1e6:.3f} µA")
print(f"  Yakopcic baseline: I_mean = {np.mean(I_yak_baseline) * 1e6:.3f} µA")

# -----------------------------------------------------------
# [4/4] Apply noise and compute robustness
# -----------------------------------------------------------
noise_levels = [0.01, 0.02, 0.05, 0.1]
print(f"\n[4/4] Testing robustness at noise levels: {[f'{n*100:.0f}%' for n in noise_levels]}")

def compute_rrmse(I_base, I_new):
    rmse = np.sqrt(np.mean((I_new - I_base) ** 2))
    rng = np.max(I_base) - np.min(I_base) + 1e-12
    return rmse / rng

rrmse_pinn, rrmse_vteam, rrmse_yak = [], [], []
rng = np.random.default_rng(seed=0)

print("\nNoise Robustness Analysis:")
print("-" * 60)

for noise_std in noise_levels:
    V_noisy = V + rng.normal(0.0, noise_std * np.std(V), size=V.shape)
    I_pinn_noisy = validator.predict_current(V_noisy, state=state_data)
    I_vteam_noisy = vteam.simulate_iv(V_noisy)
    I_yak_noisy = yakopcic.simulate_iv(V_noisy)

    rrmse_p = compute_rrmse(I_pinn_baseline, I_pinn_noisy)
    rrmse_v = compute_rrmse(I_vteam_baseline, I_vteam_noisy)
    rrmse_y = compute_rrmse(I_yak_baseline, I_yak_noisy)

    rrmse_pinn.append(rrmse_p)
    rrmse_vteam.append(rrmse_v)
    rrmse_yak.append(rrmse_y)

    print(f"Noise {noise_std*100:4.1f}%:")
    print(f"  PINN:     RRMSE = {rrmse_p:.6f}")
    print(f"  VTEAM:    RRMSE = {rrmse_v:.6f}")
    print(f"  Yakopcic: RRMSE = {rrmse_y:.6f}")

rrmse_pinn, rrmse_vteam, rrmse_yak = map(np.array, [rrmse_pinn, rrmse_vteam, rrmse_yak])
noise_pct = np.array(noise_levels) * 100

# -----------------------------------------------------------
# Results summary
# -----------------------------------------------------------
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
print(f"\nRobustness at 10% Noise:")
print(f"  PINN:     RRMSE = {rrmse_pinn[-1]:.6f}")
print(f"  VTEAM:    RRMSE = {rrmse_vteam[-1]:.6f}")
print(f"  Yakopcic: RRMSE = {rrmse_yak[-1]:.6f}")

models_scores = [
    ("PINN", rrmse_pinn[-1]),
    ("VTEAM", rrmse_vteam[-1]),
    ("Yakopcic", rrmse_yak[-1])
]
models_scores.sort(key=lambda x: x[1])

print(f"\nRobustness Ranking (lower RRMSE = better):")
for rank, (model, score) in enumerate(models_scores, 1):
    print(f"  {rank}. {model:12s} (RRMSE = {score:.6f})")

print(f"\nDegradation from Clean → 10% Noise:")
print(f"  PINN:     {rrmse_pinn[-1] / rrmse_pinn[0]:.2f}× increase")
print(f"  VTEAM:    {rrmse_vteam[-1] / rrmse_vteam[0]:.2f}× increase")
print(f"  Yakopcic: {rrmse_yak[-1] / rrmse_yak[0]:.2f}× increase")

# -----------------------------------------------------------
# Figure 1: Noise vs RRMSE
# -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(noise_pct, rrmse_pinn, label="PINN", color="#d95f02", marker='o', linewidth=2.2, markersize=7)
ax.plot(noise_pct, rrmse_vteam, label="VTEAM", color="#1b9e77", linestyle="--", linewidth=2.2, marker='s', markersize=7)
ax.plot(noise_pct, rrmse_yak, label="Yakopcic", color="#7570b3", linestyle=":", linewidth=2.5, marker='^', markersize=7)
ax.set_xlabel("Input Noise Level (%)", fontsize=12)
ax.set_ylabel("RRMSE in Predicted Current", fontsize=12)
ax.set_title("Model Performance Degradation under Noise", fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig("noise_robustness_comparison.png", dpi=600, bbox_inches='tight')
plt.close(fig)

# -----------------------------------------------------------
# Figure 2: Combined Layout (RRMSE + Degradation Ratio)
# -----------------------------------------------------------
fig2, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

# (a) RRMSE vs noise
axes[0].plot(noise_pct, rrmse_pinn, label="PINN", color="#d95f02", marker='o', linewidth=2.2)
axes[0].plot(noise_pct, rrmse_vteam, label="VTEAM", color="#1b9e77", linestyle="--", linewidth=2.2)
axes[0].plot(noise_pct, rrmse_yak, label="Yakopcic", color="#7570b3", linestyle=":", linewidth=2.5)
axes[0].set_xlabel("Noise Level (%)", fontsize=11)
axes[0].set_ylabel("RRMSE", fontsize=11)
axes[0].set_title("(a) RRMSE vs Noise", fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=10)

# (b) Relative degradation
axes[1].plot(noise_pct, rrmse_pinn / rrmse_pinn[0], label="PINN", color="#d95f02", marker='o', linewidth=2.2)
axes[1].plot(noise_pct, rrmse_vteam / rrmse_vteam[0], label="VTEAM", color="#1b9e77", linestyle="--", linewidth=2.2)
axes[1].plot(noise_pct, rrmse_yak / rrmse_yak[0], label="Yakopcic", color="#7570b3", linestyle=":", linewidth=2.5)
axes[1].set_xlabel("Noise Level (%)", fontsize=11)
axes[1].set_ylabel("Relative Degradation (×)", fontsize=11)
axes[1].set_title("(b) Degradation Ratio", fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=10)

plt.tight_layout(pad=2.0)
plt.savefig("noise_robustness_combined.png", dpi=600, bbox_inches='tight')
plt.close(fig2)

# -----------------------------------------------------------
# Final summary
# -----------------------------------------------------------
print("\n" + "=" * 80)
print("EXPERIMENT 5 COMPLETE")
print("=" * 80)
best_model = models_scores[0][0]
print("\nKey Findings:")
print(f"  • {best_model} shows best robustness under input noise.")
print(f"  • PINN maintains {'low' if rrmse_pinn[-1] < 0.03 else 'moderate'} degradation at 10% noise.")
print(f"  • Physics-informed training {'enhances' if rrmse_pinn[-1] < min(rrmse_vteam[-1], rrmse_yak[-1]) else 'maintains'} resilience.")
print(f"\nFinal Figures Saved:")
print("  → noise_robustness_comparison.png")
print("  → noise_robustness_combined.png")
print("=" * 80)
