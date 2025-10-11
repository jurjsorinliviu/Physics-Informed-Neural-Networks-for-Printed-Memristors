"""
Experiment 10b (Enhanced): Dual-Panel Arrhenius Analysis of Retention Degradation
-------------------------------------------------------------------------------
Reads:  arrhenius_retention_fit_data.csv
Outputs:
  - arrhenius_dualpanel.png
  - arrhenius_dualpanel.pdf
  - arrhenius_fit_results.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import sys
import os

# --- Make console robust on Windows ---
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# --- Load dataset ---
data_file = "arrhenius_retention_fit_data.csv"
if not os.path.exists(data_file):
    raise FileNotFoundError(f"❌ Cannot find {data_file} in current directory.")

data = pd.read_csv(data_file)
print("Loaded dataset:")
print(data)

# --- Physical constants ---
k_B = 8.617333262e-5  # eV/K

# --- Compute inverse temperature ---
data["1_over_T"] = 1.0 / data["Temperature_K"]

# --- Linear regression (Arrhenius) ---
x = data["1_over_T"]
y = data["ln_tau_rel"]
slope, intercept, r_value, p_value, std_err = linregress(x, y)
Ea = slope * k_B  # eV
Ea_err = std_err * k_B

print("\nFitted Arrhenius parameters:")
print(f"  Activation energy (Ea): {Ea:.4f} ± {Ea_err:.4f} eV")
print(f"  Pre-exponential constant (A): {intercept:.4f}")
print(f"  R² = {r_value**2:.5f}")

# --- Save fit results ---
fit_results = pd.DataFrame({
    "Parameter": ["Ea (eV)", "Ea_err (eV)", "Intercept", "R_squared"],
    "Value": [Ea, Ea_err, intercept, r_value**2]
})
fit_results.to_csv("arrhenius_fit_results.csv", index=False)

# --- Generate predictions for plotting ---
x_fit = np.linspace(x.min() * 0.98, x.max() * 1.02, 200)
y_fit = intercept + slope * x_fit

# --- Figure: Dual Panel ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- Panel A: Arrhenius fit (ln τ_rel vs 1/T) ---
axes[0].scatter(x, y, s=70, color='tab:blue', label='Data', zorder=3)
axes[0].plot(x_fit, y_fit, 'r-', lw=2.5,
             label=f'Fit: Ea = {Ea:.3f} eV')
axes[0].set_xlabel("1 / Temperature (1/K)", fontsize=11)
axes[0].set_ylabel("ln(τ_rel)", fontsize=11)
axes[0].set_title("(a) Arrhenius Fit of Retention Degradation", fontsize=13, fontweight='bold')
axes[0].legend(frameon=True)
axes[0].grid(alpha=0.3)

# Optional: secondary x-axis showing Temperature (K)
secax = axes[0].secondary_xaxis('top', functions=(lambda invT: 1/invT, lambda T: 1/T))
secax.set_xlabel("Temperature (K)", fontsize=10)

# --- Panel B: Retention drop (%) vs T ---
axes[1].semilogy(data["Temperature_K"], data["Drop_percent"],
                 'o-', color='tab:orange', lw=2.5, markersize=6)
axes[1].set_xlabel("Temperature (K)", fontsize=11)
axes[1].set_ylabel("Retention drop at t_obs (%)", fontsize=11)
axes[1].set_title("(b) Retention Acceleration with Temperature", fontsize=13, fontweight='bold')
axes[1].grid(True, which='both', linestyle='--', alpha=0.4)

# --- Adjust layout and save ---
plt.tight_layout()
plt.savefig("arrhenius_dualpanel.png", dpi=300, bbox_inches="tight")
plt.savefig("arrhenius_dualpanel.pdf", bbox_inches="tight")
plt.show()

print("\nFigures saved:")
print("  - arrhenius_dualpanel.png")
print("  - arrhenius_dualpanel.pdf")
print("Results saved:")
print("  - arrhenius_fit_results.csv")

print("\nKey Insights:")
print(f" • Ea = {Ea:.3f} eV → indicates thermally activated retention degradation.")
print(" • Right panel correlates degradation (%) with T.")
print(" • Together, they offer a clear visual link between microscopic kinetics and macroscopic loss.")
