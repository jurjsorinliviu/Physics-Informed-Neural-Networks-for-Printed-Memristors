# ================================================================
# EXPERIMENT 11: Lifetime Projection via Arrhenius Reliability Model
# ================================================================
# Objective:
#   Use activation energy (Ea) from Experiment 10b to project
#   retention lifetime across operating temperatures.
# ---------------------------------------------------------------
# Author: Sorin
# Date: 2025-10-10
# ---------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.constants import Boltzmann as kB
import sys
from pathlib import Path

# ---------------------------------------------------------------
# Make console robust on Windows (avoid UnicodeEncodeError)
# ---------------------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

print("EXPERIMENT 11: Lifetime Projection via Arrhenius Reliability Model")
print("=" * 80)

# ---------------------------------------------------------------
# 1. Load Arrhenius fit results
# ---------------------------------------------------------------
file_path = Path("arrhenius_fit_results.csv")
if not file_path.exists():
    raise FileNotFoundError("Missing file: arrhenius_fit_results.csv")

df_fit = pd.read_csv(file_path)
print(f"Loaded file: {file_path}")
print(df_fit.head())

# --- Detect layout and extract parameters ---
if "Parameter" in df_fit.columns and "Value" in df_fit.columns:
    # Key–value layout (vertical)
    df_fit = df_fit.set_index("Parameter")
    Ea = float(df_fit.loc["Ea (eV)", "Value"])
    A = float(df_fit.loc["Intercept", "Value"])
    R2 = float(df_fit.loc["R_squared", "Value"]) if "R_squared" in df_fit.index else np.nan
else:
    # Wide layout (horizontal)
    col_ea = next((c for c in df_fit.columns if "Ea" in c), None)
    col_a = next((c for c in df_fit.columns if c.lower().startswith("a")), None)
    col_r2 = next((c for c in df_fit.columns if "R2" in c or "r2" in c.lower()), None)

    if col_ea is None:
        raise ValueError("Could not find activation energy column (expected Ea_eV or similar).")
    if col_a is None:
        raise ValueError("Could not find pre-exponential/intercept column.")

    Ea = float(df_fit.loc[0, col_ea])
    A = float(df_fit.loc[0, col_a])
    R2 = float(df_fit.loc[0, col_r2]) if col_r2 else np.nan

print(f"Extracted parameters:")
print(f"  Ea = {Ea:.3f} eV")
print(f"  A  = {A:.4f}")
if not np.isnan(R2):
    print(f"  R² = {R2:.5f}")
print("-" * 80)

# ---------------------------------------------------------------
# 2. Lifetime model (Arrhenius relation)
# ---------------------------------------------------------------
kB_eV = 8.617333262145e-5  # eV/K

def tau_eV(T_K):
    """Return projected lifetime (s) for temperature (K)."""
    return np.exp(A + Ea / (kB_eV * T_K))

# ---------------------------------------------------------------
# 3. Simulate across temperature range
# ---------------------------------------------------------------
temps = np.arange(250, 401, 10)  # 250–400 K
tau_values = tau_eV(temps)
tau_hours = tau_values / 3600
tau_years = tau_values / (3600 * 24 * 365)

df_life = pd.DataFrame({
    "Temperature_K": temps,
    "Lifetime_s": tau_values,
    "Lifetime_h": tau_hours,
    "Lifetime_years": tau_years
})
df_life.to_csv("arrhenius_lifetime_projection.csv", index=False)
print("Saved lifetime table: arrhenius_lifetime_projection.csv")

# ---------------------------------------------------------------
# 4. Dual-panel visualization
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# (a) Arrhenius (ln τ vs 1/T)
invT = 1 / temps
axes[0].plot(invT, np.log(tau_values), "o-", color="firebrick", lw=2)
axes[0].set_xlabel("1 / Temperature (1/K)")
axes[0].set_ylabel("ln(τ) [s]")
axes[0].set_title("(a) Arrhenius Lifetime Projection", fontweight="bold")
axes[0].grid(True, ls="--", alpha=0.6)
axes[0].invert_xaxis()

# (b) Lifetime vs Temperature (log scale)
axes[1].plot(temps, tau_years, "o-", color="navy", lw=2)
axes[1].set_yscale("log")
axes[1].set_xlabel("Temperature (K)")
axes[1].set_ylabel("Lifetime (years)")
axes[1].set_title("(b) Projected Retention Lifetime vs Temperature", fontweight="bold")
axes[1].grid(True, ls="--", alpha=0.6)

plt.tight_layout()
plt.savefig("arrhenius_lifetime_projection.png", dpi=400, bbox_inches="tight")
plt.savefig("arrhenius_lifetime_projection.pdf", dpi=400, bbox_inches="tight")

print("Figures saved:")
print("  - arrhenius_lifetime_projection.png")
print("  - arrhenius_lifetime_projection.pdf")
print("-" * 80)

# ---------------------------------------------------------------
# 5. Summary printout
# ---------------------------------------------------------------
print("Projected Retention Lifetime Summary:")
print(df_life.to_string(index=False, formatters={
    "Lifetime_s": "{:.3e}".format,
    "Lifetime_h": "{:.3e}".format,
    "Lifetime_years": "{:.3e}".format
}))

print("\nKey Insights:")
print(f" • Activation energy {Ea:.3f} eV → thermally activated retention degradation.")
print(f" • Lifetime drops exponentially with temperature (Arrhenius trend).")
print(f" • Data exported for direct inclusion in reliability analysis section.")
print("=" * 80)
