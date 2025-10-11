# =====================================================================
#  experiment13b_reliability_map_final.py
#  Publication-grade reliability + acceleration dual-panel visualization
#  Integrates:
#     - Exp.10b: Arrhenius fit (Ea, A)
#     - Exp.12b: Self-heating bias data
#     - Exp.9 : Endurance window degradation
#  Outputs: exp13b_dualpanel_reliability.[png|pdf] + CSV summaries
# =====================================================================

import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Console safety (Windows UTF-8)
# ---------------------------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Constants
kB = 8.617333262e-5  # eV/K
line = lambda ch="=", n=80: ch * n

print(line())
print("EXPERIMENT 13b: Endurance–Thermal Reliability Map (Final Color Edition)")
print(line())
print("Goal: Combine Arrhenius (Ea,A) + Self-Heating (Exp.12b) + Endurance (Exp.9)")
print("Adds Bias–Acceleration panel with color-coded scientific styling.")
print("-" * 80)

# ---------------------------------------------------------------------
# [1] Load Arrhenius Parameters
# ---------------------------------------------------------------------
arrhenius_file = "arrhenius_fit_results.csv"
if not os.path.exists(arrhenius_file):
    raise FileNotFoundError("Missing arrhenius_fit_results.csv from Exp.10b.")

df_arr = pd.read_csv(arrhenius_file)
Ea = float(df_arr.loc[df_arr["Parameter"].str.contains("Ea"), "Value"].values[0])
A = float(df_arr.loc[df_arr["Parameter"].str.contains("Intercept"), "Value"].values[0])
R2 = float(df_arr.loc[df_arr["Parameter"].str.contains("R_squared"), "Value"].values[0])
print(f"Arrhenius parameters: Ea={Ea:.4f} eV, A={A:.4f}, R²={R2:.5f}")

# ---------------------------------------------------------------------
# [2] Load Endurance Data
# ---------------------------------------------------------------------
endurance_file = "endurance_cycling_data.csv"
if os.path.exists(endurance_file):
    end_df = pd.read_csv(endurance_file)
    if "Cycle" in end_df.columns and any("R_window" in c for c in end_df.columns):
        try:
            first_fail = end_df[end_df["R_window"] < 10]["Cycle"].iloc[0]
            endurance_hours = first_fail * 0.024  # assume 1 cycle = 86.4 s
            print(f"Endurance: first cycle below 10 ≈ {first_fail:.0f} → {endurance_hours:.3e} h")
        except IndexError:
            endurance_hours = np.nan
            print("Endurance: no failure detected or data unavailable.")
    else:
        endurance_hours = np.nan
        print(f"! Endurance CSV missing expected columns. Found: {list(end_df.columns)}")
else:
    endurance_hours = np.nan
    print("! Endurance file not found; skipping endurance contribution.")

# ---------------------------------------------------------------------
# [3] Load Self-Heating Data (Exp. 12b)
# ---------------------------------------------------------------------
self_files = sorted(glob.glob("self_heating_*V.csv"))
if not self_files:
    raise FileNotFoundError("No self_heating_*V.csv files found (from Exp. 12b).")

heating_data = []
for f in self_files:
    bias_str = os.path.basename(f).replace("self_heating_", "").replace("V.csv", "")
    try:
        bias = float(bias_str)
    except ValueError:
        continue

    df = pd.read_csv(f)
    if "Temperature_K" not in df.columns:
        print(f"  • Skipping {f} (no Temperature_K column)")
        continue
    T95 = np.percentile(df["Temperature_K"], 95)
    heating_data.append((bias, T95, f))
    print(f"  • {bias:>4.2f} V  |  T₉₅ = {T95:.2f} K  |  {f}")

if not heating_data:
    raise RuntimeError("No valid self-heating CSVs found with Temperature_K column.")

biases, T95s, _ = zip(*heating_data)
biases, T95s = np.array(biases), np.array(T95s)

# ---------------------------------------------------------------------
# [4] Compute Lifetime Projections
# ---------------------------------------------------------------------
def lifetime_hours(T):
    ln_tau = A + Ea / (kB * T)
    return np.exp(ln_tau) / 3600  # convert s → h

retention_h = lifetime_hours(T95s)
endurance_h = np.full_like(retention_h, endurance_hours)
combined_h = np.minimum(retention_h, endurance_h)

results = pd.DataFrame({
    "Bias_V": biases,
    "T95_K": T95s,
    "Retention_h": retention_h,
    "Endurance_h": endurance_h,
    "Combined_h": combined_h,
})
results.to_csv("exp13b_reliability_map.csv", index=False)

# ---------------------------------------------------------------------
# [5] Compute Acceleration Factors
# ---------------------------------------------------------------------
base = retention_h[0]
AF = base / retention_h
af_df = pd.DataFrame({"Bias_V": biases, "T95_K": T95s, "Accel_Factor": AF})
af_df.to_csv("exp13b_acceleration_factors.csv", index=False)

# ---------------------------------------------------------------------
# [6] Generate Dual-Panel Figure (Color-Enhanced)
# ---------------------------------------------------------------------
plt.rcParams.update({
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "font.family": "DejaVu Sans",
})

fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2))

# --- Left: Reliability Map ---
axes[0].plot(biases, retention_h, "o-", color="#1f77b4", lw=2.5, label="Retention (Arrhenius)")
axes[0].plot(biases, endurance_h, "s--", color="#ff7f0e", lw=2.2, label="Endurance (Cycles)")
axes[0].plot(biases, combined_h, "D-", color="#2ca02c", lw=2.2, label="Combined Lifetime")

axes[0].set_yscale("log")
axes[0].set_xlabel("Bias Voltage (V)")
axes[0].set_ylabel("Lifetime (hours)")
axes[0].set_title("Self-Heating Accelerated Reliability Map", fontweight="bold")
axes[0].grid(True, alpha=0.35, which="both")
axes[0].legend(frameon=False)

# --- Right: Acceleration Factor ---
axes[1].semilogy(biases, AF, "r-o", lw=2.5, markersize=7, label="Acceleration Factor")
axes[1].set_xlabel("Bias Voltage (V)")
axes[1].set_ylabel("Acceleration Factor (τ₀.₀₅V / τ_V)", color="r")
axes[1].tick_params(axis="y", colors="r")
axes[1].set_title("Bias-Induced Lifetime Acceleration", fontweight="bold")
axes[1].grid(True, which="both", alpha=0.35)
axes[1].legend(frameon=False, loc="upper left")

plt.tight_layout()
plt.savefig("exp13b_dualpanel_reliability.png", dpi=600, bbox_inches="tight")
plt.savefig("exp13b_dualpanel_reliability.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------
# [7] Summary
# ---------------------------------------------------------------------
print("\n" + line())
print("RESULTS SUMMARY")
print(line())
for b, T, r, e, c in zip(biases, T95s, retention_h, endurance_h, combined_h):
    print(f"  V={b:>4.2f} V | T₉₅={T:6.2f} K | Ret={r:.3e} h | End={e:.3e} h | Comb={c:.3e} h")
print(line())
print("Notes:")
print(" • Blue–Orange–Green palette for reliability components.")
print(" • Red acceleration factor visually links to bias stress.")
print(" • Publication-grade composite exported as exp13b_dualpanel_reliability.[png|pdf].")
print(line())
print("EXPERIMENT 13b + 13c COMPLETE — FINAL MANUSCRIPT VERSION")
print(line())
