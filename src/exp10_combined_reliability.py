# ======================================================================
#  experiment10_long_term_reliability.py  —  FULL SCRIPT
#  EXPERIMENT 10: Long-Term Reliability Modeling
#  Coupled effects: Temperature (T), endurance cycles (N), retention (τ)
# ======================================================================

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Make console robust on Windows (avoid UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# If your PINN stack is available, you can enable it; otherwise the
# synthetic fallback below will be used (keeps this script self-contained).
USE_PINN = False
try:
    if not USE_PINN:
        raise ImportError
    from mainPINNmodel import PrintedMemristorPINN
    from TrainingFrameworkwithNoiseInjection import PINNTrainer
except Exception:
    USE_PINN = False

print("="*80)
print("EXPERIMENT 10: Long-Term Reliability Modeling (T × Cycles × Retention)")
print("="*80)
print("\nObjective: Jointly quantify endurance window collapse and temperature-")
print("           accelerated retention loss; produce line plots + heatmaps.")
print("Protocol:  Program ON/OFF baselines → sweep cycles & temperature →")
print("           compute R_OFF/R_ON and retention drop at fixed time.")
print("-"*80)

# -----------------------------------------------------------------------------
# Parameters (feel free to tweak for your device family)
# -----------------------------------------------------------------------------
V_read = 0.2                        # read voltage (V)
T_list = np.array([250, 275, 300, 325, 350, 375])  # Kelvin
N_cycles = 200                      # endurance range (0..N)
cycles = np.arange(0, N_cycles + 1)

# Baselines at nominal T0 (300 K), will be set from PINN or synthetic
T0 = 300.0                          # reference temperature (K)

# Synthetic baseline if PINN not used
I_on0_synth  = 3.0e-3               # 3 mA (typical ON current at V_read)
I_off0_synth = 1.2e-4               # 120 µA (OFF at V_read)

# Endurance (fatigue) law parameters
#  - ON current decreases with cycles (filament thinning/rupture)
#  - OFF current increases slightly (trap generation/leak)
C_on   = 350.0   # e-folding cycles for ON-current decay
b_off  = 0.08    # OFF log-growth factor
#   I_on(N)  = I_on0 * exp(-N/C_on)
#   I_off(N) = I_off0 * (1 + b_off * log10(1+N))

# Temperature scaling (conductance increases with T slightly)
alpha_T = 3.5e-3  # per Kelvin (empirical mild slope)
#   I(T) = I(T0) * exp[alpha_T * (T - T0)]

# Retention model (stretched exponential)
#   I_ret(t) = I_inf + (I0 - I_inf) * exp(-(t/τ)^β)
# We report normalized drop at a fixed observation time t_obs.
t_obs = 1.0e5     # seconds (~1.16 days)
beta  = 0.9       # stretch factor, 0.7–1 typical

# Arrhenius acceleration for retention time constant τ(T)
#   τ(T) = τ0 * exp[ Ea/kB * (1/T - 1/T0) ]
kB = 8.617333262145e-5  # eV/K
Ea_eV = 0.45            # activation energy (eV) — tune to your tech
tau0 = 3.0e6            # s at T0 (≈ 34.7 days baseline)

# -----------------------------------------------------------------------------
# Optional: get baseline ON/OFF from PINN (if available)
# -----------------------------------------------------------------------------
def get_baselines_from_pinn():
    """Train a small PINN and extract ON/OFF currents at V_read."""
    print("\n[PINN] Initializing & training…")
    pinn = PrintedMemristorPINN(
        hidden_layers=4, neurons_per_layer=128,
        input_features=("voltage", "state"), random_seed=42,
        trainable_params=("ohmic_conductance",)
    )
    trainer = PINNTrainer(pinn, learning_rate=2e-4, seed=42, state_mixing=0.2)
    V, I, X, _ = trainer.load_experimental_data(
        "printed_memristor_training_data.csv",
        concentration_label="10_percent_PMMA", device_id=0,
        use_noisy_columns=False
    )
    trainer.train(
        epochs=600, voltage=V, current=I, state=X,
        noise_std=0.0, variability_bound=0.0,
        verbose_every=200, max_physics_weight=0.1
    )

    def measure_at_state(x_target):
        # one-step read (assumes model maps (V, x) -> I)
        inp = np.array([[V_read, float(x_target)]], dtype=np.float32)
        I_pred, _ = pinn.model(inp, training=False)
        return float(I_pred.numpy()[0, 0])

    I_on0  = measure_at_state(0.85)
    I_off0 = measure_at_state(0.10)
    print(f"[PINN] Baselines at {T0:.0f} K: I_on0={I_on0*1e6:.1f} uA, I_off0={I_off0*1e6:.1f} uA")
    return I_on0, I_off0

if USE_PINN:
    try:
        I_on0_nom, I_off0_nom = get_baselines_from_pinn()
    except Exception as e:
        print(f"[WARN] PINN baseline failed, falling back to synthetic. Details: {e}")
        I_on0_nom, I_off0_nom = I_on0_synth, I_off0_synth
else:
    I_on0_nom, I_off0_nom = I_on0_synth, I_off0_synth

# -----------------------------------------------------------------------------
# Helper models (endurance, temperature, retention)
# -----------------------------------------------------------------------------
def endurance_update(I_on0, I_off0, N):
    """Return (I_on(N), I_off(N)) from endurance fatigue laws."""
    I_onN  = I_on0 * np.exp(-N / C_on)
    I_offN = I_off0 * (1.0 + b_off * np.log10(1.0 + N))
    return I_onN, I_offN

def temperature_scale(I, T):
    """Scale current for temperature."""
    return I * np.exp(alpha_T * (T - T0))

def arrhenius_tau(T):
    """Arrhenius dependence for retention time constant."""
    return tau0 * np.exp((Ea_eV / kB) * (1.0/T - 1.0/T0))

def retention_drop_fraction(t, tau_T, beta=0.9):
    """
    Fractional additional relaxation (0…1) by time t at temperature T.
    We use the classic stretched exponential, evaluated as the fraction
    of the ON/OFF gap that has relaxed by time t.
    """
    # Clamp to avoid overflow on extreme params
    x = np.clip((t / np.maximum(tau_T, 1e-6)) ** beta, 0.0, 700.0)
    return 1.0 - np.exp(-x)

# -----------------------------------------------------------------------------
# Simulation loops
# -----------------------------------------------------------------------------
print("\n[1/3] Building T×Cycles grids…")
T_grid, N_grid = np.meshgrid(T_list, cycles)  # shapes: (N_cycles+1, len(T_list))

print("[2/3] Evaluating endurance window & retention metrics…")
I_on_grid  = np.zeros_like(T_grid, dtype=float)
I_off_grid = np.zeros_like(T_grid, dtype=float)
Rwin_grid  = np.zeros_like(T_grid, dtype=float)  # R_OFF/R_ON ~ I_ON/I_OFF (at fixed V)
drop_grid  = np.zeros_like(T_grid, dtype=float)  # normalized retention drop (% of window)

for j, T in enumerate(T_list):
    tau_T = arrhenius_tau(T)
    drop_frac_T = retention_drop_fraction(t_obs, tau_T, beta=beta)  # scalar for that T

    for i, N in enumerate(cycles):
        # Endurance impact at T0
        I_onN, I_offN = endurance_update(I_on0_nom, I_off0_nom, N)
        # Temperature scaling for both states
        I_onTN  = temperature_scale(I_onN,  T)
        I_offTN = temperature_scale(I_offN, T)

        # Store currents
        I_on_grid[i, j]  = I_onTN
        I_off_grid[i, j] = I_offTN

        # Resistance window proxy at read (assuming R~V/I)
        Rwin_grid[i, j] = (I_onTN / np.maximum(I_offTN, 1e-12))

        # Retention: fraction of ON/OFF gap relaxed by t_obs at this T
        # We report the *normalized current* I/I0 ~ 1 - drop_frac over window.
        drop_grid[i, j] = drop_frac_T  # same for all cycles at given T (first-order)

print("[3/3] Exporting CSV and generating figures…")

# -----------------------------------------------------------------------------
# Save tidy CSV of the grid
# -----------------------------------------------------------------------------
out_rows = []
for i, N in enumerate(cycles):
    for j, T in enumerate(T_list):
        out_rows.append([
            int(N),
            float(T),
            float(I_on_grid[i, j]),
            float(I_off_grid[i, j]),
            float(Rwin_grid[i, j]),
            float(arrhenius_tau(T)),
            float(drop_grid[i, j])
        ])
out = np.array(out_rows, dtype=float)
header = "cycle,T_K,I_on_A,I_off_A,Rwin_Ion_over_Ioff,tau_s,retention_drop_frac_at_tobs"
np.savetxt("exp10_reliability_grid.csv", out, delimiter=",", header=header, comments='')
print("  • Saved: exp10_reliability_grid.csv")

# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11
})

# Figure 1: Lines — window vs cycles for several temperatures
fig1, ax = plt.subplots(1, 2, figsize=(11, 4.2))

# (a) Rwin vs cycles
for j, T in enumerate(T_list):
    ax[0].plot(cycles, Rwin_grid[:, j], label=f"{int(T)} K", linewidth=2)
ax[0].set_xlabel("Cycle number")
ax[0].set_ylabel("Resistance window  (R_OFF / R_ON)  ≈  I_ON / I_OFF")
ax[0].set_title("(a) Endurance Window vs. Cycles")
ax[0].grid(True, alpha=0.3)
ax[0].legend(title="Temperature")

# (b) Retention drop fraction vs temperature (at t_obs)
ax[1].plot(T_list, drop_grid[0, :]*100.0, "-o", linewidth=2)
ax[1].set_xlabel("Temperature (K)")
ax[1].set_ylabel(f"Retention drop at t={t_obs:.0f}s  (%)")
ax[1].set_title("(b) Retention Acceleration with Temperature")
ax[1].grid(True, alpha=0.3)

fig1.tight_layout()
fig1.savefig("exp10_reliability_lines.png", bbox_inches="tight")
plt.close(fig1)
print("  • Saved: exp10_reliability_lines.png")

# Figure 2: Heatmaps — window (cycles×T) and retention drop (cycles×T)
fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4.6))

im0 = ax2[0].imshow(Rwin_grid, aspect="auto", origin="lower",
                    extent=[T_list.min(), T_list.max(), cycles.min(), cycles.max()])
ax2[0].set_xlabel("Temperature (K)")
ax2[0].set_ylabel("Cycle number")
ax2[0].set_title("Heatmap: Resistance window (I_ON / I_OFF)")
cbar0 = fig2.colorbar(im0, ax=ax2[0])
cbar0.set_label("I_ON / I_OFF")

im1 = ax2[1].imshow(drop_grid*100.0, aspect="auto", origin="lower",
                    extent=[T_list.min(), T_list.max(), cycles.min(), cycles.max()],
                    vmin=0, vmax=max(0.1, drop_grid.max()*100))
ax2[1].set_xlabel("Temperature (K)")
ax2[1].set_ylabel("Cycle number")
ax2[1].set_title(f"Heatmap: Retention drop at t={int(t_obs)} s (%)")
cbar1 = fig2.colorbar(im1, ax=ax2[1])
cbar1.set_label("Drop (%)")

fig2.tight_layout()
fig2.savefig("exp10_reliability_heatmaps.png", bbox_inches="tight")
plt.close(fig2)
print("  • Saved: exp10_reliability_heatmaps.png")

# -----------------------------------------------------------------------------
# Console summary
# -----------------------------------------------------------------------------
def pct(x): return f"{100.0*x:.2f}%"

print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

# Window degradation factor (start→end) for each T
for T in T_list:
    j = np.where(T_list == T)[0][0]
    start = Rwin_grid[0, j]
    end   = Rwin_grid[-1, j]
    factor = end / max(start, 1e-12)
    print(f"  T={int(T)} K: window {start:.1f} → {end:.1f}  (×{factor:.3f})")

print("\nRetention (normalized drop) at t_obs:")
for T in T_list:
    j = np.where(T_list == T)[0][0]
    print(f"  T={int(T)} K: drop = {pct(drop_grid[0, j])}")

print("\nFigures saved:")
print("  - exp10_reliability_lines.png")
print("  - exp10_reliability_heatmaps.png")
print("Data saved:")
print("  - exp10_reliability_grid.csv")

print("\n" + "="*80)
print("EXPERIMENT 10 COMPLETE")
print("="*80)
