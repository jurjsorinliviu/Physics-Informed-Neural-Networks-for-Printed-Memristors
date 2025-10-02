import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

def generate_memristor_cycle(n_points=500, pmma_conc=10.0, device_id=0, cycle_id=0):
    """
    Generate one I-V cycle for a printed Ag/PMMA:PVA/ITO memristor
    Based on Strutwolf et al. 2021 characteristics
    """
    # Voltage sweep: -2V -> +2V -> -2V
    V_up = np.linspace(-2.0, 2.0, n_points // 2)
    V_down = np.linspace(2.0, -2.0, n_points - n_points // 2)
    voltage = np.concatenate([V_up, V_down])
    
    # Device-to-device and cycle-to-cycle variability
    # Higher PMMA content = slightly higher SET voltage, more stable
    base_vset = 0.9 + 0.02 * (pmma_conc / 10.0)
    base_vreset = -1.0 - 0.01 * (pmma_conc / 10.0)
    
    # Add stochastic variation
    vset = base_vset + np.random.normal(0, 0.1)
    vreset = base_vreset + np.random.normal(0, 0.1)
    
    # Resistance states (with variability)
    R_on = 1000 * (1.0 + np.random.normal(0, 0.05))  # ~1kΩ ±5%
    R_off = 100000 * (1.0 + np.random.normal(0, 0.1))  # ~100kΩ ±10%
    
    # Leakage conductance
    G_leak = 1e-6
    
    # Current generation with physical mechanisms
    current = np.zeros_like(voltage)
    state = 0.0  # Internal state (0=HRS, 1=LRS)
    
    for i, V in enumerate(voltage):
        if i < n_points // 2:  # Positive sweep
            # SET transition with SCLC-like behavior
            if V >= vset:
                # Gradual transition to LRS
                state = min(1.0, state + 0.02)
                # ON state with some nonlinearity
                I_on = V / R_on
                # Space-charge limited contribution
                I_sclc = 1e-9 * (V ** 2) if abs(V) > 0.1 else 0
                current[i] = I_on + I_sclc
            else:
                # HRS with ohmic leakage
                current[i] = G_leak * V + (state * V / R_on)
        else:  # Negative sweep  
            # RESET transition
            if V <= vreset:
                # Gradual transition to HRS
                state = max(0.0, state - 0.02)
                # Mainly ohmic with reducing conductance
                current[i] = G_leak * V + (state * V / R_on)
            else:
                # Maintain previous state
                if state > 0.5:  # Still in LRS
                    current[i] = V / R_on
                else:  # Back to HRS
                    current[i] = V / R_off
    
    # Add measurement noise (typical for printed devices)
    noise_std = 5e-7  # 0.5 µA
    current += np.random.normal(0, noise_std, size=current.shape)
    
    # Compute state variable (normalized)
    state_var = np.abs(current) / (np.max(np.abs(current)) + 1e-12)
    state_var = np.clip(state_var, 0, 1)
    
    return voltage, current, state_var

# Generate dataset with multiple concentrations and devices
data_records = []

concentrations = [5.0, 10.0, 15.0, 20.0]
concentration_labels = {
    5.0: "5_percent_PMMA",
    10.0: "10_percent_PMMA", 
    15.0: "15_percent_PMMA",
    20.0: "20_percent_PMMA"
}

devices_per_concentration = 5
cycles_per_device = 2  # Multiple cycles to show variability

for conc in concentrations:
    for dev_id in range(devices_per_concentration):
        for cycle_id in range(cycles_per_device):
            V, I, x = generate_memristor_cycle(
                n_points=500,
                pmma_conc=conc,
                device_id=dev_id,
                cycle_id=cycle_id
            )
            
            # Add noisy versions for training robustness
            V_noisy = V + np.random.normal(0, 0.01, size=V.shape)
            I_noisy = I + np.random.normal(0, 5e-7, size=I.shape)
            
            for i in range(len(V)):
                data_records.append({
                    'voltage': V[i],
                    'current': I[i],
                    'state': x[i],
                    'pmma_concentration': conc,
                    'concentration_label': concentration_labels[conc],
                    'device_id': dev_id,
                    'cycle_id': cycle_id,
                    'voltage_noisy': V_noisy[i],
                    'current_noisy': I_noisy[i]
                })

# Create DataFrame
df = pd.DataFrame(data_records)

# Save to CSV
df.to_csv('printed_memristor_training_data.csv', index=False)

print(f"Generated {len(df)} data points")
print(f"Concentrations: {concentrations}")
print(f"Devices per concentration: {devices_per_concentration}")
print(f"Cycles per device: {cycles_per_device}")
print(f"\nDataset summary:")
print(df.describe())
print(f"\nSample data:")
print(df.head(10))