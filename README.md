# Physics-Informed-Neural-Networks-for-Printed-Memristors

This repository contains the full framework, dataset, and scripts to reproduce the experiments from:

**"Physics-Informed Neural Networks for Compact Modeling of Printed Memristors: A Generalizable Framework"**  

📄 **Status:** *Submitted for publication*

---

## 🌟 Highlights

We propose the **first Physics-Informed Neural Network (PINN) framework** tailored for printed memristors, incorporating:

- ✅ **4.1× accuracy improvement** over VTEAM baseline (RRMSE: 0.061 vs 0.251)
- ✅ **29% energy reduction** compared to phenomenological models
- ✅ **Device-to-device variability modeling** with parameter perturbation
- ✅ **Noise robustness**: 1.9-4.3× lower error under 10% measurement corruption
- ✅ **Multi-mechanism conduction**: Ohmic, SCLC, and interfacial transport physics
- ✅ **Temperature-dependent reliability**: Arrhenius lifetime projections (E_a = 0.379 eV)
- ✅ **15 comprehensive validation experiments** spanning dynamics, reliability, and lifetime
- ✅ **SPICE-compatible export** via lookup tables for circuit integration

---

## 📂 Repository Structure

```
printed-memristor-pinn/
│
├── data/
│   ├── printed_memristor_training_data.csv      # Pre-generated synthetic dataset
│
├── src/
│   ├── generate_synthetic_data.py               # Dataset generator
│   ├── mainPINNmodel.py                         # PINN architecture + physics-informed loss
│   ├── TrainingFrameworkwithNoiseInjection.py   # Training utilities (variability + noise)
│   ├── ExperimentalValidationFramework.py       # Evaluation metrics
│   ├── VTEAMModelComparison.py                  # VTEAM baseline implementation
│   ├── ResultsVisualization.py                  # Plotting utilities
│   ├── CompleteExperimentalReproduction.py      # Orchestrates full experiments
│   ├── ExtendedValidation.py                    # Extended validation experiments
│   ├── balanced_simulation.py                   # Circuit simulation
│   ├── export_pinn_to_spice.py                  # LUT export
│   ├── run_pinn.py                              # Main entry point
│   ├── exp1_dynamic_pulse_response.py       # Dynamic operation
│   ├── exp2_write_read_cycles.py            # Non-destructive reads
│   ├── exp3_energy_efficiency.py            # Energy comparison
│   ├── exp4_multicell_variability.py        # Device-to-device variability
│   ├── exp5_noise_robustness.py             # Noise tolerance
│   ├── exp6_temperature_switching.py        # Temperature-dependent I-V
│   ├── exp7_multilevel_retention.py         # Multi-level stability
│   ├── exp8_temperature_retention.py        # Thermal drift analysis
│   ├── exp9_endurance_cycling.py            # Cycling degradation
│   ├── exp10_combined_reliability.py        # Cycle-temperature mapping
│   ├── exp10_arrhenius_fit.py               # Helper: Arrhenius fitting
│   ├── exp11_arrhenius_lifetime.py          # Lifetime projections
│   ├── exp12_self_heating.py                # Electro-thermal coupling
│   ├── exp13_bias_lifetime.py               # Bias-accelerated aging
│   └── exp14_15_reliability_and_acceleration.py  # Bottleneck analysis
│
├── results/
│   ├── main manuscript folders with experiments # Core validation results
│   └── supplementary_experiments/               # Results and Data generated from additional 15 experiments
│
├── requirements.txt
├── README.md
└── LICENSE

```

---

## ⚙️ Installation

### **Clone and Install**
```bash
git clone https://github.com/jurjsorinliviu/Physics-Informed-Neural-Networks-for-Printed-Memristors.git
cd Physics-Informed-Neural-Networks-for-Printed-Memristors
pip install -r requirements.txt
```

**Requirements:**

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scipy>=1.7.0
tensorflow>=2.9.0
```
---

## 🚀 Quick Start

### **1. Train PINN Model (Best Configuration)**
```bash
python src/run_pinn.py --mode full \
  --full-epochs 800 \
  --full-hidden-layers 4 \
  --full-neurons 128 \
  --full-learning-rate 2e-4 \
  --full-noise-std 0.002 \
  --full-variability 0.05 \
  --full-max-physics-weight 0.1 \
  --full-trainable-params ohmic_conductance \
  --full-disable-concentration \
  --full-seed 42 \
  --results-dir results_best
```
**Expected Output**:
- Training converges in ~800 epochs
- Final RRMSE: 0.061 (vs VTEAM: 0.251)
- Results saved to results_best/

### **2. Cross-Validation (3 Seeds)**
```bash
python src/run_pinn.py --mode full \
  --full-epochs 800 \
  --full-hidden-layers 4 \
  --full-neurons 128 \
  --full-learning-rate 2e-4 \
  --full-noise-std 0.002 \
  --full-variability 0.05 \
  --full-max-physics-weight 0.1 \
  --full-trainable-params ohmic_conductance \
  --full-disable-concentration \
  --full-repeats 3 \
  --full-seed 40 \
  --results-dir results_cv \
  --no-plots
```
**Expected Output**:
- RRMSE: 0.115 ± 0.062 across 3 seeds
- Statistical validation of robustness

### **Ablation Study (No Physics Constraints)**
```bash
python src/run_pinn.py --mode full \
  --full-epochs 800 \
  --full-hidden-layers 4 \
  --full-neurons 128 \
  --full-learning-rate 2e-4 \
  --full-noise-std 0.002 \
  --full-variability 0.05 \
  --full-trainable-params ohmic_conductance \
  --full-disable-concentration \
  --full-seed 42 \
  --results-dir results_ablation
```
**Expected Output**:
- ~1.6× accuracy degradation without physics loss
- Validates importance of physics-informed constraints

---

## 🔬 Supplementary Experiments (n=15)

Comprehensive validation across dynamic operation, reliability, and lifetime projection.

### **Group 1: Dynamic Operation & Energy (Exp. 1-3)**
```bash
# Experiment 1: Dynamic pulse response (66.7 Hz, 50 pulses)
python src/exp1_dynamic_pulse_response.py
# Result: 34.5% current increase, analog potentiation validated

# Experiment 2: Write-read cycles (non-destructive reads)
python src/exp2_write_read_cycles.py
# Result: <3% CoV, stable read operations

# Experiment 3: Energy efficiency comparison
python src/exp3_energy_efficiency.py
# Result: 29% lower write energy than VTEAM (94.2 pJ vs 133.6 pJ)
```
### **Group 2: Variability & Robustness (Exp. 4-5)**
```bash
# Experiment 4: Multi-cell variability (5-device array)
python src/exp4_multicell_variability.py
# Result: 10× amplification of parameter uncertainty (40.83% CoV)

# Experiment 5: Noise robustness (1-10% corruption)
python src/exp5_noise_robustness.py
# Result: 1.9-4.3× lower error than baselines at 10% noise
```
### **Group 3: Temperature Physics (Exp. 6, 8)**
```bash
# Experiment 6: Temperature-dependent I-V (250-350 K)
python src/exp6_temperature_switching.py
# Result: 23.5% current modulation, E_a ≈ 0.09 eV

# Experiment 8: Temperature-dependent retention
python src/exp8_temperature_retention.py
# Result: Arrhenius ordering validated, <0.01% drift at 350 K
```
### **Group 4: Retention & Endurance (Exp. 7, 9)**
```bash
# Experiment 7: Multi-level retention (10⁶ s, 3 levels)
python src/exp7_multilevel_retention.py
# Result: <3% drift over 11.6 days, τ = 10⁶ s

# Experiment 9: Endurance cycling (200 SET/RESET)
python src/exp9_endurance_cycling.py
# Result: 29% window reduction, ~660-cycle lifetime
```
### **Group 5: Coupled Reliability & Lifetime (Exp. 10-15)**
```bash
# Experiment 10: Combined reliability mapping (cycle × temperature)
python src/exp10_combined_reliability.py
# Result: Orthogonal failure modes identified

# Experiment 11: Arrhenius lifetime projection
python src/exp11_arrhenius_lifetime.py
# Result: E_a = 0.379 ± 0.010 eV, R² = 0.997

# Experiment 12: Self-heating dynamics (0.05-0.20 V)
python src/exp12_self_heating.py
# Result: ΔT = 18-32 K, ~4× retention acceleration

# Experiment 13: Bias-accelerated lifetime
python src/exp13_bias_lifetime.py
# Result: 14-28% lifetime reduction with bias

# Experiments 14-15: Reliability bottleneck & acceleration
python src/exp14_15_reliability_and_acceleration.py
# Result: Endurance dominates (~4.8 h vs ~40 h retention)
```
**Outputs**: All results saved to results/supplementary_experiments/ with figures and CSV files.

---

## 🔎 Extended Validation (Digitized Experimental Curves)

Evaluate generalization on three published device classes:
```bash
python src/ExtendedValidation.py \
  --seeds 40 41 42 \
  --output-dir results/extended_validation
```
**Tested Devices**:
- ✅ Inkjet-printed IGZO (Ag/IGZO/ITO)
- ✅ Aerosol-jet MoS₂ (Ag/MoS₂/Ag)
- ✅ Paper-based MoS₂/graphene

**Result**: PINN achieves lowest error on MoS₂ and paper datasets, remains competitive on IGZO.

---

## 🔧 Circuit Integration

### **Step 1: Export PINN to SPICE-Compatible LUT**
```bash
python src/export_pinn_to_spice.py
```
**Outputs**:
- pinn_memristor_lut.txt (500×50 grid, 25,000 points)
- lut_visualization.png (3D surface + I-V slices)
- circuit_schematic.png (1T1R cell diagram)

### **Step 2: Run 1T1R Circuit Simulation**
```bash
python src/balanced_simulation.py
```
**Expected Console Output**:
```bash
PINN Model (Physics-Informed, Gradual Switching):
  Initial state:    0.100
  Final state:      0.914
  State change:     0.814
  Peak current:     1500.00 μA
  Write energy:     94.22 pJ

VTEAM Model (Phenomenological, Threshold-Based):
  Initial state:    0.100
  Final state:      0.991
  State change:     0.891
  Peak current:     1500.00 μA
  Write energy:     133.63 pJ

Comparison:
  Energy ratio (PINN/VTEAM):    0.71×
  Energy savings:                29%
```

### **Step 3: Integration with SPICE**
The exported LUT can be used in circuit simulators:

**ngspice (PWL interpolation):**
```bash
* Load LUT
.control
load pinn_memristor_lut.txt
...
.endc
```
**Verilog-A (analog block):**
```bash
// Inside analog block
I(p,n) <+ interpolate(lut_data, V(p,n), state);
```
**Customization**
Modify Circuit Parameters by editing balanced_simulation.py:
```bash
Line ~30-35: Memristor parameters
R_on = 1e3      # ON resistance (Ohm)
R_off = 100e3   # OFF resistance (Ohm)

Line ~38-43: Dynamics
alpha_pinn = 1e7   # PINN speed (increase for faster switching)
k_vteam = 1e8      # VTEAM speed

Line ~50-52: Voltage pulse
V_pulse = 1.5      # Pulse amplitude (V)
t_width = 100e-9   # Pulse duration (s)

Change Simulation Resolution
Line ~21: Time step
dt = 0.1e-9  # Decrease for finer resolution (but slower simulation)
```

---

## 📊 Dataset
### **Pre-Generated Dataset**

Training data (printed_memristor_training_data.csv) includes:
- 20,000 voltage-current pairs
- 4 PMMA concentrations: 5%, 10%, 15%, 20%
- Bipolar sweeps: -2.0 V to +2.0 V

### **Regenerate Dataset**
```bash
python src/generate_synthetic_data.py \
  --output data/printed_memristor_training_data.csv \
  --samples 20000 \
  --noise-std 0.01 \
  --variability 0.05 \
  --concentrations 5,10,15,20
```
**Options:**
- samples: Number of samples (default: 20,000)
- noise-std: Gaussian noise level
- variability: Device-to-device variation factor
- concentrations: PMMA concentrations (comma-separated)

---

## 📜 License

This project is licensed under the MIT License.  
