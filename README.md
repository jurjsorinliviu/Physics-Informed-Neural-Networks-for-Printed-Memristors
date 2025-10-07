# Physics-Informed-Neural-Networks-for-Printed-Memristors

This repository contains the full framework, dataset, and scripts to reproduce the experiments from:  

**“Physics-Informed Neural Networks for Compact Modeling of Printed Memristors: A Generalizable Framework."**  - a paper soon to be submitted to a journal

We propose the first **Physics-Informed Neural Network (PINN)** framework tailored for printed memristors, incorporating variability, noise robustness, and multi-mechanism conduction.  

---

## 📂 Repository Structure

```
printed-memristor-pinn/
│
├── data/
│   └── printed_memristor_training_data.csv      # Pre-generated synthetic dataset
│
├── src/
│   ├── generate_synthetic_data.py               # Dataset generator
│   ├── mainPINNmodel.py                         # PINN architecture + physics-informed loss
│   ├── TrainingFrameworkwithNoiseInjection.py   # Training utilities (variability + noise)
│   ├── ExperimentalValidationFramework.py       # Evaluation metrics (RRMSE, robustness, variability)
│   ├── VTEAMModelComparison.py                  # VTEAM baseline implementation
│   ├── ResultsVisualization.py                  # Plotting utilities (I–V curves, distributions, metrics)
│   ├──CompleteExperimentalReproduction.py       # Orchestrates full experiments
│   ├──ExtendedValidation.py                     # Extended Validation experiments
│   ├──balanced_simulation.py                    # Circuit Simulation
│   ├──export_pinn_to_spice.py                   # LUT export
│   └── run_pinn.py                              # Main entry point
│
├── results/   # Contains example outputs (metrics, plots) and is populated with new results when running the code
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/jurjsorinliviu/Physics-Informed-Neural-Networks-for-Printed-Memristors.git
cd printed-memristor-pinn
pip install -r requirements.txt
```

**Minimal requirements:**

```
numpy
pandas
matplotlib
scipy
tensorflow>=2.9
```

---

## 📊 Dataset

A **pre-generated dataset** (`printed_memristor_training_data.csv`) is provided in `data/`.  
It contains 20,000 voltage–current pairs across 4 PMMA concentrations (5%, 10%, 15%, 20%).  

To **regenerate the dataset** from scratch:

```bash
python src/generate_synthetic_data.py --output data/printed_memristor_training_data.csv
```

Options:  

- `--samples` : number of samples (default: 20,000)  
- `--noise-std` : Gaussian noise level  
- `--variability` : device-to-device variability factor  
- `--concentrations` : PMMA concentrations (default: 5,10,15,20%)  

---

## ▶️ Usage

All experiments are run through `run_pinn.py`.  

### 1. **Best single experiment** (seed 42)

```bash
python src/run_pinn.py --mode full   --full-epochs 800 --full-hidden-layers 4 --full-neurons 128   --full-learning-rate 2e-4 --full-noise-std 0.002   --full-variability 0.05 --full-max-physics-weight 0.1   --full-trainable-params ohmic_conductance   --full-disable-concentration   --full-seed 42   --results-dir results_final_best
```

### 2. **Cross-validation** (3 seeds)

```bash
python src/run_pinn.py --mode full   --full-epochs 800 --full-hidden-layers 4 --full-neurons 128   --full-learning-rate 2e-4 --full-noise-std 0.002   --full-variability 0.05 --full-max-physics-weight 0.1   --full-trainable-params ohmic_conductance   --full-disable-concentration   --full-repeats 3 --full-seed 40   --results-dir results_final_cv --no-plots
```

### 3. **Ablation study** (no physics term, fixed params)

```bash
python src/run_pinn.py --mode full   --full-epochs 800 --full-hidden-layers 4 --full-neurons 128   --full-learning-rate 2e-4 --full-noise-std 0.002   --full-variability 0.05   --full-trainable-params ohmic_conductance   --full-disable-concentration   --full-seed 42   --results-dir results_ablation_no_physics_fixed
```

Results (metrics + plots) are saved in the directory passed to `--results-dir`.

---
## 🔎 Extended Validation (digitized experimental curves)

We evaluate generalization on digitized I–V curves from three published device classes:
- Inkjet-printed IGZO (Ag/IGZO/ITO)
- Aerosol-jet MoS₂ (Ag/MoS₂/Ag)
- Paper-based MoS₂/graphene (Graphene/MoS₂/Au)

Run:
```bash
python src/ExtendedValidation.py \
  --seeds 40 41 42 \
  --output-dir results/extended_validation
```
---
## CIRCUIT INTEGRATION GUIDE
Detailed guide for reproducing circuit simulation results.

**Overview**

The circuit integration demonstrates how trained PINN models can be deployed in circuit simulation environments by:
1. Exporting lookup tables (LUT)
2. Generating behavioral models
3. Simulating 1T1R memory cell transient response

**Step-by-Step Instructions**

Step 1: Train PINN Model
```bash
python CompleteExperimentalReproduction.py
```
Step 2: Export LUT and Generate Schematics
```bash
python export_pinn_to_spice.py
```
Outputs:

pinn_memristor_lut.txt - 500×50 grid (25,000 points)
lut_visualization.png - 3D surface + I-V slices
circuit_schematic.png - 1T1R cell diagram

The LUT file format:
* Format: V(V) x(normalized) I(A)
-2.000000 0.000000 -2.345e-05
-2.000000 0.020408 -2.356e-05
...
  
Step 3: Run Circuit Simulation
```bash
python balanced_simulation.py
```

Expected Console Output:
```bash
PINN Model (Physics-Informed, Gradual Switching):
  Initial state:    0.100
  Final state:      0.914
  State change:     0.814
  Peak current:     1500.00 uA
  Write energy:     94.22 pJ

VTEAM Model (Phenomenological, Threshold-Based):
  Initial state:    0.100
  Final state:      0.991
  State change:     0.891
  Peak current:     1500.00 uA
  Write energy:     133.63 pJ

Comparison:
  Energy ratio (PINN/VTEAM):    0.71x
  Switching time (PINN):        27.5 ns
  Switching time (VTEAM):       18.6 ns
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

**Integration with SPICE**

The exported LUT (pinn_memristor_lut.txt) can be used in SPICE via:

a) ngspice (table-based behavioral source):

```bash
* Load LUT and use PWL interpolation
.control
load pinn_memristor_lut.txt
...
.endc
```

b) Verilog-A (read LUT in analog block):
```bash
// Inside analog block
I(p,n) <+ interpolate(lut_data, V(p,n), state);
```

---
## 📈 Reproduced Results

- **4.1× improvement** in RRMSE compared to VTEAM baseline.  
- **Robust up to 10% noise levels (as tested)** in synthetic datasets.  
- **Variability-aware training reproduces the variability trends embedded in the synthetic dataset** (SET voltages, ON/OFF resistances).  
- Framework is **exportable to Verilog-A/SPICE compact models** for potential circuit-level integration.  

---
## 📜 License

This project is licensed under the MIT License.  
