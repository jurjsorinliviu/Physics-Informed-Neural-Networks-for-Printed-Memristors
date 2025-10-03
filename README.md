# Physics-Informed-Neural-Networks-for-Printed-Memristors

This repository contains the full framework, dataset, and scripts to reproduce the experiments from:  

**“Physics-Informed Neural Networks for Compact Modeling of Printed Memristors: A Generalizable Framework."**  - a paper soon to be submitted to a journal

We propose the first **Physics-Informed Neural Network (PINN)** framework tailored for printed memristors, incorporating variability, noise robustness, and multi-mechanism conduction. We outline a pathway to a Verilog-A compact model for use in SPICE-class simulators, with implementation and solver-stability validation left for future work.  

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
│   ├──CompleteExperimentalReproduction.py      # Orchestrates full experiments
│   ├──ExtendedValidation.py                     # Extended Validation experiments
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
## 📈 Reproduced Results

- **4.1× improvement** in RRMSE compared to VTEAM baseline.  
- **Robust up to 10% noise levels (as tested)** in synthetic datasets.  
- **Variability-aware training reproduces the variability trends embedded in the synthetic dataset** (SET voltages, ON/OFF resistances).  
- Framework is **exportable to Verilog-A/SPICE compact models** for potential circuit-level integration.  

---
## 📜 License

This project is licensed under the MIT License.  
