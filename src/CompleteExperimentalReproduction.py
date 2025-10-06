from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from ExperimentalValidationFramework import ExperimentalValidator
from ResultsVisualization import plot_experimental_results
from TrainingFrameworkwithNoiseInjection import PINNTrainer
from VTEAMModelComparison import VTEAMModel
from mainPINNmodel import PrintedMemristorPINN


DEFAULT_DATASET = Path(__file__).with_name("printed_memristor_training_data.csv")


def _train_test_split(
    *arrays: np.ndarray,
    ratio: float = 0.8,
    seed: int = 42,
) -> tuple[np.ndarray, ...]:
    if not arrays:
        raise ValueError("At least one array required for train/test split.")
    length = len(arrays[0])
    for arr in arrays:
        if len(arr) != length:
            raise ValueError("All arrays must share the same length for splitting.")
    rng = np.random.default_rng(seed)
    indices = np.arange(length)
    rng.shuffle(indices)
    split_idx = int(ratio * length)
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    splitted: list[np.ndarray] = []
    for arr in arrays:
        splitted.append(arr[train_idx])
        splitted.append(arr[test_idx])
    return tuple(splitted)


def _persist_metrics(metrics: dict[str, float | int | str], results_dir: Path | str) -> None:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "metrics.json"
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    csv_path = results_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])


def run_complete_experiments(
    dataset_path: Path | str = DEFAULT_DATASET,
    concentration_label: Optional[str] = "10_percent_PMMA",
    device_id: Optional[int] = 0,
    epochs: int = 800,
    noise_std: float = 0.002,
    variability_bound: float = 0.05,
    learning_rate: float = 2e-4,
    verbose_every: int = 50,
    max_physics_weight: float = 0.1,
    hidden_layers: int = 4,
    neurons_per_layer: int = 128,
    state_mixing: float = 0.2,
    seed: int = 42,
    trainable_params: tuple[str, ...] | None = ("ohmic_conductance",),
    use_concentration_feature: bool = False,
    results_dir: Path | str = Path("results"),
    show_plots: bool = False,
) -> dict[str, object]:
    print("=== Printed Memristor PINN Framework Demonstration ===")
    print("Using physics-based synthetic validation data")

    input_features = ("voltage", "state")
    if use_concentration_feature:
        input_features = ("voltage", "state", "concentration")

    pinn = PrintedMemristorPINN(
        hidden_layers=hidden_layers,
        neurons_per_layer=neurons_per_layer,
        input_features=input_features,
        random_seed=seed,
        trainable_params=trainable_params,
    )
    trainer = PINNTrainer(
        pinn,
        learning_rate=learning_rate,
        seed=seed,
        state_mixing=state_mixing,
    )
    dataset_path = Path(dataset_path)

    print("\n1. Loading experimental dataset...")
    voltage, current, state, concentration = trainer.load_experimental_data(
        dataset_path,
        concentration_label=concentration_label,
        device_id=device_id,
        use_noisy_columns=True,
    )
    print(f"Loaded {voltage.size} samples from {dataset_path.name}.")

    (
        V_train,
        V_test,
        I_train,
        I_test,
        x_train,
        x_test,
        c_train,
        c_test,
    ) = _train_test_split(voltage, current, state, concentration, ratio=0.8, seed=seed)

    print("\n2. Training PINN model...")
    loss_history = trainer.train(
        epochs=epochs,
        voltage=V_train,
        current=I_train,
        state=x_train,
        concentration=c_train if use_concentration_feature else None,
        noise_std=noise_std,
        variability_bound=variability_bound,
        verbose_every=verbose_every,
        max_physics_weight=max_physics_weight,
    )
    
    # EXTRACT AND PRINT TRAINING TIME
    training_time_min = 0.0
    if loss_history:
        training_time_min = loss_history[-1].get('training_time_minutes', 0.0)
        print(f"Training completed in {training_time_min:.2f} minutes")

    validator = ExperimentalValidator(pinn, seed=seed)
    print("\n3. Evaluating PINN and baseline models...")
    I_pinn_pred, pinn_inference_time = validator.predict_current(
        V_test,
        x_test,
        concentration=c_test if use_concentration_feature else None,
    )
    
    # CALCULATE AND PRINT INFERENCE TIME
    inference_time_per_1000_pts = (pinn_inference_time / V_test.size) * 1000
    print(f"PINN inference: {inference_time_per_1000_pts*1000:.2f} ms per 1000 points")
    
    vteam = VTEAMModel()
    I_vteam_pred, vteam_inference_time = vteam.simulate_iv(V_test)
    vteam_inference_per_1000 = (vteam_inference_time / V_test.size) * 1000
    print(f"VTEAM inference: {vteam_inference_per_1000*1000:.2f} ms per 1000 points")

    pinn_rrmse = validator.calculate_rrmse(I_pinn_pred, I_test)
    vteam_rrmse = validator.calculate_rrmse(I_vteam_pred, I_test)
    print(f"PINN RRMSE on hold-out set: {pinn_rrmse:.3f}")
    print(f"VTEAM RRMSE on hold-out set: {vteam_rrmse:.3f}")

    print("\n4. Noise robustness analysis...")
    noise_levels = [0.01, 0.02, 0.05, 0.1]
    robustness_results = validator.calculate_noise_robustness(
        V_test,
        I_pinn_pred,
        noise_levels,
        state=x_test,
        concentration=c_test if use_concentration_feature else None,
    )
    for noise_std_level, rrmse in robustness_results.items():
        print(f"Noise {noise_std_level * 100:.1f}% -> Robustness RRMSE {rrmse:.3f}")

    print("\n5. Statistical variability studies...")
    stats_results = validator.statistical_validation(n_cycles=100)
    set_mu, set_sigma = stats_results["set_voltage_distribution"]
    print(f"SET Voltage Mean {set_mu:.3f} V | Std {set_sigma:.3f} V")
    print(f"ON Resistance CoV {stats_results['on_resistance_cv']:.3f}")

    final_history = loss_history[-1] if loss_history else {}
    trainable_param_values = pinn.get_trainable_param_values()
    improvement = vteam_rrmse / pinn_rrmse if pinn_rrmse > 0 else float("inf")

    metrics_summary: dict[str, float | int | str] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "seed": int(seed),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "noise_std": float(noise_std),
        "variability_bound": float(variability_bound),
        "max_physics_weight": float(max_physics_weight),
        "hidden_layers": int(hidden_layers),
        "neurons_per_layer": int(neurons_per_layer),
        "state_mixing": float(state_mixing),
        "use_concentration_feature": bool(use_concentration_feature),
        "trainable_params": ",".join(trainable_params) if trainable_params else "",
        "train_samples": int(V_train.size),
        "test_samples": int(V_test.size),
        "training_time_minutes": float(training_time_min),
        "pinn_inference_ms_per_1000pts": float(inference_time_per_1000_pts * 1000),
        "vteam_inference_ms_per_1000pts": float(vteam_inference_per_1000 * 1000),
        "pinn_rrmse": float(pinn_rrmse),
        "vteam_rrmse": float(vteam_rrmse),
        "improvement_factor": float(improvement),
        "set_voltage_mean": float(set_mu),
        "set_voltage_std": float(set_sigma),
        "on_resistance_cv": float(stats_results["on_resistance_cv"]),
        "final_total_loss": float(final_history.get("total_loss", float("nan"))),
        "final_data_loss": float(final_history.get("data_loss", float("nan"))),
        "final_physics_loss": float(final_history.get("physics_loss", float("nan"))),
        "final_lambda_physics": float(final_history.get("lambda_physics", float("nan"))),
    }
    for key, value in robustness_results.items():
        metrics_summary[f"noise_rrmse_{key:.3f}"] = float(value)
    for key, value in trainable_param_values.items():
        metrics_summary[f"param_{key}"] = float(value)

    results_dir = Path(results_dir)
    _persist_metrics(metrics_summary, results_dir)

    results: dict[str, object] = {
        "pinn": pinn,
        "trainer": trainer,
        "validator": validator,
        "loss_history": loss_history,
        "voltage_train": V_train,
        "current_train": I_train,
        "state_train": x_train,
        "concentration_train": c_train,
        "voltage_test": V_test,
        "current_test": I_test,
        "state_test": x_test,
        "concentration_test": c_test,
        "pinn_pred_test": I_pinn_pred,
        "vteam_pred_test": I_vteam_pred,
        "pinn_rrmse": pinn_rrmse,
        "vteam_rrmse": vteam_rrmse,
        "robustness_results": robustness_results,
        "statistical_results": stats_results,
        "noise_levels": noise_levels,
        "hidden_layers": hidden_layers,
        "neurons_per_layer": neurons_per_layer,
        "metrics_summary": metrics_summary,
    }

    print("\n6. Generating figures...")
    plot_experimental_results(results, output_dir=results_dir, show=show_plots)

    print("\n=== Summary ===")
    print(f"PINN RRMSE: {pinn_rrmse:.3f}")
    print(f"VTEAM RRMSE: {vteam_rrmse:.3f}")
    print(f"Improvement factor over VTEAM: {improvement:.2f}x")
    print(f"Noise robustness (2%): {robustness_results[0.02]:.3f}")
    print(f"Training time: {training_time_min:.2f} minutes")
    print(f"PINN inference: {inference_time_per_1000_pts*1000:.2f} ms/1000pts")
    print(f"VTEAM inference: {vteam_inference_per_1000*1000:.2f} ms/1000pts")

    return results


if __name__ == "__main__":
    run_complete_experiments()