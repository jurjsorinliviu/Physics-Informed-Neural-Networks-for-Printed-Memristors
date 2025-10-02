from __future__ import annotations

import json
import pandas as pd
import argparse
import csv
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np

from CompleteExperimentalReproduction import DEFAULT_DATASET, run_complete_experiments
from ExperimentalValidationFramework import ExperimentalValidator
from ResultsVisualization import plot_cross_validation_results
from TrainingFrameworkwithNoiseInjection import PINNTrainer
from VTEAMModelComparison import VTEAMModel
from mainPINNmodel import PrintedMemristorPINN



def _build_results_dict(
    voltage,
    current,
    state,
    concentration,
    history,
    V_test,
    I_test,
    x_test,
    c_test,
    pinn_pred,
    vteam_pred,
) -> dict[str, object]:
    return {
        "voltage_train": voltage,
        "current_train": current,
        "state_train": state,
        "concentration_train": concentration,
        "loss_history": history,
        "voltage_test": V_test,
        "current_test": I_test,
        "state_test": x_test,
        "concentration_test": c_test,
        "pinn_pred_test": pinn_pred,
        "vteam_pred_test": vteam_pred,
    }


def _resolve_input_features(use_concentration: bool) -> tuple[str, ...]:
    return ("voltage", "state", "concentration") if use_concentration else ("voltage", "state")


def run_synthetic_demo(
    epochs: int,
    learning_rate: float,
    samples: int,
    test_samples: int,
    noise_std: float,
    variability: float,
    hidden_layers: int,
    neurons: int,
    verbose_every: int,
    max_physics_weight: float,
    seed: int,
    state_mixing: float,
    trainable_params: tuple[str, ...],
    use_concentration_feature: bool,
) -> dict[str, object]:
    print("=== Printed Memristor PINN Synthetic Demo ===")
    pinn = PrintedMemristorPINN(
        hidden_layers=hidden_layers,
        neurons_per_layer=neurons,
        input_features=_resolve_input_features(use_concentration_feature),
        random_seed=seed,
        trainable_params=trainable_params,
    )
    trainer = PINNTrainer(
        pinn,
        learning_rate=learning_rate,
        seed=seed,
        state_mixing=state_mixing,
    )

    V_train, I_train, x_train, c_train = trainer.generate_synthetic_data(n_samples=samples)
    history = trainer.train(
        epochs=epochs,
        voltage=V_train,
        current=I_train,
        state=x_train,
        concentration=c_train if use_concentration_feature else None,
        noise_std=noise_std,
        variability_bound=variability,
        verbose_every=verbose_every,
        max_physics_weight=max_physics_weight,
    )

    V_test, I_test, x_test, c_test = trainer.generate_synthetic_data(n_samples=test_samples)
    validator = ExperimentalValidator(pinn, seed=seed)
    pinn_pred = validator.predict_current(
        V_test,
        x_test,
        concentration=c_test if use_concentration_feature else None,
    )
    vteam = VTEAMModel()
    vteam_pred = vteam.simulate_iv(V_test)

    return _build_results_dict(
        V_train,
        I_train,
        x_train,
        c_train,
        history,
        V_test,
        I_test,
        x_test,
        c_test,
        pinn_pred,
        vteam_pred,
    )


def plot_synthetic_results(
    results: dict[str, object],
    output_path: Path | None = None,
    show: bool = True,
) -> None:
    loss_history = results.get("loss_history", [])
    if loss_history:
        epochs = [entry["epoch"] for entry in loss_history]
        total_loss = [entry["total_loss"] for entry in loss_history]
        plt.figure(figsize=(9, 5))
        plt.semilogy(epochs, total_loss)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(True, alpha=0.3)
        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path / "synthetic_training_loss.png", dpi=300)
        if show:
            plt.show()
        plt.close()

    V_test = results.get("voltage_test")
    I_test = results.get("current_test")
    pinn_pred = results.get("pinn_pred_test")
    vteam_pred = results.get("vteam_pred_test")
    if V_test is not None and pinn_pred is not None:
        plt.figure(figsize=(9, 5))
        plt.scatter(V_test, I_test, s=10, alpha=0.4, label="Synthetic", color="#3a7cbc")
        plt.plot(V_test, pinn_pred, label="PINN", linewidth=2.0, color="#d95f02")
        if vteam_pred is not None:
            plt.plot(V_test, vteam_pred, label="VTEAM", linestyle="--", linewidth=2.0)
        plt.xlabel("Voltage (V)")
        plt.ylabel("Current (A)")
        plt.title("Synthetic I-V Comparison")
        plt.grid(True, alpha=0.3)
        plt.legend()
        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path / "synthetic_iv.png", dpi=300)
        if show:
            plt.show()
        plt.close()


def aggregate_metrics(metrics_list: list[Mapping[str, float | int | str]]) -> dict[str, float | int | list[int]]:
    numeric_keys = {
        key
        for metrics in metrics_list
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    aggregate: dict[str, float | int | list[int]] = {
        "num_runs": len(metrics_list),
        "seeds": [int(metrics.get("seed", idx)) for idx, metrics in enumerate(metrics_list)],
    }
    for key in sorted(numeric_keys):
        values = [float(metrics[key]) for metrics in metrics_list if key in metrics]
        if not values:
            continue
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        ci = float(1.96 * std / np.sqrt(len(values))) if len(values) > 1 else 0.0
        aggregate[f"{key}_mean"] = mean
        aggregate[f"{key}_std"] = std
        aggregate[f"{key}_ci95"] = ci
    return aggregate


def persist_metrics_collection(
    metrics_list: list[Mapping[str, float | int | str]],
    aggregate: Mapping[str, float | int | list[int]],
    output_dir: Path | str,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_path = output_dir / "metrics_runs.csv"
    fieldnames = sorted({key for metrics in metrics_list for key in metrics.keys()})
    with runs_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metrics in metrics_list:
            writer.writerow(metrics)

    aggregate_path = output_dir / "metrics_aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    aggregate_csv_path = output_dir / "metrics_aggregate.csv"
    with aggregate_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in aggregate.items():
            writer.writerow([key, value])


def _parse_trainable_params(raw: Iterable[str] | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if not raw:
        return default
    return tuple(raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PINN experiments for printed memristors.")
    parser.add_argument("--mode", choices=["synthetic", "full"], default="synthetic", help="Synthetic demo or full experimental reproduction.")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs for the synthetic demo.")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Learning rate for the synthetic demo.")
    parser.add_argument("--samples", type=int, default=1000, help="Synthetic samples for training.")
    parser.add_argument("--test-samples", type=int, default=600, help="Synthetic samples for evaluation.")
    parser.add_argument("--noise-std", type=float, default=0.02, help="Relative noise level injected during training.")
    parser.add_argument("--variability", type=float, default=0.1, help="Maximum relative parameter variability.")
    parser.add_argument("--hidden-layers", type=int, default=3, help="Hidden layers in the PINN (synthetic demo).")
    parser.add_argument("--neurons", type=int, default=64, help="Neurons per hidden layer in the PINN (synthetic demo).")
    parser.add_argument("--verbose-every", type=int, default=50, help="Training log frequency for synthetic demo.")
    parser.add_argument("--max-physics-weight", type=float, default=0.3, help="Upper bound for physics loss weighting in the synthetic demo.")
    parser.add_argument("--trainable-params", nargs="*", default=["ohmic_conductance"], help="Physics parameters learned from data in the synthetic demo.")
    parser.add_argument("--use-concentration-feature", action="store_true", help="Include concentration as an additional input feature in the synthetic demo.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Directory to store outputs.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATASET, help="Dataset path for full experiments.")
    parser.add_argument("--concentration", type=str, default="10_percent_PMMA", help="Dataset concentration label.")
    parser.add_argument("--device-id", type=int, default=0, help="Device identifier for dataset filtering.")
    parser.add_argument("--full-epochs", type=int, default=600, help="Epochs for full experimental reproduction.")
    parser.add_argument("--full-max-physics-weight", type=float, default=0.3, help="Upper bound for physics loss weighting during full experiments.")
    parser.add_argument("--full-hidden-layers", type=int, default=3, help="Hidden layers for the full experiment PINN.")
    parser.add_argument("--full-neurons", type=int, default=64, help="Neurons per hidden layer for the full experiment PINN.")
    parser.add_argument("--no-plots", action="store_true", help="Disable matplotlib visualisation display.")
    parser.add_argument("--full-noise-std", type=float, default=0.02, help="Training noise level for full experiments.")
    parser.add_argument("--full-variability", type=float, default=0.15, help="Parameter variability for full experiments.")
    parser.add_argument("--full-learning-rate", type=float, default=5e-4, help="Learning rate for full experiments.")
    parser.add_argument("--full-verbose-every", type=int, default=50, help="Logging cadence for full experiments.")
    parser.add_argument("--full-repeats", type=int, default=1, help="Number of repeated runs with different seeds.")
    parser.add_argument("--full-seed", type=int, default=42, help="Base random seed for full experiments.")
    parser.add_argument("--full-state-mixing", type=float, default=0.2, help="State mixing coefficient for full experiments.")
    parser.add_argument("--full-trainable-params", nargs="*", default=["ohmic_conductance"], help="Physics parameters learned from data in full experiments.")
    parser.add_argument("--full-disable-concentration", action="store_true", help="Disable concentration feature in full experiments.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for synthetic demo.")
    parser.add_argument("--state-mixing", type=float, default=0.2, help="State mixing coefficient for synthetic data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "full":
        results_dir = Path(args.results_dir)
        metrics_runs: list[dict[str, float | int | str]] = []
        seeds = [args.full_seed + idx for idx in range(args.full_repeats)]
        trainable_params = _parse_trainable_params(args.full_trainable_params, ("ohmic_conductance",))
        use_concentration = not args.full_disable_concentration
        for seed in seeds:
            run_dir = results_dir if args.full_repeats == 1 else results_dir / f"seed_{seed}"
            results = run_complete_experiments(
                dataset_path=args.data_path,
                concentration_label=args.concentration,
                device_id=args.device_id,
                epochs=args.full_epochs,
                noise_std=args.full_noise_std,
                variability_bound=args.full_variability,
                learning_rate=args.full_learning_rate,
                verbose_every=args.full_verbose_every,
                max_physics_weight=args.full_max_physics_weight,
                hidden_layers=args.full_hidden_layers,
                neurons_per_layer=args.full_neurons,
                state_mixing=args.full_state_mixing,
                seed=seed,
                trainable_params=trainable_params,
                use_concentration_feature=use_concentration,
                results_dir=run_dir,
                show_plots=not args.no_plots,
            )
            metrics_runs.append(dict(results["metrics_summary"]))

        if len(metrics_runs) > 1:
            aggregate = aggregate_metrics(metrics_runs)
            persist_metrics_collection(metrics_runs, aggregate, results_dir)
            plot_cross_validation_results(metrics_runs, aggregate, results_dir, show=not args.no_plots)
            print("\n=== Cross-Validation Summary ===")
            print("Seeds:", aggregate.get("seeds", []))
            pinn_mean = aggregate.get("pinn_rrmse_mean", float("nan"))
            pinn_std = aggregate.get("pinn_rrmse_std", 0.0)
            pinn_ci = aggregate.get("pinn_rrmse_ci95", 0.0)
            vteam_mean = aggregate.get("vteam_rrmse_mean", float("nan"))
            vteam_std = aggregate.get("vteam_rrmse_std", 0.0)
            vteam_ci = aggregate.get("vteam_rrmse_ci95", 0.0)
            print(f"PINN RRMSE: {pinn_mean:.3f} ± {pinn_std:.3f} (95% CI ±{pinn_ci:.3f})")
            print(f"VTEAM RRMSE: {vteam_mean:.3f} ± {vteam_std:.3f} (95% CI ±{vteam_ci:.3f})")
        return

    trainable_params = _parse_trainable_params(args.trainable_params, ("ohmic_conductance",))
    results = run_synthetic_demo(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        samples=args.samples,
        test_samples=args.test_samples,
        noise_std=args.noise_std,
        variability=args.variability,
        hidden_layers=args.hidden_layers,
        neurons=args.neurons,
        verbose_every=args.verbose_every,
        max_physics_weight=args.max_physics_weight,
        seed=args.seed,
        state_mixing=args.state_mixing,
        trainable_params=trainable_params,
        use_concentration_feature=args.use_concentration_feature,
    )
    plot_synthetic_results(results, output_path=args.results_dir, show=not args.no_plots)



    # # Save results dictionary to JSON
    # with open("results_dump.json", "w") as f:
    #    json.dump(results, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

    # # Save the key arrays to CSV for plotting
    # results_df = pd.DataFrame({
    # "voltage_test": results["voltage_test"],
    # "current_test": results["current_test"],
    # "pinn_pred_test": results["pinn_pred_test"],
    # "vteam_pred_test": results["vteam_pred_test"],
    # })
    # results_df.to_csv("results_dump.csv", index=False)


if __name__ == "__main__":
    main()
