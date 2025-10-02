from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _ensure_output_dir(path: Path | str) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def plot_experimental_results(
    results: Mapping[str, object],
    output_dir: Path | str = Path("results"),
    show: bool = False,
) -> None:
    output_dir = _ensure_output_dir(output_dir)

    loss_history = results.get("loss_history", [])
    if loss_history:
        epochs = [entry["epoch"] for entry in loss_history]
        total_loss = [entry["total_loss"] for entry in loss_history]
        data_loss = [entry["data_loss"] for entry in loss_history]
        physics_loss = [entry["physics_loss"] for entry in loss_history]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.semilogy(epochs, total_loss, label="Total Loss", linewidth=2.0)
        ax.semilogy(epochs, data_loss, label="Data Loss", linestyle="--", linewidth=2.0)
        ax.semilogy(epochs, physics_loss, label="Physics Loss", linestyle="--", linewidth=2.0)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("PINN Training Convergence")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "training_convergence.png", dpi=300)
        if show:
            plt.show()
        plt.close(fig)

        param_keys = [key for key in loss_history[0].keys() if key.startswith("param_")]
        if param_keys:
            fig, ax = plt.subplots(figsize=(9, 5))
            for key in param_keys:
                trace = [entry.get(key, np.nan) for entry in loss_history]
                ax.plot(epochs, trace, label=key.replace("param_", ""), linewidth=2.0)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Parameter value")
            ax.set_title("Trainable Physical Parameters")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / "trainable_parameters.png", dpi=300)
            if show:
                plt.show()
            plt.close(fig)

    V_test = np.asarray(results.get("voltage_test", []), dtype=float)
    I_test = np.asarray(results.get("current_test", []), dtype=float)
    pinn_pred = np.asarray(results.get("pinn_pred_test", []), dtype=float)
    vteam_pred = np.asarray(results.get("vteam_pred_test", []), dtype=float)
    if V_test.size and pinn_pred.size:
        fig, ax = plt.subplots(figsize=(9, 5))
        if I_test.size:
            ax.scatter(V_test, I_test, s=10, alpha=0.4, label="Measured", color="#3a7cbc")
        ax.plot(V_test, pinn_pred, label="PINN", linewidth=2.0, color="#d95f02")
        if vteam_pred.size:
            ax.plot(V_test, vteam_pred, label="VTEAM", linestyle="--", linewidth=2.0, color="#1b9e77")
        ax.set_xlabel("Voltage (V)")
        ax.set_ylabel("Current (A)")
        ax.set_title("I-V Characteristics Comparison")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "iv_comparison.png", dpi=300)
        if show:
            plt.show()
        plt.close(fig)

    stats = results.get("statistical_results", {})
    set_voltages = np.asarray(stats.get("set_voltages", []), dtype=float)
    on_resistances = np.asarray(stats.get("on_resistances", []), dtype=float)
    if set_voltages.size and on_resistances.size:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(set_voltages, bins=15, alpha=0.75, edgecolor="black")
        axes[0].set_xlabel("SET Voltage (V)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("SET Voltage Distribution")

        axes[1].hist(on_resistances, bins=15, alpha=0.75, edgecolor="black")
        axes[1].set_xlabel("ON Resistance (Ohm)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("ON Resistance Distribution")

        fig.tight_layout()
        fig.savefig(output_dir / "statistical_variability.png", dpi=300)
        if show:
            plt.show()
        plt.close(fig)


def plot_cross_validation_results(
    per_run_metrics: Sequence[Mapping[str, float]],
    aggregate_metrics: Mapping[str, float],
    output_dir: Path | str,
    show: bool = False,
) -> None:
    if not per_run_metrics:
        return
    output_dir = _ensure_output_dir(output_dir)
    pinn_vals = np.array([metrics.get("pinn_rrmse", np.nan) for metrics in per_run_metrics], dtype=float)
    vteam_vals = np.array([metrics.get("vteam_rrmse", np.nan) for metrics in per_run_metrics], dtype=float)

    fig, ax = plt.subplots(figsize=(6, 4))
    positions = np.arange(2)
    means = [np.nanmean(pinn_vals), np.nanmean(vteam_vals)]
    stds = [np.nanstd(pinn_vals, ddof=1) if len(pinn_vals) > 1 else 0.0,
            np.nanstd(vteam_vals, ddof=1) if len(vteam_vals) > 1 else 0.0]
    colors = ["#d95f02", "#1b9e77"]
    labels = ["PINN", "VTEAM"]
    ax.bar(positions, means, yerr=stds, capsize=6, color=colors, alpha=0.85)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("RRMSE")
    ax.set_title("Cross-Validation RRMSE Summary")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "cross_validation_rrmse.png", dpi=300)
    if show:
        plt.show()
    plt.close(fig)

    # Optional: noise robustness box plot if data present
    noise_keys = sorted(
        {key for metrics in per_run_metrics for key in metrics if key.startswith("noise_rrmse_")}
    )
    if noise_keys:
        fig, ax = plt.subplots(figsize=(8, 4))
        xticks = []
        data = []
        for key in noise_keys:
            label = key.replace("noise_rrmse_", "")
            xticks.append(label)
            data.append([metrics.get(key, np.nan) for metrics in per_run_metrics])
        ax.boxplot(data, labels=xticks)
        ax.set_xlabel("Noise level (%)")
        ax.set_ylabel("RRMSE")
        ax.set_title("Noise Robustness Across Runs")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(Path(output_dir) / "cross_validation_noise_robustness.png", dpi=300)
        if show:
            plt.show()
        plt.close(fig)
