from __future__ import annotations

import time
import numpy as np
import tensorflow as tf


class ExperimentalValidator:
    """Evaluation utilities for trained printed memristor PINNs."""

    def __init__(self, trained_pinn, seed: int | None = None):
        self.pinn = trained_pinn
        self.dtype = trained_pinn.dtype
        self.input_features = tuple(getattr(trained_pinn, "input_features", ("voltage", "state")))
        self._rng = np.random.default_rng(seed)

    def _build_inputs(
        self,
        voltage: np.ndarray,
        state: np.ndarray | None,
        concentration: np.ndarray | None,
    ) -> tf.Tensor:
        voltage = np.asarray(voltage, dtype=np.float32)
        if state is None:
            state = np.zeros_like(voltage, dtype=np.float32)
        else:
            state = np.asarray(state, dtype=np.float32)
        concentration_tensor = None
        if "concentration" in self.input_features:
            if concentration is None:
                raise ValueError("Concentration feature required but not provided.")
            concentration_tensor = np.asarray(concentration, dtype=np.float32)
        feature_columns = []
        for feature in self.input_features:
            if feature == "voltage":
                feature_columns.append(tf.convert_to_tensor(voltage, dtype=self.dtype))
            elif feature == "state":
                feature_columns.append(tf.convert_to_tensor(state, dtype=self.dtype))
            elif feature == "concentration":
                feature_columns.append(tf.convert_to_tensor(concentration_tensor, dtype=self.dtype))
            else:
                raise ValueError(f"Unsupported input feature '{feature}'.")
        stacked = tf.stack(feature_columns, axis=1)
        return stacked

    def predict_current(
        self,
        voltage: np.ndarray,
        state: np.ndarray | None = None,
        concentration: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        inference_start = time.time()
        
        inputs = self._build_inputs(voltage, state, concentration)
        current_pred, _ = self.pinn.model(inputs, training=False)
        result = current_pred.numpy().flatten()
        
        inference_time = time.time() - inference_start
        
        return result, inference_time

    def calculate_rrmse(self, I_pred: np.ndarray, I_meas: np.ndarray) -> float:
        mse = np.mean((I_pred - I_meas) ** 2)
        rmse = np.sqrt(mse)
        range_meas = np.max(I_meas) - np.min(I_meas) + 1e-12
        return float(rmse / range_meas)

    def calculate_noise_robustness(
        self,
        V_clean: np.ndarray,
        baseline_prediction: np.ndarray,
        noise_levels: list[float],
        state: np.ndarray | None = None,
        concentration: np.ndarray | None = None,
    ) -> dict[float, float]:
        results: dict[float, float] = {}
        V_clean = np.asarray(V_clean, dtype=np.float32)
        baseline_prediction = np.asarray(baseline_prediction, dtype=np.float32)
        state = np.asarray(state, dtype=np.float32) if state is not None else None
        concentration = np.asarray(concentration, dtype=np.float32) if concentration is not None else None

        for noise_std in noise_levels:
            V_noisy = V_clean + self._rng.normal(
                0.0, noise_std * (np.std(V_clean) + 1e-12), size=V_clean.shape
            )
            noisy_pred, _ = self.predict_current(V_noisy, state, concentration)
            robustness_rrmse = self.calculate_rrmse(noisy_pred, baseline_prediction)
            results[float(noise_std)] = robustness_rrmse
        return results

    def statistical_validation(self, n_cycles: int = 50) -> dict[str, object]:
        set_voltages = []
        on_resistances = []
        for _ in range(n_cycles):
            V_cycle, I_cycle, _ = self.generate_cycle_with_variability()
            set_voltages.append(self.detect_set_voltage(V_cycle, I_cycle))
            on_resistances.append(self.calculate_on_resistance(V_cycle, I_cycle))

        set_voltages = np.array(set_voltages)
        on_resistances = np.array(on_resistances)
        return {
            "set_voltage_distribution": (float(np.mean(set_voltages)), float(np.std(set_voltages))),
            "on_resistance_cv": float(np.std(on_resistances) / (np.mean(on_resistances) + 1e-12)),
            "set_voltages": set_voltages.tolist(),
            "on_resistances": on_resistances.tolist(),
        }

    def generate_cycle_with_variability(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        V = np.linspace(-2.0, 2.0, 500)
        set_threshold = 1.0 + self._rng.normal(0.0, 0.1)
        reset_threshold = -1.0 + self._rng.normal(0.0, 0.1)
        on_resistance_var = 1e3 * (1.0 + self._rng.normal(0.0, 0.05))
        I = np.zeros_like(V)
        for i, v in enumerate(V):
            if v >= set_threshold and v > 0:
                I[i] = v / on_resistance_var
            elif v <= reset_threshold and v < 0:
                I[i] = v / (on_resistance_var * 100.0)
            else:
                I[i] = v / (on_resistance_var * 10.0)
        state = np.abs(I) / (np.max(np.abs(I)) + 1e-12)
        return V.astype(np.float32), I.astype(np.float32), state.astype(np.float32)

    def detect_set_voltage(self, V: np.ndarray, I: np.ndarray) -> float:
        dI_dV = np.gradient(I, V)
        positive_indices = np.where(V > 0)[0]
        if positive_indices.size == 0:
            return float(0.0)
        set_idx = positive_indices[np.argmax(dI_dV[positive_indices])]
        return float(V[set_idx])

    def calculate_on_resistance(self, V: np.ndarray, I: np.ndarray) -> float:
        on_region = (V > 1.0) & (V < 1.5)
        if np.any(on_region):
            return float(np.mean(V[on_region] / (I[on_region] + 1e-12)))
        return float(1e6)