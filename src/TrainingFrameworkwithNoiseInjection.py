from __future__ import annotations

import time
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from typing import Optional, Tuple


class PINNTrainer:
    """Training utilities and data handling for the memristor PINN."""

    def __init__(
        self,
        pinn_model,
        learning_rate: float = 1e-3,
        seed: int | None = None,
        state_mixing: float = 0.2,
    ) -> None:
        self.pinn = pinn_model
        self.seed = seed
        self.state_mixing = float(np.clip(state_mixing, 0.0, 1.0))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_history: list[dict[str, float]] = []
        self._rng = np.random.default_rng(seed)
        self.include_concentration_feature = "concentration" in getattr(self.pinn, "input_features", ())
        if self.seed is not None:
            tf.keras.utils.set_random_seed(self.seed)

    # ------------------------------------------------------------------
    # Data utilities
    # ------------------------------------------------------------------
    def _compute_state_variable(
        self,
        current: np.ndarray,
        concentration: np.ndarray | None = None,
    ) -> np.ndarray:
        norm_current = np.abs(current) / (np.max(np.abs(current)) + 1e-12)
        if concentration is None:
            return np.clip(norm_current, 0.0, 1.0)
        conc = (concentration - np.min(concentration)) / (
            np.max(concentration) - np.min(concentration) + 1e-12
        )
        mixed = (1.0 - self.state_mixing) * norm_current + self.state_mixing * conc
        return np.clip(mixed, 0.0, 1.0)

    def load_experimental_data(
        self,
        csv_path: str | Path,
        concentration_label: Optional[str] = None,
        device_id: Optional[int] = None,
        use_noisy_columns: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        df = pd.read_csv(csv_path)
        if concentration_label is not None:
            df = df[df["concentration_label"] == concentration_label]
        if device_id is not None:
            df = df[df["device_id"] == device_id]
        if df.empty:
            raise ValueError("No data matches the provided filters.")

        voltage_col = "voltage_noisy" if use_noisy_columns and "voltage_noisy" in df else "voltage"
        current_col = "current_noisy" if use_noisy_columns and "current_noisy" in df else "current"

        voltage = df[voltage_col].to_numpy(dtype=np.float32)
        current = df[current_col].to_numpy(dtype=np.float32)
        concentration = df.get("pmma_concentration", pd.Series(np.zeros_like(current))).to_numpy(dtype=np.float32)
        state = self._compute_state_variable(current, concentration)
        return voltage, current, state.astype(np.float32), concentration.astype(np.float32)

    def generate_synthetic_data(
        self,
        n_samples: int = 1000,
        concentration_levels: tuple[float, ...] = (5.0, 10.0, 15.0),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        voltage_up = np.linspace(-2.0, 2.0, n_samples // 2, dtype=np.float32)
        voltage_down = np.linspace(2.0, -2.0, n_samples - n_samples // 2, dtype=np.float32)
        voltage = np.concatenate([voltage_up, voltage_down])
        current = np.zeros_like(voltage)

        for i, V in enumerate(voltage):
            if V >= 1.0 and i < voltage.shape[0] // 2:
                current[i] = 1e-3 * (1.0 - np.exp(-10.0 * (V - 1.0))) + 1e-6 * V
            elif V <= -1.0 and i >= voltage.shape[0] // 2:
                current[i] = 1e-6 * V - 1e-3 * (1.0 - np.exp(10.0 * (V + 1.0)))
            else:
                current[i] = 1e-6 * V

        current += self._rng.normal(0.0, 5e-7, size=current.shape)
        concentration = self._rng.choice(concentration_levels, size=voltage.shape)
        state = self._compute_state_variable(current, concentration)
        return voltage, current, state.astype(np.float32), concentration.astype(np.float32)

    def add_synthetic_noise(
        self, voltage: np.ndarray, current: np.ndarray, noise_std: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        v_noise = self._rng.normal(0.0, noise_std * (np.std(voltage) + 1e-12), size=voltage.shape)
        i_noise = self._rng.normal(0.0, noise_std * (np.std(current) + 1e-12), size=current.shape)
        return voltage + v_noise, current + i_noise

    def parameter_variability(self, params: dict[str, float], variation_bound: float) -> dict[str, float]:
        varied: dict[str, float] = {}
        for key, value in params.items():
            if hasattr(self.pinn, "trainable_param_keys") and key in self.pinn.trainable_param_keys:
                continue
            delta = self._rng.uniform(-variation_bound, variation_bound)
            varied[key] = value * (1.0 + delta)
        return varied

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def _build_inputs(
        self,
        V_tensor: tf.Tensor,
        x_tensor: tf.Tensor,
        c_tensor: Optional[tf.Tensor],
    ) -> tf.Tensor:
        feature_columns = []
        for feature in self.pinn.input_features:
            if feature == "voltage":
                feature_columns.append(tf.reshape(V_tensor, [-1, 1]))
            elif feature == "state":
                feature_columns.append(tf.reshape(x_tensor, [-1, 1]))
            elif feature == "concentration":
                if c_tensor is None:
                    raise ValueError("Concentration feature requested but no concentration tensor provided.")
                feature_columns.append(tf.reshape(c_tensor, [-1, 1]))
            else:
                raise ValueError(f"Unsupported input feature '{feature}'.")
        return tf.concat(feature_columns, axis=1)

    def train_step(
        self,
        V_train: tf.Tensor,
        I_train: tf.Tensor,
        x_train: tf.Tensor,
        lambda_data: float,
        lambda_physics: float,
        params: dict[str, float],
        c_train: Optional[tf.Tensor] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        train_vars = self.pinn.trainable_variables
        with tf.GradientTape() as tape:
            inputs = self._build_inputs(V_train, x_train, c_train)
            I_pred, x_deriv_pred = self.pinn.model(inputs, training=True)
            I_target = tf.reshape(I_train, [-1, 1])
            data_loss = tf.reduce_mean(tf.square(I_pred - I_target))
            physics_loss = self.pinn.physics_loss(
                V_train,
                I_pred,
                x_train,
                x_deriv_pred=x_deriv_pred,
                params=params,
            )
            total_loss = lambda_data * data_loss + lambda_physics * physics_loss

        gradients = tape.gradient(total_loss, train_vars)
        grad_var_pairs = [(g, v) for g, v in zip(gradients, train_vars) if g is not None]
        if grad_var_pairs:
            self.optimizer.apply_gradients(grad_var_pairs)
        return total_loss, data_loss, physics_loss

    def train(
        self,
        epochs: int = 500,
        voltage: Optional[np.ndarray] = None,
        current: Optional[np.ndarray] = None,
        state: Optional[np.ndarray] = None,
        concentration: Optional[np.ndarray] = None,
        noise_std: float = 0.02,
        variability_bound: float = 0.1,
        verbose_every: int = 50,
        max_physics_weight: float = 0.4,
    ) -> list[dict[str, float]]:
        if voltage is None or current is None or state is None:
            voltage, current, state, concentration = self.generate_synthetic_data()
        if self.include_concentration_feature and concentration is None:
            raise ValueError("Concentration data is required when using the concentration feature.")

        voltage = np.asarray(voltage, dtype=np.float32)
        current = np.asarray(current, dtype=np.float32)
        state = np.asarray(state, dtype=np.float32)
        concentration = np.asarray(concentration, dtype=np.float32) if concentration is not None else None

        self.loss_history = []
        lambda_data = 1.0
        min_physics_weight = 0.0
        max_physics_weight = max(min_physics_weight, max_physics_weight)

        c_tensor = tf.convert_to_tensor(concentration, dtype=self.pinn.dtype) if concentration is not None else None

        # START TIMER
        training_start_time = time.time()

        for epoch in range(epochs):
            progress = epoch / max(epochs - 1, 1)
            lambda_physics = (
                min_physics_weight
                + (max_physics_weight - min_physics_weight) * np.clip(progress, 0.0, 1.0)
            )

            V_noisy, I_noisy = self.add_synthetic_noise(voltage, current, noise_std)
            varied_params = self.parameter_variability(self.pinn.physical_params, variability_bound)

            V_tensor = tf.convert_to_tensor(V_noisy, dtype=self.pinn.dtype)
            I_tensor = tf.convert_to_tensor(I_noisy, dtype=self.pinn.dtype)
            x_tensor = tf.convert_to_tensor(state, dtype=self.pinn.dtype)

            total_loss, data_loss, physics_loss = self.train_step(
                V_tensor,
                I_tensor,
                x_tensor,
                lambda_data,
                lambda_physics,
                varied_params,
                c_train=c_tensor,
            )

            history_entry: dict[str, float] = {
                "epoch": int(epoch),
                "total_loss": float(total_loss.numpy()),
                "data_loss": float(data_loss.numpy()),
                "physics_loss": float(physics_loss.numpy()),
                "lambda_physics": float(lambda_physics),
            }
            for key, value in self.pinn.get_trainable_param_values().items():
                history_entry[f"param_{key}"] = float(value)
            self.loss_history.append(history_entry)

            if verbose_every and epoch % verbose_every == 0:
                print(
                    f"Epoch {epoch:04d} | Total {history_entry['total_loss']:.3e} "
                    f"| Data {history_entry['data_loss']:.3e} | Physics {history_entry['physics_loss']:.3e} "
                    f"| w_phys {lambda_physics:.2f}"
                )

        # END TIMER AND STORE
        training_end_time = time.time()
        training_duration = training_end_time - training_start_time
        
        if self.loss_history:
            self.loss_history[-1]['training_time_seconds'] = training_duration
            self.loss_history[-1]['training_time_minutes'] = training_duration / 60.0

        return self.loss_history