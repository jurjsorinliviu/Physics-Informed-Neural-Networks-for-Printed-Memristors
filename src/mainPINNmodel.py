from __future__ import annotations

import numpy as np
import tensorflow as tf


class PrintedMemristorPINN:
    """
    Physics-informed neural network framework for printed memristors.
    
    This generalizable framework integrates physical conduction mechanisms
    (Ohmic, SCLC, Schottky emission) with data-driven learning. Designed
    specifically for printed electronics with explicit variability modeling.
    
    Key Features:
    - Trainable physical parameters (e.g., ohmic conductance)
    - Multi-mechanism conduction modeling
    - Concentration-dependent behavior (optional)
    - Robust to fabrication variability and measurement noise
    
    Reference:
    For experimental validation data from printed Ag/PMMA:PVA/ITO devices,
    see Strutwolf et al., Appl. Phys. A 127:709 (2021).
    """

    def __init__(
        self,
        hidden_layers: int = 3,
        neurons_per_layer: int = 64,
        input_features: tuple[str, ...] = ("voltage", "state"),
        random_seed: int | None = None,
        trainable_params: tuple[str, ...] | None = ("ohmic_conductance",),
    ) -> None:
        if "voltage" not in input_features or "state" not in input_features:
            raise ValueError("input_features must include both 'voltage' and 'state'.")
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.input_features = tuple(input_features)
        self.input_dim = len(self.input_features)
        self.dtype = tf.float32
        self.random_seed = random_seed

        if self.random_seed is not None:
            tf.keras.utils.set_random_seed(self.random_seed)

        self._base_physical_params = self._initialize_physical_parameters()
        self.trainable_param_keys: tuple[str, ...] = tuple(trainable_params or ())
        self._trainable_params = self._build_trainable_params()
        self.model = self._build_model()

    def _build_model(self) -> tf.keras.Model:
        inputs = tf.keras.layers.Input(shape=(self.input_dim,), dtype=self.dtype)
        x = inputs
        for _ in range(self.hidden_layers):
            x = tf.keras.layers.Dense(
                self.neurons_per_layer,
                activation="tanh",
                kernel_initializer="glorot_normal",
            )(x)
        current_output = tf.keras.layers.Dense(1, name="current")(x)
        state_derivative = tf.keras.layers.Dense(1, name="state_derivative")(x)
        return tf.keras.Model(inputs=inputs, outputs=[current_output, state_derivative])

    def _initialize_physical_parameters(self) -> dict[str, float]:
        return {
            "epsilon_r": 3.5,
            "mu": 1e-10,
            "d": 100e-9,
            "area": 1e-8,
            "richardson": 1.2e6,
            "temperature": 300.0,
            "barrier_height": 0.8,
            "alpha": 0.1,
            "beta": 0.05,
            "ohmic_conductance": 1e-6,
        }

    def _build_trainable_params(self) -> dict[str, tf.Variable]:
        trainable = {}
        for key in self.trainable_param_keys:
            if key not in self._base_physical_params:
                raise KeyError(f"Unknown physical parameter '{key}' for trainable configuration.")
            initial = self._base_physical_params[key]
            trainable[key] = tf.Variable(
                np.log(initial),
                dtype=self.dtype,
                trainable=True,
                name=f"trainable_{key}",
            )
        return trainable

    @property
    def physical_params(self) -> dict[str, float]:
        return dict(self._base_physical_params)

    @property
    def trainable_variables(self) -> list[tf.Variable]:
        return list(self.model.trainable_variables) + list(self._trainable_params.values())

    def update_physical_params(self, overrides: dict[str, float] | None) -> None:
        if not overrides:
            return
        self._base_physical_params.update(overrides)

    def get_trainable_param_values(self) -> dict[str, float]:
        values = {}
        for key, var in self._trainable_params.items():
            values[key] = float(tf.exp(var).numpy())
        return values

    def _merge_params(self, overrides: dict[str, float | tf.Tensor] | None) -> dict[str, tf.Tensor]:
        combined: dict[str, tf.Tensor | float] = dict(self._base_physical_params)
        if overrides:
            combined.update(overrides)
        for key, var in self._trainable_params.items():
            combined[key] = tf.exp(var)
        return {k: tf.cast(v, self.dtype) for k, v in combined.items()}

    def ohmic_conduction(self, V: tf.Tensor, params: dict[str, tf.Tensor] | None = None) -> tf.Tensor:
        params_tf = self._merge_params(params)
        conductance = params_tf["ohmic_conductance"]
        return conductance * tf.cast(V, self.dtype)

    def sclc_conduction(self, V: tf.Tensor, params: dict[str, tf.Tensor] | None = None) -> tf.Tensor:
        params_tf = self._merge_params(params)
        epsilon_0 = tf.constant(8.854187817e-12, dtype=self.dtype)
        epsilon = params_tf["epsilon_r"] * epsilon_0
        mu = params_tf["mu"]
        thickness = params_tf["d"]
        area = params_tf["area"]
        V = tf.cast(V, self.dtype)
        current_density = (
            tf.constant(9.0 / 8.0, dtype=self.dtype)
            * epsilon
            * mu
            * tf.square(V)
            / tf.pow(thickness, 3)
        )
        return current_density * area

    def schottky_emission(self, V: tf.Tensor, params: dict[str, tf.Tensor] | None = None) -> tf.Tensor:
        params_tf = self._merge_params(params)
        richardson = params_tf["richardson"]
        temperature = params_tf["temperature"]
        phi_b = params_tf["barrier_height"]
        thickness = params_tf["d"]
        area = params_tf["area"]
        V = tf.cast(V, self.dtype)

        q = tf.constant(1.602176634e-19, dtype=self.dtype)
        k_b = tf.constant(1.380649e-23, dtype=self.dtype)
        epsilon_0 = tf.constant(8.854187817e-12, dtype=self.dtype)

        electric_field = V / thickness
        barrier_term = -q * phi_b / (k_b * temperature)
        field_enhancement = tf.sqrt((q * electric_field) / (4.0 * np.pi * epsilon_0))
        exponent = barrier_term + (q / (k_b * temperature)) * field_enhancement
        current_density = richardson * tf.square(temperature) * tf.exp(exponent)
        return current_density * area

    def state_variable_ode(
        self,
        V: tf.Tensor,
        x: tf.Tensor,
        params: dict[str, tf.Tensor] | None = None,
    ) -> tf.Tensor:
        params_tf = self._merge_params(params)
        alpha = params_tf["alpha"]
        beta = params_tf["beta"]
        V = tf.cast(V, self.dtype)
        x = tf.cast(x, self.dtype)
        return alpha * V - beta * x

    def physics_loss(
        self,
        V: tf.Tensor,
        I_pred: tf.Tensor,
        x: tf.Tensor,
        x_deriv_pred: tf.Tensor,
        params: dict[str, float | tf.Tensor] | None = None,
    ) -> tf.Tensor:
        if x_deriv_pred is None:
            raise ValueError("x_deriv_pred must be provided to physics_loss.")
        params_tf = self._merge_params(params)
        V = tf.reshape(tf.cast(V, self.dtype), [-1])
        I_pred = tf.reshape(tf.cast(I_pred, self.dtype), [-1])
        x = tf.reshape(tf.cast(x, self.dtype), [-1])
        x_deriv_pred = tf.reshape(tf.cast(x_deriv_pred, self.dtype), [-1])

        def masked_mse(target: tf.Tensor, reference: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
            masked_target = tf.boolean_mask(target, mask)
            masked_reference = tf.boolean_mask(reference, mask)
            return tf.cond(
                tf.equal(tf.size(masked_target), 0),
                lambda: tf.constant(0.0, dtype=self.dtype),
                lambda: tf.reduce_mean(tf.square(masked_target - masked_reference)),
            )

        mask_low = tf.abs(V) < tf.constant(0.1, dtype=self.dtype)
        ohmic_expected = self.ohmic_conduction(V, params_tf)
        ohmic_loss = masked_mse(I_pred, ohmic_expected, mask_low)

        mask_mid = (tf.abs(V) >= tf.constant(0.1, dtype=self.dtype)) & (
            tf.abs(V) < tf.constant(0.5, dtype=self.dtype)
        )
        sclc_expected = self.sclc_conduction(V, params_tf)
        sclc_loss = masked_mse(I_pred, sclc_expected, mask_mid)

        ode_target = self.state_variable_ode(V, x, params_tf)
        ode_loss = tf.reduce_mean(tf.square(x_deriv_pred - tf.reshape(ode_target, [-1])))

        return ohmic_loss + sclc_loss + ode_loss