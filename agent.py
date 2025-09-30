import numpy as np
import tensorflow as tf

class EdgeAgent:
    def __init__(self, node_id, nominal_voltage):
        self.node_id = node_id
        self.nominal_voltage = nominal_voltage
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def optimize(self, load, fault_state):
        # Simulate voltage regulation
        input_data = np.array([[load, fault_state]], dtype=np.float32)
        control_signal = self.model.predict(input_data, verbose=0)[0][0]
        voltage = self.nominal_voltage * (1 + 0.1 * control_signal)
        voltage = np.clip(voltage, 0.9 * self.nominal_voltage, 1.1 * self.nominal_voltage)
        
        # Calculate efficiency
        resistance = 0.1  # Ohm
        efficiency = 0.9 if not fault_state else 0.85  # 90% normal, 85% during fault
        power_loss = (load * 1000 / voltage)**2 * resistance + voltage * (load * 1000 / voltage) * (1 - efficiency)
        efficiency = 1 - power_loss / (load * 1000)
        return voltage, efficiency * 100

    def handle_fault(self):
        # Simulate fault response (0.5s)
        return 0.5
