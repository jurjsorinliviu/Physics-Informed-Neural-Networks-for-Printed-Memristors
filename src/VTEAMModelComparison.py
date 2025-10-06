import time
import numpy as np


class VTEAMModel:
    """Implementation of VTEAM model for comparison"""
    def __init__(self):
        self.k_on = 1e-3
        self.k_off = 1e-3
        self.alpha_on = 3
        self.alpha_off = 3
        self.v_on = 1.0
        self.v_off = -1.0
        self.w_max = 1.0
        self.w_min = 0.0
        self.r_on = 1e3
        self.r_off = 1e6
        
    def predict_current(self, V, w):
        """Predict current using VTEAM model"""
        R = self.r_off + (self.r_on - self.r_off) * (w / self.w_max)
        return V / R
    
    def state_derivative(self, V, w):
        """State variable derivative for VTEAM"""
        if V > self.v_on:
            return self.k_on * ((V / self.v_on) - 1) ** self.alpha_on
        elif V < self.v_off:
            return self.k_off * ((V / self.v_off) - 1) ** self.alpha_off
        else:
            return 0.0
    
    def simulate_iv(self, voltage_sweep):
        """Simulate complete I-V characteristics"""
        start_time = time.time()
        
        w = self.w_min
        current = np.zeros_like(voltage_sweep)
        
        for i, V in enumerate(voltage_sweep):
            # Update state variable
            dw_dt = self.state_derivative(V, w)
            w = np.clip(w + dw_dt * 1e-3, self.w_min, self.w_max)
            
            # Calculate current
            current[i] = self.predict_current(V, w)
        
        elapsed = time.time() - start_time
        return current, elapsed