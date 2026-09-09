from dataclasses import dataclass
import numpy as np

@dataclass
class FieldState:
    rho: np.ndarray
    vel: np.ndarray
    press: np.ndarray
    B: np.ndarray
    energy: np.ndarray
    time: float = 0.0

    def __post_init__(self):
        spatial_shape = self.rho.shape
        assert self.press.shape == spatial_shape
        assert self.energy.shape == spatial_shape
        assert self.vel.shape == (3, *spatial_shape)
        assert self.B.shape == (3, *spatial_shape)
