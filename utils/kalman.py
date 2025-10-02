"""Lightweight constant-velocity Kalman filter used for ball smoothing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class KalmanFilter4D:
    """Simple 2D constant velocity Kalman filter.

    The state vector is ``[x, y, vx, vy]`` expressed in pixel coordinates and
    pixels per frame.  The filter works on a per-frame cadence, so callers are
    expected to invoke :meth:`predict` for every frame and :meth:`update` when a
    detection is available.
    """

    dt: float = 1.0
    process_var: float = 5.0
    meas_var: float = 25.0

    def __post_init__(self) -> None:
        self._state: Optional[np.ndarray] = None
        self._covariance: Optional[np.ndarray] = None

        self._F = np.array(
            [[1.0, 0.0, self.dt, 0.0],
             [0.0, 1.0, 0.0, self.dt],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        self._H = np.array(
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        q = self.process_var
        r = self.meas_var
        self._Q = np.array(
            [[q, 0.0, 0.0, 0.0],
             [0.0, q, 0.0, 0.0],
             [0.0, 0.0, q, 0.0],
             [0.0, 0.0, 0.0, q]],
            dtype=np.float32,
        )
        self._R = np.array(
            [[r, 0.0],
             [0.0, r]],
            dtype=np.float32,
        )

    def init_state(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0) -> None:
        """Initialise the filter with a measurement."""
        self._state = np.array([[x], [y], [vx], [vy]], dtype=np.float32)
        self._covariance = np.eye(4, dtype=np.float32) * self.meas_var

    def predict(self) -> None:
        """Propagate state forward by one frame."""
        if self._state is None:
            return
        self._state = self._F @ self._state
        self._covariance = self._F @ self._covariance @ self._F.T + self._Q

    def update(self, x: float, y: float) -> None:
        """Incorporate a position measurement."""
        if self._state is None:
            self.init_state(x, y)
            return

        measurement = np.array([[x], [y]], dtype=np.float32)
        y_tilde = measurement - (self._H @ self._state)
        S = self._H @ self._covariance @ self._H.T + self._R
        K = self._covariance @ self._H.T @ np.linalg.inv(S)
        self._state = self._state + K @ y_tilde
        identity = np.eye(self._covariance.shape[0], dtype=np.float32)
        self._covariance = (identity - K @ self._H) @ self._covariance

    def get_state(self) -> Tuple[float, float, float, float]:
        """Return the current state vector."""
        if self._state is None:
            return 0.0, 0.0, 0.0, 0.0
        return tuple(float(x) for x in self._state.flatten())
