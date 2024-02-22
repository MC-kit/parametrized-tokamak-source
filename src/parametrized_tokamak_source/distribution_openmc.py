from __future__ import annotations

from typing import Callable

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from scipy import interpolate as sp
from scipy.constants import e as eV

import pandas as pd

HERE = Path(__file__).parent

PF = Callable[[npt.NDArray[float]], npt.NDArray[float]]

@dataclass
class PlasmaParams:
    psi: npt.NDArray
    a: PF
    sh: PF
    k: PF
    delta: PF
    Ti: PF
    N: PF

    def __post_init__(self):
        pass

    @classmethod
    def from_file(cls, path_to_parametrization: Path) -> PlasmaParams:
        if path_to_parametrization is None:
            path_to_parametrization = HERE / "data/iter-parametrization.csv"
        data = pd.read_csv(path_to_parametrization)
        psi_params = data["psi"].to_numpy()
        a_params = data["a"].to_numpy()
        sh_params = data["sh"].to_numpy()
        k_params = data["k"].to_numpy()
        delta_params = data["delta"].to_numpy()
        ti_params = data["Ti"].to_numpy()
        n_params = data["N"].to_numpy()

        return cls(
            psi=psi_params,
            a=sp.interp1d(
                psi_params, a_params, kind="cubic", fill_value="extrapolate", bounds_error=False
            ),
            sh=sp.interp1d(
                psi_params, sh_params, kind="cubic", fill_value="extrapolate", bounds_error=False
            ),
            k=sp.interp1d(
                psi_params, k_params, kind="cubic", fill_value="extrapolate", bounds_error=False
            ),
            delta=sp.interp1d(
                psi_params, delta_params, kind="cubic", fill_value="extrapolate", bounds_error=False
            ),
            Ti=sp.interp1d(
                psi_params, ti_params, kind="cubic", fill_value="extrapolate", bounds_error=False
            ),
            N=sp.interp1d(
                psi_params, n_params, kind="cubic", fill_value="extrapolate", bounds_error=False
            ),
        )

    def R(self, psi, t, r_p):
        """Computing R (psi, t)."""
        return (r_p + self.sh(psi) + self.a(psi) * (np.cos(t) - self.delta(psi) * np.sin(t) ** 2)) * 100 # m -> cm

    def Z(self, psi, t, z_p):
        """Computing Z (psi, t)."""
        return (z_p + self.a(psi) * self.k(psi) * np.sin(t)) * 100 # m -> cm


def total_power(x: npt.NDArray[float]) -> npt.NDArray[float]:
    """Converts total neutron production to energy production, MW."""
    return 17.1e6 * eV * x


def psi_calc(
    pp: PlasmaParams, sample_size: int, R_P: float, Z_P: float,
) -> tuple[npt.NDArray[float], npt.NDArray[float], npt.NDArray[float]]:

    #_R and _Z is calculated on normal scattered points (psi,t).

    psi = np.random.triangular(0.0, 1.0, 1.0, sample_size)
    t = np.random.uniform(0.0, 2 * np.pi, sample_size)
    _R = pp.R(psi, t, R_P)
    _Z = pp.Z(psi, t, Z_P)

    return _R, _Z, psi
