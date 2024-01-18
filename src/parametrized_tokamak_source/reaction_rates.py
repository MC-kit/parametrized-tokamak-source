"""Module reaction_rates.

    Computes fusion reaction rates.
"""
from __future__ import annotations

from typing import Final, cast

import numpy as np
import numpy.typing as npt


class ReactionRate:
    """Base class implementing common algorithm by Bosch&Hale.

    Note
    ----
        See *Improved formulas for fusion cross-sections and thermal particles*
        Bosch & Hale, Nuclear Fusion vol. 32, No 4 (1992)
    """

    def __init__(self, c1, c2, c3, c4, c5, c6, c7, m_r_c, b_g, t_min=0.1):
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.c5 = c5
        self.c6 = c6
        self.c7 = c7
        self.m_r_c = m_r_c
        self.b_g_sq = b_g**2
        self.t_min = t_min
        assert t_min > 0.0, "Minimal temperature (t_min) should be above 0.0 keV"

    def __call__(self, t: npt.NDArray[float]) -> npt.NDArray[float]:
        # pylint: disable=no-member
        if np.isscalar(t):
            if t < self.t_min:
                return cast(npt.NDArray[float], 0.0)
            _theta = self._theta(t)
            e = (self.b_g_sq / (4.0 * _theta)) ** (1.0 / 3.0)
            # pylint: disable=no-member
            return self.c1 * _theta * np.sqrt(e / (self.m_r_c * t**3)) * np.exp(-3 * e)
        res = np.empty(t.shape)
        i = t < self.t_min
        res[i] = 0.0
        i = np.invert(i)  # self.t_min <= t
        _t = t[i]
        _theta = self._theta(_t)
        e = (self.b_g_sq / (4.0 * _theta)) ** (1.0 / 3.0)
        res[i] = self.c1 * _theta * np.sqrt(e / (self.m_r_c * _t**3)) * np.exp(-3 * e)
        return res

    def _theta(self, t):
        """Helper function theta(temperature)."""
        res = t / (
            1
            - (t * (self.c2 + t * (self.c4 + t * self.c6)))
            / (1 + t * (self.c3 + t * (self.c5 + t * self.c7)))
        )
        return res


class ReactionRateDT(ReactionRate):
    """DT reaction <sigma*v> by Bloch&Hale, ccm/sec."""

    def __init__(self, t_min=0.1):
        ReactionRate.__init__(
            self,
            c1=1.17302e-9,
            c2=1.51361e-2,
            c3=7.51886e-2,
            c4=4.60643e-3,
            c5=1.35000e-2,
            c6=-1.06750e-4,
            c7=1.36600e-5,
            m_r_c=1124656,
            b_g=34.3827,
            t_min=t_min,
        )


class ReactionRateDD(ReactionRate):
    """DD reaction <sigma*v> by Bloch&Hale, ccm/sec."""

    def __init__(self, t_min=0.1):
        ReactionRate.__init__(
            self,
            c1=5.43360e-12,
            c2=5.85778e-3,
            c3=7.68222e-3,
            c4=None,
            c5=-2.96400e-6,
            c6=None,
            c7=None,
            m_r_c=937814,
            b_g=31.3970,
            t_min=t_min,
        )

    def _theta(self, t):
        """Helper function theta(temperature)."""
        return t / (1 - t * self.c2 / (1 + t * (self.c3 + t * self.c5)))


RRDT: Final = ReactionRateDT()
RRDD: Final = ReactionRateDD()
