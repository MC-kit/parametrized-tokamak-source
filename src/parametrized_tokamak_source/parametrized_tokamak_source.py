from __future__ import annotations

import numpy as np

import openmc
import parametrized_tokamak_source.distribution_openmc as do
import parametrized_tokamak_source.reaction_rates as rr


def make_openmc_sources(
    plasma_params: do.PlasmaParams,
    intensity: IntensityDT | IntensityDD,
    angles: tuple[float, float] = (0, 2 * np.pi),
    sample_size: int = 10000,
) -> list[openmc.IndependentSource]:
    """Creates a list of OpenMC Sources() objects. The created sources are
    ring sources based on the .RZ coordinates between two angles. The
    energy of the sources are Muir energy spectra with ion temperatures
    based on .temperatures. The strength of the sources (their probability)
    is based on .strengths.

    Args:
        plasma_params (parametrized_tokamak_source.do.PlasmaParams): plasma parameters
        intensity (parametrized_tokamak_source.do.IntensityDT | parametrized_tokamak_source.do.IntensityDD): plasma distribution
        angles ((float, float), optional): rotation of the ring source.
        Defaults to (0, 2*np.pi).
        sample_size (int, optional): number of samples in the plasma source.
        Defaults to 10000.

    Returns:
        list: list of openmc.IndependentSource()
    """

    params = do.psi_calc(plasma_params, sample_size)
    r, z, psi = params
    neutron_source_density = intensity(psi) * r
    strengths = neutron_source_density / sum(neutron_source_density)

    sources = []
    # create a ring source for each sample in the plasma source
    for i in range(len(strengths)):
        my_source = openmc.IndependentSource()

        # extract the RZ values accordingly
        radius = openmc.stats.Discrete([r[i]], [1])
        z_values = openmc.stats.Discrete([z[i]], [1])
        angle = openmc.stats.Uniform(a=angles[0], b=angles[1])

        # create a ring source
        my_source.space = openmc.stats.CylindricalIndependent(
            r=radius, phi=angle, z=z_values, origin=(0.0, 0.0, 0.0)
        )

        my_source.angle = openmc.stats.Isotropic()
        my_source.energy = openmc.stats.muir(e0=14080000.0, m_rat=5.0, kt=plasma_params.Ti(psi)[i])

        # the strength of the source (its probability) is given by
        # self.strengths
        my_source.strength = strengths[i]

        # append to the list of sources
        sources.append(my_source)
    return sources


class IntensityDT:
    def __init__(self, plasma_params: do.PlasmaParams, dt_fraction, coeff=1e19**2 * 1e-6):
        """Computes DT unit intensity.

        With default coeff normalizes output per cubic meter.
        1e19 - scale for concentration per cubic meter.
        1e-6 - scale for reaction rate
        """
        self.pp = plasma_params
        self.dt_fraction = dt_fraction
        self.dt = rr.ReactionRateDT()
        self.coeff = coeff

    def __call__(self, psi):
        return (
            self.dt(self.pp.Ti(psi))
            * self.pp.N(psi) ** 2
            * self.dt_fraction
            * (1 - self.dt_fraction)
        )

    def norm(self, x):
        return x * self.coeff


class IntensityDD:
    def __init__(self, plasma_params: do.PlasmaParams, coeff=1e19**2 * 1e-6):
        """Computes DT unit intensity.

        With default coeff normalizes output per cubic meter.
        1e19 - scale for concentration per cubic meter.
        1e-6 - scale for sig_v
        """
        self.pp = plasma_params
        self.dd = rr.ReactionRateDD()
        self.coeff = coeff

    def __call__(self, psi):
        return self.dd(self.pp.Ti(psi)) * self.pp.N(psi) ** 2 * 0.5

    def norm(self, x):
        return x * self.coeff
