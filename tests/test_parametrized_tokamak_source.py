import pytest
from pathlib import Path

import parametrized_tokamak_source as pts
from parametrized_tokamak_source.distribution_openmc import PlasmaParams
from parametrized_tokamak_source.parametrized_tokamak_source import IntensityDT, IntensityDD

HERE = Path(__file__).parent


def test_iter_parametrization_dt():
    params_file = HERE.parent / "src/parametrized_tokamak_source/data/iter-parametrization.csv"
    assert params_file.exists()
    plasma_params = PlasmaParams.from_file(params_file)
    source = pts.make_openmc_sources(
        sample_size=1000,
        plasma_params=plasma_params,
        intensity=IntensityDT(plasma_params, dt_fraction=0.5),
    )

def test_iter_parametrization_dd():
    params_file = HERE.parent / "src/parametrized_tokamak_source/data/iter-parametrization.csv"
    assert params_file.exists()
    plasma_params = PlasmaParams.from_file(params_file)
    source = pts.make_openmc_sources(
        sample_size=1000,
        plasma_params=plasma_params,
        intensity=IntensityDD(plasma_params),
    )
