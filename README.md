# Parametrized tokamak source for OpenMC

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
git clone git@github.com:MC-kit/parametrized-tokamak-source.git

pip install .
```

## Usage:

```python
import parametrized_tokamak_source as pts

# Loading plasma params from file

my_source_params = pts.distribution_openmc.PlasmaParams.from_file('path/to/parametrization.csv')

# Making an OpenMC source distribution

my_source = pts.parametrized_tokamak_source.make_openmc_sources(
    sample_size=1000, 
    plasma_params=my_source_params,
    intensity=pts.parametrized_tokamak_source.IntensityDD(my_source_params)
)

# Creating OpenMC source based on our (input) plasma source

settings = openmc.Settings()
settings.source = my_source
settings.particles = 10000
settings.batches = 10

settings.export_to_xml()
```


## Acknowledgments

Inspiration and code snippets:
* [openmc-plasma-source](https://github.com/fusion-energy/openmc-plasma-source)
* [OpenMC](https://github.com/openmc-dev/openmc)

