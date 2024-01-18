# ITER SOURCE DEFENITION FOR OPENMC



## Usage:

```
import iter_tokamak_source

my_plasma = iter_tokamak_source.ITER_TokamakSource(sample_size=100000)
```
## Plotting:

```
import numpy as np
from plot_tokamak_source import scatter_tokamak_source, plot_tokamak_source_3D

scatter_tokamak_source(my_plasma, quantity='strength', aspect='equal') # To create 2D-plot
plot_tokamak_source_3D(my_plasma, quantity='strength', angles=[0, 0.5 * np.pi]) # To create 3D-plot
```

## Acknowledgments

Inspiration, code snippets, etc.
* [openmc-plasma-source](https://github.com/fusion-energy/openmc-plasma-source)
