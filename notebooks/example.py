# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import parametrized_tokamak_source.parametrized_tokamak_source as pts

# %%
plasma_params = pts.do.PlasmaParams.from_file('../src/parametrized_tokamak_source/data/iter-parametrization.csv')

# %%
my_plasma = pts.make_openmc_sources(plasma_params=plasma_params, sample_size=1000, intensity=pts.IntensityDD(plasma_params))

# %%
import openmc
# Create a single material
iron = openmc.Material()
iron.set_density("g/cm3", 5.0)
iron.add_element("Fe", 1.0)
mats = openmc.Materials([iron])

# Create a 5 cm x 5 cm box filled with iron
cells = []
inner_box1 = openmc.ZCylinder(r=600.0)
inner_box2 = openmc.ZCylinder(r=1400.0)
outer_box = openmc.model.rectangular_prism(4000.0, 4000.0, boundary_type="vacuum")
cells += [openmc.Cell(fill=iron, region=-inner_box1)]
cells += [openmc.Cell(fill=None, region=+inner_box1 & -inner_box2)]
cells += [openmc.Cell(fill=iron, region=+inner_box2 & outer_box)]
geometry = openmc.Geometry(cells)

# Tell OpenMC we're going to use our custom source
settings = openmc.Settings()
settings.run_mode = "fixed source"
settings.batches = 10
settings.particles = 10000
settings.source = my_plasma

# Finally, define a mesh tally so that we can see the resulting flux
mesh = openmc.RegularMesh()
mesh.lower_left = (-2000.0, -2000.0)
mesh.upper_right = (2000.0, 2000.0)
mesh.dimension = (4000, 4000)

tally = openmc.Tally()
tally.filters = [openmc.MeshFilter(mesh)]
tally.scores = ["flux"]
tallies = openmc.Tallies([tally])

model = openmc.model.Model(
    materials=mats, geometry=geometry, settings=settings, tallies=tallies
)

#settings.export_to_xml()
#model.run(tracks=True)
