# make_dataset.py -- builds the raw dataset and nothing else. mtg.demo_data
# loads a dataset, records its identity/source on ds.attrs, prints its
# dimensions, and returns the ds plus the names of its target, missing (cloud)
# flag, and land flag variables. This file is recorded verbatim in the model
# bundle (as make_dataset.py) so the dataset build is reproducible.
#
# Examples of other datasets you could load with mtg.demo_data():
#   dataset = "globcolour";   region = "arabian sea";   time_slice = None
#   dataset = "indian-ocean"; region = "indian ocean"; time_slice = slice("01-01-2000", "12-31-2000")
import mindthegap as mtg

dataset = "io-shared-public"  # pace, globcolour, indian-ocean, or synthetic
region = "indian ocean"       # or [lat_min, lat_max, lon_min, lon_max]
time_slice = slice("01-01-2010", "01-01-2020")

# target is the variable name of the variable being gap-filled; _flag are the names of the flags in the dataset.
ds, target, missing_flag, land_flag = mtg.demo_data(
    dataset,
    region=region,
    time_slice=time_slice,
    smoke_test=False,
)
