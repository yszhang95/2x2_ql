#!/usr/bin/env python3

# %%

import h5py

# %%
offset = 0.1
# offset = 0.2
offset_str = str(offset).replace(".", "p")

# %%
with h5py.File("../threshold_summary.hdf5") as fin:
    with h5py.File(f"scaled_down{offset_str}_threshold_summary.hdf5", "w") as fout:
        for key in fin.keys():
            data = fin[key]["threshold"][:]
            data["Q"] *= 1 - offset  # Scale the thresholds by a factor of 1.5
            fout.create_dataset(key, data=data)

    with h5py.File(f"scaled_up{offset_str}_threshold_summary.hdf5", "w") as fout:
        for key in fin.keys():
            data = fin[key]["threshold"][:]
            data["Q"] *= 1 + offset  # Scale the thresholds by a factor of 1.5
            fout.create_dataset(f"{key}/threshold", data=data)
