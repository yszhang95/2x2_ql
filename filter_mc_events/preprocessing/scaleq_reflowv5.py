#!/usr/bin/env python3

import numpy as np
import h5py
import os
import sys
import argparse


def reflow_hdf5(input_file, output_file, scaleq=1.0):
    """
    Copies the HDF5 file structure, scaling the 'Q' field in all hits/data structured arrays by scaleq.
    Args:
        input_file (str): Path to the input HDF5 file.
        output_file (str): Path to the output HDF5 file.
        scaleq (float): Scale factor to multiply hits['Q'] values by.
    """
    def copy_group(g_in, g_out):
        for name, item in g_in.items():
            if isinstance(item, h5py.Group):
                subgroup = g_out.create_group(name)
                copy_group(item, subgroup)
            elif isinstance(item, h5py.Dataset):
                # Check for deselected/hits/data or selected/hits/data
                fullpath = item.name
                if fullpath.endswith('/hits/data'):
                    data = item[:]
                    data_new = data.copy()
                    if 'Q' in data.dtype.names:
                        data_new['Q'] *= scaleq
                    else:
                        raise ValueError(f"Dataset {fullpath} does not contain 'Q' field.")
                    g_out.create_dataset(name, data=data_new, dtype=data.dtype)
                else:
                    g_out.create_dataset(name, data=item[:], dtype=item.dtype)
    with h5py.File(input_file, 'r') as f_in, h5py.File(output_file, 'w') as f_out:
        copy_group(f_in, f_out)
    print(f"Reflowed {input_file} to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Reflow HDF5 file to float32 format.")
    parser.add_argument("input_file", type=str, help="Path to the input HDF5 file.")
    parser.add_argument("output_file", type=str, help="Path to the output HDF5 file.")
    parser.add_argument("--scaleq", type=float, default=1.0, help="Scale factor to apply to the data (default: 1.0).")

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} does not exist.")
        sys.exit(1)

    reflow_hdf5(args.input_file, args.output_file, args.scaleq)


if __name__ == "__main__":
    main()
