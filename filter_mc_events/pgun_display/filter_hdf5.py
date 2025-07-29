import h5py
import numpy as np
import argparse
import os


def load_event_list(path):
    return np.loadtxt(path, dtype=int, delimiter=',',)


def filter_events(finpath, event_list):
    with h5py.File(finpath, 'r') as fin:
        source_path = fin['/source_file'][()].decode('utf-8')
        sel = fin['/hits/selected/data'][:]
        dist = fin['picked/trk_dist/data'][:]
        dist_src = fin['picked/trk_dist_src/data'][:]
        event_ids = fin['/picked/event_id'][:]
        run_ids = fin['/picked/run_id'][:]

    with h5py.File(finpath.replace("hits.hdf5", "effq.hdf5"), 'r') as fq:
        selq = fq['/hits/selected/data'][:]

    selq_out = []
    sel_out = []
    sel_src_out = []
    dist_out = []
    dist_src_out = []

    with h5py.File(source_path, 'r') as fsrc:
        sel_src = fsrc['/selected/hits/data'][:]

    for (ievent, irun) in event_list:
        mask = (sel['event_id'] == ievent) & (sel['run_id'] == irun)
        if not np.any(mask):
            print(f"Event {ievent} Run {irun} not found in {finpath}")
            continue
        msrc = (sel_src['event_id'] == irun)
        if not np.any(msrc):
            print(f"Event {ievent} Run {irun} not found in source file {source_path}")
            continue
        md = (event_ids == ievent) & (run_ids == irun)
        mq = (selq['event_id'] == ievent) & (selq['run_id'] == irun)
        selq_out.append(selq[mq])
        sel_out.append(sel[mask])
        sel_src_out.append(sel_src[msrc])
        dist_out.append(dist[md])
        dist_src_out.append(dist_src[md])
    if len(sel_out):
        sel_out = np.concatenate(sel_out)
        selq_out = np.concatenate(selq_out)
        dist_out = np.concatenate(dist_out)
        dist_src_out = np.concatenate(dist_src_out)
    else:
        sel_out = np.empty((0,), dtype=sel.dtype)
    if len(sel_src_out):
        sel_src_out = np.concatenate(sel_src_out)
    else:
        sel_src_out = np.empty((0,), dtype=sel_src.dtype)

    return sel_out, selq_out, sel_src_out, dist_out, dist_src_out


def save_output(foutpath, data, dsname):
    with h5py.File(foutpath, 'w') as fout:
        if not isinstance(data, (list, tuple)):
            data = (data,)
        if not isinstance(dsname, (list, tuple)):
            dsname = (dsname,)
        if len(data) != len(dsname):
            raise ValueError("Data and dataset names must have the same length.")
        for d, n in zip(data, dsname):
            fout.create_dataset(n, data=d)
            print(f"Data saved to {foutpath} under dataset {n}")


def main():

    parser = argparse.ArgumentParser(description="Filter events from HDF5 file based on event list.")
    parser.add_argument("finpath", type=str, help="Path to the input HDF5 file.")
    parser.add_argument("event_list", type=str, help="Path to the event list file.")
    parser.add_argument("-o", "--output", type=str, default="filtered_events.hdf5",
                        help="Path to the output HDF5 file (default: filtered_events.hdf5).")
    args = parser.parse_args()

    foutpath = args.output
    base, ext = os.path.splitext(foutpath)
    foutpath_src = f"{base}_source{ext}"

    if "hits.hdf5" not in foutpath:
        raise ValueError("Output file name must contain 'hits.hdf5' to match the input file structure.")

    event_list = load_event_list(args.event_list)
    sel_out, selq_out, sel_src_out, dist, dist_src = filter_events(args.finpath, event_list)
    save_output(foutpath, (sel_out, dist), ('/hits', '/picked/trk_dist/data'))
    save_output(foutpath.replace("hits.hdf5", "effq.hdf5"), selq_out,  '/effq')
    save_output(foutpath_src, (sel_src_out, dist_src), ('/selected/hits/data', '/picked/trk_dist_src/data'))


if __name__ == "__main__":
    main()
