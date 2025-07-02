import sys
import numpy as np
import h5py

if len(sys.argv) < 3:
    print("pick_single_seg.py in_path out_path 1 2 3 # 1 2 3 are event ids")
    exit(-1)

in_path = sys.argv[1]
out_path = sys.argv[2]
if not (isinstance(in_path, str) and isinstance(out_path, str)):
    print("pick_single_seg.py in_path out_path 1 2 3 # 1 2 3 are event ids")

event_list = set()
try:
    event_list = set(int(i) for i in sys.argv[3:])
except ValueError:
    event_list = None


with h5py.File(in_path) as fin:
    segments = fin['segments'][:]
    source_file = fin['segments'].attrs['source_file']

uq_events_set = set(int(i) for i in np.unique(segments['event_id']))
if len(event_list) == 0:
    event_list = uq_events_set
print(event_list)

selected_events = uq_events_set & event_list
if len(selected_events) == 0 :
    print(f"None of events {event_list} in the file {in_path}")
    exit(-1)

output_hits = []

for ie in selected_events:
    em = segments['event_id'] == ie
    aligned_dtype = np.dtype([
    # 8-byte aligned float64s ? must start at 96, 104, etc.
    ('t_start',      np.float64),  # offset 96
    ('t0_start',     np.float64),  # offset 104
    ('t0_end',       np.float64),  # offset 112
    ('t0',           np.float64),  # offset 120
    ('t_end',        np.float64),  # offset 128
    ('t',            np.float64),  # offset 136
    ('vertex_id',    np.uint64),   # offset 8
    ('event_id',     np.uint32),   # offset 0
    ('segment_id',   np.uint32),   # offset 4
    ('traj_id',      np.uint32),   # offset 16
    ('file_traj_id', np.uint32),   # offset 20
    ('n_electrons',  np.uint32),   # offset 24
    ('pdg_id',       np.int32),    # offset 28
    ('pixel_plane',  np.int32),    # offset 32
    ('x_end',        np.float32),  # offset 36
    ('y_end',        np.float32),  # offset 40
    ('z_end',        np.float32),  # offset 44
    ('x_start',      np.float32),  # offset 48
    ('y_start',      np.float32),  # offset 52
    ('z_start',      np.float32),  # offset 56
    ('dx',           np.float32),  # offset 60
    ('tran_diff',    np.float32),  # offset 64
    ('long_diff',    np.float32),  # offset 68
    ('dEdx',         np.float32),  # offset 72
    ('dE',           np.float32),  # offset 76
    ('n_photons',    np.float32),  # offset 80
    ('x',            np.float32),  # offset 84
    ('y',            np.float32),  # offset 88
    ('z',            np.float32),  # offset 92

], align=True)
    segments_aligned = np.empty(segments.shape, dtype=aligned_dtype)
    for name in aligned_dtype.fields:
        segments_aligned[name] = segments[name]

    output_hits.append(np.atleast_1d(segments_aligned[em][0].astype(aligned_dtype)))

segments_out = np.concatenate(output_hits).astype(aligned_dtype)
for name in segments_out.dtype.names:
    print(f"{name:15s} offset = {segments_out.dtype.fields[name][1]}")

with h5py.File(out_path, 'w') as fout:
    fout['segments'] = segments_out

