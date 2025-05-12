import streamlit as st
import numpy as np
import h5py
import matplotlib.pyplot as plt
from asymmetry_prep import (  # replace with real names
    split_sorted_dataset,
    sel_uni_pxl,
    yz_line,
    proj_yz,
    xline,
    plot_three_views
)
import numpy.lib.recfunctions as rfn

st.set_page_config(layout="wide")

@st.cache_resource
def load_data():
    fh5 = h5py.File('/home/yousen/Public/ndlar_shared/data/many_muons.hdf5', 'r')
    xyzs = fh5['/hits/position']
    qs = fh5['/hits/charge']
    eids = fh5['/hits/event_id']
    tids = fh5['/hits/tpc_id']
    pids = fh5['/hits/grid_index']
    groups = list(split_sorted_dataset(eids))
    return xyzs, qs, eids, tids, pids, groups

xyzs, qs, eids, tids, pids, groups = load_data()

ie = st.slider("Select Event Index", 0, len(groups)-1, 0)

st.title("Event + TPC Viewer")
ie = st.slider("Event Index (ie)", 0, len(groups)-1, 0)
sel_tpc_ids = st.multiselect("Select TPC IDs", list(range(70)), default=[0])
eid, em = groups[ie]

args = np.argsort(tids[em])
tpcids = tids[em][args]
charges = qs[em][args]
positions = xyzs[em][args]
pixels = pids[em][args]

dtype = np.dtype([
    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('Q', 'f4'),
    ('t_drift', 'f8'), ('io_group', 'i4'), ('io_channel', 'i8'),
    ('event_id', 'i4'), ('drift_direction', 'i4')
])
large_offset = 10_000
entries = []
for ih in range(charges.shape[0]):
    drift = 1 if tpcids[ih] % 2 else -1
    entry = positions[ih].tolist() + [float(charges[ih])/1E3, float(pixels[ih,2]), int(tpcids[ih]),
                                      int(pixels[ih,0] * large_offset + pixels[ih,1]), int(eid), drift]
    entries.append(tuple(entry))
hits = np.array(entries, dtype=dtype)
mask = np.zeros(len(hits), dtype=bool)
for sel_id in sel_tpc_ids:
    tpc_mask = (hits['io_group'] == sel_id)
    mask |= tpc_mask
hits = hits[mask]

if len(hits) < 20:
    st.warning("Less than 20 hits found for selected TPCs and event.")
else:
    uni_pxls, extended_hits = sel_uni_pxl(hits, att='t_drift', yposlabel='io_channel', zposlabel='io_channel')
    k_yz, b_yz, _ = yz_line(uni_pxls)
    d_tg, d_nm = proj_yz(extended_hits, k_yz, b_yz)
    extended_hits = rfn.append_fields(extended_hits, names=('d_tg', 'd_nm'), data=(d_tg, d_nm), usemask=False)
    k_x, b_x, _ = xline(extended_hits[extended_hits['Q'] > 5], reflabel='y')
    ref_x = k_x * extended_hits['y'] + b_x
    dx = extended_hits['x'] - ref_x
    sign = np.where(extended_hits['io_group'] % 2, 1.0, -1.0)
    dx *= sign
    extended_hits = rfn.append_fields(extended_hits, names='dx', data=dx, usemask=False)

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    plot_three_views(uni_pxls, axs=axs, all_hits=extended_hits)
    st.pyplot(fig)
