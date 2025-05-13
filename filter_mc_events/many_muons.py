#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import re
import os
import h5py
from collections import defaultdict
import numpy.lib.recfunctions as rfn

def export_event_array(npz_path, prefix):
    f = np.load(npz_path)

    # Match pattern only for exact keys like 'effq_tpcX_batchY'
    pattern = re.compile(rf'{prefix}_tpc(\d+)_batch(\d+)$')
    matched_keys = [k for k in f.files if pattern.match(k)]

    all_positions = []
    all_charges = []
    all_event_ids = []
    all_tpc_ids = []
    all_indices = []

    for key in matched_keys:
        match = pattern.match(key)
        tpc = int(match.group(1))
        batch = int(match.group(2))

        data = f[key]
        num_hits = data.shape[0]
        event_id = int(f[f'event_id_tpc{tpc}_batch{batch}'])
        event_ids = np.full(num_hits, event_id, dtype=int)
        tpc_ids = np.full(num_hits, tpc, dtype=int)

        all_positions.append(data[:, :3])
        all_charges.append(data[:, -1])
        all_event_ids.append(event_ids)
        all_tpc_ids.append(tpc_ids)
        all_indices.append(f[f'{key}_location'])

    # Concatenate all arrays
    positions = np.vstack(all_positions)
    charges = np.hstack(all_charges)
    event_ids = np.hstack(all_event_ids)
    tpc_ids = np.hstack(all_tpc_ids)
    indices = np.vstack(all_indices)

    # Sort by event_id
    sort_indices = np.argsort(event_ids)
    positions = positions[sort_indices]
    charges = charges[sort_indices]
    event_ids = event_ids[sort_indices]
    tpc_ids = tpc_ids[sort_indices]
    indices = indices[sort_indices]
    return positions, charges, event_ids, tpc_ids, indices

# with h5py.File('many_muons.hdf5', 'w') as fout:
#     for k in ['hits', 'effq']:
#         arrs = export_event_array("output_hits.npz", "mu_jsons/data", prefix=k)
#         g = fout.create_group(k)
#         g.create_dataset('position', data=arrs[0])
#         g.create_dataset('charge', data=arrs[1])
#         g.create_dataset('event_id', data=arrs[2])
#         g.create_dataset('tpc_id', data=arrs[3])
#         g.create_dataset('grid_index', data=arrs[4])
#output_dir = './nonoise_pid13_unipolar'
output_dir = './nonoise_pid13'
with h5py.File(f'{output_dir}/many_muon_hits.hdf5', 'w') as fout:
    # pos, qs, eids, tids, inds = export_event_array("/home/yousen/Public/ndlar_shared/data/tred_2x2_2025010/waveforms.npz", ".", prefix='hits')
    # pos, qs, eids, tids, inds = export_event_array("nonoise_pid13_unipolar/waveforms_nonoise_pid13_unipolar.npz", prefix='hits')
    pos, qs, eids, tids, inds = export_event_array("nonoise_pid13/waveforms_pid13_ndlar.npz", prefix='hits')
    io_group = tids # there might be mismatch
    io_channel = inds[:,0] * 5000 + inds[:,1]
    # print(io_group.shape, io_channel.shape, inds.shape)
    pos_dtype = {
        'x': 'f4',
        'y': 'f4',
        'z': 'f4',
    }
    t_drift = inds[:,3].astype(np.float32) * 0.05
    # print(t_drift.max()/0.05)
    # uqc, inv = np.unique(io_group*100000000 + io_channel, return_inverse=True)
    chid = np.stack([io_group, io_channel, eids], axis=1)
    # print(chid.shape)
    uqc, inv = np.unique(chid, return_inverse=True, axis=0)
    # print(inv.shape)
    # for i, uq in enumerate(uqc):
    #     m = i == inv
    #     ok = len(np.unique(t_drift[m])) == len(t_drift[m])
    #     if not ok:
    #         print('t', t_drift[m], 'tpc', io_group[m], 'ch', io_channel[m], 'pos', pos[m], 'q', qs[m], 'eid', eids[m], 'tid', tids[m], 'inds', inds[m])
    pos = np.core.records.fromarrays(pos.T, names='x,y,z', formats='f4,f4,f4')
    hits = np.array(pos)

    hits = rfn.append_fields(hits, names=['Q', 't_drift'], data=[qs/1E3, t_drift], usemask=False)
    hits = rfn.append_fields(hits, names=['io_group', 'io_channel', 'event_id'], data=[io_group, io_channel, eids], usemask=False)

    fout.create_dataset('hits', data=hits)



