import h5py
import numpy as np

f = h5py.File('/home/yousen/Public/ndlar_shared/data/tred_2x2_2025010/MiniRun5_1E19_RHC.convert2h5.0000000.EDEPSIM.hdf5')

traj = f['trajectories']
segm = f['segments']

events = np.unique(traj['event_id'])

segments_out = {
    13 : [],
    2212 : [],
    211 : [],
}
itrk = 0
for i, eid in enumerate(events):
    emt = traj['event_id'] == eid
    ems = segm['event_id'] == eid
    particles = traj[emt]
    tracks = segm[ems]
    for pid in [13, 2212, 211]:# no electron == 11
        pars = particles[np.abs(particles['pdg_id']) == pid]
        tids = pars['file_traj_id']
        for tid in tids:
            tks = tracks[tracks['file_traj_id'] == tid]
            tks['event_id'] = itrk
            segments_out[pid].append(tks)
            itrk += 1

for k, v in segments_out.items():
    ds = np.concatenate(v)
    with h5py.File(f'segments_pid{k}.hdf5', 'w') as fout:
        fout.create_dataset('segments', data=ds)
