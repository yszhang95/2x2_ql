import h5py
import numpy as np
import numpy.lib.recfunctions as rfn

import sys
import os

# prefix = 'web/many_muon_hits'
prefix = sys.argv[1]
try:
    key = sys.argv[2]
except IndexError:
    key = 'hits'
dname = os.path.dirname(prefix)
out_arr = []
out2_arr = []
directions = []
points = []
event_ids = []
with open(os.path.join(dname, 'event_list.txt'), 'r') as fin:
    lines = fin.readlines()
    for l in lines:
        l = l.strip()
        if l[0] == '#':
            continue
        with h5py.File(f'{prefix}_evt{l}.hdf5', 'r') as fh5:
            ds = fh5['/hits/selected/data'][:]
            ds2 = fh5['/hits/deselected/data'][:]
            direction = fh5['/picked/direction/data'][:]
            pts = fh5['/picked/points/data'][:]

            if 'event_id' in ds.dtype.names:
                out = ds
                out2 = ds2
            else:
                out = rfn.append_fields(np.array(ds, dtype=ds.dtype), names=['event_id',], data=[np.full((len(ds), ), fill_value=int(l)),], usemask=False)
                out2 = rfn.append_fields(np.array(ds2, dtype=ds2.dtype), names=['event_id',], data=[np.full((len(ds2), ), fill_value=int(l)),], usemask=False)

            if len(out['event_id']):
                event_ids.append(out['event_id'][0])
            elif len(out2['event_id']):
                event_ids.append(out2['event_id'][0])
            else:
                raise ValueError

            if pts.shape == (2,):
                points.append(np.array([pts['x'][0], pts['y'][0], pts['z'][0], pts['x'][1], pts['y'][1], pts['z'][1]]).reshape(1, 6))
            elif pts.shape == (2, 3):
                points.append(pts.reshape(1, 6))

            directions.append(direction.reshape(1,3))

            out['Q'] = out['Q']
            print(out['Q'][0])

            out_arr.append(out)
            out2_arr.append(out2)

with h5py.File(f'{prefix}_selected.hdf5', 'w') as fh5:
    fh5.create_group('selected').create_group(key).create_dataset('data', data=np.concatenate(out_arr))
    fh5.create_group('deselected').create_group(key).create_dataset('data', data=np.concatenate(out2_arr))
    picked = fh5.create_group('picked')
    picked.create_dataset('direction', data=np.concatenate(directions))
    picked.create_dataset('points', data=np.concatenate(points))
    picked.create_dataset('event_id', data=np.array(event_ids))
