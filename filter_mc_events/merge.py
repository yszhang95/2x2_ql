import h5py
import numpy as np
import numpy.lib.recfunctions as rfn

prefix = 'web/many_muon_hits'
out_arr = []
with open('web/event_list.txt', 'r') as fin:
    lines = fin.readlines()
    for l in lines:
        l = l.strip()
        with h5py.File(f'{prefix}_evt{l}.hdf5', 'r') as fh5:
            ds = fh5['/hits/selected/data'][:]

            if 'event_id' in ds.dtype.names:
                out = ds
            else:
                out = rfn.append_fields(np.array(ds, dtype=ds.dtype), names=['event_id',], data=[np.full((len(ds), ), fill_value=int(l)),], usemask=False)
            out['Q'] = out['Q']
            print(out['Q'][0])

            out_arr.append(out)

with h5py.File(f'{prefix}_selected.hdf5', 'w') as fh5:
    fh5.create_group('selected').create_group('hits').create_dataset('data', data=np.concatenate(out_arr))
