import h5py
import numpy as np
import os
import ipywidgets as widgets
from IPython.display import display, clear_output
import k3d

# Load and group HDF5 data
def load_files(input2_path):
    print("Input2_path:", input2_path)
    with h5py.File(input2_path, 'r') as f2:
        source_path = f2['/source_file'][()].decode('utf-8')
        sel2 = f2['/hits/selected/data'][:]
        des2 = f2['/hits/deselected/data'][:]
        pts2 = f2['/picked/points/data'][:]
        eids = f2['/picked/event_id'][:]
        rids = f2['/picked/run_id'][:]
        dq2  = f2['/picked/dqdx_raw/data'][:]

    if not os.path.isabs(source_path):
        source_path = os.path.join(os.path.dirname(input2_path), source_path)
    with h5py.File(source_path, 'r') as f1:
        sel1 = f1['/selected/hits/data'][:]
        des1 = f1['/deselected/hits/data'][:]

    records = []
    for i, rec in enumerate(pts2):
        eid = eids[i]
        rid = rids[i]
        pmin = pts2[i, :3].tolist()
        pmax = pts2[i, 3:].tolist()

        mask2_sel = (sel2['event_id'] == eid) & (sel2['run_id'] == rid)
        mask2_des = (des2['event_id'] == eid) & (des2['run_id'] == rid)
        h2_sel = np.vstack([sel2[mask2_sel]['x'], sel2[mask2_sel]['y'], sel2[mask2_sel]['z']]).T
        h2_des = np.vstack([des2[mask2_des]['x'], des2[mask2_des]['y'], des2[mask2_des]['z']]).T

        mask1_sel = (sel1['event_id'] == rid)
        mask1_des = (des1['event_id'] == rid)
        h1_sel = np.vstack([sel1[mask1_sel]['x'], sel1[mask1_sel]['y'], sel1[mask1_sel]['z']]).T
        h1_des = np.vstack([des1[mask1_des]['x'], des1[mask1_des]['y'], des1[mask1_des]['z']]).T

        records.append({
            'event_id': eid,
            'run_id': rid,
            'line': {'p_min': pmin, 'p_max': pmax},
            'dqdx_raw': float(dq2[i]),
            'pgun': {'selected': h2_sel, 'deselected': h2_des},
            'source': {'selected': h1_sel, 'deselected': h1_des}
        })
    return records

# Visualization

def draw_record_k3d(record):
    plot = k3d.plot(grid_visible=False, camera_auto_fit=True)

    def add_points(plot, points, color, point_size=0.5, name="hits"):
        if points.size > 0:
            plot += k3d.points(points.astype(np.float32),
                               point_size=point_size,
                               color=color, name=name)

    # Input 1
    add_points(plot, record['pgun']['selected'], 0x0000ff, name="pgun sel")   # Blue
    add_points(plot, record['pgun']['deselected'], 0x00ffff, name="pgun desel") # Cyan

    # Input 2
    add_points(plot, record['source']['selected'], 0xff0000, name="source sel")   # Red
    add_points(plot, record['source']['deselected'], 0xffa500, name="source desel") # Orange

    # Draw line from picked points
    line_pts = np.array([record['line']['p_min'], record['line']['p_max']], dtype=np.float32)
    plot += k3d.line(line_pts, color=0x000000, shader="simple", name="pgun track")

    plot.display()

def interactive_browser_k3d(records):
    options = [(f"E:{r['event_id']} R:{r['run_id']}", i) for i, r in enumerate(records)]
    dropdown = widgets.Dropdown(options=options, description='Event:')
    out = widgets.Output()

    def on_change(change):
        with out:
            clear_output(wait=True)
            draw_record_k3d(records[change['new']])

    dropdown.observe(on_change, names='value')

    display(dropdown, out)
    with out:
        draw_record_k3d(records[0])  # Show the first one by default

# USAGE
# path2 = 'your_input2_file.h5'
# records = load_files(path2)
# interactive_browser(records)
