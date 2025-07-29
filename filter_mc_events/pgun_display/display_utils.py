import h5py
import numpy as np
import os
import ipywidgets as widgets
from IPython.display import display, HTML
import k3d
# from itertools import product


# Load and group HDF5 data
def load_files(input2_path):
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
def add_bounding_boxes(plot, all_bounds, color=0x888888, width=0.5, name='volume'):
    import numpy as np

    cube_edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # top
        [0, 4], [1, 5], [2, 6], [3, 7],  # vertical
    ], dtype=np.float32)

    all_vertices = []
    all_indices = []
    offset = 0

    for bounds in all_bounds:
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds

        # Properly ordered corners
        corners = np.array([
            [xmin, ymin, zmin],  # 0
            [xmax, ymin, zmin],  # 1
            [xmax, ymax, zmin],  # 2
            [xmin, ymax, zmin],  # 3
            [xmin, ymin, zmax],  # 4
            [xmax, ymin, zmax],  # 5
            [xmax, ymax, zmax],  # 6
            [xmin, ymax, zmax],  # 7
        ], dtype=np.float32)

        all_vertices.append(corners)
        all_indices.append(cube_edges + offset)
        offset += 8

    vertices = np.vstack(all_vertices)
    indices = np.vstack(all_indices)

    box_lines = k3d.lines(vertices=vertices, indices=indices, color=color, indices_type='segment', width=width)
    box_lines.name = name
    box_lines.pickable = False
    plot += box_lines


def init_k3d_plot():

    # def add_bounding_box(plot, bounds, color=0x888888, width=0.5):
    #     (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    #     corners = np.array(list(product([xmin, xmax], [ymin, ymax], [zmin, zmax])), dtype=np.float32)
    #     edges = [
    #         [0, 1], [0, 2], [0, 4],
    #         [1, 3], [1, 5],
    #         [2, 3], [2, 6],
    #         [3, 7],
    #         [4, 5], [4, 6],
    #         [5, 7],
    #         [6, 7]
    #     ]
    #     for i1, i2 in edges:
    #         segment = np.array([corners[i1], corners[i2]], dtype=np.float32)
    #         line = k3d.line(segment, color=color, width=width, shader="simple", group="Detector Volume")
    #         line.visible = True      # still show the lines
    #         line.pickable = False    # optional: disable mouse picking
    #         plot += line


    # Add all static volumes once
    boundaries = [
        [[ 3.069, 33.34125], [-62.076, 62.076], [ 2.462, 64.538]],
        [[63.931, 33.65875], [-62.076, 62.076], [ 2.462, 64.538]],
        [[ 3.069, 33.34125], [-62.076, 62.076], [-64.538, -2.462]],
        [[63.931, 33.65875], [-62.076, 62.076], [-64.538, -2.462]],
        [[-63.931, -33.65875], [-62.076, 62.076], [ 2.462, 64.538]],
        [[ -3.069, -33.34125], [-62.076, 62.076], [ 2.462, 64.538]],
        [[-63.931, -33.65875], [-62.076, 62.076], [-64.538, -2.462]],
        [[ -3.069, -33.34125], [-62.076, 62.076], [-64.538, -2.462]],
    ]

    plot = k3d.plot(grid_visible=False, grid_auto_fit=True, camera_auto_fit=True, height=700,)
    add_bounding_boxes(plot, boundaries)
    # for box in boundaries:
    #     add_bounding_box(plot, box)

    return plot


def draw_record_k3d(record, plot, objects):
    def update_points(name, points):
        if points.size == 0:
            objects[name].positions = np.empty((0, 3), dtype=np.float32)
        else:
            objects[name].positions = points.astype(np.float32)

    update_points('pgun_sel', record['pgun']['selected'])
    update_points('pgun_desel', record['pgun']['deselected'])
    update_points('source_sel', record['source']['selected'])
    update_points('source_desel', record['source']['deselected'])

    # Line
    line_pts = np.array([record['line']['p_min'], record['line']['p_max']], dtype=np.float32)
    objects['track'].positions = line_pts


def interactive_browser_k3d(records):
    options = [(f"E:{r['event_id']} R:{r['run_id']}, "
                f"raw mean dQ/dx:{r['dqdx_raw']:.2f}",
                i) for i, r in enumerate(records)]
    dropdown = widgets.Dropdown(options=options, description='Event:',
                                layout=widgets.Layout(min_width='400px'))
    out = widgets.Output()
    label_display = widgets.Text(value=options[0][0], layout=widgets.Layout(width='300px'), disabled=True)

    # Navigation buttons
    prev_button = widgets.Button(description='← Prev', layout=widgets.Layout(width='80px'))
    next_button = widgets.Button(description='Next →', layout=widgets.Layout(width='80px'))

    # Initialize static plot and dynamic objects
    plot = init_k3d_plot()
    objects = {
        'pgun_sel': k3d.points(np.empty((0, 3), dtype=np.float32), color=0x0000ff, point_size=0.5, name='pgun_sel'),
        'pgun_desel': k3d.points(np.empty((0, 3), dtype=np.float32), color=0x00ffff, point_size=0.5, name='pgun_desel'),
        'source_sel': k3d.points(np.empty((0, 3), dtype=np.float32), color=0xff0000, point_size=0.5, name='source_sel'),
        'source_desel': k3d.points(np.empty((0, 3), dtype=np.float32), color=0xffa500, point_size=0.5, name='source_desel'),
        'track': k3d.line(np.empty((2, 3), dtype=np.float32), color=0x000000, shader="simple", name='track')
    }

    for obj in objects.values():
        plot += obj

    def update_view(index):
        dropdown.value = index
        label_display.value = options[index][0]

    def on_dropdown_change(change):
        with out:
            # clear_output(wait=True)
            draw_record_k3d(records[change['new']], plot, objects)
            label_display.value = options[change['new']][0]

    def on_prev_clicked(b):
        if dropdown.value > 0:
            update_view(dropdown.value - 1)

    def on_next_clicked(b):
        if dropdown.value < len(records) - 1:
            update_view(dropdown.value + 1)

    # Wire it up
    dropdown.observe(on_dropdown_change, names='value')
    prev_button.on_click(on_prev_clicked)
    next_button.on_click(on_next_clicked)

    # save event_id, run_id

    selected_list = []
    selected_output = widgets.Select(
        options=[],
        rows=6,
        description='Selected:',
        layout=widgets.Layout(width='300px')
    )

    add_button = widgets.Button(description='Add Current', button_style='success')
    remove_button = widgets.Button(description='Remove Selected', button_style='warning')
    save_button = widgets.Button(description='Save to File', button_style='info')

    def update_selected_output():
        selected_output.options = [f"E:{eid} R:{rid}" for eid, rid in selected_list]

    def on_add_clicked(b):
        idx = dropdown.value
        eid = records[idx]['event_id']
        rid = records[idx]['run_id']
        pair = (eid, rid)
        if pair not in selected_list:
            selected_list.append(pair)
            update_selected_output()

    def on_remove_clicked(b):
        if selected_output.index is not None and selected_output.index < len(selected_list):
            selected_list.pop(selected_output.index)
            update_selected_output()

    def on_save_clicked(b):
        with open("selected_event_run_ids.txt", "w") as f:
            for eid, rid in selected_list:
                f.write(f"{eid},{rid}\n")
        print("Saved to selected_event_run_ids.txt")

    add_button.on_click(on_add_clicked)
    remove_button.on_click(on_remove_clicked)
    save_button.on_click(on_save_clicked)

    # Layout
    nav_row = widgets.HBox([prev_button, next_button, label_display])

    selection_controls = widgets.HBox([add_button, remove_button, save_button])
    selection_panel = widgets.VBox([selected_output, selection_controls])

    ui = widgets.VBox([dropdown, nav_row, selection_panel, out, plot])

    display(ui)
    # Initial display
    with out:
        draw_record_k3d(records[dropdown.value], plot, objects)
