# app.py
import os
import numpy as np
import h5py
from threading import Lock
from flask import Flask, request, render_template_string, abort

import k3d

# -----------------------------
# Data loading (adapted from your notebook)
# -----------------------------
def load_files(input2_path):
    with h5py.File(input2_path, 'r') as f2:
        # source_path = f2['/source_file'][()].decode('utf-8')
        sel = f2['/hits/selected/data'][:]
        des = f2['/hits/deselected/data'][:]
        pts = f2['/picked/points/data'][:]
        pick_eids = f2['/picked/event_id/data'][:]
        pick_io_group = f2['/picked/io_group/data'][:]

    eids = set(np.unique(sel['event_id'])) | set(np.unique(des['event_id']))

    records = []
    for i, eid in enumerate(eids):
        # rid = int(rids[i])
        j = (pick_eids == eid)
        if len(j) == 0:
            pmin = np.empty((0, ))
            pmax = np.empty((0, ))
        else:
            pmin = pts[j, :3].tolist()
            pmax = pts[j, 3:].tolist()

        mask_sel = (sel['event_id'] == eid)
        mask_des = (des['event_id'] == eid)

        h_sel = np.vstack([sel[mask_sel]['x'], sel[mask_sel]['y'], sel[mask_sel]['z']]).T if np.any(mask_sel) else np.empty((0,3))
        h_des = np.vstack([des[mask_des]['x'], des[mask_des]['y'], des[mask_des]['z']]).T if np.any(mask_des) else np.empty((0,3))

        records.append({
            'event_id': eid,
            'line': {'p_min': pmin, 'p_max': pmax},
            'source': {'selected': h_sel, 'deselected': h_des}
        })
    return records


# -----------------------------
# K3D helpers (volumes + scene)
# -----------------------------
def add_bounding_boxes(plot, all_bounds, color=0x888888, width=0.5, name='volume'):
    cube_edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # top
        [0, 4], [1, 5], [2, 6], [3, 7],  # vertical
    ], dtype=np.uint32)

    vertices_list = []
    indices_list = []
    offset = 0
    for bounds in all_bounds:
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
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
        vertices_list.append(corners)
        indices_list.append(cube_edges + offset)
        offset += 8

    vertices = np.vstack(vertices_list).astype(np.float32)
    indices = np.vstack(indices_list).astype(np.uint32)

    box_lines = k3d.lines(vertices=vertices, indices=indices, color=color, indices_type='segment', width=width)
    box_lines.name = name
    box_lines.pickable = False
    plot += box_lines


def init_k3d_plot():
    # detector volumes (copied from your code)
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

    plot = k3d.plot(grid_visible=False, grid_auto_fit=True, camera_auto_fit=True, height=700)
    add_bounding_boxes(plot, boundaries)
    return plot


def plot_for_record(rec):
    plot = init_k3d_plot()

    def add_points(arr, color, size, name):
        if arr.size == 0:
            return
        pts = arr.astype(np.float32)
        obj = k3d.points(pts, color=color, point_size=size, name=name)
        return obj

    # point clouds
    selected = add_points(rec['source']['selected'], 0xff0000, 0.5, 'source_sel')
    if selected is not None:
        plot += selected
    plot += add_points(rec['source']['deselected'],0xffa500, 0.5, 'source_desel')

    # line
    if selected is not None:
        line_pts = np.array([rec['line']['p_min'], rec['line']['p_max']], dtype=np.float32)
        line_pts = line_pts.reshape(-1, 3)
        indices = np.arange(len(line_pts))
        indices = indices.reshape(2, -1).T  # pairs of points
        plot += k3d.lines(line_pts, indices=indices,
                          color=0x000000, indices_type='segment',
                          shader="simple", name='track')


    return plot


# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)
RECORDS = []

_loaded = False
_load_lock = Lock()

def _load_records():
    """Load records once from env var (supports INPUT2_PATH or input2_path)."""
    global RECORDS
    path = os.environ.get("INPUT2_PATH") or os.environ.get("input2_path")
    if not path:
        raise RuntimeError("Set INPUT2_PATH (or input2_path) to your .h5 path.")
    if not os.path.exists(path):
        raise RuntimeError(f"INPUT2_PATH points to a missing file: {path}")
    RECORDS = load_files(path)
    print(f"Loaded {len(RECORDS)} records from {path}")

def _ensure_loaded():
    """Lazy, thread-safe one-time init; call at the top of any route."""
    global _loaded
    if _loaded:
        return
    with _load_lock:
        if not _loaded:
            _load_records()
            _loaded = True


INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>K3D + Flask viewer</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 16px; }
    .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    .col { display: flex; flex-direction: column; gap: 8px; }
    select, button { padding: 6px 10px; font-size: 14px; }
    #viewer { width: 100%; height: 720px; border: 0; }
    #selected { width: 320px; height: 180px; }
    .muted { color: #666; }
  </style>
</head>
<body>
  <h2>K3D Browser</h2>

  <div class="row">
    <label for="record">Event:</label>
    <select id="record">
      {% for i, r in enumerate(records) %}
        <option value="{{ i }}">
          E:{{ r['event_id'] }}
        </option>
      {% endfor %}
    </select>
    <button id="prev">← Prev</button>
    <button id="next">Next →</button>
    <span class="muted" id="label"></span>
  </div>

  <div class="row" style="margin-top:8px">
    <div class="col">
      <textarea id="selected" readonly placeholder="Selected E,R pairs will appear here..."></textarea>
      <div class="row">
        <button id="add" style="background:#16a34a;color:white;">Add Current</button>
        <button id="remove" style="background:#f59e0b;color:white;">Remove Last</button>
        <button id="save" style="background:#0ea5e9;color:white;">Save to File</button>
      </div>
    </div>
  </div>

  <iframe id="viewer" src="/view/0"></iframe>

  <script>
    const dd = document.getElementById('record');
    const viewer = document.getElementById('viewer');
    const label = document.getElementById('label');

    const addBtn = document.getElementById('add');
    const removeBtn = document.getElementById('remove');
    const saveBtn = document.getElementById('save');
    const selectedBox = document.getElementById('selected');

    function currentLabel() {
      return dd.options[dd.selectedIndex].text;
    }
    function updateLabel() { label.textContent = currentLabel(); }
    function goTo(idx) {
      if (idx < 0 || idx >= dd.options.length) return;
      dd.value = String(idx);
      viewer.src = '/view/' + idx; // load fresh K3D snapshot
      updateLabel();
    }

    document.getElementById('prev').onclick = () => goTo(parseInt(dd.value) - 1);
    document.getElementById('next').onclick = () => goTo(parseInt(dd.value) + 1);
    dd.onchange = () => goTo(parseInt(dd.value));
    updateLabel();

    // selection list helpers
    function parseSelected() {
      return selectedBox.value
        .split('\\n')
        .map(s => s.trim())
        .filter(Boolean);
    }
    function setSelected(lines) {
      selectedBox.value = lines.join('\\n');
    }
    addBtn.onclick = () => {
      const txt = currentLabel();
      const tag = txt.split(',')[0].trim().replace('Event:', '').replace('Event', '').trim() || txt;
      const lines = parseSelected();
      if (!lines.includes(tag)) {
        lines.push(tag);
        setSelected(lines);
      }
    };
    removeBtn.onclick = () => {
      const lines = parseSelected();
      lines.pop();
      setSelected(lines);
    };
    saveBtn.onclick = async () => {
      const payload = { items: parseSelected() };
      const res = await fetch('/save', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
      if (res.ok) { alert('Saved to selected_event_run_ids.txt'); }
      else { alert('Save failed'); }
    };
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    _ensure_loaded()
    if not RECORDS:
        abort(500, "No records loaded. Set INPUT2_PATH env var before running.")
    return render_template_string(INDEX_HTML, records=RECORDS, enumerate=enumerate)

@app.route("/view/<int:index>")
def view(index: int):
    _ensure_loaded()
    if index < 0 or index >= len(RECORDS):
        abort(404)
    plot = plot_for_record(RECORDS[index])
    # Return a self-contained HTML snapshot with the live K3D viewer.
    # (K3D docs: plot.get_snapshot() generates a standalone HTML.)  # noqa
    return plot.get_snapshot()  #

# your routes (/, /view/<int:index>, /save) stay the same

if __name__ == "__main__":
    _load_records()
    app.run(debug=True)
