#!/usr/bin/env python3
# coding: utf-8

# pick_pgun.py
import numpy as np
import numpy.lib.recfunctions as rfn
import h5py
import argparse
import sklearn


# event_id matching
# event_id/1000 --> event_id
# (event_id/100) % 10 --> io_group

def load_source_data(source_hdf5):
    """
    Load picked points and event_ids from source HDF5 file.
    """
    with h5py.File(source_hdf5, 'r') as f:
        # selected = f['/hits/selected/data'][:]
        # deselected = f['/hits/deselected/data'][:]
        selected = f['/selected/hits/data'][:]
        deselected = f['/deselected/hits/data'][:]
        points = f['picked/points/data'][:]
        event_ids = f['picked/event_id/data'][:]
        io_group_ids = f['picked/io_group/data'][:]
    # split points according to event_id
    lower_z = points[:, 2] < points[:, 5]  # False -->0 ,True --> 1
    p_min = np.where(lower_z[:, np.newaxis], points[:, :3], points[:, 3:])
    p_max = np.where(~lower_z[:, np.newaxis], points[:, :3], points[:, 3:])
    # I want a list of arrays, each array corresponds to an event_id
    selected_data = []
    deselected_data = []
    eids = io_group_ids + event_ids * 10
    for eid in eids:
        smask = (selected['event_id'] * 10 + selected['io_group']) == eid
        dmask = (deselected['event_id'] * 10 + deselected['io_group']) == eid
        sel = selected[smask]
        desel = deselected[dmask]
        selected_data.append(sel)
        deselected_data.append(desel)
    return p_min, p_max, eids, selected_data, deselected_data


def proj_dist(points, p_min, p_max, threshold):
    """
    For each set of points, determine if they are within threshold distance
    from the line segment defined by p_min and p_max.
    """
    line_vec = p_max - p_min
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        raise ValueError("line length is zero")
    line_unitvec = line_vec / line_len
    point_vec = points - p_min[np.newaxis, :]
    t = np.dot(point_vec, line_unitvec)
    t = np.clip(t, 0, line_len)
    nearest = p_min[np.newaxis, :] + t[:, None] * line_unitvec[np.newaxis, :]
    dists = np.linalg.norm(points - nearest, axis=1)
    return dists < threshold


def split_events(points):
    """
    Split points into a list of arrays according to event_id.
    """
    args = np.argsort(points['event_id'])
    sorted_pts = points[args]
    _, idx = np.unique(sorted_pts['event_id'], return_index=True)
    pts = [sorted_pts[idx[i]:idx[i+1]] for i in range(len(idx)-1)]
    pts.append(sorted_pts[idx[-1]:])
    return pts


def compute_dqdx(selected_hits, bin_width=2.0, qthres=10, nvalid=5):
    """
    Compute dQ/dx along the given direction using selected hits.
    selected_hits: structured array with fields 'x', 'y', 'z', 'Q'
    bin_width: width of each bin in cm
    qthres: minimum total charge to consider the hit valid
    """
    highq = selected_hits['Q'] > qthres

    highq_hits = selected_hits[highq]

    if len(highq_hits) < nvalid:
        # not enought hits above threshold
        return np.array([]), np.array([]), np.array([np.nan]*3), \
            np.array([np.nan]*3)
        # raise RuntimeError("No hits above Q threshold; cannot compute dQ/dx")

    centroid = np.mean(np.vstack([highq_hits['x'], highq_hits['y'],
                                  highq_hits['z']]).T, axis=0)
    pca = sklearn.decomposition.PCA(n_components=1)

    hit_xyz = np.vstack([highq_hits['x'], highq_hits['y'], highq_hits['z']]).T
    pca.fit(hit_xyz - centroid)

    direction = pca.components_[0]
    if direction[-1] < 0:
        direction = -direction

    # Project selected hits onto direction
    hit_xyz = np.stack([selected_hits['x'], selected_hits['y'],
                        selected_hits['z']], axis=1)
    hit_q = selected_hits['Q']

    relative_positions = hit_xyz - centroid
    projections = np.dot(relative_positions, direction)

    proj_min, proj_max = projections.min(), projections.max()
    bin_edges = np.arange(proj_min, proj_max + bin_width, bin_width)
    if bin_edges[-1] < proj_max:
        bin_edges = np.append(bin_edges, proj_max)

    hist, _ = np.histogram(projections, bins=bin_edges, weights=hit_q)
    bin_widths = np.diff(bin_edges)
    dqdx = hist / bin_widths
    bin_centers = bin_edges[:-1] + bin_widths / 2

    # 3d point
    p_min = proj_min * direction + centroid  # shape (3,)
    p_max = proj_max * direction + centroid  # shape (3,)

    return bin_centers, dqdx, p_min, p_max


def main():
    parser = argparse.ArgumentParser(description="Generate hdf5 from hits or"
                                     " effq according to picked data.")
    parser.add_argument("source_hdf5",
                        help="Path to the source HDF5 file (required).")
    parser.add_argument("pgun_hdf5",
                        help="Patern of path to the hits HDF5 file (required)."
                        " e.g. a_eventid{}_b.hdf5")
    parser.add_argument("out_hdf5",
                        help="Path to the output HDF5 file (required).")
    parser.add_argument("--dtype", choices=["hits", "effq"],
                        default="hits", help="hits or effq")
    parser.add_argument("--no_sel", action="store_true",
                        help="Do not select hits, use all hits in the source HDF5 file.")

    args = parser.parse_args()

    fout_hdf5 = args.out_hdf5
    if ("hits.hdf5" not in fout_hdf5) and ("effq.hdf5" not in fout_hdf5):
        raise ValueError(f"Output HDF5 file name {fout_hdf5} must contain 'hits.hdf5' or "
                         "'effq.hdf5' to indicate the type of data.")

    # global parameters
    dist_thres = 2  # cm
    dqdx_diff = 1E5
    dx_width = 2.0
    qthres = 10 if args.dtype == "hits" else 0

    p_min, p_max, event_ids, selected_source, deselected_source = \
        load_source_data(args.source_hdf5)

    # load pgun data

    with h5py.File(args.pgun_hdf5, 'r') as fpgun:
        points = fpgun[args.dtype][:]

    sel_pts = []
    desel_pts = []
    pts_minmax = []
    dqdx_raw = []
    eids = []
    dqdx_raw_src = []
    trk_dist = []
    trk_dist_src = []
    for ie in range(len(event_ids)):
        print(f"Event {ie}: id={event_ids[ie]}, p_min={p_min[ie]},"
              f" p_max={p_max[ie]} from {args.source_hdf5}")
        # match event_id in pgun data
        io_group = event_ids[ie] % 10
        # FIXME: hardwired numbers
        if io_group in [5, 6]: # module 2;
            continue
        iemask = points['event_id']//100 == event_ids[ie]
        # pts_per_run = split_events(points[iemask])
        # for pts in pts_per_run:
        #     xyzs = [np.vstack([p['x'], p['y'], p['z']]).T for p in pts]
        #     proj_masks = [proj_dist(xyzs[j], p_min[ie], p_max[ie], dist_thres)
        #                   for j in range(len(xyzs))]
        #     selected_pts = [pts[i][proj_masks[i]] for i in range(len(pts))]
        #     deselected_pts = [pts[i][~proj_masks[i]] for i in range(len(pts))]
        pts = split_events(points[iemask])
        xyzs = [np.vstack([p['x'], p['y'], p['z']]).T for p in pts]
        proj_masks = [proj_dist(xyzs[j], p_min[ie], p_max[ie], dist_thres)
              for j in range(len(xyzs))]
        selected_pts = [pts[i][proj_masks[i]] for i in range(len(pts))]
        deselected_pts = [pts[i][~proj_masks[i]] for i in range(len(pts))]

        # compute dQdx in selected hits in source
        _, dqdx_per_source, p1, p2 = compute_dqdx(selected_source[ie],
                                                dx_width, qthres)

        dist_src = np.linalg.norm(p2 - p1)

        if len(dqdx_per_source) == 0:
            continue
        # mean dqdx of the an run/event in the source
        dqdx_mean_per_source = np.mean(dqdx_per_source)
        # filter out mean dqdx awayfrom the dqdx_mean by 2;

        dropped = []

        proj_min = []
        proj_max = []
        mean_dqdx_per_event = []
        eid_per_event = []
        rid_per_event = []
        dqdx_raw_src_per_event = []
        trk_dist_per_event = []
        trk_dist_src_per_event = []
        for j, (sel, desel) in enumerate(zip(selected_pts, deselected_pts)):
            _, dqdx, pt_min, pt_max = compute_dqdx(sel, dx_width, qthres)
            proj_min.append(pt_min)
            proj_max.append(pt_max)
            mean_dqdx_per_event.append(np.mean(dqdx))
            eid_per_event.append(sel['event_id'][0])
            rid_per_event.append(event_ids[ie])
            dqdx_raw_src_per_event.append(dqdx_mean_per_source)
            trk_dist_per_event.append(np.linalg.norm(pt_max - pt_min))
            trk_dist_src_per_event.append(dist_src)
            if len(dqdx) == 0:
                dropped.append(True)
                continue
            if np.abs(np.mean(dqdx) - dqdx_mean_per_source) > dqdx_diff:
                # print(f"Filtering out hit {j} with mean dqdx {np.mean(dqdx)}")
                # pop out the element
                dropped.append(True)
            else:
                dropped.append(False)
        # does not drop anything in effq
        if args.dtype == "effq":
            dropped = []
        if args.no_sel:
            # if no selection, do not drop anything
            dropped = []
        for j in range(len(dropped)):
            ind = len(dropped) - j - 1
            if dropped[ind]:
                del deselected_pts[ind]
                del selected_pts[ind]
                del mean_dqdx_per_event[ind]
                del proj_min[ind]
                del proj_max[ind]
                del eid_per_event[ind]
                del rid_per_event[ind]
                del dqdx_raw_src_per_event[ind]
                del trk_dist_per_event[ind]
                del trk_dist_src_per_event[ind]

        if len(selected_pts):
            # add a new field run id to event_ids from source ...
            selected_pts = [rfn.append_fields(
                sel, 'run_id', np.full(len(sel), event_ids[ie]),
                usemask=False) for sel in selected_pts]
            deselected_pts = [rfn.append_fields(
                desel, 'run_id', np.full(len(desel), event_ids[ie]),
                usemask=False) for desel in deselected_pts]
            sel_pts.append(np.concatenate(selected_pts))
            pts_minmax.append(
                np.hstack([np.array(proj_min), np.array(proj_max)])
            )
            dqdx_raw.append(np.array(mean_dqdx_per_event))
            eids.append(np.array(eid_per_event))
            dqdx_raw_src.append(np.array(dqdx_raw_src_per_event))
            trk_dist.append(np.array(trk_dist_per_event))
            trk_dist_src.append(np.array(trk_dist_src_per_event))

            assert np.isnan(pts_minmax[-1]).sum() == 0, "NaN in pts_minmax"
            # desel_pts is meaningful only sel_pts passes quality
            if len(deselected_pts):
                desel_pts.append(np.concatenate(deselected_pts))
            else:
                empty = np.zeros((0,), dtype=deselected_pts[0].dtype)
                desel_pts.append(empty)

    with h5py.File(args.out_hdf5, 'w') as fout:
        fout.create_dataset('/selected/hits/data',
                            data=np.concatenate(sel_pts))
        fout.create_dataset('/deselected/hits/data',
                            data=np.concatenate(desel_pts))
        pts_minmax = np.concatenate(pts_minmax)
        fout.create_dataset('picked/points/data', data=pts_minmax)
        direction = pts_minmax[:, 3:] - pts_minmax[:, :3]
        direction = direction / np.linalg.norm(direction, axis=1)[:, None]
        fout.create_dataset('picked/direction/data', data=direction)
        fout.create_dataset('picked/dqdx_raw/data',
                            data=np.concatenate(dqdx_raw))
        fout.create_dataset('picked/dqdx_raw_src/data',
                            data=np.concatenate(dqdx_raw_src))
        fout.create_dataset('picked/trk_dist/data',
                            data=np.concatenate(trk_dist))
        fout.create_dataset('picked/trk_dist_src/data',
                            data=np.concatenate(trk_dist_src))
        fout.create_dataset('picked/event_id', data=np.concatenate(eids))
        fout.create_dataset('source_file', data=np.bytes_(args.source_hdf5))


if __name__ == "__main__":
    main()
