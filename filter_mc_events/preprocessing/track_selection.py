# %% [markdown]
# # Track Selection
# - Cluster hits using DBSCAN
# PCA and centroid are employed.

# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from numpy.lib import recfunctions as rfn


def load_file(finpath):
    """Load hits from h5 file and append event_id to hits."""
    with h5py.File(finpath, "r") as fin:
        hits = fin["charge/calib_prompt_hits/data"][:]

        events = fin["charge/events/data"][:]
        hits_events_ref = fin["charge/events/ref/charge/calib_prompt_hits/ref"]
        hits_events = events[hits_events_ref[:, 0]]
        hits = rfn.append_fields(hits, "event_id", hits_events["id"], usemask=False)
        hits = rfn.append_fields(
            hits, "n_ext_trigs", hits_events["n_ext_trigs"], usemask=False
        )
        # TODO; I need information whether an event can have >1 external trigs.
        # print(fin["charge/ext_trigs/ref/charge/events/ref_region"][:30],
        #       len(fin["charge/ext_trigs/data"]),
        #       np.sum(events["n_ext_trigs"] > 0),
        #       len(fin["charge/events/data"]))
        # print(fin["charge/events/ref/charge/ext_trigs/ref"][:10],
        # len(fin["charge/events/ref/charge/ext_trigs/ref"]))
        # trig_io_group = np.full(-1, len(events))

        uni_event_ids = np.unique(events["id"])

    return hits, uni_event_ids


def filter_min_n_ext_trigs(hits, n_ext_trigs):
    """Filter hits based on number of external triggers in their events."""
    return hits[hits["n_ext_trigs"] >= n_ext_trigs]


def load_event(hits, event_id):
    event_mask = hits["event_id"] == event_id
    return hits[event_mask]


def load_io_group(hits, io_group):
    io_group_mask = hits["io_group"] == io_group
    return hits[io_group_mask]


def cluster_hits(hits, eps=3, min_samples=5, min_samples_total=30):
    xyz = np.vstack((hits["x"], hits["y"], hits["z"])).T
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(xyz)
    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    cluster_hits_in_event = []

    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        n_hits_in_cluster = np.sum(cluster_mask)
        if n_hits_in_cluster < min_samples_total:
            labels[cluster_mask] = -1
            continue
        cluster_hits = rfn.append_fields(
            hits[cluster_mask],
            "cluster_id",
            np.full(n_hits_in_cluster, cluster_id),
            usemask=False,
        )
        cluster_hits_in_event.append(cluster_hits)

    return cluster_hits_in_event


def track_fitting(hits, pca_tolerance=0.1, cut_fraction=0.2):
    ok = True
    xyz = np.vstack((hits["x"], hits["y"], hits["z"])).T
    centroid = np.mean(xyz, axis=0)
    points = xyz - centroid

    pca = PCA(n_components=2)
    pca.fit(points)
    explained_ratio = pca.explained_variance_ratio_[0]
    direction = pca.components_[0]
    if explained_ratio < pca_tolerance:
        ok = False
        return ok, None, None, None, None

    projections = points @ direction

    proj_min = np.min(projections)
    proj_max = np.max(projections)
    range_cut = (proj_max - proj_min) * cut_fraction
    lower_cut = proj_min + range_cut
    upper_cut = proj_max - range_cut

    mask = (projections > lower_cut) & (projections < upper_cut)
    selected_hits = hits[mask]

    selected_xyz = np.vstack(
        (selected_hits["x"], selected_hits["y"], selected_hits["z"])
    ).T
    centroid = np.mean(selected_xyz, axis=0)
    pca = PCA(n_components=2)
    pca.fit(selected_xyz - centroid)
    direction = pca.components_[0]
    explained_ratio = pca.explained_variance_ratio_[0]
    if explained_ratio < pca_tolerance:
        ok = False
        return ok, None, None, None, None

    dropped_hits = hits[~mask]

    return ok, direction, centroid, selected_hits, dropped_hits


def plot_track(hits, selected, dropped, direction, centroid, eid, io_group, cluster_id):
    fig = plt.figure(figsize=(12, 10))

    # Split hits into out-cluster (not selected)
    out_cluster = hits[hits["cluster_id"] != cluster_id]
    in_cluster = selected

    # Get projection of in-cluster hits onto direction
    diffs = np.stack(
        [
            in_cluster["x"] - centroid[0],
            in_cluster["y"] - centroid[1],
            in_cluster["z"] - centroid[2],
        ],
        axis=1,
    )
    projections = np.dot(diffs, direction / np.linalg.norm(direction))
    t_min = projections.min()
    t_max = projections.max()
    t = np.linspace(t_min, t_max, 100)
    points = centroid + np.outer(t, direction)

    # Plot X vs Y
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.scatter(
        out_cluster["x"],
        out_cluster["y"],
        c="grey",
        label="Other clusters",
        marker=".",
        alpha=0.5,
    )
    ax1.scatter(
        dropped["x"],
        dropped["y"],
        c="orange",
        label="Dropped",
        marker="x",
        s=2,
    )
    ax1.scatter(
        in_cluster["x"],
        in_cluster["y"],
        c="b",
        label=f"Cluster {cluster_id}",
        marker="o",
        s=2,
    )
    ax1.plot(points[:, 0], points[:, 1], color="r", linewidth=2, label="Fitted line")
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")
    ax1.set_title("X vs Y")
    ax1.legend()

    # Plot Y vs Z
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(
        out_cluster["y"],
        out_cluster["z"],
        c="grey",
        label="Other clusters",
        marker=".",
        alpha=0.5,
    )
    ax2.scatter(
        dropped["y"],
        dropped["z"],
        c="orange",
        label="Dropped",
        marker="x",
        s=2,
    )
    ax2.scatter(
        in_cluster["y"],
        in_cluster["z"],
        c="b",
        label=f"Cluster {cluster_id}",
        marker="o",
        s=2,
    )
    ax2.plot(points[:, 1], points[:, 2], color="r", linewidth=2, label="Fitted line")
    ax2.set_xlabel("Y (mm)")
    ax2.set_ylabel("Z (mm)")
    ax2.set_title("Y vs Z")
    ax2.legend()

    # Plot X vs Z
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(
        out_cluster["x"],
        out_cluster["z"],
        c="grey",
        label="Other clusters",
        marker=".",
        alpha=0.5,
    )
    ax3.scatter(
        dropped["x"],
        dropped["z"],
        c="orange",
        label="Dropped",
        marker="x",
        s=2,
    )
    ax3.scatter(
        in_cluster["x"],
        in_cluster["z"],
        c="b",
        label=f"Cluster {cluster_id}",
        marker="o",
        s=2,
    )
    ax3.plot(points[:, 0], points[:, 2], color="r", linewidth=2, label="Fitted line")
    ax3.set_xlabel("X (mm)")
    ax3.set_ylabel("Z (mm)")
    ax3.set_title("X vs Z")
    ax3.legend()

    # 3D plot
    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    ax4.scatter(
        out_cluster["x"],
        out_cluster["y"],
        out_cluster["z"],
        c="grey",
        marker=".",
        alpha=0.5,
        label="Other clusters",
    )
    ax4.scatter(
        dropped["x"],
        dropped["y"],
        dropped["z"],
        c="orange",
        marker="x",
        s=2,
        label="Dropped",
    )
    ax4.scatter(
        in_cluster["x"],
        in_cluster["y"],
        in_cluster["z"],
        c="b",
        marker="o",
        label=f"Cluster {cluster_id}",
        s=2,
    )
    ax4.plot(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        color="r",
        linewidth=2,
        label="Fitted line",
    )
    ax4.set_xlabel("X (mm)")
    ax4.set_ylabel("Y (mm)")
    ax4.set_zlabel("Z (mm)")
    ax4.set_title("3D View")
    ax4.legend()

    fig.suptitle(
        f"Event {eid} - IO Group {io_group} - Cluster {cluster_id}", fontsize=16
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def main():
    finpath = "/home/yousen/Public/ndlar_shared/data_reflowv5_20250708/packet-0050015-2024_07_08_13_37_49_CDT.FLOW.hdf5"

    n_min_hits_global = 20

    eps = 3
    min_samples = 5  # 3cm/sqrt(2) / 0.4434cm ~ 4.7 -> 4

    min_samples_total = 30  # arbitrary

    hits, uni_event_ids = load_file(finpath)
    hits = filter_min_n_ext_trigs(hits, n_ext_trigs=1)

    for eid in uni_event_ids[:10]:
        event_hits = load_event(hits, eid)
        # print(f"Event {eid} has {len(event_hits)} hits.")
        for io_group in np.unique(event_hits["io_group"]):
            io_group_hits = load_io_group(event_hits, io_group)
            if len(io_group_hits) < n_min_hits_global:
                continue
            # print(f"  IO group {io_group} has {len(io_group_hits)} hits.")
            clustered_hits = cluster_hits(
                io_group_hits,
                eps=eps,
                min_samples=min_samples,
                min_samples_total=min_samples_total,
            )

            track_hits = []

            for selected_hits in clustered_hits:
                track_ok, direction, centroid, fitted_hits, dropped_hits = (
                    track_fitting(selected_hits, pca_tolerance=0.1, cut_fraction=0.15)
                )
                if not track_ok:
                    continue
                cluster_id = np.unique(selected_hits["cluster_id"])
                print(cluster_id)
                assert len(cluster_id) == 1, "More than one cluster found"
                # print(f"Event {eid}, IO group {io_group},"
                #       f" Cluster {cluster_id}: Track found with direction"
                #       f"{direction} and centroid {centroid}, {len(fitted_hits)}"
                #       f" hits used for fitting.")
                track_hits.append(
                    (fitted_hits, dropped_hits, direction, centroid, cluster_id[0])
                )
            selected_track = track_hits[np.argmax([len(th[0]) for th in track_hits])]

            clustered_hits = np.concatenate(clustered_hits)
            print(len(clustered_hits), len(selected_track[0]))


            plot_track(
                hits=clustered_hits,
                selected=selected_track[0],
                dropped=selected_track[1],
                direction=selected_track[2],
                centroid=selected_track[3],
                edi=eid,
                io_group=io_group,
                cluster_id=selected_track[4],
            )


main()

if __name__ == "__main__":
    main()
