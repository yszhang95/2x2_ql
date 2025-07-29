#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import uproot
import h5py
import numpy as np

import numpy.lib.recfunctions as rfn
import math

import sys

import matplotlib
import matplotlib.pyplot as plt

import os

from scipy.spatial.distance import pdist, squareform, cdist

from sklearn.neighbors import KDTree

from sklearn.cluster import DBSCAN

from sklearn.linear_model import LinearRegression

from hist import Hist
import scipy

def normalized_dx(s_y, s_z):
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_sy2 = 1.0 / s_y**2
        inv_sz2 = 1.0 / s_z**2

    denom = 1 + inv_sy2 + inv_sz2

    # Handle infinite slope (dx = 0) ? normalized dx = 0
    if np.isinf(denom):
        return 0.0
    else:
        return 1 / np.sqrt(denom)


def split_sorted_dataset(keys):
    uqk, idx_split = np.unique(keys, return_index=True)
    # print(len(uqk), len(idx_split))
    idx_split = list(idx_split) + [len(keys)]
    for i, uq in enumerate(uqk):
        yield uq, slice(idx_split[i], idx_split[i+1])

def plot_three_views(hits, axs=None, **kwargs):
    # ridx, cidx
    axis_dict = {
        # y vs. x
        (0, 0) : { 'label' : {'x' : 'x [cm]', 'y' : 'y [cm]'},
                    'key' : {'x' : 'x', 'y' : 'y'}},
        # (0, 1) : # empty
        (1, 0) : { 'label' : {'x' : 'x [cm]', 'y' : 'z [cm]'},
                  'key' : {'x' : 'x', 'y' : 'z'}},
        (1, 1) : { 'label' : {'x' : 'y [cm]', 'y' : 'z [cm]'},
                  'key' : {'x' : 'y', 'y' : 'z'}}
    }
    all_hits = kwargs.get('all_hits', None)
    if isinstance(all_hits, np.ndarray):
        for k, v in axis_dict.items():
            axs[k[0], k[1]].scatter(all_hits[v['key']['x']], all_hits[v['key']['y']], label='all hits')
    for k, v in axis_dict.items():
        axs[k[0], k[1]].scatter(hits[v['key']['x']], hits[v['key']['y']], label=kwargs.get('label', 'selected'))

    for k, v in axis_dict.items():
        axs[k[0], k[1]].set_xlabel(v['label']['x'])
        axs[k[0], k[1]].set_ylabel(v['label']['y'])
        axs[k[0], k[1]].legend()

    # return fig


def line1D(X, y):
    reg = LinearRegression().fit(X.reshape(-1, 1), y)
    return reg.coef_[0], reg.intercept_

def yz_line(hits):
    '''
    z = ky + b
    dz = kdy
    return k, b, (1/sqrt(1+k**2), k/sqrt(1+k**2))
    tagent vector = (1/sqrt(1+k**2), k/sqrt(1+k**2))
    '''
    # reg = LinearRegression().fit(hits['y'].reshape(-1, 1), hits['z'])
    # k = reg.coef_[0]
    # b = reg.intercept_
    k, b = line1D(hits['y'], hits['z'])
    return k, b, (1/np.sqrt(1+k**2), k/np.sqrt(1+k**2))

def proj_yz(hits, k_yz, b_yz):
    '''
    z = ky + b
    '''
    tg_yz = np.array([1/np.sqrt(1+k_yz**2), k_yz/np.sqrt(1+k_yz**2)])
    o_yz = (0, b_yz)
    points_yz = np.column_stack([hits['y'], hits['z']]) - o_yz
    d_tg = np.dot(points_yz, tg_yz)
    # for i in range(10):
    #     print(tg_yz[0]* points_yz[i][0] + tg_yz[1]* points_yz[i][1] - d_tg[i], points_yz[i], rock_muon_hits['y'][i], rock_muon_hits['z'][i], o_yz)
    d_nm = np.cross(points_yz, tg_yz)
    # for i in range(10):
    #     print(-tg_yz[0]* points_yz[i][1] + tg_yz[1]* points_yz[i][0] - d_nm[i], points_yz[i], rock_muon_hits['y'][i], rock_muon_hits['z'][i], o_yz, tg_yz)
    return d_tg, d_nm

def xline(hits, reflabel='d_tg'):
    '''
    x = k* reflabel + b
    '''
    k, b = line1D(hits[reflabel], hits['x'])
    return k, b, (1/np.sqrt(1+k**2), k/np.sqrt(1+k**2))


def plot_line(k, b, ax, **kwargs):
    x0, x1 = ax.get_xlim()
    x = np.arange(x0, x1, step=0.01)
    y = k * x + b
    ax.plot(x, y, label='line fit', c=kwargs.get('color'))
    ax.legend()

from scipy.spatial import KDTree

def find_closest_point_2dgrid(p, points):
    """
    Find the (i1, i2) index of the point in a 2D grid of 2D points closest to `p`.

    Parameters:
        p (array-like): A 2D point (shape: (2,))
        points (ndarray): A grid of 2D points (shape: (N1, N2, 2))

    Returns:
        tuple: (i1, i2) index of the closest point
    """
    flat_points = points.reshape(-1, 2)
    tree = KDTree(flat_points)

    # Query all at once
    distances, flat_indices = tree.query(p)

    # Convert flat indices back to (i1, i2)
    i1, i2 = np.unravel_index(flat_indices, points.shape[:2])

    return distances, (i1, i2)

# not validated
def sel_uni_pxl(hits, att='Q', yposlabel='y', zposlabel='z', thresholds=None):
    thres  = np.zeros(hits.shape[0], dtype=float)
    points_yz = np.column_stack([hits['io_group'], hits[yposlabel], hits[zposlabel]])

    points_yz1000 = points_yz[:,1:] * 1000
    if thresholds is None:
        thresholds = []
        with h5py.File('threshold_summary.hdf5', 'r') as fthres:
            for ig in range(1,9):
                thresholds.append(fthres[f'io_group{ig}/threshold'][:])
                m = points_yz[:,0] == ig
                tgtyz1000 = np.stack([thresholds[-1]['y_i'], thresholds[-1]['z_i']], axis=-1)
                d2d, indices_2d = find_closest_point_2dgrid(points_yz1000[m], tgtyz1000)
                m2 = d2d > 100
                if len(thres[m]) > 0:
                    thres[m] = thresholds[-1][indices_2d]['Q']
                    thres[m][m2] = 1E19

    clustering = DBSCAN(eps=0.0001, min_samples=1).fit(points_yz)
    selected_hit = []
    unique_labels = np.unique(clustering.labels_)
    totQ = np.zeros(hits.shape[0], dtype=float)
    totN = np.zeros(hits.shape[0], dtype=int)
    totQ_cp = totQ.copy()
    totN_cp = totN.copy()
    accQ = np.zeros(hits.shape[0], dtype=float)
    avg_i = np.zeros(hits.shape[0], dtype=float)
    dt = np.zeros(hits.shape[0], dtype=float)
    tinterval = np.zeros(hits.shape[0], dtype=float)
    tindex = np.zeros(hits.shape[0], dtype=int)

    dist = pdist((points_yz - clustering.components_).reshape(-1, 1))
    # if np.sum(dist)>1E-6:
    #     print(np.sum(dist))
    # print(np.sum(dist))
    print('negative Q', hits['Q'][hits['Q']<0])

    for ilabel, unique_label in enumerate(unique_labels):
        m = clustering.labels_ == unique_label
        idxs = np.asarray(m).nonzero()[0]
        d = hits[m]
        selected_hit.append(d[np.argmax(d[att])])
        totQ[ilabel] = np.sum(d['Q'])
        totN[ilabel] = len(d)
        totQ_cp[m] = np.sum(d['Q'])
        totN_cp[m] = len(d)
        sorted_indices = np.argsort(d['t_drift'])
        dt[m] = hits['t_drift'][m] - selected_hit[-1]['t_drift']
        q = 0

        io_group = d['io_group'][0]
        threshold = thres[m][0]
        for i in range(len(sorted_indices)):
            q += d['Q'][sorted_indices[i]]
            accQ[idxs[sorted_indices[i]]] = np.float64(q)
            if i == 0:
                avg_i[idxs[sorted_indices[i]]] = (d['Q'][sorted_indices[i]]-threshold)/17
                tinterval[idxs[sorted_indices[i]]] = 17
            else:
                avg_i[idxs[sorted_indices[i]]] = d['Q'][sorted_indices[i]] / (d['t_drift'][sorted_indices[i]] - d['t_drift'][sorted_indices[i-1]])
                tinterval[idxs[sorted_indices[i]]] = d['t_drift'][sorted_indices[i]] - d['t_drift'][sorted_indices[i-1]]
            tindex[idxs[sorted_indices[i]]] = i
    uni_pxl = np.array(selected_hit, dtype=hits.dtype)
    totQ = totQ[:len(uni_pxl)]
    totN = totN[:len(uni_pxl)]
    uni_pxl = rfn.append_fields(uni_pxl, names=['totQ', 'totN'], data=[totQ, totN], usemask=False)
    extended_hits = rfn.append_fields(hits, names=['totQ', 'totN'], data=[totQ_cp, totN_cp], usemask=False)
    extended_hits = rfn.append_fields(extended_hits, names=['accQ', 'dt', 'avg_i', 'thres'], data=[accQ, dt, avg_i, thres], usemask=False)
    extended_hits = rfn.append_fields(extended_hits, names=['tinterval', 'tindex'], data=[tinterval, tindex], usemask=False)
    return uni_pxl, extended_hits

# Use existing functions from your provided code: split_sorted_dataset, sel_uni_pxl, yz_line, proj_yz, xline, plot_three_views

# Load file once
def prep_per_event(hits, highq_thres=None):
    uni_pxls, extended_hits = sel_uni_pxl(hits, att='Q', yposlabel='y', zposlabel='z')

    k_yz, b_yz, tg_yz = yz_line(uni_pxls)
    d_tg, d_nm = proj_yz(extended_hits, k_yz, b_yz)
    extended_hits = rfn.append_fields(extended_hits, names=('d_tg', 'd_nm'), data=(d_tg, d_nm), usemask=False)

    if highq_thres is not None:
        xyhits = extended_hits[extended_hits['Q'] > highq_thres]
    else:
        xyhits = uni_pxls
    k_x, b_x, _ = xline(xyhits, reflabel='y')
    ref_x = k_x * extended_hits['y'] + b_x

    # direction
    reg_xy = LinearRegression().fit(uni_pxls['y'][:,None], uni_pxls['x'])
    reg_xz = LinearRegression().fit(uni_pxls['z'][:,None], uni_pxls['x'])
    para = normalized_dx(reg_xy.coef_[0], reg_xz.coef_[0])

    dx = extended_hits['x'] - ref_x
    sign = np.where(extended_hits['io_group'] % 2, 1.0, -1.0)
    dx *= sign
    extended_hits = rfn.append_fields(extended_hits, names='dx', data=dx, usemask=False)
    # if not (para < 0.05):
    #     # print('filtered', para)
    #     # print(reg.coef_[0], reg.coef_[1])
    #     return np.array([], dtype=extended_hits.dtype)
    # else:
    #     # print('----------------- kept', para, np.degrees(np.arcsin(np.abs(para))), extended_hits['event_id'][0])
    #     # print('dx', np.max(extended_hits['x'])-np.min(extended_hits['x']), 'dy', np.max(extended_hits['y']), np.min(extended_hits['y']), 'dz', np.max(extended_hits['z']), np.min(extended_hits['z']))
    #     pass
    return extended_hits


with uproot.recreate('track_data.root') as f:

    # fh5 = h5py.File('/home/yousen/Public/ndlar_shared/data/data_reflowv5_20250510/packet-0050015-2024_07_08_13_37_49_CDT.FLOW_selected.hdf5', 'r')
    # fh5 = h5py.File('check_selection/selected_data_small_angle/packet-0050015-2024_07_08_13_37_49_CDT.FLOW_selected.hdf5')
    try:
        path = sys.argv[1]
    except IndexError:
        path = '/home/yousen/Public/ndlar_shared/data_reflowv5_20250708/packet-0050015-2024_07_08_13_37_49_CDT.FLOW_selected.hdf5'
    fh5 = h5py.File(path)
    # fh5 = h5py.File("web/many_muon_hits_selected.hdf5")
    dh5 = fh5['/selected/hits/data']
    data = dh5[:][np.argsort(dh5['event_id'])]
    groups = list(split_sorted_dataset(data['event_id']))

    out_hits = []
    distances = []
    for g, slc in groups:
        dg = data[slc]
        for itpc in [1,2,3,4,7,8]:
            inhits = dg[dg['io_group'] == itpc]
            if len(inhits) == 0:
                continue
            hits = prep_per_event(inhits)
            if len(hits) < 20:
                continue
            xyz = np.vstack([hits['x'], hits['y'], hits['z']]).T
            max_dist = np.max(pdist(xyz))
            distances.append(float(max_dist))
            out_hits.append(hits)
    # print(out_hits)
    f['selected_data/hits'] = np.concatenate(out_hits)
    f['selected_data/distances'] = { "distance" : np.array(distances)}
