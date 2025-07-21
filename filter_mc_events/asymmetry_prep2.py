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

ismc = False
thresholds = None
thres_yis = None
thres_zis = None
def load_threshold(threshold):
    '''
    A map of io group from data to MC should be done.
    Hard-coded map is provided for 2x2 geometry.
    Assume thresholds are aligned from lower to high ends.
    '''
    global thresholds
    global thres_yis
    global thres_zis
    try:
        threshold = float(threshold)
        thresholds = [threshold, ] * 8
        return
    except ValueError:
        pass
    # io_groups = [1, 2, 3, 4, 5, 6, 7, 8]
    # io_indices_tred = [1, 0, 3, 2, 5, 4, 7, 6]
    thresholds = []
    thres_yis = []
    thres_zis = []
    if ismc:
        orders = [2,1,4,3,6,5,8,7]
    else:
        orders = list(range(1,9))

    with h5py.File(threshold, 'r') as fthres:
        for ig in orders:
            thresholds.append(fthres[f'io_group{ig}/threshold']['Q'][:])
            thres_yis.append(fthres[f'io_group{ig}/threshold']['y_i'][:])
            thres_zis.append(fthres[f'io_group{ig}/threshold']['z_i'][:])

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

# not validated
def sel_uni_pxl(hits, att='Q', yposlabel='y', zposlabel='z'):
    points_yz = np.column_stack([hits['io_group'], hits[yposlabel], hits[zposlabel]])
    clustering = DBSCAN(eps=0.0001, min_samples=1).fit(points_yz)
    selected_hit = []
    unique_labels = np.unique(clustering.labels_)
    totQ = np.zeros(hits.shape[0], dtype=float)
    totN = np.zeros(hits.shape[0], dtype=int)
    totQ_cp = totQ.copy()
    totN_cp = totN.copy()
    accQ = np.zeros(hits.shape[0], dtype=float)
    ch_thres = np.zeros(hits.shape[0], dtype=float)
    avg_i = np.zeros(hits.shape[0], dtype=float)
    dt = np.zeros(hits.shape[0], dtype=float)
    tinterval = np.zeros(hits.shape[0], dtype=float)
    tindex = np.zeros(hits.shape[0], dtype=int)

    y_i = np.rint(hits['y'] * 1000).astype(int)
    z_i = np.rint(hits['z'] * 1000).astype(int)

    dist = pdist((points_yz - clustering.components_).reshape(-1, 1))
    # if np.sum(dist)>1E-6:
    #     print(np.sum(dist))
    # print(np.sum(dist))
    # print('negative Q', hits['Q'][hits['Q']<0])

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
        for i in range(len(sorted_indices)):
            # io_group = d['io_group'][sorted_indices[i]]
            io_group = d['io_group'][0]
            # fixme: hard coded
            io_group_idx = io_group if ismc else io_group-1
            if not ismc and (io_group_idx == 5 or io_group_idx == 6):
                continue

            q += d['Q'][sorted_indices[i]]
            accQ[idxs[sorted_indices[i]]] = np.float64(q)
            if i == 0:
                try:
                    thres = float(thresholds[io_group_idx])
                except TypeError:
                    if ismc:
                        iy = d['flatten_index'][0] // d['flatten_stride'][0]
                        iz = d['flatten_index'][0] % d['flatten_stride'][0]
                    else:
                        raise NotImplementedError()
                    thres = thresholds[io_group_idx][iy, iz]
                    # print(iy, iz)
                # print(thres)
                if thres < 2:
                    thres = 1E16
                avg_i[idxs[sorted_indices[i]]] = (d['Q'][sorted_indices[i]]-thres)/17
                tinterval[idxs[sorted_indices[i]]] = 17
                ch_thres[m] = thres
            else:
                tinterval[idxs[sorted_indices[i]]] = (d['t_drift'][sorted_indices[i]] - d['t_drift'][sorted_indices[i-1]]) / 0.1
                avg_i[idxs[sorted_indices[i]]] = d['Q'][sorted_indices[i]] / tinterval[idxs[sorted_indices[i]]]
            tindex[idxs[sorted_indices[i]]] = i
            # if d['Q'][sorted_indices[i]] < ch_thres[idxs[sorted_indices[i]]]:
            #     print('1', d[sorted_indices[i]], ch_thres[idxs[sorted_indices[i]]], thres_yis[io_group_idx][iy,iz], thres_zis[io_group_idx][iy,iz])
            #     print('2', d[sorted_indices[i]], thres, thres_yis[io_group_idx][iy,iz], thres_zis[io_group_idx][iy,iz])
    uni_pxl = np.array(selected_hit, dtype=hits.dtype)
    totQ = totQ[:len(uni_pxl)]
    totN = totN[:len(uni_pxl)]
    uni_pxl = rfn.append_fields(uni_pxl, names=['totQ', 'totN'], data=[totQ, totN], usemask=False)
    extended_hits = rfn.append_fields(hits, names=['totQ', 'totN'], data=[totQ_cp, totN_cp], usemask=False)
    extended_hits = rfn.append_fields(extended_hits, names=['accQ', 'dt', 'avg_i', 'thres'], data=[accQ, dt, avg_i, ch_thres], usemask=False)
    extended_hits = rfn.append_fields(extended_hits, names=['tinterval', 'tindex'], data=[tinterval, tindex], usemask=False)
    return uni_pxl, extended_hits

def prep_per_event(hits, itpc=None, highq_thres=None):

    uni_pxls, extended_hits = sel_uni_pxl(hits, att='Q', yposlabel='y', zposlabel='z')

    k_yz, b_yz, tg_yz = yz_line(uni_pxls)
    d_tg, d_nm = proj_yz(extended_hits, k_yz, b_yz)
    extended_hits = rfn.append_fields(extended_hits, names=('d_tg', 'd_nm'), data=(d_tg, d_nm), usemask=False)
    d_tg, d_nm = proj_yz(uni_pxls, k_yz, b_yz)
    uni_pxls = rfn.append_fields(uni_pxls, names=('d_tg', 'd_nm'), data=(d_tg, d_nm), usemask=False)

    if highq_thres is not None:
        xyhits = extended_hits[extended_hits['Q'] > highq_thres]
    else:
        xyhits = uni_pxls
    k_x, b_x, _ = xline(xyhits, reflabel='y')
    # k_x, b_x, _ = xline(xyhits, reflabel='d_tg')
    ref_x = k_x * extended_hits['y'] + b_x
    # ref_x = k_x * extended_hits['d_tg'] + b_x

    # direction
    reg_xy = LinearRegression().fit(uni_pxls['y'][:,None], uni_pxls['x'])
    reg_xz = LinearRegression().fit(uni_pxls['z'][:,None], uni_pxls['x'])
    para = normalized_dx(reg_xy.coef_[0], reg_xz.coef_[0])

    dx = extended_hits['x'] - ref_x
    sign = np.where(extended_hits['io_group'] % 2, 1.0, -1.0)
    dx *= sign
    extended_hits = rfn.append_fields(extended_hits, names='dx', data=dx, usemask=False)

    # if not (para < 0.05):
    #     # print(reg.coef_[0], reg.coef_[1])
    #     return np.array([], dtype=extended_hits.dtype)
    return extended_hits

finpath = sys.argv[1]
fdir = os.path.dirname(finpath)
ffname = os.path.basename(finpath)
fprefix = os.path.splitext(ffname)[0]
threshold = sys.argv[2]
try:
    ismc = bool(sys.argv[3])
except IndexError:
    ismc = True
if not ismc:
    raise NotImplementedError()

load_threshold(threshold)
with uproot.recreate(f'{fdir}/{fprefix}.root') as f:

    # Load file once
    fh5 = h5py.File(f'{finpath}', 'r')
    # hits = fh5['/selected/hits'][:]
    if '/selected/hits/data' in fh5:
        hits = fh5['/selected/hits/data'][:]
    elif '/hits' in fh5:
        hits = fh5['/hits'][:]
    else:
        raise KeyError("Neither '/selected/hits/data' nor '/hits' found in the file.")

    eids = hits['event_id']
    hits = hits[np.argsort(eids)]
    eids = hits['event_id']
    print(hits['event_id'])
    groups = list(split_sorted_dataset(eids))
    print(groups)
    out_hits = []
    distances = []
    for ie in range(len(groups)):
        print(ie)
        for itpc in range(70):
            if itpc in [4,5]:
                continue
            sel_hits = hits[groups[ie][1]]
            sel_hits = sel_hits[sel_hits['io_group'] == itpc]
            # print(np.unique(sel_hits['event_id']))
            if len(sel_hits) < 10:
                continue
            extended_hits = prep_per_event(sel_hits)
            if len(extended_hits) < 10:
                print('filtered 2')
                continue
            xyz = np.vstack([extended_hits['x'], extended_hits['y'], extended_hits['z']]).T
            max_dist = np.max(pdist(xyz))
            distances.append(float(max_dist))
            out_hits.append(extended_hits)
            print(ie, extended_hits['event_id'][0], max_dist)
    #f[f'event{ie}/hits'] = hits
    f['mu_ndlar/hits'] = np.concatenate(out_hits)
    f['mu_ndlar/distances'] = { "distance" : np.array(distances)}
    print('d', distances)
    print('written', f'{fdir}/{fprefix}.root')
