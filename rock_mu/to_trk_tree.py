#!/usr/bin/env python3
import uproot
import awkward as ak
import h5py

import numpy as np
import numpy.lib.recfunctions as rfn

import sys

from scipy.spatial.distance import pdist, squareform, cdist

from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression

import argparse
import logging

import numba as nb

import matplotlib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument( '--loglevel',
                     default='info',
                     help='Provide logging level. Example --loglevel debug, default=warning' )

args = parser.parse_args()

logging.basicConfig()
logger = logging.getLogger(__name__).getChild('trk_h5totree_converter')
logger.setLevel(level=args.loglevel.upper())



class trk_h5totree_converter:
    def __init__(self):
        self._data = {}
        self._tinterval_1st = 18 # [0.1us]; must be consistent with hits['t_drift']
        self._Qthres = 5 # will be done channel by channel; unit must be consistent with hits['Q']

    def load_trk(self, f_name, itrk):
        trk, hits = self._load_hdf5(f_name, itrk)
        hits = self._pxl_labels(hits)

        ak_hits = self._group_hits(hits)
        ak_hits = self._drop_empty_pxl(ak_hits)
        sorted_ak_hits = self._sort_hits(ak_hits)

        ref_tdrift_hits = self._flattened_max_hits(sorted_ak_hits)
        ref_Q_hits = self._flattened_max_hits(sorted_ak_hits, key='Q')

        analyzed_hits = self._simple_analysis(sorted_ak_hits, ref_tdrift_hits, projviews_png='{}_trk{}.png'.format(f_name, itrk))
        if not self._data.get(f_name, None):
            self._data[f_name] = {}
        if isinstance(self._data[f_name], dict):
            self._data[f_name][itrk] = analyzed_hits
        else:
            raise NotImplementedError

    def _load_hdf5(self, f_name, itrk):
        logger.info('loading file {}'.format(f_name))
        f = h5py.File(f_name, 'r')

        logger.info('loading {}th track'.format(itrk))
        trks = f['/analysis/rock_muon_tracks/data']
        trk = trks[itrk]

        logger.info('loading associated hits')
        hits = f['/charge/calib_prompt_hits/data']

        # references
        trk2hit_refs = f['/analysis/rock_muon_tracks/ref/charge/calib_prompt_hits/ref/']
        m = trk2hit_refs[:,1][trk2hit_refs [:,0] == itrk]

        trk2hits = hits[m]

        # begin debug
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('trk info')
            logger.debug(trk.dtype)
            logger.debug(trk)
            logger.debug('furthest hits info')
            logger.debug('n hits = {}'.format(len(trk2hits)))
            logger.debug('xmin = {}'.format(np.min(trk2hits['x'])))
            logger.debug('xmax = {}'.format(np.max(trk2hits['x'])))
            logger.debug('ymin = {}'.format(np.min(trk2hits['y'])))
            logger.debug('ymax = {}'.format(np.max(trk2hits['y'])))
            logger.debug('zmin = {}'.format(np.min(trk2hits['z'])))
            logger.debug('zmax = {}'.format(np.max(trk2hits['z'])))
        # end debug

        return trk, trk2hits

    def _simple_analysis(self, hits, flat_hits, **metadata):
        '''

        '''
        flat_all_hits_np = ak.flatten(hits).to_numpy()
        flat_hits_np = flat_hits.to_numpy()

        reflabel='y'
        refaxislabel='y [cm]'

        k_yz, b_yz, tg_yz = self.__yz_line(flat_all_hits_np)
        logger.debug('k_yz:{}, b_yz:{}, tg_yz:{}'.format(k_yz, b_yz, tg_yz))

        # # d_tg, d_nm = proj_yz(hits, k_yz, b_yz)

        k_x, b_x, _ = self.__xline(flat_hits_np, reflabel=reflabel)
        logger.debug('reflabel:{}, k_x:{}, b_x:{}'.format(reflabel, k_x, b_x) )

        ref_x = k_x * hits[reflabel] + b_x
        dx = hits['x'] - ref_x
        sign = ak.where(hits['io_group'] % 2, ak.ones_like(hits['io_group']), -1*ak.ones_like(hits['io_group']))
        dx = dx * sign
        logger.debug('first 5 ref_x: {}'.format(ref_x[:5].to_list()))
        logger.debug('first 5 x: {}, iogroup: {}, dx sign: {}'.format(hits['x'][:5].to_list(), hits['io_group'][:5].tolist(), sign[:5].to_list()))
        logger.debug('first 5 ref {}: {}'.format(reflabel, hits[reflabel][:5].to_list()))
        logger.debug('first 5 dx: {}'.format(dx[:5].to_list()))

        dt = self.__dt(hits)
        tinterval = self.__tinterval(hits)
        totQ = self.__totQ(hits)
        dQ = self.__dQ(hits)
        Qint = self.__Qint(hits)
        dQint = self.__dQint(hits)
        accQ = self.__accQ(hits)
        indices = self.__index_perpxl(hits)

        hits_new = ak.copy(hits)
        hits_new['dx'] = dx
        hits_new['dt'] = dt

        hits_new['tinterval'] = tinterval

        hits_new['totQ'] = totQ
        hits_new['Qint'] = Qint

        hits_new['dQ'] = dQ
        hits_new['dQint'] = dQint

        hits_new['accQ'] = accQ

        hits_new['index'] = indices

        logger.debug('first 3 hits (old): {}'.format(hits[:3].to_list()))
        logger.debug('first 3 hits (new): {}'.format(hits_new[:3].to_list()))

        ### plotting

        fig, axs = plt.subplots(4, 2, figsize=(5*2, 5*4))

        self.__plot_three_views(flat_hits_np, axs, all_hits=flat_all_hits_np)
        axs[0,1].scatter(flat_all_hits_np[reflabel], flat_all_hits_np['x'], label='all hits')
        axs[0,1].scatter(flat_hits_np[reflabel], flat_hits_np['x'], label='selected')
        axs[0,1].set_xlabel(refaxislabel)
        axs[0,1].set_ylabel('x [cm]')
        self.__plot_line(ax=axs[1,1], k=k_yz, b=b_yz, color='g')
        self.__plot_line(ax=axs[0,1], k=k_x, b=b_x, color='g')

        # sc0 = axs[2,0].scatter(flat_hits_np['y'], flat_hits_np['z'], c=flat_hits_np['totQ'], s=1,
        #     norm=matplotlib.colors.LogNorm(), cmap='rainbow')
        # sc1 = axs[2,1].scatter(flat_hits_np['y'], flat_hits_np['z'], c=flat_hits_np['totN'], s=1,
        #     norm=matplotlib.colors.BoundaryNorm(boundaries=np.arange(0, 5, 1), ncolors=256),
        #     cmap='RdBu_r')
        # fig.colorbar(sc0,
        #     ax=axs[2,0], orientation='vertical', label='total charge at pixel')
        # fig.colorbar(sc1,
        #     ax=axs[2,1], orientation='vertical', label='N hits at pixel', extend='max')

        fig.savefig(metadata['projviews_png'])

        return hits_new

    # validated
    def __dt(self, hits):
        if np.sum(ak.num(hits['t_drift']) == 0) != 0 :
            raise NotImplementedError
        dt = ak.copy(hits['t_drift'])
        dt_max = ak.max(dt, axis=-1, keepdims=True)
        dt = dt - dt_max
        logger.debug('first 5 t: {}'.format(hits['t_drift'][:5].to_list()))
        logger.debug('first 5 dt: {}'.format(dt[:5].to_list()))
        dt = ak.enforce_type(dt, 'var * float64')
        return dt

    # validated
    def __tinterval(self, hits):
        if np.sum(ak.num(hits['Q']) == 0) != 0 :
            raise NotImplementedError

        tinterval = hits['t_drift'][:, 1:] - hits['t_drift'][:, :-1]
        # the single element case should be properly handled?
        tinterval = ak.concatenate([ak.Array([[self._tinterval_1st]]), tinterval], axis=-1)
        tinterval = ak.where(ak.num(hits['t_drift'])>0, tinterval, ak.Array([[]]))

        logger.debug('__tinterval')
        logger.debug('first 5 t: {}'.format(hits['t_drift'][:5].to_list()))
        logger.debug('first 5 tinterval: {}'.format(tinterval[:5].to_list()))
        logger.debug('first 5 t of single-hit pixel: {}'.format(hits['t_drift'][ak.num(hits['t_drift'])==1][:5].to_list()))
        logger.debug('first 5 of single-hit pixel: {}'.format(tinterval[ak.num(hits['t_drift'])==1][:5].to_list()))
        return tinterval

    # validated
    def __totQ(self, hits):
        totQ = ak.sum(hits['Q'], axis=-1, keepdims=True)
        totQ = totQ + ak.zeros_like(hits['Q'])
        logger.debug('__totQ')
        logger.debug('first 5 Q: {}'.format(hits['Q'][:5].to_list()))
        logger.debug('first 5 totQ: {}'.format(totQ[:5].to_list()))
        return totQ

    # validated
    def __Qint(self, hits):
        if np.sum(ak.num(hits['Q']) == 0) != 0:
            raise NotImplementedError

        first_elements = ak.firsts(hits['Q']) - self._Qthres
        rest_elements = hits['Q'][:,1:]
        Qint = ak.concatenate([first_elements[..., None], rest_elements], axis=-1)
        Qint = ak.enforce_type(Qint , 'var * float64')

        logger.debug('__Qint')
        logger.debug('first 3 Q: {}'.format(hits['Q'][:3].to_list()))
        logger.debug('first 3 Qint: {}'.format(Qint[:3].to_list()))
        logger.debug('first 3 Q with n==1: {}'.format(hits['Q'][ak.num(hits['Q'],axis=-1)==1][:3].to_list()))
        logger.debug('first 3 Qint with n==1: {}'.format(Qint[ak.num(Qint,axis=-1)==1][:3].to_list()))
        logger.debug('first 3 Q with n==2: {}'.format(hits['Q'][ak.num(hits['Q'],axis=-1)==2][:3].to_list()))
        logger.debug('first 3 Qint with n==2: {}'.format(Qint[ak.num(Qint,axis=-1)==2][:3].to_list()))

        return Qint

    # validated
    def __index_perpxl(self, hits):
        if np.sum(ak.num(hits['Q']) == 0) != 0:
            raise NotImplementedError
        Q = hits['Q']
        indices = ak.ones_like(Q, dtype=np.int32)
        indices = ak.Array([np.cumsum(idxs).to_list() for idxs in indices]) - 1

        for i in range(1,3+1):
            logger.debug('first 3 indices with n=={}: {}'.format(i, indices[ak.num(indices,axis=-1)==i][:3].to_list()))
            logger.debug('first 3 indices with n=={}: {}'.format(i, indices[ak.num(indices,axis=-1)==i][:3].to_list()))
            logger.debug('first 3 indices with n=={}: {}'.format(i, indices[ak.num(indices,axis=-1)==i][:3].to_list()))
        return indices

    # validated
    def __accQ(self, hits):
        # discussions https://stackoverflow.com/a/65185301 seem obsolete
        # copy from https://stackoverflow.com/a/65174020
        Q = hits['Q']
        totQ = self.__totQ(hits)
        accQ = ak.Array([np.cumsum(qs).to_list() for qs in Q])

        logger.debug('__accQ')
        for i in range(1,3+1):
            logger.debug('first 3 Q with n=={}: {}'.format(i, Q[ak.num(Q,axis=-1)==i][:3].to_list()))
            logger.debug('first 3 accQ with n=={}: {}'.format(i, accQ[ak.num(accQ,axis=-1)==i][:3].to_list()))
            logger.debug('first 3 totQ with n=={}: {}'.format(i, totQ[ak.num(totQ,axis=-1)==i][:3].to_list()))

        return accQ

    # validated
    def __dQ(self, hits):
        if np.sum(ak.num(hits['Q']) == 0) != 0:
            raise NotImplementedError
        Q = hits['Q']
        dQ = Q[:, 1:] - Q[:, :-1]
        dQfirsts = ak.firsts(Q)
        # the single element case should be properly handled?
        dQ = ak.concatenate([dQfirsts[..., None], dQ], axis=-1)
        dQ = ak.enforce_type(dQ, 'var * float64')

        logger.debug('__dQ')
        for i in range(1,3+1):
            logger.debug('first 3 Q with n=={}: {}'.format(i, Q[ak.num(Q,axis=-1)==i][:3].to_list()))
            logger.debug('first 3 dQ with n=={}: {}'.format(i, dQ[ak.num(dQ,axis=-1)==i][:3].to_list()))
        return dQ


    # validated
    def __dQint(self, hits):
        if np.sum(ak.num(hits['Q']) == 0) != 0:
            raise NotImplementedError
        Qint = self.__Qint(hits)
        dQint = Qint[:, 1:] - Qint[:, :-1]
        # the single element case should be properly handled?
        dQint = ak.concatenate([ak.firsts(Qint)[...,None], dQint], axis=-1)
        dQint = ak.enforce_type(dQint, 'var * float64')

        logger.debug('__dQint')
        for i in range(1,3+1):
            logger.debug('first 3 Qint with n=={}: {}'.format(i, Qint[ak.num(Qint,axis=-1)==i][:3].to_list()))
            logger.debug('first 3 dQint with n=={}: {}'.format(i, dQint[ak.num(dQint,axis=-1)==i][:3].to_list()))

        return dQint

    def _pxl_labels(self, hits, tol=1E-6):
        points_yz = np.column_stack([hits['io_group'], hits['io_channel'], hits['y'], hits['z']])
        clustering = DBSCAN(eps=0.0001, min_samples=1).fit(points_yz)
        # check if DBSCAN works
        d = points_yz-clustering.components_
        assert np.all(np.linalg.norm(d)<tol)
        logger.debug('sum of all differences along each component {}'.format(np.sum(d, axis=0)))
        return rfn.append_fields(hits, names=['label'], data=[clustering.labels_], usemask=False)

    def _group_hits(self, hits):
        X = hits
        y = np.argsort(X['label'])
        Xsplit = np.split(X[y], np.unique(X[y]['label'], return_index=True)[1][1:])
        Xdict = [
            [{field: row[field] for field in row.dtype.names} for row in rec_array]
            for rec_array in Xsplit
        ]
        ak_hits = ak.from_iter(Xdict)
        logger.debug('is sorted for labels? {}'.format(np.all(np.diff(ak.flatten(ak_hits['label']))>=0)))
        logger.debug('first three elements: {}'.format(ak_hits[0:3].tolist()))
        logger.debug('last three elements: {}'.format(ak_hits[-3:].tolist()))
        return ak_hits
    def _drop_empty_pxl(self, hits):
        l = ak.num(hits, axis=1)
        no_empty = hits[l>0]
        logger.debug('are all non-empty? {}'.format(np.all(ak.num(no_empty, axis=1))))
        return no_empty

    def _sort_hits(self, hits, key='t_drift'):
        args = ak.argsort(hits[key], axis=1)
        sorted_hits = hits[args]
        logger.debug('is sorted for all pixels according to {}? {}'.format(key,
            np.all(ak.all(sorted_hits[key][:,:-1] <= sorted_hits[key][:,1:], axis=-1))
        ))
        logger.debug('key of first five elements: {}'.format(sorted_hits[key][0:5].tolist()))
        logger.debug('key of last five elements: {}'.format(sorted_hits[key][-5:].tolist()))
        return sorted_hits

    def _flattened_max_hits(self, hits, key='t_drift'):
        args = ak.argmax(hits[key], axis=1, keepdims=True)
        max_hits = hits[args]
        flat_hits = ak.flatten(max_hits)
        logger.debug('first three elements and their max for {}: {}, {}'.format(key, hits[key][:3].tolist(), max_hits[key][:3].to_list()))
        logger.debug('last three elements and their max for {}: {}, {}'.format(key, hits[key][-3:].tolist(), max_hits[key][-3:].to_list()))
        return flat_hits

    def __line1D(self, X, y):
        reg = LinearRegression().fit(X.reshape(-1, 1), y)
        return reg.coef_[0], reg.intercept_

    def __yz_line(self, hits):
        '''
        z = ky + b
        dz = kdy
        return k, b, (1/sqrt(1+k**2), k/sqrt(1+k**2))
        tagent vector = (1/sqrt(1+k**2), k/sqrt(1+k**2))
        '''
        k, b = self.__line1D(hits['y'], hits['z'])
        return k, b, (1/np.sqrt(1+k**2), k/np.sqrt(1+k**2))

# does not work given the input is awkward array
#    def __proj_yz(self, hits, k_yz, b_yz):
#
#        '''
#        z = ky + b
#        '''
#        tg_yz = np.array([1/np.sqrt(1+k_yz**2), k_yz/np.sqrt(1+k_yz**2)])
#        o_yz = (0, b_yz)
#        points_yz = np.column_stack([hits['y'], hits['z']]) - o_yz
#        d_tg = np.dot(points_yz, tg_yz)
#        # for i in range(10):
#        #     print(tg_yz[0]* points_yz[i][0] + tg_yz[1]* points_yz[i][1] - d_tg[i], points_yz[i], rock_muon_hits['y'][i], rock_muon_hits['z'][i], o_yz)
#        d_nm = np.cross(points_yz, tg_yz)
#        # for i in range(10):
#        #     print(-tg_yz[0]* points_yz[i][1] + tg_yz[1]* points_yz[i][0] - d_nm[i], points_yz[i], rock_muon_hits['y'][i], rock_muon_hits['z'][i], o_yz, tg_yz)
#        return d_tg, d_nm
    def __xline(self, hits, reflabel='y'):
        '''
        x = k* reflabel + b
        '''
        k, b = self.__line1D(hits[reflabel], hits['x'])
        return k, b, (1/np.sqrt(1+k**2), k/np.sqrt(1+k**2))

    def __plot_line(self, k, b, ax, **kwargs):
        x0, x1 = ax.get_xlim()
        x = np.arange(x0, x1, step=0.01)
        y = k * x + b
        ax.plot(x, y, label='line fit', c=kwargs.get('color'))
        ax.legend()

    def __plot_three_views(self, hits, axs=None, **kwargs):
        #if axs is None:
    #    fig, axs = plt.subplots(2, 2, figsize=(5*2, 5*2))
    #    assert fig == axs[0,0].get_figure()
    #elif isinstance(axs, np.ndarray):
    #    if axs.shape == (2, 2):
    #        fig = axs[0,0].get_figure()
    #    else:
    #        raise TypeError('Axes passed in are not 2x2')
    #else:
    #    raise TypeError('axs is not properly given')

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

    def write(self):
        for k, v in self._data.items():
            f = uproot.recreate('{}.root'.format(k))
            for i, a in v.items():
                f['trk{}/hits'.format(i)] = a
            f.close()

if __name__ == '__main__':
    converter = trk_h5totree_converter()
    converter.load_trk('/home/yousen/Public/ndlar_shared/data/packet-0050017-2024_07_08_15_13_35_CDT.FLOW.rock_mu.h5', 33)
    #converter.load_trk('/home/yousen/Public/ndlar_shared/data/packet-0050015-2024_07_08_13_37_49_CDT.FLOW.rock_mu.h5', 33)
    converter.write()
