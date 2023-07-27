#!/usr/bin/env python
'''
Author: Li Hao
Date: 2022-04-11 09:41:21
LastEditTime: 2023-02-25 08:34:37
LastEditors: Li Hao
Description: 
FilePath: /pyMC/lib/Spoints.py

 May the force be with you!
'''

import os
from pyscf.lib import logger
from pyMC.lib import LebedevGrid, FibonacciGrid, LegendreGrid
path = os.path.abspath(os.path.dirname(__file__))


class Spoints:
    def __init__(self):
        self.Tdistrion = 'LebedevGrid'
    def make_sph_samples(self, Npoints):
        '''make_sph_samples: makes sample points and weights on spherical surface.
    
        Parameters
        ----------
        Args:
            Npoints : int
                Nubmber of sample points in spin space, the same as Ndirect.
        
        Returns:
        ----------
            directions : numpy.array
                Projection directions, the same as Nx.
            weights : numpy.array
                Weights.
        '''
        if self.Tdistrion == 'LebedevGrid':
            ang_grids = LebedevGrid.MakeLebedevGrid(Npoints)
            directions = ang_grids[:,:3].copy(order='F')
            weights = ang_grids[:,3].copy()
        elif self.Tdistrion == 'FibonacciGrid':
            ang_grids = FibonacciGrid.MakeFibonacciGrid(Npoints)
            directions = ang_grids[:,:3].copy(order='F')
            weights = ang_grids[:,3].copy()
        elif self.Tdistrion == 'LegendreGrid' or self.Tdistrion == 'SphericalDesign':
            assert isinstance(Npoints,tuple), 'A tuple (n_theta,n_phi) is needed for LegendreGrid.'
            n_theta,n_phi = Npoints
            ang_grids = LegendreGrid.MakeLegendreGrid(n_theta,n_phi)
            directions = ang_grids[:,:3].copy(order='F')
            weights = ang_grids[:,3].copy()
        else :
            raise ValueError('Only lebedev, Fibonacci, Legendre and spherical design are implemented!')
        return directions, weights
    
        # try:
        #     directions = numpy.load(path + '/NX_tot.npy',allow_pickle=True)[Npoints]
        #     weights = numpy.load('NX_tot.npy',allow_pickle=True)[Npoints]
        # except:
        #     raise ValueError(f'Grid {Npoints} is not implemented for method {self.Tdistrion}!')