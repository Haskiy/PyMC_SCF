#!/usr/bin/env python
'''
Author: Pu Zhichen
Date: 2021-03-03 18:44:27
LastEditTime: 2022-04-12 09:53:40
LastEditors: Li Hao
Description: 
    A subroutine to implement Legendre grids.
FilePath: \pyMC\grids_util\grids_hss.py

 May the force be with you!
'''


import time
import ctypes
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf import gto
from pyscf.dft import gen_grid, radi
from pyMC.grids_mc import legendre

GAUSS_LEGENDRE_THETA = numpy.asarray((
    10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 84))

libdft = lib.load_library('libdft')

   

def gen_atomic_grids(mol, atom_grid={}, radi_method=radi.gauss_chebyshev,
                     level=3, prune=gen_grid.nwchem_prune, **kwargs):
    '''Generate number of radial grids and angular grids for the given molecule.

    Returns:
        A dict, with the atom symbol for the dict key.  For each atom type,
        the dict value has two items: one is the meshgrid coordinates wrt the
        atom center; the second is the volume of that grid.
    '''
    if isinstance(atom_grid, (list, tuple)):
        atom_grid = dict([(mol.atom_symbol(ia), atom_grid)
                          for ia in range(mol.natm)])
    atom_grids_tab = {}
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)

        if symb not in atom_grids_tab:
            chg = gto.charge(symb)
            if symb in atom_grid:
                n_rad, n_ang = atom_grid[symb]
                if n_ang not in gen_grid.LEBEDEV_NGRID:
                    if n_ang in gen_grid.LEBEDEV_ORDER:
                        logger.warn(mol, 'n_ang %d for atom %d %s is not '
                                    'the supported Lebedev angular grids. '
                                    'Set n_ang to %d', n_ang, ia, symb,
                                    gen_grid.LEBEDEV_ORDER[n_ang])
                        n_ang = gen_grid.LEBEDEV_ORDER[n_ang]
                    else:
                        raise ValueError('Unsupported angular grids %d' % n_ang)
            else:
                n_rad = gen_grid._default_rad(chg, level)
                n_ang = gen_grid._default_ang(chg, level)
            rad, dr = radi_method(n_rad, chg, ia, **kwargs)

            rad_weight = 4*numpy.pi * rad**2 * dr

            if callable(prune):
                angs = prune(chg, rad, n_ang)
            else:
                angs = [n_ang] * n_rad
            logger.debug(mol, 'atom %s rad-grids = %d, ang-grids = %s',
                         symb, n_rad, angs)

            angs = numpy.array(angs)
            coords = []
            vol = []
            for n in sorted(set(angs)):
                grid = numpy.empty((n,4))
                libdft.MakeAngularGrid(grid.ctypes.data_as(ctypes.c_void_p),
                                       ctypes.c_int(n))
                idx = numpy.where(angs==n)[0]
                for i0, i1 in gen_grid.prange(0, len(idx), 12):  # 12 radi-grids as a group
                    coords.append(numpy.einsum('i,jk->jik',rad[idx[i0:i1]],
                                               grid[:,:3]).reshape(-1,3))
                    vol.append(numpy.einsum('i,j->ji', rad_weight[idx[i0:i1]],
                                            grid[:,3]).ravel())
            atom_grids_tab[symb] = (numpy.vstack(coords), numpy.hstack(vol))
    return atom_grids_tab

def gen_atomic_grids_gauss_legendre(mol, atom_grid={}, radi_method=radi.gauss_chebyshev,
                     level=3, prune=None, **kwargs):
    '''Generate number of radial grids and angular grids for the given molecule.
        the angular part using the Gauss-Legendre method, reference:
        
        $BDFHOME/source/numint_util/grid_angular.F90
        subroutine grid_angular_legendre

    Returns:
        A dict, with the atom symbol for the dict key.  For each atom type,
        the dict value has two items: one is the meshgrid coordinates wrt the
        atom center; the second is the volume of that grid.
    '''
    if isinstance(atom_grid, (list, tuple)):
        atom_grid = dict([(mol.atom_symbol(ia), atom_grid)
                          for ia in range(mol.natm)])
    atom_grids_tab = {}
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)

        if symb not in atom_grids_tab:
            chg = gto.charge(symb)
            if symb in atom_grid:
                n_rad, n_theta, n_phi = atom_grid[symb]
                if n_theta not in GAUSS_LEGENDRE_THETA:
                    raise ValueError('Unsupported theta grids %d' % n_theta)
            else:
                raise ValueError('Gauss-Legendre grids needs all the grids parameters'
                                 ' of radii, theta, phi')
            rad, dr = radi_method(n_rad, chg, ia, **kwargs)
            # ! Note : the 4 pi in Lebedev is because the weights of angular part is not
            # !         producted py 4pi, so the 4pi is producted here, but the Legendre
            # !         part has already considered the angel part, so 4pi is removed.
            rad_weight = rad**2 * dr

            if callable(prune):
                raise NotImplementedError('Prune is not implemented in Gauss-Legendre prat')
            else:
                angs = [n_theta*n_phi] * n_rad
            logger.debug(mol, 'atom %s rad-grids = %d, ang-grids = %s',
                         symb, n_rad, angs)

            angs = numpy.array(angs)
            coords = []
            vol = []
            for n in sorted(set(angs)):
                grid = numpy.empty((n,4))
                grid = legendre.grid_angular_legendre(n_theta, n_phi)
                idx = numpy.where(angs==n)[0]
                for i0, i1 in gen_grid.prange(0, len(idx), 12):  # 12 radi-grids as a group
                    coords.append(numpy.einsum('i,jk->jik',rad[idx[i0:i1]],
                                               grid[:,:3]).reshape(-1,3))
                    vol.append(numpy.einsum('i,j->ji', rad_weight[idx[i0:i1]],
                                            grid[:,3]).ravel())
            big_grid = numpy.einsum('i,j->ji', numpy.array([1.0]),
                                            grid[:,3]).ravel()
            numpy.save('big_grid_weight', big_grid)
            
            atom_grids_tab[symb] = (numpy.vstack(coords), numpy.hstack(vol))
    return atom_grids_tab


class Grids_hss(gen_grid.Grids):
    def __init__(self,mol):
        gen_grid.Grids.__init__(self,mol)
            
    def gen_atomic_grids(self, mol, atom_grid=None, radi_method=None,
                         level=None, prune=None, **kwargs):
        if atom_grid is None: atom_grid = self.atom_grid
        if radi_method is None: radi_method = self.radi_method
        if level is None: level = self.level
        if prune is None: prune = self.prune
        return gen_atomic_grids(mol, atom_grid, self.radi_method, level, prune, **kwargs)
    
    def gen_atomic_grids_gauss_legendre(self, mol, atom_grid=None, radi_method=None,
                         level=None, prune=None, **kwargs):
        if atom_grid is None: 
            atom_grid = self.atom_grid
        if radi_method is None: radi_method = self.radi_method
        if level is None: level = self.level
        if prune is not None: prune = None
        return gen_atomic_grids_gauss_legendre(mol, atom_grid, self.radi_method, level, prune, **kwargs)