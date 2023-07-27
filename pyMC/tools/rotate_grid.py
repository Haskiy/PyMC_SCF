#!/usr/bin/env python
'''
Author: Pu Zhichen
Date: 2021-04-08 15:40:33
LastEditTime: 2021-06-08 08:16:38
LastEditors: Pu Zhichen
Description: 
FilePath: \pyMC\tools\rotate_grid.py

 May the force be with you!
'''

import time
import numpy
import pyscf
import scipy
from pyscf import lib
from pyscf import gto
from pyscf import df
from pyscf.dft import numint
from pyMC.tools import Dmatrix, rotate_dm, group_proj
import scipy.linalg
from pyscf import __config__
from scipy.spatial.transform import Rotation as R
from pyMC.lib import group_info

def rotate_grid(coords, mol_tot, group = 'D3'):
    """Rotate the numerical grids for each of the group operator of the given group

    Args:
        coords (numpy array 2D): [ngrid,3]
        natom (int): number of the atoms
        group (str, optional): name of the group. Defaults to 'D3'.

    Returns:
        coords_rot [numpy array 3D]: [ng, ngrid, 3] ng is the total number of the operators of the group.
    """
    ngrid = coords.shape[0]
    ng, rotvec, theta_vec = group_proj.rotation_info(mol_tot, 
                                        group.upper())
    coords_rot = numpy.zeros((ng//2,ngrid,3))
    for ig in range(ng//2):
        T = _get_rotation_matrix(rotvec[ig], theta_vec[ig])
        coords_rot[ig] = numpy.einsum('ij,nj->ni',T,coords)
    return coords_rot
        
        
def _get_rotation_matrix(rotvec, theta):
    """Using scipy.spatial.transform to generate the rotation matrix

    Args:
        rotvec (numpy array 1D): rotation axis vector
        theta (float): rotation angle

    Returns:
        T (numpy array 2D): Rotation matrix
    """
    r = R.from_rotvec(theta*rotvec)
    T = r.as_matrix()
    return T

def _read_file(name, istep, npart=1):
    """Read M or Bxc from binary npy files, generated in numint_gksm.py
    

    Args:
        name (str): name of the files
        istep (int): the number of the SCF step
        npart (int, optional): number of grid patches used in numint_gksm.py. Defaults to 1.

    Returns:
        X [numpy 2D array]: 
    """
    name0 = name + '_step_' + str(istep) + '_ipart_1.npy'
    X = numpy.load(name0)
    for ipart in range(2,npart+1):
        X1 = numpy.load(name + '_step_' + str(istep) + '_ipart_' + str(ipart) + '.npy')
        X = numpy.concatenate((X,X1), axis = 1)
    return X

def get_rotated_M(equivelent_grids_table, step_list, npart, mol_tot, group = 'D3', MorBxc = 'M'):
    """This subroutine rotate the M or Bxc vector.

    Args:
        equivelent_grids_table (dictionary): equivelent grids for each rotation operator.
        step_list (list): Steps that will be analysed
        npart (int): number of parts that are partiioned in numint_gksm.py
        natom (int): number of atoms
        group (str, optional): name of the group. Defaults to 'D3'.
        MorBxc (str, optional): whether to generate M or Bxc. Defaults to 'M'.

    Returns:
        M_rot_dict (dictionary): 
            {istep : M_rot[ig, 3, ngrid]}
            istep --> the istep'th circle of the SCF calculation.
            ig --> the ig'th operator of the group. 
            ngrid --> number of the grids.
    """
    for ikey in equivelent_grids_table:
        ngrid = len(equivelent_grids_table[ikey])
        break
    M = numpy.zeros((3,ngrid))
    ng, rotvec, theta_vec = group_proj.rotation_info(mol_tot, 
                                        group.upper())
    M_rot_dict = {i : numpy.zeros((ng//2,3,ngrid)) for i in step_list}
    for istep in step_list:
        if MorBxc.upper() == 'M':
            M[0] = _read_file('Mx', istep, npart)[0]
            M[1] = _read_file('My', istep, npart)[0]
            M[2] = _read_file('Mz', istep, npart)[0]
        elif MorBxc.upper() == 'BXC':
            M[0] = _read_file('Bxc_d0', istep, npart)
            M[1] = _read_file('Bxc_d1', istep, npart)
            M[2] = _read_file('Bxc_d2', istep, npart)
        for ig in range(ng//2):
            T = _get_rotation_matrix(rotvec[ig], theta_vec[ig])
            M_tmp = numpy.einsum('ij,jn->in',T,M)
            M_rot_dict[istep][ig] = M_tmp
            
    return M_rot_dict
        
        