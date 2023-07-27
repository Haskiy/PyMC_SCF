#!/usr/bin/env python
'''
Author: Pu Zhichen
Date: 2021-03-29 10:16:13
LastEditTime: 2021-06-08 08:14:56
LastEditors: Pu Zhichen
Description: 
FilePath: \pyMC\tools\degenerate_degree.py
May the force be with you!
'''

import time
import numpy
import pyscf
import os
import scipy
from pyscf import lib
from pyscf import gto
from pyscf import df
from pyscf.dft import numint
from pyMC.tools import Dmatrix, rotate_dm, rotate_grid
import scipy.linalg
from pyscf import __config__
from scipy.spatial.transform import Rotation as R
from pyMC.lib import group_info

def _get_equivelent_grids(coords, coords_rot, THRESHOLD = 1.0E-6):
    """This subroutine gets the equivelent grids, when performed group operators.
        equivelent grids are those having the same space coordinates after the rotation
        with the original grids before rotation.

    Args:
        coords (numpy array 2D): [ngrid, 3] original coordinates of grids
        coords_rot (numpy array 3D): [ng, ngrid, 3] ng is the number of group operators for single group.
        THRESHOLD ([float], optional): Thresholds for dicide whether two grids are the same. Defaults to 1.0E-6.

    Returns:
        equivelent_grids_table [dict]: Dictionary for equivalent grids.
            The structure of equivelent_grids_table is {ig : {i : irot}}. 
            ig --> the number of the group operator.
            i --> the number of the original grids
            irot --> equivelent grids
            It should be noted that it's the original grid i have the same position when rotation have been
                performed on original grid irot.
            For example, if there is and element {1 : {2 : 10}}, which means for the 1st rotation operation.
            10th grid will have the same position with 2nd grid after the 1st rotation.
    """
    # ! THIS is aborted for low efficiency, _get_equivelent_grids_direct is prefered
    # TODO : Increase the efficiency
    # TODO : This subroutine is some unreadable, should be rewrite.
    ng,ngrid = coords_rot.shape[:2]
    equivelent_grids_table= {}
    # * sort by x component, then by y, then by z, using the lexsort
    idx_co = numpy.lexsort(coords[:,::-1].T)
    
    for ig in range(ng):
        # print('ig ',ig)
        coords_tmp = coords_rot[ig].copy()
        equivelent_grids_table[ig] = {}
        idx_co_rot = numpy.lexsort(coords_tmp[:,::-1].T)
        idx_compare = list(set((numpy.where(numpy.abs(coords[idx_co]-coords_tmp[idx_co_rot])>THRESHOLD)[0]).tolist()))
        for igrid in range(ngrid):
            if igrid in idx_compare:
                # print(igrid)
                # ! Note that idx_i_in_rot is the index in the idx_compare
                idx_i_in_rot = numpy.where(\
                    (numpy.abs(coords_tmp[idx_co_rot[idx_compare]]-coords[idx_co[igrid]])<=THRESHOLD).all(axis=1))[0]
                # if igrid == 0:
                #     import pdb
                #     pdb.set_trace()
                # print(idx_i_in_rot)
                equivelent_grids_table[ig][idx_co[igrid]] = idx_co_rot[idx_compare[int(idx_i_in_rot)]]
                idx_compare.pop(int(idx_i_in_rot))
            else:
                equivelent_grids_table[ig][idx_co[igrid]] = idx_co_rot[igrid]
        coords_tmp = None
    # * It should be noted that it means the rotated grid i has the same coordinates with the original j            
    return equivelent_grids_table


def _get_equivelent_grids_direct(coords, coords_rot, THRESHOLD = 8):
    """This subroutine gets the equivelent grids, when performed group operators.
        equivelent grids are those having the same space coordinates after the rotation
        with the original grids before rotation.

    Args:
        coords (numpy array 2D): [ngrid, 3] original coordinates of grids
        coords_rot (numpy array 3D): [ng, ngrid, 3] ng is the number of group operators for single group.
        THRESHOLD ([float], optional): Thresholds for dicide whether two grids are the same. Defaults to 8.
            Which means 8 digits are reserved for judging whether two grids are the same.

    Returns:
        equivelent_grids_table [dict]: Dictionary for equivalent grids.
            The structure of equivelent_grids_table is {ig : {i : irot}}. 
            ig --> the number of the group operator.
            i --> the number of the original grids
            irot --> equivelent grids
            It should be noted that it's the original grid i have the same position when rotation have been
                performed on original grid irot.
            For example, if there is and element {1 : {2 : 10}}, which means for the 1st rotation operation.
            10th grid will have the same position with 2nd grid after the 1st rotation.
    """
    ng,ngrid = coords_rot.shape[:2]
    equivelent_grids_table= {}
    # * sort by x component, then by y, then by z, using the lexsort
    idx_co = numpy.lexsort((numpy.around(coords[:,2], decimals=THRESHOLD)
                            , numpy.around(coords[:,1], decimals=THRESHOLD)
                            , numpy.around(coords[:,0], decimals=THRESHOLD)))
    
    for ig in range(ng):
        # print('ig ',ig)
        coords_tmp = numpy.around(coords_rot[ig], decimals=THRESHOLD)
        equivelent_grids_table[ig] = {}
        idx_co_rot = numpy.lexsort((coords_tmp[:,2], coords_tmp[:,1], coords_tmp[:,0]))
        idx_compare = list(set((numpy.where(numpy.abs(coords[idx_co]-coords_tmp[idx_co_rot])>1.0E-5)[0]).tolist()))
        if len(idx_compare) == 0:
            for igrid in range(ngrid):
                equivelent_grids_table[ig][idx_co[igrid]] = idx_co_rot[igrid]
        else:
            for igrid in range(ngrid):
                for jgrid in range(ngrid):
                    if numpy.abs(coords[igrid,0] - coords_tmp[jgrid,0]) <=1E-8\
                        and numpy.abs(coords[igrid,1] - coords_tmp[jgrid,1]) <=1E-8 \
                        and numpy.abs(coords[igrid,2] - coords_tmp[jgrid,2]) <=1E-8 :
                        equivelent_grids_table[ig][igrid] = jgrid
                        break
        coords_tmp = None
    # * It should be noted that it means the rotated grid i has the same coordinates with the original j            
    return equivelent_grids_table

def _get_average_M(equivelent_grids_table, M_rot):
    """get the average M on ORIGINAL grids after all operators have been perfomed on grids.coords

    Args:
        equivelent_grids_table (dict): {ig : {i : irot}}. 
            ig --> the number of the group operator.
            i --> the number of the original grids
            irot --> equivelent grids
        M_rot (dict): {istep : M_rot[ig, 3, ngrid]}
            istep --> the istep'th circle of the SCF calculation.
            ig --> the ig'th operator of the group. 
            ngrid --> number of the grids.

    Returns:
        M_average (dict): {istep : M[3, ngrid]}
            istep --> the istep'th circle of the SCF calculation.
            ngrid --> number of the grids.
    """
    equal_list = list(equivelent_grids_table.keys())
    step_list = list(M_rot.keys())
    ng, nx, ngrid = M_rot[step_list[0]].shape
    M_average = {i : numpy.zeros((3,ngrid)) for i in step_list}

    for istep in step_list:
        for ig in equal_list:
            # TODO : This part is low efficiency
            for igrid in range(ngrid):
                M_average[istep][:,igrid] += M_rot[istep][ig,:,equivelent_grids_table[ig][igrid]]/ng
            
    return M_average
            
 
def _get_diviance_M(equivelent_grids_table, M_rot, M_average):
    """This subroutine calculates the diviance of the M of each grids.
        It should be noted that each symmetrical equivelent grid should have the same M vector.
        However, there exists symmetry breaking, thus the equivelent grid may not have the same M vector.

    Args:
        equivelent_grids_table (dict): {ig : {i : irot}}. 
            ig --> the number of the group operator.
            i --> the number of the original grids
            irot --> equivelent grids
        M_rot (dict): Rotated M vectors.
            {istep : M_rot[ig, 3, ngrid]} 
            istep --> the istep'th circle of the SCF calculation.
            ig --> the ig'th operator of the group. 
            ngrid --> number of the grids.
        M_average (dict): Average M vector
            {istep : M[3, ngrid]}
            istep --> the istep'th circle of the SCF calculation.
            ngrid --> number of the grids.

    Returns:
        M_diviance (dict): Diviance vector, which is defined as \vec{M} - \bar{M}.
            {istep : M_diviance[ig, 3, ngrid]} 
            istep --> the istep'th circle of the SCF calculation.
            ig --> the ig'th operator of the group. 
            ngrid --> number of the grids.
    """
    equal_list = list(equivelent_grids_table.keys())
    step_list = list(M_rot.keys())
    ng, nx, ngrid = M_rot[step_list[0]].shape
    M_diviance = {i : numpy.zeros((ng, 3, ngrid)) for i in step_list}
    for istep in step_list:
        for ig in equal_list:
            for igrid in range(ngrid):
                M_diviance[istep][ig,:,igrid] = M_rot[istep][ig,:,equivelent_grids_table[ig][igrid]] \
                                              - M_average[istep][:,igrid]
    return M_diviance


def _get_scaler_diviance(M_diviance):
    """This subroutine gets the scaler Deviance for vector Deviance.
        There are 2 scaler deviance, whose definition is in Returns.

    Args:
        M_diviance (dict): Diviance vector, which is defined as \vec{M} - \bar{M}.
            {istep : M_diviance[ig, 3, ngrid]} 
            istep --> the istep'th circle of the SCF calculation.
            ig --> the ig'th operator of the group. 
            ngrid --> number of the grids.

    Returns:
        deviance ( dict {istep : numpy.array([ng, ngrid])} ): This is scalar deviance, which saves the norm of 
            each deviance vector.
        deviance_tot ( dict {istep : float} ) : This is scalar deviance, which saves the sum or all the norms of each
            vector.
    """
    step_list = list(M_diviance.keys())
    ng, nx, ngrid = M_diviance[step_list[0]].shape
    deviance = {i : numpy.zeros((ng,ngrid)) for i in step_list}
    deviance_tot = {}
    for istep in step_list:
        deviance[istep] = numpy.linalg.norm(M_diviance[istep], axis = 1)
        deviance_tot[istep] = numpy.sum(deviance[istep], axis = 1)
    return deviance,deviance_tot

def energy_level_degeneracy(energy, threshold = 1.0E-6):
    if type(energy) is not numpy.ndarray:
        energy = numpy.asarray(energy)
    ntot = energy.shape[0]
    degene_dict = {}   
    degene_dict['pair1'] = [i for i in range(ntot-1) if numpy.abs(energy[i] - energy[i+1]) <= threshold]
    degene_dict['pair2'] = [i+1 for i in degene_dict['pair1']]
    degene_dict['degree'] = [energy[i] - energy[i+1] for i in degene_dict['pair1']]
    
    return degene_dict        
    
def irep_degeneracy(C):
    """[summary]

    Args:
        C (numpy array 2D): C matrix
        
    Returns:
        comp [numpy array]
    """
    ntot = C.shape[0]
    comp = numpy.zeros((ntot))
    comp[:] = numpy.linalg.norm(C,axis = 0)
    
    return comp

def get_M_symmetry_break(natom, group = 'D3', step=-1, MorBxc = 'M', THRESHOLD = 8):
    """This subroutine examines the degree of how the symmetry breaking.
    Rotate all the M in space by the operations of the group. Calculating the average \vec{M}
    for each symmetry grid, and calculate the diviance of the \vec{M} for each operations. 

    Args:
        natom (int): number of the atoms
        group (str, optional): Name of the group. Defaults to 'D3'.
        step (int, optional): which step of the SCF calculation is calculated. Defaults to -1.
        MorBxc (str, optional): whether examines \vec{M} or \vec{Bxc}. Defaults to 'M'.

    Raises:
        ValueError: step should be -1 or >=1.

    Returns:
        equivelent_grids_table [dict]: Dictionary for equivalent grids.
            The structure of equivelent_grids_table is {ig : {i : irot}}. 
            ig --> the number of the group operator.
            i --> the number of the original grids
            irot --> equivelent grids
            It should be noted that it's the original grid i have the same position when rotation have been
                performed on original grid irot.
            For example, if there is and element {1 : {2 : 10}}, which means for the 1st rotation operation.
            10th grid will have the same position with 2nd grid after the 1st rotation.
            
        deviance ( dict {istep : numpy.array([ng, ngrid])} ): This is scalar deviance, which saves the norm of 
            each deviance vector.
            
        deviance_tot ( dict {istep : float} ) : This is scalar deviance, which saves the sum or all the norms of each
            vector.    
            
        M_rot (dict): Rotated M vectors.
            {istep : M_rot[ig, 3, ngrid]} 
            istep --> the istep'th circle of the SCF calculation.
            ig --> the ig'th operator of the group. 
            ngrid --> number of the grids.
            
        M_average (dict): Average M vector
            {istep : M[3, ngrid]}
            istep --> the istep'th circle of the SCF calculation.
            ngrid --> number of the grids.
            
        M_diviance (dict): Diviance vector, which is defined as \vec{M} - \bar{M}.
            {istep : M_diviance[ig, 3, ngrid]} 
            istep --> the istep'th circle of the SCF calculation.
            ig --> the ig'th operator of the group. 
            ngrid --> number of the grids.
    """
    npart = 0 #1 based
    nstep = 0 #0 based
    file_list = os.listdir(os.getcwd())
        
    for file in file_list:
        if file.startswith('Mx_step'):
            if int(file.split('_')[2]) > nstep:
                nstep = int(file.split('_')[2])
            if int((file.split('_')[4]).split('.')[0]) > npart:
                npart = int(file.split('_')[4].split('.')[0])
    nstep += 1 
    coords = numpy.loadtxt('coords_1.txt')
    ngrid = coords.shape[0]
    coords_labels = numpy.zeros((ngrid))
    coords_rot = rotate_grid.rotate_grid(coords, natom, group)
    # * equivelent_grids_table
    #   First, Key is the number of the rotation operators.
    #   key-->grids_number of original grids
    #   value-->grids_number of rotated grids
    equivelent_grids_table = _get_equivelent_grids_direct(coords, coords_rot, THRESHOLD)
    if step >=1:
        step_list = [i for i in range(0,nstep,step)]
    elif step == -1:
        step_list = [nstep-1]
    else:
        raise ValueError('Input step if wrong, only -1 and positive integers can be used.')
    M_rot = rotate_grid.get_rotated_M(equivelent_grids_table, step_list, npart, natom, group, MorBxc)
    M_average = _get_average_M(equivelent_grids_table, M_rot)
    M_diviance = _get_diviance_M(equivelent_grids_table, M_rot, M_average)
    deviance, deviance_tot = _get_scaler_diviance(M_diviance)
    
    return equivelent_grids_table, deviance_tot, deviance, M_rot, M_average, M_diviance