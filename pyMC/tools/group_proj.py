#!/usr/bin/env python
'''
Author: Pu Zhichen
Date: 2021-03-22 14:14:26
LastEditTime: 2021-06-22 09:37:44
LastEditors: Pu Zhichen
Description: 
    Group utils.
FilePath: \pyMC\tools\group_proj.py

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
from pyMC.tools import Dmatrix, rotate_dm
import scipy.linalg
from pyscf import __config__
from scipy.spatial.transform import Rotation as R
from pyMC.lib import group_info



def rotation_info(mol_tot, groupname = 'D5', operator = 'CHI', rep = 'A1'):
    """Just A wrap

    Args:
        see _group_info

    Returns:
        
    """
    ng, theta, Aalpha, salpha, \
        atom_change, rotvec, theta_vec = _group_info(mol_tot, groupname, operator, rep)
    return ng, rotvec, theta_vec

def _get_rotvec(mol):
    natom = mol.natm
    atomcoords_list = numpy.zeros((natom,3))
    axis = numpy.zeros((natom,3))
    for i in range(natom):
        atomcoords_list[i] = mol.atom_coord(i)
    center = numpy.sum(atomcoords_list,axis = 0)/natom
    zaxis = numpy.cross(atomcoords_list[0]-center, atomcoords_list[1]-center)
    axis = atomcoords_list - center
    ng = group_info.NG[mol.groupname]
    nx = numpy.zeros((ng,3))
    nghalfhalf = ng//4
    nghalf = ng//2
    if mol.groupname in ['D3', 'D5']:
        nx[:nghalfhalf] = zaxis
        nx[nghalfhalf:nghalf] = axis
    elif mol.groupname == 'C3':
        nx[:nghalf] = zaxis
    else:
        raise NotImplementedError("Other basis are now not implemented")
    nx[ng//2:] = nx[:ng//2]
    
    return nx
        
def  _get_euler(nx, theta):
    ng = nx.shape[0]
    euler = numpy.zeros((ng//2,3))
    for ig in range(ng//2):
        r = R.from_rotvec(nx[ig]*theta[ig])
        euler[ig] = r.as_euler('ZYZ', degrees=False) # Note that this part uses the intrinsic axis !
    return euler

def _get_atom_change(mol_tot, theta):
    natom = mol_tot.natm
    ng = theta.shape[0]
    atomcoords_list = numpy.zeros((natom,3))
    atom_change = numpy.zeros((ng,natom)).astype(int)
    for i in range(natom):
        atomcoords_list[i] = numpy.around(mol_tot.atom_coord(i), decimals = 6)
    for i in range(natom):
        for ig in range(ng):
            r = R.from_euler('ZYZ', theta[ig]) # Note that this part uses the intrinsic axis !
            coords_rot = numpy.around(r.as_matrix()@mol_tot.atom_coord(i).round(9), decimals = 6)
            # * NOTE that this is 0-based
            atom_change[ig,i] = int(numpy.argwhere(numpy.all((atomcoords_list -coords_rot)==0, axis=1)))     
    return atom_change
            
       

def _group_info(mol_tot, groupname = 'D5', operator = 'CHI', rep = 'A1', auto = True):
    """ get some group informations about the group.
        It should be noted that ALL the informations is for DOUBLE GROUP !

    Args:
        natom (int): number of atoms
        groupname (str, optional): Name of the group. Defaults to 'D5'.
        operator (str, optional): It can be set to 'CHI'(onlyg this is implemented). Defaults to 'CHI'.
        rep (str, optional): The ireducible representation. Defaults to 'A1'.

    Raises:
        NotImplementedError: Only D3 and D5 double group is implemented. If want to use more groups.
            Add the information in lib.group_info.py

    Returns:
        ng [int]: number of the operations in this group.
        theta [numpy array 2D]: Euler angles for rotation in real spaces.
        Characters
        Demension of the ireps
        Atom rotation patter
        Rotation axis vector, used in spin space rotation
        Rotation axis angles, used in spin space rotation
    """
    # TODO: It should be noted that all the rotations should have just only one rotvec representation.
    # TODO:     other representations such as Euler angles can be translated using scipy.spatial.transform
    # TODO: Now, this part is some of awkward.
    natom = mol_tot.natm
    ng = group_info.NG[groupname]
    theta = numpy.zeros((ng,3))
    atom_change = numpy.zeros((ng,natom))
    theta_vec = group_info.U_ROTATE[groupname]['theta']
    if auto:
        rotvec = _get_rotvec(mol_tot)
    else:
        rotvec = group_info.U_ROTATE[groupname]['nx']
    norm = numpy.sqrt(rotvec[:,0]**2 + rotvec[:,1]**2 + rotvec[:,2]**2)
    # ! Note that the rotation vector is normalized here.
    rotvec[:,0] = rotvec[:,0]/norm[:]
    rotvec[:,1] = rotvec[:,1]/norm[:]
    rotvec[:,2] = rotvec[:,2]/norm[:]    
    if auto:
        theta[:ng//2] = _get_euler(rotvec, theta_vec)
        theta[ng//2:] = theta[:ng//2]
        atom_change[:ng//2] = _get_atom_change(mol_tot, theta[:ng//2])
        atom_change[ng//2:] = atom_change[:ng//2]
    else:
        theta[:ng//2] = group_info.THETA[groupname]
        theta[ng//2:] = group_info.THETA[groupname]
        atom_change[:ng//2] = group_info.ATOM_CHANGE[groupname] - 1
        atom_change[ng//2:] = group_info.ATOM_CHANGE[groupname] - 1    
         
    
    if groupname not in group_info.GROUP:
        raise NotImplementedError(groupname + ' is not implemented')
    if operator == 'CHI':
        return ng, theta, group_info.CHI[groupname][rep], \
                    group_info.SALPHA[groupname][rep], atom_change.astype(numpy.int), \
                    rotvec, theta_vec
    elif operator == 'MATRIX':
        rep_chi = group_info.MATRIX_2_CHI[groupname][rep]
        return ng, theta, group_info.MATRIX_REP[groupname][rep], \
                    group_info.SALPHA[groupname][rep_chi], atom_change.astype(numpy.int), \
                    rotvec, theta_vec
        
# TODO : Rotation of orbital has been done many times. It should be put out.
def project_2_SO(mol, mol_tot, C, group = 'D5', operator = 'CHI'
                 , rep = 'A1', method = 'DIRECT', Double = True):
    """Construct the Character projection operator and do it on the C matrix.

    Args:
        mol (gto class in pyscf): saving one atom which construct the whole cluster.
        mol_tot (gto class in pyscf or mole_sym class in pyMC): saving the whold cluster.
        C (numpy array 2D complex): C matrix
        group (str, optional): Name of the group. Defaults to 'D5'.
        operator (str, optional): It can be set to 'CHI'(only this is implemented). Defaults to 'CHI'.
        rep (str, optional): The ireducible representation. Defaults to 'A1'.
        method (str, optional): Using different methods for space. Defaults to 'DIRECT'.
        Double (bool, optional): Wherther use double group. Defaults to True.

    Returns:
        C_f [numpy array 2D]: C matrix done by Character projection operator
    """
    natom = mol_tot.natm
    nao = C.shape[0]//natom//2 # total basis for alpha 
    naoatom = nao*natom
    ng, theta, Aalpha, salpha, \
        atom_change, rotvec, theta_vec = _group_info(mol_tot, group.upper(), operator.upper(), rep.upper())
    C_f = numpy.zeros((nao*2*natom,C.shape[-1]),dtype=numpy.complex128)
    if method.upper() == 'ORIGINAL':
        raise ValueError('ORIGINAL method has been aborted, if you do want to use, please comment this line.')
        for ig in range(ng):
            C_rot = rotate_dm.rotate_mo_coeff(mol, natom, C, theta[ig])
            C_rot *= Aalpha[ig].conj()*salpha/ng
            for iatm in range(natom):
                offset = atom_change[ig,iatm]
                
                C_f[offset*nao:(offset+1)*nao]+= C_rot[iatm*nao:(iatm+1)*nao]
                C_f[offset*nao+naoatom:(offset+1)*nao+naoatom]+= C_rot[iatm*nao+naoatom:(iatm+1)*nao+naoatom]
    elif method.upper() == 'DIRECT':
        if Double is False:
            raise ValueError('Only double group is used, if you do want to use, please comment this line.')
            ng = ng//2 
            # nx, theta_nx = rotate_dm.euler_to_rotvec_2(theta)
            for ig in range(ng):
                C_rot = rotate_dm.rotate_mo_coeff_direct_single(mol, natom, C, (theta[ig], rotvec[ig], theta_vec[ig]))
                # print((theta[ig], rotvec[ig], theta_vec[ig]),ig)
                numpy.save('C_rot'+str(ig),C_rot)
                C_rot_debug = C_rot.copy()
                C_rot_debug1 = C_rot.copy()
                C_rot *= Aalpha[ig].conj()*salpha/ng
                
                for iatm in range(natom):
                    offset = atom_change[ig,iatm]

                    C_f[offset*nao:(offset+1)*nao]+= C_rot[iatm*nao:(iatm+1)*nao]
                    C_f[offset*nao+naoatom:(offset+1)*nao+naoatom]+= C_rot[iatm*nao+naoatom:(iatm+1)*nao+naoatom]
                    C_rot_debug[offset*nao:(offset+1)*nao] = C_rot_debug1[iatm*nao:(iatm+1)*nao]
                    C_rot_debug[offset*nao+naoatom:(offset+1)*nao+naoatom] = C_rot_debug1[iatm*nao+naoatom:(iatm+1)*nao+naoatom]
                    
                numpy.save('C_rot_final'+str(ig),C_rot_debug)
            
            return C_f
        nghalf = ng//2
        theta[nghalf:] = theta[:nghalf]
        # nx, theta_nx = rotate_dm.euler_to_rotvec_2(theta)
        if not mol_tot.dirac4c:
            for ig in range(ng):
                
                C_rot = rotate_dm.rotate_mo_coeff_direct(mol, natom, C, (theta[ig], rotvec[ig], theta_vec[ig]))
                # print((theta[ig], rotvec[ig], theta_vec[ig]),ig)
                # print("111")
                # numpy.save('C_rot'+str(ig),C_rot)
                C_rot_debug = C_rot.copy()
                C_rot_debug1 = C_rot.copy()
                C_rot *= Aalpha[ig].conj()*salpha/ng
                
                for iatm in range(natom):
                    offset = atom_change[ig,iatm]

                    C_f[offset*nao:(offset+1)*nao]+= C_rot[iatm*nao:(iatm+1)*nao]
                    C_f[offset*nao+naoatom:(offset+1)*nao+naoatom]+= C_rot[iatm*nao+naoatom:(iatm+1)*nao+naoatom]
                    C_rot_debug[offset*nao:(offset+1)*nao] = C_rot_debug1[iatm*nao:(iatm+1)*nao]
                    C_rot_debug[offset*nao+naoatom:(offset+1)*nao+naoatom] = C_rot_debug1[iatm*nao+naoatom:(iatm+1)*nao+naoatom]
                    
                # numpy.save('C_rot_final'+str(ig),C_rot_debug)
        else:
            for ig in range(ng):
                
                C_rot = rotate_dm.rotate_mo_coeff_direct_4c(mol, natom, C, (theta[ig], rotvec[ig], theta_vec[ig]))
                # print((theta[ig], rotvec[ig], theta_vec[ig]),ig)
                # print("111")
                # numpy.save('C_rot'+str(ig),C_rot)
                C_rot_debug = C_rot.copy()
                C_rot_debug1 = C_rot.copy()
                C_rot *= Aalpha[ig].conj()*salpha/ng
                
                for iatm in range(natom):
                    offset = atom_change[ig,iatm]

                    C_f[offset*nao:(offset+1)*nao]+= C_rot[iatm*nao:(iatm+1)*nao]
                    C_f[offset*nao+naoatom:(offset+1)*nao+naoatom]+= C_rot[iatm*nao+naoatom:(iatm+1)*nao+naoatom]
                    C_rot_debug[offset*nao:(offset+1)*nao] = C_rot_debug1[iatm*nao:(iatm+1)*nao]
                    C_rot_debug[offset*nao+naoatom:(offset+1)*nao+naoatom] = C_rot_debug1[iatm*nao+naoatom:(iatm+1)*nao+naoatom]
                    
                # numpy.save('C_rot_final'+str(ig),C_rot_debug)
        
    return C_f
    
    
def get_symm_averaged_Fock(mol, mol_tot, Fock, group = 'D5'):
    natom = mol_tot.natm
    C = numpy.linalg.eigh(Fock)[1]
    nao = C.shape[0]//natom//2 # total basis for alpha 
    naoatom = nao*natom
    ng, theta, Aalpha, salpha, \
        atom_change, rotvec, theta_vec = _group_info(mol_tot, group.upper())
    C_f = numpy.zeros((nao*2*natom, nao*2*natom), dtype=numpy.complex128)
    Fock_rot_mo = numpy.zeros((nao*2*natom, nao*2*natom), dtype=numpy.complex128)
    nghalf = ng//2
    theta[nghalf:] = theta[:nghalf]
    
    for ig in range(ng):
        
        C_rot = rotate_dm.rotate_mo_coeff_direct(mol, natom, C, (theta[ig], rotvec[ig], theta_vec[ig]))
        # print((theta[ig], rotvec[ig], theta_vec[ig]),ig)
        # numpy.save('C_rot'+str(ig),C_rot)
        
        for iatm in range(natom):
            offset = atom_change[ig,iatm]

            C_f[offset*nao:(offset+1)*nao] = C_rot[iatm*nao:(iatm+1)*nao]
            C_f[offset*nao+naoatom:(offset+1)*nao+naoatom] = C_rot[iatm*nao+naoatom:(iatm+1)*nao+naoatom]
    
        Fock_rot_mo+= C_f.conj().T@Fock@C_f
    
    Fock_rot_mo = Fock_rot_mo/ng
    Fock_rot_ao = C_f@Fock_rot_mo@C_f.conj().T
    Fock_rot_ao = (Fock_rot_ao + Fock_rot_ao.conj().T)*0.5
    
    return Fock_rot_ao


def project_2_equal_basis(mol, mol_tot, C, Aalpha, nbasis, group = 'D5'):
    """Construct the Shift projection operator and do it on the C matrix.
       And generate one equivelent basis to coefficients C.

    Args:
        mol (gto class in pyscf): [description]
        natom (int): number of atoms
        C (numpy array 2D complex): C matrix
        Aalpha (numpy.array): The coefficients or group operators, used in projection operator.
        nbasis (integer): the dimension of the multi-dimensional irrep
        group (str, optional): Name of the group. Defaults to 'D5'.

    Returns:
        C_f [numpy array 2D]: C matrix done by Character projection operator
    """
    
    if mol_tot.dirac4c:
        rotate_func = rotate_dm.rotate_mo_coeff_direct_4c
    else:
        rotate_func = rotate_dm.rotate_mo_coeff_direct
    natom = mol_tot.natm
    nao = C.shape[0]//natom//2 # total basis for alpha 
    norbital = C.shape[1]
    naoatom = nao*natom
    
    # ! NOTE : only use this subroutine to get the information of group.
    ng, theta, Aalpha_aborted, salpha_aborted, \
        atom_change, rotvec, theta_vec = _group_info(mol_tot, group.upper(), 'CHI', 'A1')
    C_f = numpy.zeros((nao*2*natom,norbital),dtype=numpy.complex128)
    
    nghalf = ng//2
    theta[nghalf:] = theta[:nghalf]

    for ig in range(ng):
        
        C_rot = rotate_func(mol, natom, C, (theta[ig], rotvec[ig], theta_vec[ig]))
        # print((theta[ig], rotvec[ig], theta_vec[ig]),ig)
        # numpy.save('C_rot'+str(ig),C_rot)
        C_rot_debug = C_rot.copy()
        C_rot_debug1 = C_rot.copy()
        C_rot *= Aalpha[ig].conj()*nbasis/ng
        
        for iatm in range(natom):
            offset = atom_change[ig,iatm]

            C_f[offset*nao:(offset+1)*nao]+= C_rot[iatm*nao:(iatm+1)*nao]
            C_f[offset*nao+naoatom:(offset+1)*nao+naoatom]+= C_rot[iatm*nao+naoatom:(iatm+1)*nao+naoatom]
            C_rot_debug[offset*nao:(offset+1)*nao] = C_rot_debug1[iatm*nao:(iatm+1)*nao]
            C_rot_debug[offset*nao+naoatom:(offset+1)*nao+naoatom] = C_rot_debug1[iatm*nao+naoatom:(iatm+1)*nao+naoatom]
            
        # numpy.save('C_rot_final'+str(ig),C_rot_debug)
        
    return C_f