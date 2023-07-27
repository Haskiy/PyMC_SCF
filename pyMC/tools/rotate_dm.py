#!/usr/bin/env python
'''
Author: Pu Zhichen
Date: 2021-01-18 09:04:43
LastEditTime: 2022-08-06 07:05:18
LastEditors: Li Hao
Description: 
    Rotation of the DM or MO coefficients.
FilePath: /pyMC/tools/rotate_dm.py

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
from pyMC.tools import Dmatrix
from pyMC.tools import group_proj
import scipy.linalg
from pyscf import __config__
from collections import Counter
from scipy.spatial.transform import Rotation as R

# TODO :WTX? ONLY CLUSTER WITH SAME ATOMS CAN GET INITIAL DM.
# TODO : multi-atom clusters.
# TODO : Just using the already written codes for D, U and so on the get the Whole matrix, and just write
# TODO :    a interface to regroup the coefficients. By this means, smallest changes will be done, but, of couse, 
# TODO :    some memory and calculation will be wasted.

# TODO : Many subroutines is aborted. Can check whether those codes are reachable, if not reachable, can be deleted.

GROUP = {
        3 : 'C3'
    }

def _get_base_l(bas_l_each_base):
    l_dict = dict(Counter(bas_l_each_base))
    bas_l = []
    for l in l_dict:
        bas_l+=[l for _ in range(int(l_dict[l]/(2*l+1)))]
    return bas_l
    

def get_gks_dm_guess(mol, dm, natom_tot, theta, rotatel = True, rotatem = True):
    r'''From one atomic density matrix to generate total density matrix,
        by rotating the known one atomic density matrix.    

    Args:
        mol : mol object of the system

        dm : A list of UKS type density matrix stored as (2,nao,nao)
            ((alpha,alphya),(beta,beta))

        natom_tot : total atoms

        theta : a list of theta angles,
            floats, rotation Euler angles, stored as (alpha,beta,gamma),
            which corresponds to first rotation along z axis with alpha,
            rotations with y' axis with beta, and rotation with z'' axis with gamma.

            Theta is in rad, and is stored as (natom_tot,3)

    Kwargs:
        rotatel : whether rotate real space part. Default : True

        rotatem : whether rotate spin part. Default : True

    Returns:
        dmr : a list of 2D arrays, which should contains 4 parts
            , stored as (alpha_alpha,alpha_beta,beta_alpha,beta_beta)

    Examples:
    >>> 
    '''
    dmaa = dm[0,:,:]
    dmbb = dm[1,:,:]
    nao = dmaa.shape[-1]
    dmab = numpy.zeros((nao,nao),dtype = numpy.complex128)
    dmba = numpy.zeros((nao,nao),dtype = numpy.complex128)
    nao_tot = nao*natom_tot
    dm_total = numpy.zeros((nao_tot*2,nao_tot*2),dtype = numpy.complex128)
    dmaa_tmp = numpy.zeros((nao,nao),dtype = numpy.complex128)
    dmbb_tmp = numpy.zeros((nao,nao),dtype = numpy.complex128)
    dmab_tmp = numpy.zeros((nao,nao),dtype = numpy.complex128)
    dmba_tmp = numpy.zeros((nao,nao),dtype = numpy.complex128)
    for i in range(natom_tot):
        dmaa_tmp, dmab_tmp, dmba_tmp, dmbb_tmp = rotate_dm(mol, 
            (dmaa,dmab,dmba,dmbb), theta[i], rotatel = rotatel, rotatem = rotatem)
        dm_total[i*nao:(i+1)*nao , i*nao:(i+1)*nao] = dmaa_tmp
        dm_total[i*nao:(i+1)*nao , nao_tot+i*nao:nao_tot+(i+1)*nao] = dmab_tmp
        dm_total[nao_tot+i*nao:nao_tot+(i+1)*nao , i*nao:(i+1)*nao] = dmba_tmp
        dm_total[nao_tot+i*nao:nao_tot+(i+1)*nao , nao_tot+i*nao:nao_tot+(i+1)*nao] = dmbb_tmp

    return dm_total


def get_gks_dm_guess_mo(mol, mo_coeff_in, natom_tot, theta, rotatel = True
                        , rotatem = True, stepwise = False, moltot= None):
    
    # mo_coeffaa = mo_coeff_in[0]
    # mo_coeffbb = mo_coeff_in[1]
    # nao = mo_coeffaa.shape[-1]
    
    # mo_coeff = numpy.asarray(scipy.linalg.block_diag(mo_coeffaa,mo_coeffbb), dtype=numpy.complex128)
    
    # nao_tot = nao*natom_tot
    # mo_coeff_tot = numpy.zeros((nao_tot*2,nao_tot*2),dtype = numpy.complex128)
    
    # for i in range(natom_tot):
    #     mo_coeff_tot[i*nao:(i+1)*nao , i*nao:(i+1)*nao] = mo_coeff[:nao,:nao]
    #     mo_coeff_tot[nao_tot+i*nao:nao_tot+(i+1)*nao , nao_tot+i*nao:nao_tot+(i+1)*nao] = mo_coeff[nao:,nao:]
        
    # U = numpy.identity(mo_coeff_tot.shape[-1], dtype=numpy.complex128)
    # D = numpy.identity(mo_coeff_tot.shape[-1], dtype=numpy.complex128)
    # if rotatel:
    #     D = cal_D(mol, nao, natom_tot, theta)[0]
    # if rotatem:
    #     U = cal_U(mol, nao, natom_tot, theta)
    # mo_coeff_f = U@D@mo_coeff_tot
    if stepwise:
        if moltot == None:
            raise ValueError('Please check the input file, no moltot input in get_gks_dm_guess_mo')
        return get_gks_dm_guess_mo_direct_stepwise(mol, moltot, mo_coeff_in, natom_tot, theta, rotatel, rotatem)
    else:
        return get_gks_dm_guess_mo_direct(mol,  mo_coeff_in, natom_tot, theta, rotatel, rotatem)

def get_gks_dm_guess_mo_direct_stepwise(mol, moltot, mo_coeff_in, natom_tot, theta, rotatel = True, rotatem = True):
    """Get the initial guess of the mo coefficients.

    Args:
        mol (pyscf.gto.Mole or pyMC.mole_symm.Mole_sym): Single atom which construce the cluster.
        natom_tot (int): number of the atoms in the cluster.
        mo_coeff_in (numpy.array): [mol.nao()*2*natom,mol.nao()*2*natom] is the mo_coeff.
        theta (tuple or numpy.array): tuple of ((euler)*natom_tot)
            euler(numpy.array): is the euler angel of (alpha, beta, gamma) or rotation in real space.
        rotatel (bool, optional): Whether rotate real space. Defaults to True. 
        rotatem (bool, optional): Whether rotate spin space. Defaults to True. 

    Returns:
        mo_coeff_f (numpy.array): mo-coefficients after rotation.
    """
    mo_coeffaa = mo_coeff_in[0]
    mo_coeffbb = mo_coeff_in[1]
    nao = mo_coeffaa.shape[-1]
    
    mo_coeff = numpy.asarray(scipy.linalg.block_diag(mo_coeffaa,mo_coeffbb), dtype=numpy.complex128)
    
    nao_tot = nao*natom_tot
    mo_coeff_tot = numpy.zeros((nao_tot*2,nao_tot*2),dtype = numpy.complex128)
    C_f = numpy.zeros((nao_tot*2,nao_tot*2),dtype = numpy.complex128)
    
    for i in range(1):
        mo_coeff_tot[i*nao:(i+1)*nao , i*nao:(i+1)*nao] = mo_coeff[:nao,:nao]
        mo_coeff_tot[nao_tot+i*nao:nao_tot+(i+1)*nao , nao_tot+i*nao:nao_tot+(i+1)*nao] = mo_coeff[nao:,nao:]
    
    nx, theta_nx = euler_to_rotvec(theta)
    U = numpy.identity(mo_coeff_tot.shape[-1], dtype=numpy.complex128)
    D = numpy.identity(mo_coeff_tot.shape[-1], dtype=numpy.complex128)
    if rotatel:
        D = cal_D(mol, nao, natom_tot, theta)[0]
    if rotatem:
        U = cal_U_direct(mol, nao, natom_tot, (nx, theta_nx))
    mo_coeff_f = U@D@mo_coeff_tot
    moltot.groupname = GROUP[natom_tot]
    ng, theta, Aalpha, salpha, \
        atom_change, rotvec, theta_vec = group_proj._group_info(moltot, GROUP[natom_tot], 'CHI', 'A')
    for ig in range(ng//2):
        C_rot = rotate_mo_coeff_direct(mol, natom_tot, mo_coeff_f, (theta[ig], rotvec[ig], theta_vec[ig]))

        offset = atom_change[ig,0]
        C_f[offset*nao:(offset+1)*nao,offset*nao:(offset+1)*nao] = C_rot[:nao,:nao]
        C_f[offset*nao:(offset+1)*nao,offset*nao+nao_tot:(offset+1)*nao+nao_tot] \
            = C_rot[:nao,nao_tot:nao+nao_tot]
        C_f[offset*nao+nao_tot:(offset+1)*nao+nao_tot,offset*nao:(offset+1)*nao] \
            = C_rot[nao_tot:nao+nao_tot,:nao]
        C_f[offset*nao+nao_tot:(offset+1)*nao+nao_tot,offset*nao+nao_tot:(offset+1)*nao+nao_tot] \
            = C_rot[nao_tot:nao+nao_tot,nao_tot:nao+nao_tot]

    return C_f


def get_gks_dm_guess_mo_direct(mol, mo_coeff_in, natom_tot, theta, rotatel = True, rotatem = True):
    """Get the initial guess of the mo coefficients.

    Args:
        mol (pyscf.gto.Mole or pyMC.mole_symm.Mole_sym): Single atom which construce the cluster.
        natom_tot (int): number of the atoms in the cluster.
        mo_coeff_in (numpy.array): [mol.nao()*2*natom,mol.nao()*2*natom] is the mo_coeff.
        theta (tuple or numpy.array): tuple of ((euler)*natom_tot)
            euler(numpy.array): is the euler angel of (alpha, beta, gamma) or rotation in real space.
        rotatel (bool, optional): Whether rotate real space. Defaults to True. 
        rotatem (bool, optional): Whether rotate spin space. Defaults to True. 

    Returns:
        mo_coeff_f (numpy.array): mo-coefficients after rotation.
    """
    
    mo_coeffaa = mo_coeff_in[0]
    mo_coeffbb = mo_coeff_in[1]
    nao = mo_coeffaa.shape[-1]
    
    mo_coeff = numpy.asarray(scipy.linalg.block_diag(mo_coeffaa,mo_coeffbb), dtype=numpy.complex128)
    
    nao_tot = nao*natom_tot
    mo_coeff_tot = numpy.zeros((nao_tot*2,nao_tot*2),dtype = numpy.complex128)
    
    for i in range(natom_tot):
        mo_coeff_tot[i*nao:(i+1)*nao , i*nao:(i+1)*nao] = mo_coeff[:nao,:nao]
        mo_coeff_tot[nao_tot+i*nao:nao_tot+(i+1)*nao , nao_tot+i*nao:nao_tot+(i+1)*nao] = mo_coeff[nao:,nao:]
    
    nx, theta_nx = euler_to_rotvec(theta)
    U = numpy.identity(mo_coeff_tot.shape[-1], dtype=numpy.complex128)
    D = numpy.identity(mo_coeff_tot.shape[-1], dtype=numpy.complex128)
    if rotatel:
        D = cal_D(mol, nao, natom_tot, theta)[0]
    if rotatem:
        U = cal_U_direct(mol, nao, natom_tot, (nx, theta_nx))
    mo_coeff_f = U@D@mo_coeff_tot

    return mo_coeff_f



def rotate_mo_coeff(mol, natom, mo_coeff_in, theta):
    raise NotImplementedError("rotate_mo_coeff has been aborted!")
    
    nao = mo_coeff_in.shape[-1]//2//natom
    D = cal_D(mol, nao, natom, numpy.array([theta]*natom))[0]
    U = cal_U(mol, nao, natom, numpy.array([theta]*natom))
    mo_coeff_f = U@D@mo_coeff_in

    return mo_coeff_f

def rotate_mo_coeff_direct(mol, natom, mo_coeff_in, theta_dict):
    """Rotate MO coefficients.
       Rotation is done in two parts, because of the  J = L + 1/2*sigma for spinor. 
       L and s are communative, so rotation of real space and rotation of
       spin space can be seperated.

    Args:
        mol (pyscf.gto or mole_symm.Mole): single atom.
        natom (int): number of atoms
        mo_coeff_in (numpy array): initial mo_coeff
        theta_dict (tuple): tuple of (euler, nx, theta)
            euler(numpy.array): is the euler angel of (alpha, beta, gamma) or rotation in real space.
            nx(numpy.array): is the rotation axis.
            theta(float): rotation angle in the spin space (valued from 0 to 4pi)

    Returns:
        [type]: [description]
    """
    theta_real, nx, theta = theta_dict
       
    nao = mo_coeff_in.shape[0]//2//natom
    D = cal_D(mol, nao, natom, numpy.array([theta_real]*natom))[0]
    U = cal_U_direct(mol, nao, natom, (numpy.array([nx]*natom), numpy.array([theta]*natom)))
    # U = cal_U(mol, nao, natom, numpy.array([theta_real]*natom))
    mo_coeff_f = D@U@mo_coeff_in

    return mo_coeff_f

def rotate_mo_coeff_direct_4c(mol, natom, mo_coeff_in, theta_dict, dirac4c = True):
    """Rotate the orbital for 4-component Dirac calculations.
       If 2c calculations will be implemented, only small changes will be done.
       Rotation is done in two parts, because of the J = L + S = L + 1/2*Sigma for 4-c orbital, or
       J = L + 1/2*sigma for spinor. L and s are communative, so rotation of real space and rotation of
       spin space can be seperated.
       
       Dmatrix is rotation of real space and the U_part is rotation of spin space.
       
       Fomula or the rotation is listed as follow:
       '
        \hat{R}_{l}\hat{R}_{s}
            = & U^{-1} D  \left(\begin{array}{cc}
                A & B \\
                C & D
            \end{array}\right)
            U C
        '
        where U is the coefficients from sph2spinor. U^{-1}==U^{\dagger}

    Args:
        mol (pyscf.gto.Mole or pyMC.mole_symm.Mole_sym): Single atom which construce the cluster.
        natom (int): number of the atoms in the cluster.
        mo_coeff_in (numpy.array): [mol.nao_2c()*2*natom,mol.nao_2c()*2*natom] is the mo_coeff
            (LL LS)
            (SL SS) 
            is how the mo_coeff is saved.
        theta_dict (tuple): tuple of (euler, nx, theta)
            euler(numpy.array): is the euler angel of (alpha, beta, gamma) or rotation in real space.
            nx(numpy.array): is the rotation axis.
            theta(float): rotation angle in the spin space (valued from 0 to 4pi)
        dirac4c (bool, optional): Whether rotate 4-c orbital or 2-c orbital. Defaults to True. 

    Raises:
        ValueError: dirac4c = False --> rotate 2c spinors. Which is not been implemented.

    Returns:
        Cf (numpy.array): mo-coefficients after rotation.
    """
    if not dirac4c:
        raise ValueError('It should be noted that only Dirac 4c calculations can be done NOW !')
    theta_real, nx, theta = theta_dict
    
    nao2c_atm = mo_coeff_in.shape[0]//2//natom
    nao = nao2c_atm//2
    # get the U_sph2spinor_part, Dpart and U_spin_part matrix.
    # * Note that: part means only get the [natom*nao2c,natom*nao2c] for LL part
    U_sph2spinor_part = cal_sph2spinor_matrix(mol, natom)
    # Dpart is the rotation matrix (also know as Wigner-D function) for sphrical basis for LL part.
    Dpart = cal_D(mol, nao, natom, numpy.array([theta_real]*natom))[0]
    U_spin_part = cal_U_direct(mol, nao, natom, (numpy.array([nx]*natom), numpy.array([theta]*natom)))
    
    # get the full matrix for L and S
    U_sph2spinor = numpy.array(scipy.linalg.block_diag(U_sph2spinor_part,U_sph2spinor_part))
    D = numpy.array(scipy.linalg.block_diag(Dpart,Dpart))
    U_spin = numpy.array(scipy.linalg.block_diag(U_spin_part,U_spin_part))
    
    Cf = U_sph2spinor.conj().T@D@U_spin@U_sph2spinor@mo_coeff_in
    
    return Cf


def rotate_mo_coeff_direct_4c_debug(mol, natom, mo_coeff_in, theta_dict, dirac4c = True):
    """Rotate the orbital for 4-component Dirac calculations.
       If 2c calculations will be implemented, only small changes will be done.
       Rotation is done in two parts, because of the J = L + S = L + 1/2*Sigma for 4-c orbital, or
       J = L + 1/2*sigma for spinor. L and s are communative, so rotation of real space and rotation of
       spin space can be seperated.
       
       Dmatrix is rotation of real space and the U_part is rotation of spin space.
       
       Fomula or the rotation is listed as follow:
       '
        \hat{R}_{l}\hat{R}_{s}
            = & U^{-1} D  \left(\begin{array}{cc}
                A & B \\
                C & D
            \end{array}\right)
            U C
        '
        where U is the coefficients from sph2spinor. U^{-1}==U^{\dagger}

    Args:
        mol (pyscf.gto.Mole or pyMC.mole_symm.Mole_sym): Single atom which construce the cluster.
        natom (int): number of the atoms in the cluster.
        mo_coeff_in (numpy.array): [mol.nao_2c()*2*natom,mol.nao_2c()*2*natom] is the mo_coeff
            (LL LS)
            (SL SS) 
            is how the mo_coeff is saved.
        theta_dict (tuple): tuple of (euler, nx, theta)
            euler(numpy.array): is the euler angel of (alpha, beta, gamma) or rotation in real space.
            nx(numpy.array): is the rotation axis.
            theta(float): rotation angle in the spin space (valued from 0 to 4pi)
        dirac4c (bool, optional): Whether rotate 4-c orbital or 2-c orbital. Defaults to True. 

    Raises:
        ValueError: dirac4c = False --> rotate 2c spinors. Which is not been implemented.

    Returns:
        Cf (numpy.array): mo-coefficients after rotation.
    """
    if not dirac4c:
        raise ValueError('It should be noted that only Dirac 4c calculations can be done NOW !')
    theta_real, nx, theta = theta_dict
    
    nao2c_atm = mo_coeff_in.shape[0]//2//natom
    nao = nao2c_atm//2
    U_sph2spinor_part = numpy.zeros((nao2c_atm*natom,nao2c_atm*natom),dtype = numpy.complex128) 
    # get the U_sph2spinor_part, Dpart and U_spin_part matrix.
    # * Note that: part means only get the [natom*nao2c,natom*nao2c] for LL part
    # * This can be replaced by mol.sph2spinor_coeff().
    U_tmp = mol.sph2spinor_coeff()
    U_sph2spinor_part[:nao*natom] = U_tmp[0]
    U_sph2spinor_part[nao*natom:] = U_tmp[1]
    # Dpart is the rotation matrix (also know as Wigner-D function) for sphrical basis for LL part.
    Dpart = cal_D(mol, nao, natom, numpy.array([theta_real]*natom))[0]
    U_spin_part = cal_U_direct(mol, nao, natom, (numpy.array([nx]*natom), numpy.array([theta]*natom)))
    
    # get the full matrix for L and S
    U_sph2spinor = numpy.array(scipy.linalg.block_diag(U_sph2spinor_part,U_sph2spinor_part))
    D = numpy.array(scipy.linalg.block_diag(Dpart,Dpart))
    U_spin = numpy.array(scipy.linalg.block_diag(U_spin_part,U_spin_part))
    
    Cf = numpy.linalg.inv(U_sph2spinor)@U_spin@D@U_sph2spinor@mo_coeff_in
    
    return Cf
    


def rotate_mo_coeff_direct_single(mol, natom, mo_coeff_in, theta_dict):
    theta_real, nx, theta = theta_dict
    
    nao = mo_coeff_in.shape[-1]//2//natom
    D = cal_D(mol, nao, natom, numpy.array([theta_real]*natom))[0]
    # U = cal_U_direct(mol, nao, natom, (numpy.array([nx]*natom), numpy.array([theta]*natom)))
    # U = cal_U(mol, nao, natom, numpy.array([theta_real]*natom))
    mo_coeff_f = D@mo_coeff_in

    return mo_coeff_f

def euler_to_rotvec(theta):
    natom = theta.shape[0]
    theta_nx = numpy.zeros((natom))
    nx = numpy.zeros((natom,3))
    for i in range(natom):
        r = R.from_euler('ZYZ', theta[i], degrees=False) # Note that this part uses the intrinsic axis !
        v = r.as_rotvec()
        theta_nx[i] = numpy.linalg.norm(v)
        if numpy.linalg.norm(v) == 0:
            nx[i] = numpy.array([0.0, 0.0, 1.0])
        else:
            nx[i] = v/numpy.linalg.norm(v)
    
    return nx, theta_nx


def euler_to_rotvec_2(theta):
    nghalf = theta.shape[0]//2
    nx = numpy.zeros((nghalf*2,3))
    theta_nx = numpy.zeros((nghalf*2))
    for i in range(nghalf):
        r = R.from_euler('ZYZ', theta[i], degrees=False) # Note that this part uses the intrinsic axis !
        v = r.as_rotvec()
        theta_nx[i] = numpy.linalg.norm(v)
        if numpy.linalg.norm(v) == 0:
            nx[i] = numpy.array([0.0, 0.0, 1.0])
        else:
            nx[i] = v/numpy.linalg.norm(v)
    nx[nghalf:] = nx[:nghalf]
    theta_nx[nghalf:] = theta_nx[:nghalf] + 2*numpy.pi
    
    return nx, theta_nx

def get_gks_dm_T_guess(mol, dm, natom_tot, theta, rotatel = True, rotatem = True):
    r'''From one atomic density matrix to generate total density matrix,
        by rotating the known one atomic density matrix.    

    Args:
        mol : mol object of the system

        dm : A list of UKS type density matrix stored as (2,nao,nao)
            ((alpha,alphya),(beta,beta))

        natom_tot : total atoms

        theta : a list of theta angles,
            floats, rotation Euler angles, stored as (alpha,beta,gamma),
            which corresponds to first rotation along z axis with alpha,
            rotations with y' axis with beta, and rotation with z'' axis with gamma.

            Theta is in rad, and is stored as (natom_tot,3)

    Kwargs:
        rotatel : whether rotate real space part. Default : True

        rotatem : whether rotate spin part. Default : True

    Returns:
        dmr : a list of 2D arrays, which should contains 4 parts
            , stored as (alpha_alpha,alpha_beta,beta_alpha,beta_beta)

    Examples:
    >>> 
    '''
    dmaa = dm[0,:,:]
    dmbb = dm[1,:,:]
    nao = dmaa.shape[-1]
    dmab = numpy.zeros((nao,nao),dtype = numpy.complex128)
    dmba = numpy.zeros((nao,nao),dtype = numpy.complex128)
    nao_tot = nao*natom_tot
    dm_total = numpy.zeros((nao_tot*2,nao_tot*2),dtype = numpy.complex128)
    dmaa_tmp = numpy.zeros((nao,nao),dtype = numpy.complex128)
    dmbb_tmp = numpy.zeros((nao,nao),dtype = numpy.complex128)
    dmab_tmp = numpy.zeros((nao,nao),dtype = numpy.complex128)
    dmba_tmp = numpy.zeros((nao,nao),dtype = numpy.complex128)
    for i in range(natom_tot):
        dmaa_tmp, dmab_tmp, dmba_tmp, dmbb_tmp = rotate_dm_T(mol, 
            (dmaa,dmab,dmba,dmbb), theta[i], rotatel = rotatel, rotatem = rotatem)
        dm_total[i*nao:(i+1)*nao , i*nao:(i+1)*nao] = dmaa_tmp
        dm_total[i*nao:(i+1)*nao , nao_tot+i*nao:nao_tot+(i+1)*nao] = dmab_tmp
        dm_total[nao_tot+i*nao:nao_tot+(i+1)*nao , i*nao:(i+1)*nao] = dmba_tmp
        dm_total[nao_tot+i*nao:nao_tot+(i+1)*nao , nao_tot+i*nao:nao_tot+(i+1)*nao] = dmbb_tmp

    return dm_total
        

def rotate_dm_s(dm, theta):
    r'''Rotate the density matrix in spin space

    For different denstiy matrix blocks using different formulas:
    1. alpha alpha block
    2. alpha beta block
    3. beta alpha block
    4. beta beta block
    

    Args:
        dm : a list of 2D arrays, which should contains 4 parts
            , stored as (alpha_alpha,alpha_beta,beta_alpha,beta_beta)

        theta : a list of floats, rotation Euler angles, stored as (alpha,beta,gamma),
            which corresponds to first rotation along z axis with alpha,
            rotations with y' axis with beta, and rotation with z'' axis with gamma.

            Theta is in rad.

    Kwargs:
        None

    Returns:
        dmr : a list of 2D arrays, which should contains 4 parts
            , stored as (alpha_alpha,alpha_beta,beta_alpha,beta_beta)

    Examples:
    >>> 
    '''
    alpha, beta, gamma = theta
    halfbeta = 0.5*beta
    dmaa, dmab, dmba, dmbb = dm
    # AA = 1.0 - numpy.cos(gamma)*numpy.sin(beta)
    # BB = 1.0 + numpy.cos(gamma)*numpy.sin(beta)
    # AB = numpy.exp(-1.0j*alpha-1.0j*gamma)*numpy.cos(halfbeta)*numpy.cos(halfbeta) \
    #     -numpy.exp(-1.0j*alpha+1.0j*gamma)*numpy.sin(halfbeta)*numpy.sin(halfbeta)
    # BA = numpy.exp( 1.0j*alpha+1.0j*gamma)*numpy.cos(halfbeta)*numpy.cos(halfbeta) \
    #     -numpy.exp( 1.0j*alpha-1.0j*gamma)*numpy.sin(halfbeta)*numpy.sin(halfbeta)
    # dmaar = AA * (dmaa + dmab + dmba + dmbb)
    # dmabr = AB * (dmaa + dmab + dmba + dmbb)
    # dmbar = BA * (dmaa + dmab + dmba + dmbb)
    # dmbbr = BB * (dmaa + dmab + dmba + dmbb)
    cos2 = numpy.cos(halfbeta)*numpy.cos(halfbeta)
    sin2 = numpy.sin(halfbeta)*numpy.sin(halfbeta)
    e_ialpha = numpy.exp(1.0j*alpha)
    e_igamma = numpy.exp(1.0j*gamma)
    halfsin = numpy.sin(beta)*0.5
    dmaar = cos2*dmaa - e_igamma.conj()*halfsin*dmab - e_igamma*halfsin*dmba + sin2*dmbb
    dmabr = e_ialpha.conj()*halfsin*dmaa + e_ialpha.conj()*e_igamma.conj()*cos2*dmab \
        - e_ialpha.conj()*e_igamma*sin2*dmba - e_ialpha.conj()*halfsin*dmbb
    dmbar =  e_ialpha*halfsin*dmaa - e_igamma.conj()*e_ialpha*sin2*dmab \
        + e_ialpha*e_igamma*cos2*dmba - e_ialpha*halfsin*dmbb
    dmbbr = sin2*dmaa + e_igamma.conj()*halfsin*dmab + e_igamma*halfsin*dmba + cos2*dmbb

    return (dmaar, dmabr, dmbar ,dmbbr)


def rotate_dm_l(mol, dm, theta):
    r'''Rotate the density matrix in real space

        For different denstiy matrix blocks using the same formula
    

    Args:
        mol : mol object of the system

        dm : a list of 2D arrays, which should contains 4 parts
            , stored as (alpha_alpha,alpha_beta,beta_alpha,beta_beta)

        theta : a list of floats, rotation Euler angles, stored as (alpha,beta,gamma),
            which corresponds to first rotation along z axis with alpha,
            rotations with y' axis with beta, and rotation with z'' axis with gamma.

            Theta is in rad.

    Kwargs:
        None

    Returns:
        dmr : a list of 2D arrays, which should contains 4 parts
            , stored as (alpha_alpha,alpha_beta,beta_alpha,beta_beta)

    Examples:
    >>> 
    '''
    from pyscf import gto, symm
    alpha, beta, gamma = theta
    natm = mol.natm
    nao = dm[0].shape[-1]
    shell_list = numpy.array([mol.atom_nshells(i) for i in range(natm)])
    nshell = numpy.sum(shell_list)
    llist = [mol.bas_angular(i) for i in range(nshell)]
    tmp = numpy.array([gto.nao_nr_range(mol,i,i+1) for i in range(nshell)])
    bas_off = []
    # bas_l_label = []
    bas_id = []
    for i in range(nshell):
        bas_off.append([i for i in range(tmp[i,0],tmp[i,1])])
        for _ in bas_off[i]:
            #bas_l_label.append(llist[i])
            bas_id.append(i)
    #bas_off = numpy.array(bas_off)
    # bas_l_label = numpy.array(bas_l_label)
    bas_id = numpy.array(bas_id)
    bas_m = numpy.zeros((bas_id.shape[0]),dtype=numpy.int8)
    bas_l = numpy.zeros((bas_id.shape[0]),dtype=numpy.int8)
    i = 0
    while i < bas_id.shape[0]:
        if mol.bas_angular(bas_id[i]) == 1:
            bas_m[i] = -1
            bas_m[i+1] = 0
            bas_m[i+2] = 1
            bas_l[i:i+3] = 1
            i = i+3
            continue
        if mol.bas_angular(bas_id[i]) == 2:
            bas_m[i] = -2
            bas_m[i+1] = -1
            bas_m[i+2] = 0
            bas_m[i+3] = 1
            bas_m[i+4] = 2
            bas_l[i:i+5] = 2
            i = i+5
            continue
        if mol.bas_angular(bas_id[i]) == 3:
            bas_m[i] = -3
            bas_m[i+1] = -2
            bas_m[i+2] = -1
            bas_m[i+3] = 0
            bas_m[i+4] = 1
            bas_m[i+5] = 2
            bas_m[i+6] = 3
            bas_l[i:i+7] = 3
            i = i+7
            continue
        if mol.bas_angular(bas_id[i]) == 4:
            bas_m[i  ] = -4
            bas_m[i+1] = -3
            bas_m[i+2] = -2
            bas_m[i+3] = -1
            bas_m[i+4] = 0
            bas_m[i+5] = 1
            bas_m[i+6] = 2
            bas_m[i+7] = 3
            bas_m[i+8] = 4
            bas_l[i:i+9] = 4
            i = i+9
            continue
        i = i + 1
    # if nao != bas_off[-1][-1] + 1:
    #     raise RuntimeError('Angular moment of each spherical basis sets offset wrong')
    dmr = [numpy.zeros((nao,nao),dtype=numpy.complex128),
        numpy.zeros((nao,nao),dtype=numpy.complex128),
        numpy.zeros((nao,nao),dtype=numpy.complex128),
        numpy.zeros((nao,nao),dtype=numpy.complex128)]
    i = -1
    for dmop in dm:
        i += 1
        ioff=0
        WignerDM = numpy.zeros((bas_id.shape[0],bas_id.shape[0]))
        while ioff < bas_id.shape[0]:
            WignerD = Dmatrix.Dmatrix(bas_l[ioff], alpha, beta, gamma)
            if mol.bas_angular(bas_id[ioff]) == 1:
                WignerDM[ioff:ioff+3,ioff:ioff+3] = WignerD
                ioff = ioff+3
                continue
            if mol.bas_angular(bas_id[ioff]) == 2:
                WignerDM[ioff:ioff+5,ioff:ioff+5] = WignerD
                ioff = ioff+5
                continue
            if mol.bas_angular(bas_id[ioff]) == 3:
                WignerDM[ioff:ioff+7,ioff:ioff+7] = WignerD
                ioff = ioff+7
                continue
            WignerDM[ioff,ioff] = WignerD[0,0]
            ioff = ioff + 1

        dmr[i] = WignerDM@dm[i]@WignerDM.conj().T

    return dmr
        
# def rotate_dm_l(mol, dm, theta):
#     r'''Rotate the density matrix in real space

#         For different denstiy matrix blocks using the same formula
    

#     Args:
#         mol : mol object of the system

#         dm : a list of 2D arrays, which should contains 4 parts
#             , stored as (alpha_alpha,alpha_beta,beta_alpha,beta_beta)

#         theta : a list of floats, rotation Euler angles, stored as (alpha,beta,gamma),
#             which corresponds to first rotation along z axis with alpha,
#             rotations with y' axis with beta, and rotation with z'' axis with gamma.

#             Theta is in rad.

#     Kwargs:
#         None

#     Returns:
#         dmr : a list of 2D arrays, which should contains 4 parts
#             , stored as (alpha_alpha,alpha_beta,beta_alpha,beta_beta)

#     Examples:
#     >>> 
#     '''
#     from pyscf import gto, symm
#     alpha, beta, gamma = theta
#     natm = mol.natm
#     nao = dm[0].shape[-1]
#     shell_list = numpy.array([mol.atom_nshells(i) for i in range(natm)])
#     nshell = numpy.sum(shell_list)
#     llist = [mol.bas_angular(i) for i in range(nshell)]
#     tmp = numpy.array([gto.nao_nr_range(mol,i,i+1) for i in range(nshell)])
#     bas_off = []
#     # bas_l_label = []
#     bas_id = []
#     for i in range(nshell):
#         bas_off.append([i for i in range(tmp[i,0],tmp[i,1])])
#         for _ in bas_off[i]:
#             #bas_l_label.append(llist[i])
#             bas_id.append(i)
#     #bas_off = numpy.array(bas_off)
#     # bas_l_label = numpy.array(bas_l_label)
#     bas_id = numpy.array(bas_id)
#     bas_m = numpy.zeros((bas_id.shape[0]),dtype=numpy.int8)
#     i = 0
#     while i < bas_id.shape[0]:
#         if mol.bas_angular(bas_id[i]) == 1:
#             bas_m[i] = -1
#             bas_m[i+1] = 0
#             bas_m[i+2] = 1
#             i = i+3
#             continue
#         if mol.bas_angular(bas_id[i]) == 2:
#             bas_m[i] = -2
#             bas_m[i+1] = -1
#             bas_m[i+2] = 0
#             bas_m[i+3] = 1
#             bas_m[i+4] = 2
#             i = i+5
#             continue
#         if mol.bas_angular(bas_id[i]) == 3:
#             bas_m[i] = -3
#             bas_m[i+1] = -2
#             bas_m[i+2] = -1
#             bas_m[i+3] = 0
#             bas_m[i+4] = 1
#             bas_m[i+5] = 2
#             bas_m[i+6] = 3
#             i = i+7
#             continue
#         i = i + 1
#     # if nao != bas_off[-1][-1] + 1:
#     #     raise RuntimeError('Angular moment of each spherical basis sets offset wrong')
#     dmr = (numpy.zeros((nao,nao),dtype=numpy.complex128),
#         numpy.zeros((nao,nao),dtype=numpy.complex128),
#         numpy.zeros((nao,nao),dtype=numpy.complex128),
#         numpy.zeros((nao,nao),dtype=numpy.complex128))
#     i = -1
#     for dmop in dm:
#         i += 1
#         for j in range(nao):
#             for k in range(nao):
#                 row_id = bas_id[j]
#                 col_id = bas_id[k]
#                 # row_list = numpy.where(bas_id == bas_id[j])
#                 # col_list = numpy.where(bas_id == bas_id[k])
#                 lrow = mol.bas_angular(row_id)
#                 lcol = mol.bas_angular(col_id)
#                 mrow = bas_m[j]+lrow
#                 mcol = bas_m[k]+lcol
#                 WignerD_row = symm.Dmatrix.Dmatrix(lrow, alpha, beta, gamma)
#                 WignerD_col = symm.Dmatrix.Dmatrix(lcol, alpha, beta, gamma)
#                 D1 = WignerD_row[:,mrow]
#                 D2 = WignerD_col[:,mcol]
#                 D2p = numpy.zeros((D2.shape[0]))  
#                 for m in range(D2.shape[0]):
#                     D2p[m] = (-1.0)**(m-bas_m[k])*WignerD_col[lcol-m,lcol-bas_m[k]]
#                 dmr[i][j,k] += dm[i][j,k]*numpy.einsum('i,j->',D1,D2)

#     return dmr


def rotate_dm(mol, dm, theta, rotatel = True, rotatem = True):
    r'''Rotate the density matrix in spin space

        For different denstiy matrix blocks using different formulas:
        1. alpha alpha block
        2. alpha beta block
        3. beta alpha block
        4. beta beta block
    

    Args:
        mol : mol object of the system

        dm : a list of 2D arrays, which should contains 4 parts
            , stored as (alpha_alpha,alpha_beta,beta_alpha,beta_beta)

        theta : a list of floats, rotation Euler angles, stored as (alpha,beta,gamma),
            which corresponds to first rotation along z axis with alpha,
            rotations with y' axis with beta, and rotation with z'' axis with gamma.

            Theta is in rad.

    Kwargs:
        rotatel : whether rotate real space part. Default : True

        rotatem : whether rotate spin part. Default : True

    Returns:
        dmr : a list of 2D arrays, which should contains 4 parts
            , stored as (alpha_alpha,alpha_beta,beta_alpha,beta_beta)

    Examples:
    >>> 
    '''
    if rotatel :
        dmr = rotate_dm_l(mol, dm, theta)
        dm = dmr
    if rotatem :
        dmr2 = rotate_dm_s(dm, theta)
        dm = dmr2
    dmr_final = dm
        

    return dmr_final


def rotate_dm_T(mol, dm, theta, rotatel = True, rotatem = True):
    r'''Rotate the density matrix in spin space

        For different denstiy matrix blocks using different formulas:
        1. alpha alpha block
        2. alpha beta block
        3. beta alpha block
        4. beta beta block
    

    Args:
        mol : mol object of the system

        dm : a list of 2D arrays, which should contains 4 parts
            , stored as (alpha_alpha,alpha_beta,beta_alpha,beta_beta)

        theta : a list of floats, rotation Euler angles, stored as (alpha,beta,gamma),
            which corresponds to first rotation along z axis with alpha,
            rotations with y' axis with beta, and rotation with z'' axis with gamma.

            Theta is in rad.

    Kwargs:
        rotatel : whether rotate real space part. Default : True

        rotatem : whether rotate spin part. Default : True

    Returns:
        dmr : a list of 2D arrays, which should contains 4 parts
            , stored as (alpha_alpha,alpha_beta,beta_alpha,beta_beta)

    Examples:
    >>> 
    '''
    if rotatel :
        dmr = rotate_dm_l(mol, dm, theta)
        dm = dmr
    if rotatem :
        dmr2 = rotate_dm_s(dm, theta*numpy.array([1.0,2.0,1.0]))
        dm = dmr2
    dmr_final = dm
        

    return dmr_final
    

def get_gks_dm_Time_Rever_guess(mol, dm, natom_tot, theta, rotatel = True, rotatem = True):
    r'''From one atomic density matrix to generate total density matrix,
        by rotating the known one atomic density matrix.    

    Args:
        mol : mol object of the system

        dm : A list of UKS type density matrix stored as (2,nao,nao)
            ((alpha,alphya),(beta,beta))

        natom_tot : total atoms

        theta : a list of theta angles,
            floats, rotation Euler angles, stored as (alpha,beta,gamma),
            which corresponds to first rotation along z axis with alpha,
            rotations with y' axis with beta, and rotation with z'' axis with gamma.

            Theta is in rad, and is stored as (natom_tot,3)

    Kwargs:
        rotatel : whether rotate real space part. Default : True

        rotatem : whether rotate spin part. Default : True

    Returns:
        dmr : a list of 2D arrays, which should contains 4 parts
            , stored as (alpha_alpha,alpha_beta,beta_alpha,beta_beta)

    Examples:
    >>> 
    '''
    dm_total = get_gks_dm_guess(mol, dm, natom_tot, theta, rotatel, rotatem)

    nao = dm_total.shape[-1]//2
    tmpaa = dm_total[:nao,:nao].copy()
    dm_total[:nao,:nao] = dm_total[nao:,nao:].T
    dm_total[nao:,nao:] = tmpaa
    tmpab = dm_total[:nao,nao:].copy()
    dm_total[:nao,nao:] = - tmpab.T
    tmpba = dm_total[nao:,:nao].copy()
    dm_total[nao:,:nao] = - tmpba.T

    return dm_total


def rotate_gks_dm(mol, dm, theta, rotatel = True, rotatem = True):
    r'''Rotate a density matrix.   

    Args:
        mol : mol object of the system

        dm : A list of UKS type density matrix stored as (2,nao,nao)
            ((alpha,alphya),(beta,beta))

        natom_tot : total atoms

        theta : a list of theta angles,
            floats, rotation Euler angles, stored as (alpha,beta,gamma),
            which corresponds to first rotation along z axis with alpha,
            rotations with y' axis with beta, and rotation with z'' axis with gamma.

            Theta is in rad, and is stored as (natom_tot,3)

    Kwargs:
        rotatel : whether rotate real space part. Default : True

        rotatem : whether rotate spin part. Default : True

    Returns:
        dmr : a list of 2D arrays, which should contains 4 parts
            , stored as (alpha_alpha,alpha_beta,beta_alpha,beta_beta)

    Examples:
    >>> 
    '''
    nao = dm.shape[-1]//2
    dmaa = dm[:nao,:nao]
    dmab = dm[:nao,nao:]
    dmba = dm[nao:,:nao]
    dmbb = dm[nao:,nao:]
    nao = dmaa.shape[-1]
    nao_tot = nao
    dm_total = numpy.zeros((nao_tot*2,nao_tot*2),dtype = numpy.complex128)
    dmaa_tmp = numpy.zeros((nao,nao),dtype = numpy.complex128)
    dmbb_tmp = numpy.zeros((nao,nao),dtype = numpy.complex128)
    dmab_tmp = numpy.zeros((nao,nao),dtype = numpy.complex128)
    dmba_tmp = numpy.zeros((nao,nao),dtype = numpy.complex128)
    
    dmaa_tmp, dmab_tmp, dmba_tmp, dmbb_tmp = rotate_dm(mol, 
        (dmaa,dmab,dmba,dmbb), theta, rotatel = rotatel, rotatem = rotatem)
    dm_total[:nao,:nao] = dmaa_tmp
    dm_total[:nao,nao:] = dmab_tmp
    dm_total[nao:,:nao] = dmba_tmp
    dm_total[nao:,nao:] = dmbb_tmp

    return dm_total

def rotate_gks_dm_aba(mol, dm, theta, natom, rotatel = True, rotatem = True):
    r'''Divide a density matrix into each atom, and rotate each atom in different
    Thetas.

    Args:
        mol : mol object of the system

        dm : A list of UKS type density matrix stored as (2,nao,nao)
            ((alpha,alphya),(beta,beta))

        natom_tot : total atoms

        theta : a list of theta angles,
            floats, rotation Euler angles, stored as (alpha,beta,gamma),
            which corresponds to first rotation along z axis with alpha,
            rotations with y' axis with beta, and rotation with z'' axis with gamma.

            Theta is in rad, and is stored as (natom_tot,3)

    Kwargs:
        rotatel : whether rotate real space part. Default : True

        rotatem : whether rotate spin part. Default : True

    Returns:
        dmr : a list of 2D arrays, which should contains 4 parts
            , stored as (alpha_alpha,alpha_beta,beta_alpha,beta_beta)

    Examples:
    >>> 
    '''
    nao = dm.shape[-1]//2//natom
    # dmaa = dm[:nao,:nao]
    # dmab = dm[:nao,nao:]
    # dmba = dm[nao:,:nao]
    # dmbb = dm[nao:,nao:]
    
    nao_tot = nao*natom
    
    dm_total = numpy.zeros((nao_tot*2,nao_tot*2),dtype = numpy.complex128)
    # dmaa_tmp = numpy.zeros((nao,nao),dtype = numpy.complex128)
    # dmbb_tmp = numpy.zeros((nao,nao),dtype = numpy.complex128)
    # dmab_tmp = numpy.zeros((nao,nao),dtype = numpy.complex128)
    # dmba_tmp = numpy.zeros((nao,nao),dtype = numpy.complex128)
    
    D = numpy.identity(nao_tot*2, dtype = numpy.complex128)
    Dstar = numpy.identity(nao_tot*2, dtype = numpy.complex128)
    U = numpy.identity(nao_tot*2, dtype = numpy.complex128)
    
    if rotatel and (not rotatem):
        D,Dstar = cal_D(mol, nao, natom, theta)
    if rotatem and (not rotatel):
        U = cal_U(mol, nao, natom, theta)
    if rotatel and rotatem:
        D,Dstar = cal_D(mol, nao, natom, theta)
        U = cal_U(mol, nao, natom, theta)
    
    dm_total = U@D@dm@Dstar.T@U.T.conj()
    
    return dm_total

def cal_D(mol, nao, natom, thate_list):
    Da = numpy.zeros((nao*natom, nao*natom), dtype = numpy.complex128)
    Dastar = numpy.zeros((nao*natom, nao*natom), dtype = numpy.complex128)
    iatm = -1
    for theta in thate_list:
        iatm += 1
        alpha, beta, gamma = theta
        natm = mol.natm
        shell_list = numpy.array([mol.atom_nshells(i) for i in range(natm)])
        nshell = numpy.sum(shell_list)
        llist = [mol.bas_angular(i) for i in range(nshell)]
        tmp = numpy.array([gto.nao_nr_range(mol,i,i+1) for i in range(nshell)])
        bas_off = []
        # bas_l_label = []
        bas_id = []
        for i in range(nshell):
            bas_off.append([i for i in range(tmp[i,0],tmp[i,1])])
            for _ in bas_off[i]:
                #bas_l_label.append(llist[i])
                bas_id.append(i)
        #bas_off = numpy.array(bas_off)
        # bas_l_label = numpy.array(bas_l_label)
        bas_id = numpy.array(bas_id)
        bas_m = numpy.zeros((bas_id.shape[0]),dtype=numpy.int8)
        bas_l = numpy.zeros((bas_id.shape[0]),dtype=numpy.int8)
        i = 0
        while i < bas_id.shape[0]:
            if mol.bas_angular(bas_id[i]) == 1:
                bas_m[i] = -1
                bas_m[i+1] = 0
                bas_m[i+2] = 1
                bas_l[i:i+3] = 1
                i = i+3
                continue
            if mol.bas_angular(bas_id[i]) == 2:
                bas_m[i] = -2
                bas_m[i+1] = -1
                bas_m[i+2] = 0
                bas_m[i+3] = 1
                bas_m[i+4] = 2
                bas_l[i:i+5] = 2
                i = i+5
                continue
            if mol.bas_angular(bas_id[i]) == 3:
                bas_m[i] = -3
                bas_m[i+1] = -2
                bas_m[i+2] = -1
                bas_m[i+3] = 0
                bas_m[i+4] = 1
                bas_m[i+5] = 2
                bas_m[i+6] = 3
                bas_l[i:i+7] = 3
                i = i+7
                continue
            if mol.bas_angular(bas_id[i]) == 4:
                bas_m[i  ] = -4
                bas_m[i+1] = -3
                bas_m[i+2] = -2
                bas_m[i+3] = -1
                bas_m[i+4] = 0
                bas_m[i+5] = 1
                bas_m[i+6] = 2
                bas_m[i+7] = 3
                bas_m[i+8] = 4
                bas_l[i:i+9] = 4
                i = i+9
                continue
            i = i + 1
        WignerDM = numpy.zeros((bas_id.shape[0] ,bas_id.shape[0]))
        WignerDMstar = numpy.zeros((bas_id.shape[0] ,bas_id.shape[0]))
        ioff=0
        while ioff < bas_id.shape[0]:
            WignerD = Dmatrix.Dmatrix(bas_l[ioff], alpha, beta, gamma, reorder_p= True)
            if mol.bas_angular(bas_id[ioff]) == 1:
                WignerDM[ioff:ioff+3,ioff:ioff+3] = WignerD
                WignerDMstar[ioff:ioff+3,ioff:ioff+3] = calculate_Dstar(1, WignerD)
                ioff = ioff+3
                continue
            if mol.bas_angular(bas_id[ioff]) == 2:
                WignerDM[ioff:ioff+5,ioff:ioff+5] = WignerD
                WignerDMstar[ioff:ioff+5,ioff:ioff+5] = calculate_Dstar(2, WignerD)
                ioff = ioff+5
                continue
            if mol.bas_angular(bas_id[ioff]) == 3:
                WignerDM[ioff:ioff+7,ioff:ioff+7] = WignerD
                WignerDMstar[ioff:ioff+7,ioff:ioff+7] = calculate_Dstar(3, WignerD)
                ioff = ioff+7
                continue
            if mol.bas_angular(bas_id[ioff]) == 4:
                WignerDM[ioff:ioff+9,ioff:ioff+9] = WignerD
                WignerDMstar[ioff:ioff+9,ioff:ioff+9] = calculate_Dstar(4, WignerD)
                ioff = ioff+9
                continue
            WignerDM[ioff,ioff] = WignerD[0,0]
            WignerDMstar[ioff,ioff] = WignerD[0,0]
            ioff = ioff + 1
        Da[iatm*nao:(iatm+1)*nao, iatm*nao:(iatm+1)*nao] =  WignerDM
        Dastar[iatm*nao:(iatm+1)*nao, iatm*nao:(iatm+1)*nao] =  WignerDMstar
    D = numpy.asarray(scipy.linalg.block_diag(Da,Da), dtype=numpy.complex128)
    Dstar = numpy.asarray(scipy.linalg.block_diag(Dastar,Dastar), dtype=numpy.complex128)
    return D, Dstar


def cal_U(mol, nao, natom, thate_list):
    A = numpy.identity(nao*natom, dtype = numpy.complex128)
    B = numpy.identity(nao*natom, dtype = numpy.complex128)
    C = numpy.identity(nao*natom, dtype = numpy.complex128)
    D = numpy.identity(nao*natom, dtype = numpy.complex128)
    iatm = -1
    for theta in thate_list:
        iatm += 1
        alpha, beta, gamma = theta
        halfbeta = 0.5*beta
    
        coshalf = numpy.cos(halfbeta)
        sinhalf = numpy.sin(halfbeta)
        e_ialpha = numpy.exp(0.5j*alpha)
        e_igamma = numpy.exp(0.5j*gamma)
        
        Afactor = e_ialpha.conjugate()*e_igamma.conjugate()*coshalf
        Bfactor = - e_ialpha.conjugate()*e_igamma*sinhalf
        Cfactor = e_ialpha*e_igamma.conjugate()*sinhalf
        Dfactor = e_ialpha*e_igamma*coshalf
        A[iatm*nao:(iatm+1)*nao,iatm*nao:(iatm+1)*nao] = A[iatm*nao:(iatm+1)*nao,iatm*nao:(iatm+1)*nao]*Afactor
        B[iatm*nao:(iatm+1)*nao,iatm*nao:(iatm+1)*nao] = B[iatm*nao:(iatm+1)*nao,iatm*nao:(iatm+1)*nao]*Bfactor
        C[iatm*nao:(iatm+1)*nao,iatm*nao:(iatm+1)*nao] = C[iatm*nao:(iatm+1)*nao,iatm*nao:(iatm+1)*nao]*Cfactor
        D[iatm*nao:(iatm+1)*nao,iatm*nao:(iatm+1)*nao] = D[iatm*nao:(iatm+1)*nao,iatm*nao:(iatm+1)*nao]*Dfactor
    U = numpy.asarray(scipy.linalg.block_diag(A,D), dtype=numpy.complex128)
    U[:nao*natom,nao*natom:] = B
    U[nao*natom:,:nao*natom] = C
    
    return U

def cal_U_direct(mol, nao, natom, Rotation):
    """A direct subroutine

    Args:
        mol ([type]): [description]
        nao ([type]): [description]
        natom ([type]): [description]
        Rotation ([type]): [description]

    Returns:
        [type]: [description]
    """
    nx, theta = Rotation
    A = numpy.identity(nao*natom, dtype = numpy.complex128)
    B = numpy.identity(nao*natom, dtype = numpy.complex128)
    C = numpy.identity(nao*natom, dtype = numpy.complex128)
    D = numpy.identity(nao*natom, dtype = numpy.complex128)
    normnx = numpy.sqrt(nx[:,0]**2 + nx[:,1]**2 + nx[:,2]**2)
    costheta = numpy.cos(theta*0.5*normnx)
    sintheta = numpy.sin(theta*0.5*normnx)
    
    # print(normnx[0],theta[0],costheta[0],sintheta[0])
    
    Afactor = costheta - nx[:,2]*1.0j*sintheta/normnx
    Bfactor = (-nx[:,0]*1.0j - nx[:,1])*sintheta/normnx
    Cfactor = (-nx[:,0]*1.0j + nx[:,1])*sintheta/normnx
    Dfactor = costheta + nx[:,2]*1.0j*sintheta/normnx
    # print(Afactor,Bfactor,Cfactor,Dfactor)
    # print(Afactor[0],Bfactor[0],Cfactor[0],Dfactor[0],theta[0])
    # print()
    
    for iatm in range(natom):
        A[iatm*nao:(iatm+1)*nao,iatm*nao:(iatm+1)*nao] = A[iatm*nao:(iatm+1)*nao,iatm*nao:(iatm+1)*nao]*Afactor[iatm]
        B[iatm*nao:(iatm+1)*nao,iatm*nao:(iatm+1)*nao] = B[iatm*nao:(iatm+1)*nao,iatm*nao:(iatm+1)*nao]*Bfactor[iatm]
        C[iatm*nao:(iatm+1)*nao,iatm*nao:(iatm+1)*nao] = C[iatm*nao:(iatm+1)*nao,iatm*nao:(iatm+1)*nao]*Cfactor[iatm]
        D[iatm*nao:(iatm+1)*nao,iatm*nao:(iatm+1)*nao] = D[iatm*nao:(iatm+1)*nao,iatm*nao:(iatm+1)*nao]*Dfactor[iatm]
        
    U = numpy.asarray(scipy.linalg.block_diag(A,D), dtype=numpy.complex128)
    U[:nao*natom,nao*natom:] = B
    U[nao*natom:,:nao*natom] = C
    # print(A,B,C,D)
    
    return U


def calculate_Dstar(l, Dmatrix):
    WignerDMstar = numpy.zeros((2*l+1, 2*l+1))
    if l == 1:
        WignerDMstar[0,0] = Dmatrix[1,1]
        WignerDMstar[0,1] = Dmatrix[1,0]
        WignerDMstar[0,2] =-Dmatrix[1,2]
        WignerDMstar[1,0] = Dmatrix[0,1]
        WignerDMstar[1,1] = Dmatrix[0,0]
        WignerDMstar[1,2] =-Dmatrix[0,2]
        WignerDMstar[2,0] =-Dmatrix[2,1]
        WignerDMstar[2,1] =-Dmatrix[2,0]
        WignerDMstar[2,2] = Dmatrix[2,2]
    elif l == 2:
        WignerDMstar[0] = Dmatrix[4,::-1]
        WignerDMstar[1] = Dmatrix[3,::-1]
        WignerDMstar[2] = Dmatrix[2,::-1]
        WignerDMstar[3] = Dmatrix[1,::-1]
        WignerDMstar[4] = Dmatrix[0,::-1]
        WignerDMstar[0,1] = -WignerDMstar[0,1]
        WignerDMstar[0,3] = -WignerDMstar[0,3]
        WignerDMstar[1,0] = -WignerDMstar[1,0]
        WignerDMstar[1,2] = -WignerDMstar[1,2]
        WignerDMstar[1,4] = -WignerDMstar[1,4]
        WignerDMstar[2,1] = -WignerDMstar[2,1]
        WignerDMstar[2,3] = -WignerDMstar[2,3]
        WignerDMstar[3,0] = -WignerDMstar[3,0]
        WignerDMstar[3,2] = -WignerDMstar[3,2]
        WignerDMstar[3,4] = -WignerDMstar[3,4]
        WignerDMstar[4,1] = -WignerDMstar[4,1]
        WignerDMstar[4,3] = -WignerDMstar[4,3]
    elif l == 3:
        WignerDMstar[0] = Dmatrix[6,::-1]
        WignerDMstar[1] = Dmatrix[5,::-1]
        WignerDMstar[2] = Dmatrix[4,::-1]
        WignerDMstar[3] = Dmatrix[3,::-1]
        WignerDMstar[4] = Dmatrix[2,::-1]
        WignerDMstar[5] = Dmatrix[1,::-1]
        WignerDMstar[6] = Dmatrix[0,::-1]
        WignerDMstar[0,1] = -WignerDMstar[0,1]
        WignerDMstar[0,3] = -WignerDMstar[0,3]
        WignerDMstar[0,5] = -WignerDMstar[0,5]
        WignerDMstar[1,0] = -WignerDMstar[1,0]
        WignerDMstar[1,2] = -WignerDMstar[1,2]
        WignerDMstar[1,4] = -WignerDMstar[1,4]
        WignerDMstar[1,6] = -WignerDMstar[1,6]
        
        WignerDMstar[2,1] = -WignerDMstar[2,1]
        WignerDMstar[2,3] = -WignerDMstar[2,3]
        WignerDMstar[2,5] = -WignerDMstar[2,5]
        WignerDMstar[3,0] = -WignerDMstar[3,0]
        WignerDMstar[3,2] = -WignerDMstar[3,2]
        WignerDMstar[3,4] = -WignerDMstar[3,4]
        WignerDMstar[3,6] = -WignerDMstar[3,6]
        
        WignerDMstar[4,1] = -WignerDMstar[4,1]
        WignerDMstar[4,3] = -WignerDMstar[4,3]
        WignerDMstar[4,5] = -WignerDMstar[4,5]
        WignerDMstar[5,0] = -WignerDMstar[5,0]
        WignerDMstar[5,2] = -WignerDMstar[5,2]
        WignerDMstar[5,4] = -WignerDMstar[5,4]
        WignerDMstar[5,6] = -WignerDMstar[5,6]
        
        WignerDMstar[6,1] = -WignerDMstar[6,1]
        WignerDMstar[6,3] = -WignerDMstar[6,3]
        WignerDMstar[6,5] = -WignerDMstar[6,5]
    
    return WignerDMstar

def _get_baslm(mol):
    nshell = mol.atom_nshells(0)
    tmp = numpy.array([gto.nao_nr_range(mol,i,i+1) for i in range(nshell)])
    bas_off = []
            # bas_l_label = []
    bas_id = []
    for i in range(nshell):
            bas_off.append([i for i in range(tmp[i,0],tmp[i,1])])
            for _ in bas_off[i]:
                    #bas_l_label.append(llist[i])
                bas_id.append(i)
    bas_id = numpy.array(bas_id)
    bas_m = numpy.zeros((bas_id.shape[0]),dtype=numpy.int8)
    bas_l = numpy.zeros((bas_id.shape[0]),dtype=numpy.int8)
    i = 0
    while i < bas_id.shape[0]:
        if mol.bas_angular(bas_id[i]) == 1:
            bas_m[i] = -1
            bas_m[i+1] = 0
            bas_m[i+2] = 1
            bas_l[i:i+3] = 1
            i = i+3
            continue
        if mol.bas_angular(bas_id[i]) == 2:
            bas_m[i] = -2
            bas_m[i+1] = -1
            bas_m[i+2] = 0
            bas_m[i+3] = 1
            bas_m[i+4] = 2
            bas_l[i:i+5] = 2
            i = i+5
            continue
        if mol.bas_angular(bas_id[i]) == 3:
            bas_m[i] = -3
            bas_m[i+1] = -2
            bas_m[i+2] = -1
            bas_m[i+3] = 0
            bas_m[i+4] = 1
            bas_m[i+5] = 2
            bas_m[i+6] = 3
            bas_l[i:i+7] = 3
            i = i+7
            continue
        if mol.bas_angular(bas_id[i]) == 4:
            bas_m[i  ] = -4
            bas_m[i+1] = -3
            bas_m[i+2] = -2
            bas_m[i+3] = -1
            bas_m[i+4] = 0
            bas_m[i+5] = 1
            bas_m[i+6] = 2
            bas_m[i+7] = 3
            bas_m[i+8] = 4
            bas_l[i:i+9] = 4
            i = i+9
            continue
        i = i + 1
    return bas_l,bas_m

def _get_basjjz(mol):
    basl, bas_m = _get_baslm(mol)
    nao = basl.shape[-1]
    basj = numpy.zeros((nao*2))
    basjz = numpy.zeros((nao*2))
    basl_2c = numpy.zeros((nao*2)).astype(int)
    i = 0
    jz_list = numpy.arange(-3.5,4.0,1.0)
    jz_listg = numpy.arange(-4.5,5.0,1.0)
    while i < 2*nao:
        l = basl[i//2]
        if l == 0:  # s 1/2
            basj[i:i+2] = 0.5
            basjz[i:i+2] =jz_list[3:5].copy()
            basl_2c[i:i+2] = 0
            i+=2
            continue
        elif l == 1: # p1/2 and p3/2
            basj[i:i+2] = 0.5
            basj[i+2:i+6] = 1.5
            basjz[i:i+2] = jz_list[3:5].copy()
            basjz[i+2:i+6] = jz_list[2:6].copy()
            basl_2c[i:i+6] = 1
            i+=6
            continue
        elif l == 2: # d3/2 d5/2
            basj[i:i+4] = 1.5
            basj[i+4:i+10] = 2.5
            basjz[i:i+4] = jz_list[2:6].copy()
            basjz[i+4:i+10] = jz_list[1:7].copy()
            basl_2c[i:i+10] = 2
            i+=10
            continue
        elif l == 3: #f5/2 f7/2
            basj[i:i+6] = 2.5
            basj[i+6:i+14] = 3.5
            basjz[i:i+6] = jz_list[1:7].copy()
            basjz[i+6:i+14] = jz_list[0:8].copy()
            basl_2c[i:i+14] = 3
            i+=14
            continue
        elif l == 4: #g7/2 g9/2
            basj[i:i+8] = 3.5
            basj[i+8:i+18] = 4.5
            basjz[i:i+8] = jz_list[0:8].copy()
            basjz[i+8:i+18] = jz_listg[0:10].copy()
            basl_2c[i:i+18] = 4
            i+=18
            continue
    return basj, basjz, basl_2c
            

def cal_D_r(mol, nao2c, theta):
    alpha, beta, gamma = theta
    D = numpy.zeros((nao2c,nao2c), dtype = numpy.complex128)
    basj, basjz, basl_2c = _get_basjjz(mol)
    basj_list = basj.tolist()
    basj_ind = []
    [basj_ind.append(i) for i in basj_list if i not in basj_ind]
    D_dict = {}
    for j in basj_ind:
        D_dict[j] = Dmatrix.Dmatrix_r(j, alpha, beta, gamma)
    i = 0
    while i < nao2c:
        j = basj[i]
        D[i:i+int(j*2)+1,i:i+int(j*2)+1] = D_dict[j]
        i+=int(2*j)+1
    return D
    
    
        
def get_gks_dm_guess_mo_4c(mol, mo_coeff_in, natom_tot, theta_dict, stepwise = False, moltot = None):
    """Get the initial guess of the mo coefficients.

    Args:
        mol (pyscf.gto.Mole or pyMC.mole_symm.Mole_sym): Single atom which construce the cluster.
        natom_tot (int): number of the atoms in the cluster.
        mo_coeff_in (numpy.array): [mol.nao_2c()*2*natom,mol.nao_2c()*2*natom] is the mo_coeff
            (LL LS)
            (SL SS) 
            is how the mo_coeff is saved.
        theta_dict (tuple): tuple of ((euler, nx, theta)*natom_tot)
            euler(numpy.array): is the euler angel of (alpha, beta, gamma) or rotation in real space.
            nx(numpy.array): is the rotation axis.
            theta(float): rotation angle in the spin space (valued from 0 to 4pi)
        dirac4c (bool, optional): Whether rotate 4-c orbital or 2-c orbital. Defaults to True. 

    Returns:
        Cf (numpy.array): mo-coefficients after rotation.
    """
    nao2c = mo_coeff_in.shape[-1]//2
    nao_tot = nao2c*natom_tot
    mo_coeff_tot = numpy.zeros((nao_tot*2,nao_tot*2),dtype = numpy.complex128)
    
    if stepwise:
        if moltot == None:
            raise ValueError('Please check the input file, no moltot input in get_gks_dm_guess_mo') 
        moltot.groupname = GROUP[natom_tot]
        ng, theta, Aalpha, salpha, \
            atom_change, rotvec, theta_vec = group_proj._group_info(moltot, GROUP[natom_tot], 'CHI', 'A')
        mo_coeff_f = rotate_mo_coeff_direct_4c(mol, 1, mo_coeff_in,theta_dict[0])
        for ig in range(natom_tot):
            i = atom_change[ig,0]
            mo_coeff_f2 = rotate_mo_coeff_direct_4c(mol, 1, mo_coeff_f, (theta[ig], rotvec[ig], theta_vec[ig]))
            mo_coeff_tot[i*nao2c:i*nao2c+nao2c,i*nao2c:i*nao2c+nao2c] = mo_coeff_f2[:nao2c,:nao2c]
            mo_coeff_tot[i*nao2c:i*nao2c+nao2c,nao_tot+i*nao2c:nao_tot+i*nao2c+nao2c] = mo_coeff_f2[:nao2c,nao2c:]
            mo_coeff_tot[nao_tot+i*nao2c:nao_tot+i*nao2c+nao2c,i*nao2c:i*nao2c+nao2c] = mo_coeff_f2[nao2c:,:nao2c]
            mo_coeff_tot[nao_tot+i*nao2c:nao_tot+i*nao2c+nao2c,nao_tot+i*nao2c:nao_tot+i*nao2c+nao2c] = mo_coeff_f2[nao2c:,nao2c:]
    else:        
        for i in range(natom_tot):
            mo_coeff_f = rotate_mo_coeff_direct_4c(mol, 1, mo_coeff_in,theta_dict[i])
            mo_coeff_tot[i*nao2c:i*nao2c+nao2c,i*nao2c:i*nao2c+nao2c] = mo_coeff_f[:nao2c,:nao2c]
            mo_coeff_tot[i*nao2c:i*nao2c+nao2c,nao_tot+i*nao2c:nao_tot+i*nao2c+nao2c] = mo_coeff_f[:nao2c,nao2c:]
            mo_coeff_tot[nao_tot+i*nao2c:nao_tot+i*nao2c+nao2c,i*nao2c:i*nao2c+nao2c] = mo_coeff_f[nao2c:,:nao2c]
            mo_coeff_tot[nao_tot+i*nao2c:nao_tot+i*nao2c+nao2c,nao_tot+i*nao2c:nao_tot+i*nao2c+nao2c] = mo_coeff_f[nao2c:,nao2c:]
        
    return mo_coeff_tot


def cal_sph2spinor_matrix(molsingle, natom):
    """Get the sphrical to spinor matrix, which is defined as U in the notes.
    
        NOTE that the U is DEFINED and ORGANISED as the following formula.
            \left(\begin{array}{l}
            p_{\alpha} \\
            p_{\beta}
            \end{array}\right)
            = & \sum_{\mu} \mu U_{\mu p} \\
            = & \sum_{\mu} \left(\begin{array}{l}
                \mu \\
                0
            \end{array}\right) U_{\mu p}^{\alpha}
            + \sum_{\mu} \left(\begin{array}{l}
                0 \\
                \mu
            \end{array}\right) U_{\mu p}^{\beta}

    Args:
        molsingle (mole_symm.Mole_symm type): A single atom enabling double group symmetry in 4-c DKS ]
            calculations.
        natom (int): number of atoms

    Returns:
        U: U matrix.
    """
    
    nao2c = molsingle.nao_2c()
    nao = nao2c//2
    nhalf = nao2c*natom//2
    U = numpy.zeros((nao2c*natom,nao2c*natom),dtype = numpy.complex128)
    #
    bas_l_each_base = _get_baslm(molsingle)[0]
    bas_l = _get_base_l(bas_l_each_base)
    lvalus = set(bas_l_each_base.tolist())
    # Dictionary saves the U for each L
    # Ulist[l] = (Ualpha, Ubeta) Ualtha is a [2*l+1,2*(2*l+1)] numpy array
    Ulist = {}
    for l in lvalus:
        Ulist[l] = gto.mole.sph2spinor_l(l)
    ioffset = 0
    # * NOTE U[:nao*natom,:] is the Ualpha part
    # * NOTE U[nao*natom:,:] is the Ubeta part
    for l in bas_l:
        U[ioffset:ioffset+2*l+1,ioffset*2:ioffset*2+(2*l+1)*2] = Ulist[l][0]
        ioffset_beta = ioffset+nhalf
        U[ioffset_beta:ioffset_beta+2*l+1,ioffset*2:ioffset*2+(2*l+1)*2] = Ulist[l][1]
        ioffset += 2*l + 1
        
    for iatom in range(1,natom):
        # alpha alpha part.
        U[iatom*nao:(iatom+1)*nao, iatom*nao2c:(iatom+1)*nao2c] = U[:nao, :nao2c]
        # beta beta part.
        U[nhalf+iatom*nao:nhalf+(iatom+1)*nao, iatom*nao2c:(iatom+1)*nao2c] = U[nhalf:nhalf+nao, :nao2c]
        
    return U

def get_init_guess_theta(mol, rotatez_negative = False, vortex = False,
                             target_vec_atm1 = None):
    """This subroutine can calculates the rotation information to get the initial DM.
       Informations contains euler angle of (alpha, beta, gamma) in intrinsic rotation, rotation vector,
       rotation angles.
       
       The output can be directed used in get_gks_dm_guess_mo_4c and 

    Args:
        mol (pyscf.gto.Mole object or pyMC.mole_symm.Mole_sym object): Saves the cluster geometry and other informations.

        rotatez_negative (Bool) : Default to be False.
    Returns:
        (list): [[numpy.array([alpha,beta,gamma]), numpy.array([nx,ny,nz]), theta]*natom]
            where nx,ny,nz is the rotation vector.
    """
    return get_init_guess_theta_new(mol, rotatez_negative = rotatez_negative, vortex = vortex,
                                    target_vec_atm1 = target_vec_atm1)
    # raise NotImplementedError("This is not recommended for generating theta dictionary.")
    natom = mol.natm
    if rotatez_negative:
        z1 = numpy.array([.0, .0,-1.0])
    else:
        z1 = numpy.array([.0, .0, 1.0])
    pivot = numpy.array([1.0, .0, .0])
    euler_list = []
    nx_lsit = []
    theta_list = []
    for i in range(natom):
        if not vortex:
            z2 = mol.atom_coord(i)
        else:
            zout = mol.atom_coord(i)
            z2 = numpy.cross(z1,zout)
        normz2 = numpy.linalg.norm(z2)
        vec = numpy.cross(z1, z2)
        vec = vec/numpy.linalg.norm(vec)
        costheta = numpy.dot(z1, z2)/numpy.linalg.norm(z1)/numpy.linalg.norm(z2)
        theta = numpy.arccos(costheta)
        r = R.from_rotvec(theta*vec)
        T1 = r.as_matrix()
        pivot_tmp = (T1@pivot.reshape(-1,1)).reshape(-1,)
        # vec2 = numpy.cross(pivot_tmp, z1)
        # vec2 = vec2/numpy.linalg.norm(vec2)
        costheta2 = numpy.dot(z1, pivot_tmp)/numpy.linalg.norm(z1)/numpy.linalg.norm(pivot_tmp)
        theta2 = numpy.arccos(costheta2)
        r2 = R.from_rotvec(theta2*z2/normz2)
        rf = R.from_matrix(r2.as_matrix()@T1)
        euler_list.append(rf.as_euler('ZYZ', degrees=False))
        # euler_list.append(r.as_euler('ZYZ', degrees=False))
        vecf = rf.as_rotvec()
        thetaf = numpy.linalg.norm(vecf)
        nx_lsit.append(vec)
        theta_list.append(theta)
    
    return [[euler_list[i], nx_lsit[i], theta_list[i]] for i in range(natom)]

def get_init_guess_theta_new(mol, rotatez_negative = False, vortex = False,
                             target_vec_atm1 = None):
    """This subroutine can calculates the rotation information to get the initial DM.
       Informations contains euler angle of (alpha, beta, gamma) in intrinsic rotation, rotation vector,
       rotation angles.
       
       The output can be directed used in get_gks_dm_guess_mo_4c and 

    Args:
        mol (pyscf.gto.Mole object or pyMC.mole_symm.Mole_sym object): Saves the cluster geometry and other informations.

        rotatez_negative (Bool) : Default to be False.
    Returns:
        (list): [[numpy.array([alpha,beta,gamma]), numpy.array([nx,ny,nz]), theta]*natom]
            where nx,ny,nz is the rotation vector.
    """
    natom = mol.natm
    euler_list = []
    nx_lsit = []
    theta_list = []
    if rotatez_negative:
        z1 = numpy.array([.0, .0,-1.0])
    else:
        z1 = numpy.array([.0, .0, 1.0])
    c1 = numpy.eye(3)

    for i in range(natom):
        c2 = numpy.zeros((3,3))
        # In this subroutine, we let the orginal x vector to point -z.
        c2[0,2] = -1
        # Make the orginal z axis to the new orientation
        if not vortex:
            c2[2] = mol.atom_coord(i)
        else:
            if target_vec_atm1 is None:
                zout = mol.atom_coord(i)
                c2[2] = numpy.cross(z1,zout)
            else:
                # rotate the z axis the orientation
                c2[2] = target_vec_atm1[i]
                if c2[2,2] != 0.0:
                    c2[0,0:2] = c2[2,0:2].copy()
                    vec_v = numpy.cross(c2[0],c2[2])
                    vec_v = vec_v/numpy.linalg.norm(vec_v)
                    r_v = R.from_rotvec(vec_v*numpy.pi*0.5)
                    c2[0] = -(r_v.as_matrix()@target_vec_atm1[i].reshape(-1,1)).reshape(1,-1)
                    # c2[0,2] = -(c2[2,0]**2 + c2[2,1]**2)/c2[2,2]
        # Make another axis to be in the new orientation.
        c2[2] = c2[2]/numpy.linalg.norm(c2[2])
        c2[1] = numpy.cross(c2[2], c2[0])
        alpha, beta, gamma = Dmatrix.get_euler_angles(c1, c2)
        r = R.from_euler('ZYZ', numpy.array([alpha, beta, gamma]))
        euler_list.append(numpy.array([alpha, beta, gamma]))
        # euler_list.append(r.as_euler('ZYZ', degrees=False))
        vecf = r.as_rotvec()
        thetaf = numpy.linalg.norm(vecf)
        nx_lsit.append(vecf/thetaf)
        theta_list.append(thetaf)
    
    return [[euler_list[i], nx_lsit[i], theta_list[i]] for i in range(natom)]

def get_gks_dm_guess_mo_direct_new(mol, mo_coeff_in, natom_tot, theta_dict, rotatel = True, rotatem = True):
    """Get the initial guess of the mo coefficients.

    Args:
        mol (pyscf.gto.Mole or pyMC.mole_symm.Mole_sym): Single atom which construce the cluster.
        natom_tot (int): number of the atoms in the cluster.
        mo_coeff_in (numpy.array): [mol.nao()*2*natom,mol.nao()*2*natom] is the mo_coeff.
        theta_dict (tuple or numpy.array): tuple of ((euler)*natom_tot)
            euler(numpy.array): is the euler angel of (alpha, beta, gamma) or rotation in real space.
        rotatel (bool, optional): Whether rotate real space. Defaults to True. 
        rotatem (bool, optional): Whether rotate spin space. Defaults to True. 

    Returns:
        mo_coeff_f (numpy.array): mo-coefficients after rotation.
    """
    
    mo_coeffaa = mo_coeff_in[0]
    mo_coeffbb = mo_coeff_in[1]
    nao = mo_coeffaa.shape[-1] 
    natom =  theta_dict.__len__()
    theta = numpy.array([theta_dict[i][0] for i in range(natom)])
    nx = numpy.array([theta_dict[i][1] for i in range(natom)])
    theta_nx = numpy.array([theta_dict[i][2] for i in range(natom)])
    
    mo_coeff = numpy.asarray(scipy.linalg.block_diag(mo_coeffaa,mo_coeffbb), dtype=numpy.complex128)
    
    nao_tot = nao*natom_tot
    mo_coeff_tot = numpy.zeros((nao_tot*2,nao_tot*2),dtype = numpy.complex128)
    
    for i in range(natom_tot):
        mo_coeff_tot[i*nao:(i+1)*nao , i*nao:(i+1)*nao] = mo_coeff[:nao,:nao]
        mo_coeff_tot[nao_tot+i*nao:nao_tot+(i+1)*nao , nao_tot+i*nao:nao_tot+(i+1)*nao] = mo_coeff[nao:,nao:]
        
    U = numpy.identity(mo_coeff_tot.shape[-1], dtype=numpy.complex128)
    D = numpy.identity(mo_coeff_tot.shape[-1], dtype=numpy.complex128)
    if rotatel:
        D = cal_D(mol, nao, natom_tot, theta)[0]
    if rotatem:
        U = cal_U_direct(mol, nao, natom_tot, (nx, theta_nx))
    mo_coeff_f = U@D@mo_coeff_tot

    return mo_coeff_f        

    
