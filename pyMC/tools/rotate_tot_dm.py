#!/usr/bin/env python
'''
Author: Pu Zhichen
Date: 2021-07-22 18:55:32
LastEditTime: 2021-07-23 10:14:19
LastEditors: Pu Zhichen
Description: 
FilePath: \pyMC\tools\rotate_tot_dm.py

 May the force be with you!
'''

import numpy
import scipy
from pyMC.tools import rotate_dm

def get_gks_dm_guess_mo_4c_tot(molu, natom_tot, mo_coeff_in, theta_dict):
    """Rotate the whole 4c mo_coeff, only in real space. Defaults to

    Args:
        mol (gto.mole object): Whole molecule.
        mo_coeff_in (numpy.array): 4c mo_coeff
        theta_dict (rotation info in ): (tuple): tuple of (nx, theta)
            nx(numpy.array): is the rotation axis.
            theta(float): rotation angle in the spin space (valued from 0 to 4pi)

    Returns:
        mo_coeff_f: rotated mo_coeff
    """

    mo_coeff_f = rotate_mo_coeff_direct_4c_tot(molu, natom_tot, mo_coeff_in,theta_dict)
    
        
    return mo_coeff_f


def rotate_mo_coeff_direct_4c_tot(mol, natom, mo_coeff_in, theta_dict):
    """Rotate the orbital for 4-component Dirac calculations.
       If 2c calculations will be implemented, only small changes will be done.
       Rotation is done in two parts, because of the J = L + S = L + 1/2*Sigma for 4-c orbital, or
       J = L + 1/2*sigma for spinor. L and s are communative, so rotation of real space and rotation of
       spin space can be seperated.
       
       ! NOTE: this subroutine ONLY rotates real space!
       
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
    
    nx, theta = theta_dict
    
    nao2c_atm = mo_coeff_in.shape[0]//2//natom
    nao = nao2c_atm//2
    # get the U_sph2spinor_part, Dpart and U_spin_part matrix.
    # * Note that: part means only get the [natom*nao2c,natom*nao2c] for LL part
    U_sph2spinor_part = rotate_dm.cal_sph2spinor_matrix(mol, natom)
    U_spin_part = rotate_dm.cal_U_direct(mol, nao, natom, (numpy.array([nx]*natom), numpy.array([theta]*natom)))
    import pdb
    pdb.set_trace()
    # get the full matrix for L and S
    U_sph2spinor = numpy.array(scipy.linalg.block_diag(U_sph2spinor_part,U_sph2spinor_part))
    U_spin = numpy.array(scipy.linalg.block_diag(U_spin_part,U_spin_part))
    
    Cf = U_sph2spinor.conj().T@U_spin@U_sph2spinor@mo_coeff_in
    
    return Cf


def rotate_mo_coeff_direct_4c_tot_direct(molS, mo_coeff_in, theta_dict):
    """Rotate the orbital for 4-component Dirac calculations.
       If 2c calculations will be implemented, only small changes will be done.
       Rotation is done in two parts, because of the J = L + S = L + 1/2*Sigma for 4-c orbital, or
       J = L + 1/2*sigma for spinor. L and s are communative, so rotation of real space and rotation of
       spin space can be seperated.
       
       ! NOTE: this subroutine ONLY rotates real space!
       
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
    
    nx, theta = theta_dict
    normnx = numpy.linalg.norm(nx)
    costheta = numpy.cos(theta*0.5*normnx)
    sintheta = numpy.sin(theta*0.5*normnx)
    Afactor = costheta - nx[2]*1.0j*sintheta/normnx
    Bfactor = (-nx[0]*1.0j - nx[1])*sintheta/normnx
    Cfactor = (-nx[0]*1.0j + nx[1])*sintheta/normnx
    Dfactor = costheta + nx[2]*1.0j*sintheta/normnx
    
    Ua, Ub = molS.sph2spinor_coeff()
    Wa = Afactor*Ua + Bfactor*Ub
    Wb = Cfactor*Ua + Dfactor*Ub
    W = numpy.vstack((Wa,Wb))
    U = numpy.vstack((Ua,Ub))
    Wf = numpy.linalg.inv(U)@W
    
    import pdb
    pdb.set_trace()

    Rm = numpy.array(scipy.linalg.block_diag(Wf,Wf))
    
    Cf = Rm@mo_coeff_in
    
    return Cf
