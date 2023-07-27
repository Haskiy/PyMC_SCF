#!/usr/bin/env python
r'''
Author: Pu Zhichen
Date: 2022-03-07 19:18:45
LastEditTime: 2022-03-07 19:18:48
LastEditors: Pu Zhichen
Description: 
FilePath: \pyMC\tools\rotate_dm_all.py

 May the force be with you!
'''

import numpy
from pyMC.tools import Dmatrix
import scipy.linalg
from pyMC.tools import rotate_dm
from pyscf import gto
from scipy.spatial.transform import Rotation as R

def rotate_dm_as_one(mol_old, c2, mo_coeff_tuple, spin_list, rotatel = True, rotatem = True):
    """rotate a whole molecule as one molecule.

    Args:
        mol_old (gto.Mole object): the molecule before the rotation.
        c2 (numpy.array): (3,3) the coordinate system after rotation.
        mo_coeff_tuple (tuple): (mo_coeffu, mo_coeffd) is the return of uks.mo_coeff 
        spin_list (list): a list of spins for each atom.
        rotatel (bool, optional): Whether to rotate real space. Defaults to True.
        rotatem (bool, optional): Whether to rotate spin space. Defaults to True.

    Returns:
        _type_: _description_
    """
    # ~
    mo_coeffaa, mo_coeffbb = mo_coeff_tuple
    nao = mo_coeffaa.shape[-1]
    
    # ~ 
    c1 = numpy.eye(3)
    alpha, beta, gamma = Dmatrix.get_euler_angles(c1, c2)
    # ~
    r = R.from_euler('ZYZ', numpy.array([alpha, beta, gamma]))
    vecf = r.as_rotvec()
    thetaf = numpy.linalg.norm(vecf)
    # ~
    rotate_infos = [numpy.array([alpha, beta, gamma]), 
                    vecf/thetaf,
                    thetaf]
    # ~
    mol_atoms_list = [gto.M(
                        verbose = 3,
                        atom = mol_old._atom[iatm][0]   + ' 0.0 0.0 0.0 ',
                        basis = mol_old.basis,
                        spin = spin_list[iatm])
                        for iatm in range(mol_old.natm)
                      ]
    
    # ~
    mo_coeff = numpy.asarray(scipy.linalg.block_diag(mo_coeffaa,mo_coeffbb), dtype=numpy.complex128)
    
    # ~
    D_list = [rotate_dm.cal_D(mol_atoms_list[iatm], 
                    mol_atoms_list[iatm].nao,
                    1,
                    ([rotate_infos[0]]))[0]
              for iatm in range(len(mol_atoms_list))]
    
    U_list = [rotate_dm.cal_U_direct(mol_atoms_list[iatm], 
                    mol_atoms_list[iatm].nao,
                    1,
                    (rotate_dm.euler_to_rotvec(numpy.array([rotate_infos[0]])))) 
              for iatm in range(len(mol_atoms_list))]
    
    # ~
    Daa = D_list[0][:mol_atoms_list[0].nao, :mol_atoms_list[0].nao]
    Dab = D_list[0][:mol_atoms_list[0].nao, mol_atoms_list[0].nao:]
    Dba = D_list[0][mol_atoms_list[0].nao:, :mol_atoms_list[0].nao]
    Dbb = D_list[0][mol_atoms_list[0].nao:, mol_atoms_list[0].nao:]
    Uaa = U_list[0][:mol_atoms_list[0].nao, :mol_atoms_list[0].nao]
    Uab = U_list[0][:mol_atoms_list[0].nao, mol_atoms_list[0].nao:]
    Uba = U_list[0][mol_atoms_list[0].nao:, :mol_atoms_list[0].nao]
    Ubb = U_list[0][mol_atoms_list[0].nao:, mol_atoms_list[0].nao:]
    for iatm in range(1, len(mol_atoms_list)):
        nao_iatm = mol_atoms_list[iatm].nao
        Daa1 = D_list[iatm][:nao_iatm, :nao_iatm]
        Dab1 = D_list[iatm][:nao_iatm, nao_iatm:]
        Dba1 = D_list[iatm][nao_iatm:, :nao_iatm]
        Dbb1 = D_list[iatm][nao_iatm:, nao_iatm:]
        Uaa1 = U_list[iatm][:nao_iatm, :nao_iatm]
        Uab1 = U_list[iatm][:nao_iatm, nao_iatm:]
        Uba1 = U_list[iatm][nao_iatm:, :nao_iatm]
        Ubb1 = U_list[iatm][nao_iatm:, nao_iatm:]
        
        Daa = scipy.linalg.block_diag(Daa, Daa1)
        Dab = scipy.linalg.block_diag(Dab, Dab1)
        Dba = scipy.linalg.block_diag(Dba, Dba1)
        Dbb = scipy.linalg.block_diag(Dbb, Dbb1)
        Uaa = scipy.linalg.block_diag(Uaa, Uaa1)
        Uab = scipy.linalg.block_diag(Uab, Uab1)
        Uba = scipy.linalg.block_diag(Uba, Uba1)
        Ubb = scipy.linalg.block_diag(Ubb, Ubb1)
        
    D = numpy.block([
        [Daa, Dab],
        [Dba, Dbb]
    ])
    
    U = numpy.block([
        [Uaa, Uab],
        [Uba, Ubb]
    ])
    
    if rotatel and rotatem:
        return U@D@mo_coeff
    else:
        if rotatel:
            return D@mo_coeff
        elif rotatem:
            return U@mo_coeff
    
    
    
def rotate_dm_gks(mol_old, c2, mo_coeff, spin_list, rotatel = True, rotatem = True):
    """rotate a whole molecule as one molecule.

    Args:
        mol_old (gto.Mole object): the molecule before the rotation.
        c2 (numpy.array): (3,3) the coordinate system after rotation.
        mo_coeff_tuple (tuple): (mo_coeffu, mo_coeffd) is the return of uks.mo_coeff 
        spin_list (list): a list of spins for each atom.
        rotatel (bool, optional): Whether to rotate real space. Defaults to True.
        rotatem (bool, optional): Whether to rotate spin space. Defaults to True.

    Returns:
        _type_: _description_
    """
    # ~
    nao = mo_coeff.shape[-1]//2
    
    # ~ 
    c1 = numpy.eye(3)
    alpha, beta, gamma = Dmatrix.get_euler_angles(c1, c2)
    # ~
    r = R.from_euler('ZYZ', numpy.array([alpha, beta, gamma]))
    vecf = r.as_rotvec()
    thetaf = numpy.linalg.norm(vecf)
    # ~
    rotate_infos = [numpy.array([alpha, beta, gamma]), 
                    vecf/thetaf,
                    thetaf]
    # ~
    mol_atoms_list = [gto.M(
                        verbose = 3,
                        atom = mol_old._atom[iatm][0]   + ' 0.0 0.0 0.0 ',
                        basis = mol_old.basis,
                        spin = spin_list[iatm])
                        for iatm in range(mol_old.natm)
                      ]

    # ~
    D_list = [rotate_dm.cal_D(mol_atoms_list[iatm], 
                    mol_atoms_list[iatm].nao,
                    1,
                    ([rotate_infos[0]]))[0]
              for iatm in range(len(mol_atoms_list))]
    
    U_list = [rotate_dm.cal_U_direct(mol_atoms_list[iatm], 
                    mol_atoms_list[iatm].nao,
                    1,
                    (rotate_dm.euler_to_rotvec(numpy.array([rotate_infos[0]])))) 
              for iatm in range(len(mol_atoms_list))]
    
    # ~
    Daa = D_list[0][:mol_atoms_list[0].nao, :mol_atoms_list[0].nao]
    Dab = D_list[0][:mol_atoms_list[0].nao, mol_atoms_list[0].nao:]
    Dba = D_list[0][mol_atoms_list[0].nao:, :mol_atoms_list[0].nao]
    Dbb = D_list[0][mol_atoms_list[0].nao:, mol_atoms_list[0].nao:]
    Uaa = U_list[0][:mol_atoms_list[0].nao, :mol_atoms_list[0].nao]
    Uab = U_list[0][:mol_atoms_list[0].nao, mol_atoms_list[0].nao:]
    Uba = U_list[0][mol_atoms_list[0].nao:, :mol_atoms_list[0].nao]
    Ubb = U_list[0][mol_atoms_list[0].nao:, mol_atoms_list[0].nao:]
    for iatm in range(1, len(mol_atoms_list)):
        nao_iatm = mol_atoms_list[iatm].nao
        Daa1 = D_list[iatm][:nao_iatm, :nao_iatm]
        Dab1 = D_list[iatm][:nao_iatm, nao_iatm:]
        Dba1 = D_list[iatm][nao_iatm:, :nao_iatm]
        Dbb1 = D_list[iatm][nao_iatm:, nao_iatm:]
        Uaa1 = U_list[iatm][:nao_iatm, :nao_iatm]
        Uab1 = U_list[iatm][:nao_iatm, nao_iatm:]
        Uba1 = U_list[iatm][nao_iatm:, :nao_iatm]
        Ubb1 = U_list[iatm][nao_iatm:, nao_iatm:]
        
        Daa = scipy.linalg.block_diag(Daa, Daa1)
        Dab = scipy.linalg.block_diag(Dab, Dab1)
        Dba = scipy.linalg.block_diag(Dba, Dba1)
        Dbb = scipy.linalg.block_diag(Dbb, Dbb1)
        Uaa = scipy.linalg.block_diag(Uaa, Uaa1)
        Uab = scipy.linalg.block_diag(Uab, Uab1)
        Uba = scipy.linalg.block_diag(Uba, Uba1)
        Ubb = scipy.linalg.block_diag(Ubb, Ubb1)
        
    D = numpy.block([
        [Daa, Dab],
        [Dba, Dbb]
    ])
    
    U = numpy.block([
        [Uaa, Uab],
        [Uba, Ubb]
    ])
    # import pdb
    # pdb.set_trace()
    
    if rotatel and rotatem:
        return U@D@mo_coeff
    else:
        if rotatel:
            return D@mo_coeff
        elif rotatem:
            return U@mo_coeff

def rotate_dm_gks_SSW(mol_old, c2, mo_coeff, spin_list, rotatel = True, rotatem = True):
    """rotate a whole molecule as one molecule.

    Args:
        mol_old (gto.Mole object): the molecule before the rotation.
        c2 (numpy.array): (3,3) the coordinate system after rotation.
        mo_coeff_tuple (tuple): (mo_coeffu, mo_coeffd) is the return of uks.mo_coeff 
        spin_list (list): a list of spins for each atom.
        rotatel (bool, optional): Whether to rotate real space. Defaults to True.
        rotatem (bool, optional): Whether to rotate spin space. Defaults to True.

    Returns:
        _type_: _description_
    """
    # ~
    nao = mo_coeff.shape[-1]//2
    
    # ~ 
    c1 = numpy.eye(3)
    alpha, beta, gamma = Dmatrix.get_euler_angles(c1, c2)
    # ~
    r = R.from_euler('ZYZ', numpy.array([alpha, beta, gamma]))
    vecf = r.as_rotvec()
    thetaf = numpy.linalg.norm(vecf)
    # ~
    rotate_infos = [numpy.array([alpha, beta, gamma]), 
                    vecf/thetaf,
                    thetaf]
    # ~
    mol_atoms_list = [gto.M(
                        verbose = 3,
                        atom = mol_old._atom[iatm][0]   + ' 0.0 0.0 0.0 ',
                        basis = mol_old.basis,
                        spin = spin_list[iatm])
                        for iatm in range(mol_old.natm)
                      ]

    # ~
    D_list = [rotate_dm.cal_D(mol_atoms_list[iatm], 
                    mol_atoms_list[iatm].nao,
                    1,
                    ([rotate_infos[0]]))[0]
              for iatm in range(len(mol_atoms_list))]
    
    U_list = [rotate_dm.cal_U_direct(mol_atoms_list[iatm], 
                    mol_atoms_list[iatm].nao,
                    1,
                    (rotate_dm.euler_to_rotvec(numpy.array([rotate_infos[0]])))) 
              for iatm in range(len(mol_atoms_list))]
    
    # ~
    Daa = D_list[0][:mol_atoms_list[0].nao, :mol_atoms_list[0].nao]
    Dab = D_list[0][:mol_atoms_list[0].nao, mol_atoms_list[0].nao:]
    Dba = D_list[0][mol_atoms_list[0].nao:, :mol_atoms_list[0].nao]
    Dbb = D_list[0][mol_atoms_list[0].nao:, mol_atoms_list[0].nao:]
    Uaa = U_list[0][:mol_atoms_list[0].nao, :mol_atoms_list[0].nao]
    Uab = U_list[0][:mol_atoms_list[0].nao, mol_atoms_list[0].nao:]
    Uba = U_list[0][mol_atoms_list[0].nao:, :mol_atoms_list[0].nao]
    Ubb = U_list[0][mol_atoms_list[0].nao:, mol_atoms_list[0].nao:]
    for iatm in range(1, len(mol_atoms_list)):
        nao_iatm = mol_atoms_list[iatm].nao
        Daa1 = D_list[iatm][:nao_iatm, :nao_iatm]
        Dab1 = D_list[iatm][:nao_iatm, nao_iatm:]
        Dba1 = D_list[iatm][nao_iatm:, :nao_iatm]
        Dbb1 = D_list[iatm][nao_iatm:, nao_iatm:]
        Uaa1 = U_list[iatm][:nao_iatm, :nao_iatm]
        Uab1 = U_list[iatm][:nao_iatm, nao_iatm:]
        Uba1 = U_list[iatm][nao_iatm:, :nao_iatm]
        Ubb1 = U_list[iatm][nao_iatm:, nao_iatm:]
        
        Daa = scipy.linalg.block_diag(Daa, Daa1)
        Dab = scipy.linalg.block_diag(Dab, Dab1)
        Dba = scipy.linalg.block_diag(Dba, Dba1)
        Dbb = scipy.linalg.block_diag(Dbb, Dbb1)
        Uaa = scipy.linalg.block_diag(Uaa, Uaa1)
        Uab = scipy.linalg.block_diag(Uab, Uab1)
        Uba = scipy.linalg.block_diag(Uba, Uba1)
        Ubb = scipy.linalg.block_diag(Ubb, Ubb1)
        
    D = numpy.block([
        [Daa, Dab],
        [Dba, Dbb]
    ])
    
    U = numpy.block([
        [Uaa, Uab],
        [Uba, Ubb]
    ])
    import pdb
    pdb.set_trace()
    
    if rotatel and rotatem:
        return U@D@mo_coeff
    else:
        if rotatel:
            return D@mo_coeff
        elif rotatem:
            return U@mo_coeff
  
# def rotate_dm_gks(mol_old, c2, mo_coeff, spin_list, rotatel = True, rotatem = True):
#     """rotate a whole molecule as one molecule.

#     Args:
#         mol_old (gto.Mole object): the molecule before the rotation.
#         c2 (numpy.array): (3,3) the coordinate system after rotation.
#         mo_coeff_tuple (tuple): (mo_coeffu, mo_coeffd) is the return of uks.mo_coeff 
#         spin_list (list): a list of spins for each atom.
#         rotatel (bool, optional): Whether to rotate real space. Defaults to True.
#         rotatem (bool, optional): Whether to rotate spin space. Defaults to True.

#     Returns:
#         _type_: _description_
#     """
#     # ~
#     # import pdb
#     # pdb.set_trace()
#     nao = mo_coeff.shape[-1]//2
    
#     # ~ 
#     c1 = numpy.eye(3)
#     alpha, beta, gamma = Dmatrix.get_euler_angles(c1, c2)
#     # ~
#     r = R.from_euler('ZYZ', numpy.array([alpha, beta, gamma]))
#     vecf = r.as_rotvec()
#     thetaf = numpy.linalg.norm(vecf)
#     # ~
#     rotate_infos = [numpy.array([alpha, beta, gamma]), 
#                     vecf/thetaf,
#                     thetaf]
#     # ~
#     mol_atoms_list = [gto.M(
#                         verbose = 3,
#                         atom = mol_old._atom[iatm][0]   + ' 0.0 0.0 0.0 ',
#                         basis = mol_old.basis,
#                         spin = spin_list[iatm])
#                         for iatm in range(mol_old.natm)
#                       ]
#     # D_list = [rotate_dm.cal_D(mol_atoms_list[iatm], 
#     #                 mol_atoms_list[iatm].nao,
#     #                 1,
#     #                 ([rotate_infos[0]]))[0]
#     #           for iatm in range(len(mol_atoms_list))]
    
#     U_list = [rotate_dm.cal_U_direct(mol_atoms_list[iatm], 
#                     mol_atoms_list[iatm].nao,
#                     1,
#                     (rotate_dm.euler_to_rotvec(numpy.array([rotate_infos[0]])))) 
#               for iatm in range(len(mol_atoms_list))]
    
#     # Daa = D_list[0][:mol_atoms_list[0].nao, :mol_atoms_list[0].nao]
#     # Dab = D_list[0][:mol_atoms_list[0].nao, mol_atoms_list[0].nao:]
#     # Dba = D_list[0][mol_atoms_list[0].nao:, :mol_atoms_list[0].nao]
#     # Dbb = D_list[0][mol_atoms_list[0].nao:, mol_atoms_list[0].nao:]
    
#     Uaa = U_list[0][:mol_atoms_list[0].nao, :mol_atoms_list[0].nao]
#     Uab = U_list[0][:mol_atoms_list[0].nao, mol_atoms_list[0].nao:]
#     Uba = U_list[0][mol_atoms_list[0].nao:, :mol_atoms_list[0].nao]
#     Ubb = U_list[0][mol_atoms_list[0].nao:, mol_atoms_list[0].nao:]
#     for iatm in range(1, len(mol_atoms_list)):
#         nao_iatm = mol_atoms_list[iatm].nao
#         # Daa1 = D_list[iatm][:nao_iatm, :nao_iatm]
#         # Dab1 = D_list[iatm][:nao_iatm, nao_iatm:]
#         # Dba1 = D_list[iatm][nao_iatm:, :nao_iatm]
#         # Dbb1 = D_list[iatm][nao_iatm:, nao_iatm:]
#         Uaa1 = U_list[iatm][:nao_iatm, :nao_iatm]
#         Uab1 = U_list[iatm][:nao_iatm, nao_iatm:]
#         Uba1 = U_list[iatm][nao_iatm:, :nao_iatm]
#         Ubb1 = U_list[iatm][nao_iatm:, nao_iatm:]
        
#         # Daa = scipy.linalg.block_diag(Daa, Daa1)
#         # Dab = scipy.linalg.block_diag(Dab, Dab1)
#         # Dba = scipy.linalg.block_diag(Dba, Dba1)
#         # Dbb = scipy.linalg.block_diag(Dbb, Dbb1)
#         Uaa = scipy.linalg.block_diag(Uaa, Uaa1)
#         Uab = scipy.linalg.block_diag(Uab, Uab1)
#         Uba = scipy.linalg.block_diag(Uba, Uba1)
#         Ubb = scipy.linalg.block_diag(Ubb, Ubb1)
        
#     # D = numpy.block([
#     #     [Daa, Dab],
#     #     [Dba, Dbb]
#     # ])
    
#     U = numpy.block([
#         [Uaa, Uab],
#         [Uba, Ubb]
#     ])
#     # import pdb
#     # pdb.set_trace()
    
#     if rotatel and rotatem:
#         return U@D@mo_coeff
#     else:
#         if rotatel:
#             return D@mo_coeff
#         elif rotatem:
#             return U@mo_coeff    
    
def rotate_dm_uks(mol_old, c2, mo_coeff_tuple, spin_list, rotatel = True, rotatem = True):
    """rotate a whole molecule as one molecule.

    Args:
        mol_old (gto.Mole object): the molecule before the rotation.
        c2 (numpy.array): (3,3) the coordinate system after rotation.
        mo_coeff_tuple (tuple): (mo_coeffu, mo_coeffd) is the return of uks.mo_coeff 
        spin_list (list): a list of spins for each atom.
        rotatel (bool, optional): Whether to rotate real space. Defaults to True.
        rotatem (bool, optional): Whether to rotate spin space. Defaults to True.

    Returns:
        _type_: _description_
    """
    # ~
    mo_coeffaa, mo_coeffbb = mo_coeff_tuple
    nao = mo_coeffaa.shape[-1]
    
    # ~ 
    c1 = numpy.eye(3)
    alpha, beta, gamma = Dmatrix.get_euler_angles(c1, c2)
    # ~
    r = R.from_euler('ZYZ', numpy.array([alpha, beta, gamma]))
    vecf = r.as_rotvec()
    thetaf = numpy.linalg.norm(vecf)
    # ~
    rotate_infos = [numpy.array([alpha, beta, gamma]), 
                    vecf/thetaf,
                    thetaf]
    # ~
    mol_atoms_list = [gto.M(
                        verbose = 3,
                        atom = mol_old._atom[iatm][0]   + ' 0.0 0.0 0.0 ',
                        basis = mol_old.basis,
                        spin = spin_list[iatm])
                        for iatm in range(mol_old.natm)
                      ]
    
    # ~
    mo_coeff = numpy.asarray(scipy.linalg.block_diag(mo_coeffaa,mo_coeffbb), dtype=numpy.complex128)
    
    # ~
    D_list = [rotate_dm.cal_D(mol_atoms_list[iatm], 
                    mol_atoms_list[iatm].nao,
                    1,
                    ([rotate_infos[0]]))[0]
              for iatm in range(len(mol_atoms_list))]
    
    U_list = [rotate_dm.cal_U_direct(mol_atoms_list[iatm], 
                    mol_atoms_list[iatm].nao,
                    1,
                    (rotate_dm.euler_to_rotvec(numpy.array([rotate_infos[0]])))) 
              for iatm in range(len(mol_atoms_list))]
    
    # ~
    Daa = D_list[0][:mol_atoms_list[0].nao, :mol_atoms_list[0].nao]
    Dab = D_list[0][:mol_atoms_list[0].nao, mol_atoms_list[0].nao:]
    Dba = D_list[0][mol_atoms_list[0].nao:, :mol_atoms_list[0].nao]
    Dbb = D_list[0][mol_atoms_list[0].nao:, mol_atoms_list[0].nao:]
    Uaa = U_list[0][:mol_atoms_list[0].nao, :mol_atoms_list[0].nao]
    Uab = U_list[0][:mol_atoms_list[0].nao, mol_atoms_list[0].nao:]
    Uba = U_list[0][mol_atoms_list[0].nao:, :mol_atoms_list[0].nao]
    Ubb = U_list[0][mol_atoms_list[0].nao:, mol_atoms_list[0].nao:]
    for iatm in range(1, len(mol_atoms_list)):
        nao_iatm = mol_atoms_list[iatm].nao
        Daa1 = D_list[iatm][:nao_iatm, :nao_iatm]
        Dab1 = D_list[iatm][:nao_iatm, nao_iatm:]
        Dba1 = D_list[iatm][nao_iatm:, :nao_iatm]
        Dbb1 = D_list[iatm][nao_iatm:, nao_iatm:]
        Uaa1 = U_list[iatm][:nao_iatm, :nao_iatm]
        Uab1 = U_list[iatm][:nao_iatm, nao_iatm:]
        Uba1 = U_list[iatm][nao_iatm:, :nao_iatm]
        Ubb1 = U_list[iatm][nao_iatm:, nao_iatm:]
        
        Daa = scipy.linalg.block_diag(Daa, Daa1)
        Dab = scipy.linalg.block_diag(Dab, Dab1)
        Dba = scipy.linalg.block_diag(Dba, Dba1)
        Dbb = scipy.linalg.block_diag(Dbb, Dbb1)
        Uaa = scipy.linalg.block_diag(Uaa, Uaa1)
        Uab = scipy.linalg.block_diag(Uab, Uab1)
        Uba = scipy.linalg.block_diag(Uba, Uba1)
        Ubb = scipy.linalg.block_diag(Ubb, Ubb1)
        
    D = numpy.block([
        [Daa, Dab],
        [Dba, Dbb]
    ])
    
    U = numpy.block([
        [Uaa, Uab],
        [Uba, Ubb]
    ])
    
    return U@D@mo_coeff

def rotate_dm_uks_m(mol_old, c2, mo_coeff_tuple, spin_list):
    """rotate a whole molecule as one molecule.

    Args:
        mol_old (gto.Mole object): the molecule before the rotation.
        c2 (numpy.array): (3,3) the coordinate system after rotation.
        mo_coeff_tuple (tuple): (mo_coeffu, mo_coeffd) is the return of uks.mo_coeff 
        spin_list (list): a list of spins for each atom.
        rotatel (bool, optional): Whether to rotate real space. Defaults to True.
        rotatem (bool, optional): Whether to rotate spin space. Defaults to True.

    Returns:
        _type_: _description_
    """
    # ~
    mo_coeffaa, mo_coeffbb = mo_coeff_tuple
    
    # ~ 
    c1 = numpy.eye(3)
    alpha, beta, gamma = Dmatrix.get_euler_angles(c1, c2)
    # ~
    r = R.from_euler('ZYZ', numpy.array([alpha, beta, gamma]))
    vecf = r.as_rotvec()
    thetaf = numpy.linalg.norm(vecf)
    # ~
    rotate_infos = [numpy.array([alpha, beta, gamma]), 
                    vecf/thetaf,
                    thetaf]
    # ~
    mol_atoms_list = [gto.M(
                        verbose = 3,
                        atom = mol_old._atom[iatm][0]   + ' 0.0 0.0 0.0 ',
                        basis = mol_old.basis,
                        spin = spin_list[iatm])
                        for iatm in range(mol_old.natm)
                      ]
    
    # ~
    mo_coeff = numpy.asarray(scipy.linalg.block_diag(mo_coeffaa,mo_coeffbb), dtype=numpy.complex128)
    
    # ~
    
    U_list = [rotate_dm.cal_U_direct(mol_atoms_list[iatm], 
                    mol_atoms_list[iatm].nao,
                    1,
                    (rotate_dm.euler_to_rotvec(numpy.array([rotate_infos[0]])))) 
              for iatm in range(len(mol_atoms_list))]
    
    # ~
    Uaa = U_list[0][:mol_atoms_list[0].nao, :mol_atoms_list[0].nao]
    Uab = U_list[0][:mol_atoms_list[0].nao, mol_atoms_list[0].nao:]
    Uba = U_list[0][mol_atoms_list[0].nao:, :mol_atoms_list[0].nao]
    Ubb = U_list[0][mol_atoms_list[0].nao:, mol_atoms_list[0].nao:]
    for iatm in range(1, len(mol_atoms_list)):
        nao_iatm = mol_atoms_list[iatm].nao
        Uaa1 = U_list[iatm][:nao_iatm, :nao_iatm]
        Uab1 = U_list[iatm][:nao_iatm, nao_iatm:]
        Uba1 = U_list[iatm][nao_iatm:, :nao_iatm]
        Ubb1 = U_list[iatm][nao_iatm:, nao_iatm:]
        
        Uaa = scipy.linalg.block_diag(Uaa, Uaa1)
        Uab = scipy.linalg.block_diag(Uab, Uab1)
        Uba = scipy.linalg.block_diag(Uba, Uba1)
        Ubb = scipy.linalg.block_diag(Ubb, Ubb1)
        
    
    U = numpy.block([
        [Uaa, Uab],
        [Uba, Ubb]
    ])
    
    return U@mo_coeff
