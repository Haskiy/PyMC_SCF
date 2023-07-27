#!/usr/bin/env python
r'''
Author: Pu Zhichen
Date: 2022-04-09 14:45:55
LastEditTime: 2022-04-11 14:48:53
LastEditors: Pu Zhichen
Description: 
    An example test to compute:
        1. GKSMC LDA, no numerical stable scheme
        2. GKSMC GGA, without IBP, no numerical stable scheme 
    All tests are in 4-c Dirac formalism. And the vortex pattern is used.
FilePath: \undefinedd:\PKU_msi\pyMC\tests\test_input\mol_scf\Li3_r_vortex_symm.py

 May the force be with you!
'''

import os
PATH_pyMC = os.path.abspath(os.path.dirname(__file__)).split('/pyMC')[0]

from pyscf import gto, scf, dft
import sys
import numpy
# ! NOTE: should add the path of the pyMC module.
sys.path.append(PATH_pyMC)
from pyMC import gksmc, mole_sym, grids_util
from pyMC import tools as tools_hss

molu = gto.Mole()
molu.atom = "Li 0 0 0" ###
molu.spin=1 # ^ Spin
molu.basis = "cc-pvdz" # ^ Basis
molu.symmetry=False ###
molu.max_memory = 50000
molu.build()

# ! NOTE: Single atom SCF calculation
mf = scf.dhf.DHF(molu)
mf.kernel()

molcoords = """
Li          2.309401076758503     0.000000000000000     0.000000000000000    ;
Li          -1.154700538379252    2.000000000000000     0.000000000000000    ;
Li          -1.154700538379252    -2.000000000000000    0.000000000000000    ;
"""

Ndirect = 1454
functional_list = ['SVWN','PBE']

BENCHMARK_etot = {
    'SVWN': -22.063408582513777,
    'PBE': -22.414782999671115
}

BENCHMARK_motot = {
    'SVWN': numpy.array([-1.852168654415418, -1.852125881645259, -1.852125881659543,
       -1.848519565066701, -1.848475319853977]),
    'PBE': numpy.array([-1.894337409017328, -1.894311122304204, -1.894311122313593,
       -1.890035910244095, -1.890007395699115])
}

FLAG = 'Positive'
EPSILON = 1.0E-8


molS = gto.Mole()
molS.atom = molcoords
molS.spin=3 # ^ Spin
molS.basis = "cc-pvdz" # ^ Basis
molS.max_memory = 50000
molS.build()
mfs = scf.dhf.DHF(molS)
mfs.max_cycle = 0
mfs.kernel()
S = mfs.get_ovlp()

# ! NOTE: Create calculated clusters. 
mol = mole_sym.Mole_sym() 
mol.Dsymmetry = True
mol.ovlp = S
mol.atom = molcoords 
mol.spin=3 # ^ Spin
mol.basis = "cc-pvdz" # ^ Basis
mol.max_memory = 50000 
mol.vortex = True 
mol.dirac4c = True 
mol.build(singleatom = molu) 

mo_coeffu = mf.mo_coeff
noccu = mf.mo_occ
nao = mo_coeffu.shape[-1] 
dm = mf.make_rdm1(mo_coeffu, noccu)
"""
Rotate the magnetization to z-axis
only used in 4c calculations
"""
mo_coeffu = tools_hss.rotate_utils2.get_z_oriented_atom(molu, mo_coeffu, dm) 
natom = mol.natm 
"""
rotatez_negative: whether rotate the magnetization to its opposite direction!
vortex: whether use vortex like toroidal structure or just the D3 initial guess.

theta_dict: contains all the rotation informations
    (list): [
                [numpy.array([alpha,beta,gamma]), numpy.array([nx,ny,nz]), theta]*natom
            ]
    where nx,ny,nz is the rotation vector, theta is the rotating degree.
"""
theta_dict = tools_hss.rotate_dm.get_init_guess_theta(mol, rotatez_negative = False,vortex = True) 
"""
Generating the symmetried initial guess molecule co_efficients.
"""
mo_coeffu = tools_hss.rotate_dm.get_gks_dm_guess_mo_4c(molu, mo_coeffu, natom, theta_dict)
nocc = numpy.array((noccu[:molu.nao_2c()].tolist())*natom+(noccu[molu.nao_2c():].tolist())*natom)

"""
Using the Gauss-Legendre quadrature for symmetry reason!
"""
dft.Grids.gen_atomic_grids = grids_util.Grids_hss.gen_atomic_grids_gauss_legendre

for idx, xc in enumerate(functional_list):
    mftot = gksmc.GKSMC_r(mol)
    mftot.Ndirect=Ndirect
    mftot.xc = xc # ^ functional
    mftot.grids.atom_grid = {"Li" : (40, 30, 24)} 
    # ^ (A, B, C) : A--> radial grid; B--> theta; C--> phi
    dm_tot  = mftot.make_rdm1(mo_coeffu, nocc)
    mftot.level_shift=0.8
    mftot.conv_tol = 1e-04 # ^ tolerance
    mftot.max_cycle = 40 # ^ max cycle
    mftot.kernel(dm_tot)
    
    if numpy.abs(BENCHMARK_etot[xc] - mftot.e_tot) > EPSILON:
        FLAG = 'Negative'
        print(FLAG)
        break
    else:
        print(numpy.abs(BENCHMARK_etot[xc] - mftot.e_tot))
    for mo_idx, moi in enumerate(BENCHMARK_motot[xc]):
        if numpy.abs(moi - mftot.mo_energy[84 + mo_idx]) > EPSILON:
            FLAG = 'Negative'
            print(FLAG)
            break
        else:
            print(numpy.abs(moi - mftot.mo_energy[84 + mo_idx]))