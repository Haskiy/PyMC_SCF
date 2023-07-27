#!/usr/bin/env python
r'''
Author: Pu Zhichen
Date: 2022-04-09 12:05:24
LastEditTime: 2022-04-11 14:47:36
LastEditors: Pu Zhichen
Description: 
    An example test to compute:
        1. GKSMC LDA, no numerical stable scheme
        2. GKSMC GGA, without IBP, no numerical stable scheme
        3. GKSMC GGA, with IBP, no numerical stable scheme
        4. GKSMC LDA, numerical stable scheme
        5. GKSMC GGA, without IBP, numerical stable scheme
        6. GKSMC GGA, with IBP, numerical stable scheme
    ALL WITH DOUBLE GROUP SYMMETRY
FilePath: \pyMC\tests\test_input\mol_scf\Li3_symm.py

 May the force be with you!
'''

import os
PATH_pyMC = os.path.abspath(os.path.dirname(__file__)).split('/pyMC')[0]

from pyscf import gto, lib, scf, dft
import sys
import numpy
# ! NOTE: should add the path of the pyMC module.
sys.path.append(PATH_pyMC)
from pyMC import gksmc, mole_sym, grids_util
from pyMC import tools as tools_hss
lib.misc.num_threads(n=40)

molu = gto.Mole()
molu.atom = "Li 0 0 0" ###
molu.spin=1 # ^ Spin
molu.basis = "cc-pvtz" # ^ Basis
molu.symmetry=False ###
molu.max_memory = 50000
molu.build()

# ! NOTE: Single atom SCF calculation
mf = scf.UHF(molu)
mf.kernel()

molcoords = """
Li          2.309401076758503     0.000000000000000     0.000000000000000    ;
Li          -1.154700538379252    2.000000000000000     0.000000000000000    ;
Li          -1.154700538379252    -2.000000000000000    0.000000000000000    ;
"""

THRESHOLD_list = [(-1.0, 10.0), (1.0E-10, 0.999)]
Ndirect = 1454
functional_list = ['SVWN','PBE','PBE']
IBP_list = [False, False, True]
output_dict = {}

BENCHMARK = [
    # No threshold
    # Threshold
]

# ! NOTE: Create pre-calculated molecule to generate S matrix.
molS = gto.Mole()
molS.atom = molcoords
molS.spin=9 # ^ Spin
molS.basis = "cc-pvtz" # ^ Basis
molS.max_memory = 50000
molS.build()
mfs = scf.UHF(molS)
mfs.max_cycle = 0
mfs.kernel()
S = mfs.get_ovlp()

# ! NOTE: Create calculated clusters. 
mol = mole_sym.Mole_sym() 
mol.Dsymmetry = True
mol.ovlp = S
mol.atom = molcoords 
mol.spin=9 # ^ Spin
mol.basis = "cc-pvtz" # ^ Basis
mol.max_memory = 50000 
mol.build(singleatom = molu)

mo_coeffu = mf.mo_coeff 
noccu = mf.mo_occ 
nao = mo_coeffu.shape[-1] 
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
theta_dict = tools_hss.rotate_dm.get_init_guess_theta(mol, rotatez_negative = False,vortex = False) 
"""
Generating the symmetried initial guess molecule co_efficients.
rotatem: whether rotation spin part.
rotatel: whether real space part.
"""
mo_coeffu = tools_hss.rotate_dm.get_gks_dm_guess_mo(molu, mo_coeffu, natom
            ,numpy.array([theta_dict[i][0] for i in range(3)]), rotatem = True, rotatel =  True)
nocc = numpy.array((noccu[0].tolist()*natom+noccu[1].tolist()*natom))

"""
Using the Gauss-Legendre quadrature for symmetry reason!
"""
dft.Grids.gen_atomic_grids = grids_util.Grids_hss.gen_atomic_grids_gauss_legendre


BENCHMARK_etot = {
    0:{
        0: -22.06662078753168,
        1: -22.41426934226565,
        2: -22.414269332396135
        },
    1:{
        0: -22.066620788380945,
        1: -22.41426934285122,
        2: -22.414269332981895
        },
}

BENCHMARK_motot = {
    0:{
        0: numpy.array([-1.865961840444991, -1.865933058651423, -1.865933058651423,
            -1.862704135952426, -1.86267921441756 ]),
        1: numpy.array([-1.892848364048397, -1.89281453401601 , -1.89281453401601 ,
            -1.889059661097997, -1.889027579776835]),
        2: numpy.array([-1.892852220943599, -1.892818383532987, -1.892818383532987,
            -1.889065113483104, -1.889033025114055])
        },
    1:{
        0: numpy.array([-1.865961845791655, -1.865933063997798, -1.865933063997798,
            -1.862704141369969, -1.862679219835579]),
        1: numpy.array([-1.892848366701148, -1.892814536669628, -1.892814536669628,
            -1.88905966389072 , -1.889027582564587]),
        2: numpy.array([-1.892852223601541, -1.892818386191799, -1.892818386191799,
            -1.88906511628528 , -1.889033027911274])
        },
}

FLAG = 'Positive'
EPSILON = 1.0E-8

for idxT,THRESHOLD in enumerate(THRESHOLD_list):
    for idx, xc in enumerate(functional_list):
        mftot = gksmc.GKSMC(mol)
        mftot.xc = xc
        mftot.ibp = IBP_list[idx]
        mftot.grids.atom_grid = {"Li" : (50, 30, 24)} 
        # ^ (A, B, C) : A--> radial grid; B--> theta; C--> phi
        mftot.LIBXCT_factor = THRESHOLD[0]
        mftot.MSL_factor = THRESHOLD[1]
        mftot.Ndirect = Ndirect
        mftot.max_cycle = 50
        mftot.conv_tol = 1e-08
        dm = mftot.make_rdm1(mo_coeffu, nocc)
        mftot.kernel(dm)
        
        if numpy.abs(BENCHMARK_etot[idxT][idx] - mftot.e_tot) > EPSILON:
            FLAG = 'Negative'
            print(FLAG)
            break
        else:
            print(numpy.abs(BENCHMARK_etot[idxT][idx] - mftot.e_tot))
        for mo_idx, moi in enumerate(BENCHMARK_motot[idxT][idx]):
            if numpy.abs(moi - mftot.mo_energy[mo_idx]) > EPSILON:
                FLAG = 'Negative'
                print(FLAG)
                break
            else:
                print(numpy.abs(moi - mftot.mo_energy[mo_idx]))