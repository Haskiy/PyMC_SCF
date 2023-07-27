#!/usr/bin/env python
r'''
Author: Pu Zhichen
Date: 2022-04-09 13:52:15
LastEditTime: 2022-04-11 14:48:46
LastEditors: Pu Zhichen
Description: 
    An example test to compute:
        1. GKSLC LDA, no numerical stable scheme
        2. GKSLC LDA, numerical stable scheme
    ALL WITH DOUBLE GROUP SYMMETRY
FilePath: \pyMC\tests\test_input\mol_scf\Li3_ncstate_lc_nosymm.py

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
xc = 'SVWN'
IBP = False
output_dict = {}

BENCHMARK_etot = {
    0: -22.06662078753173,
    1: -22.06662078753143
}

BENCHMARK_motot = {
    0: numpy.array([-1.865961840440106, -1.865933058651073, -1.86593305864201 ,
       -1.862704135954012, -1.862679214423746]),
    1: numpy.array([-1.865961836302225, -1.865933054503088, -1.865933054496376,
       -1.862704131111516, -1.862679209614089])
}

FLAG = 'Positive'
EPSILON = 1.0E-8


# ! NOTE: Create calculated clusters. 
mol = gto.Mole()
mol.atom = molcoords 
mol.spin=9 # ^ Spin
mol.basis = "cc-pvtz" # ^ Basis
mol.max_memory = 50000 
mol.build()

mo_coeffu = mf.mo_coeff 
noccu = mf.mo_occ 
nao = mo_coeffu.shape[-1] 
natom = mol.natm 
theta_dict = tools_hss.rotate_dm.get_init_guess_theta(mol, rotatez_negative = False,vortex = False) 

mo_coeffu = tools_hss.rotate_dm.get_gks_dm_guess_mo(molu, mo_coeffu, natom
            ,numpy.array([theta_dict[i][0] for i in range(3)]), rotatem = True, rotatel =  True)
nocc = numpy.array((noccu[0].tolist()*natom+noccu[1].tolist()*natom))

"""
Using the Gauss-Legendre quadrature for symmetry reason!
"""
dft.Grids.gen_atomic_grids = grids_util.Grids_hss.gen_atomic_grids_gauss_legendre

for idxT,THRESHOLD in enumerate(THRESHOLD_list):
    mftot = gksmc.GKSLC(mol)
    mftot.xc = xc
    mftot.ibp = IBP
    mftot.grids.atom_grid = {"Li" : (50, 30, 24)} 
    # ^ (A, B, C) : A--> radial grid; B--> theta; C--> phi
    mftot.LIBXCT_factor = THRESHOLD[0]
    mftot.max_cycle = 50
    dm = mftot.make_rdm1(mo_coeffu, nocc)
    mftot.kernel(dm)
    
    if numpy.abs(BENCHMARK_etot[idxT] - mftot.e_tot) > EPSILON:
        FLAG = 'Negative'
        print(FLAG)
        break
    else:
        print(numpy.abs(BENCHMARK_etot[idxT] - mftot.e_tot))
    for mo_idx, moi in enumerate(BENCHMARK_motot[idxT]):
        if numpy.abs(moi - mftot.mo_energy[mo_idx]) > EPSILON:
            FLAG = 'Negative'
            print(FLAG)
            break
        else:
            print(numpy.abs(moi - mftot.mo_energy[mo_idx]))
