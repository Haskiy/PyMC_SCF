#!/usr/bin/env python
r'''
Author: Pu Zhichen
Date: 2022-04-09 14:32:25
LastEditTime: 2022-04-11 13:49:16
LastEditors: Pu Zhichen
Description: 
    An example test to compute:
        1. GKSMC LDA, no numerical stable scheme
        2. GKSMC GGA, without IBP, no numerical stable scheme 
    All tests are in 4-c Dirac formalism.
FilePath: \pyMC\tests\test_input\mol_scf\Li_r_mc_no_symm.py

 May the force be with you!
'''

import os
import numpy
PATH_pyMC = os.path.abspath(os.path.dirname(__file__)).split('/pyMC')[0]

from pyscf import gto, lib
import sys
# ! NOTE: should add the path of the pyMC module.
sys.path.append(PATH_pyMC)
from pyMC import gksmc
lib.misc.num_threads(n=40)
import scipy

molcoords = """
 Li                0.00000000    0.00000000    0.00000000 ;
"""

ndirect = 1454
functional_list = ['SVWN', 'PBE']
output_dict = {}

BENCHMARK_etot = {
        'SVWN' : -7.3439729662789,
        'PBE' : -7.461748780952367
}

BENCHMARK_motot = {
     'SVWN' : numpy.array([-1.872043512635277, -1.86438635344432 , -0.116158172706127,
                -0.076647003702643, -0.048931330773349]),
     'PBE' : numpy.array([-1.897168696294109, -1.888753888404371, -0.11828453778253 ,
                -0.049172931885186, -0.049171172890383])
}

FLAG = 'Positive'
EPSILON = 1.0E-8

# ! NOTE: Create pre-calculated molecule to generate S matrix.
mol = gto.Mole()
mol.atom = molcoords
mol.spin=1 # ^ Spin
mol.charge = 0
mol.basis = "cc-pvtz" # ^ Basis
mol.max_memory = 50000
mol.build()

for XC in functional_list:
    mftot = gksmc.GKSMC_r(mol)
    mftot.xc = XC
    mftot.Ndirect = ndirect
#     mftot.max_cycle = 0
    mftot.kernel()
    
    if numpy.abs(BENCHMARK_etot[XC] - mftot.e_tot) > EPSILON:
        FLAG = 'Negative'
        print(FLAG)
        break
    else:
        print(numpy.abs(BENCHMARK_etot[XC] - mftot.e_tot))
    for mo_idx, moi in enumerate(BENCHMARK_motot[XC]):
        if numpy.abs(moi - mftot.mo_energy[60 + mo_idx]) > EPSILON:
            FLAG = 'Negative'
            print(FLAG)
            break
        else:
            print(numpy.abs(moi - mftot.mo_energy[60 + mo_idx]))
    