#!/usr/bin/env python
r'''
Author: Pu Zhichen
Date: 2022-04-09 09:42:20
LastEditTime: 2022-04-09 14:49:01
LastEditors: Pu Zhichen
Description: 
    An example test to compute:
        1. GKSMC LDA, no numerical stable scheme
        2. GKSMC GGA, without IBP, no numerical stable scheme
        3. GKSMC GGA, with IBP, no numerical stable scheme
        4. GKSMC LDA, numerical stable scheme
        5. GKSMC GGA, without IBP, numerical stable scheme
        6. GKSMC GGA, with IBP, numerical stable scheme
FilePath: \undefinedd:\PKU_msi\pyMC\tests\test_input\mol_scf\H2O_col_nosymm.py

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

molcoords = """
 O                  0.00000000    0.00000000   -0.10983178 ;
 H                  0.00000000   -0.75754080    0.47724786 ;
 H                 -0.00000000    0.75754080    0.47724786 ;
"""

THRESHOLD_list = [(-1.0, 10.0), (1.0E-10, 0.999)]
Ndirect = 1454
functional_list = ['SVWN','PBE','PBE']
IBP_list = [False, False, True]
output_dict = {}

mol = gto.Mole()
mol.atom = molcoords
mol.spin=1 # ^ Spin
mol.charge = 1
mol.basis = "cc-pvtz" # ^ Basis
mol.max_memory = 50000
mol.build()

BENCHMARK_etot = {
    0:{
        0: -75.42034051646813,
        1: -75.90923181270902,
        2: -75.9092318126624
        },
    1:{
        0: -75.42034051646883,
        1: -75.9092318127094,
        2: -75.9092318126626
        },
}

BENCHMARK_motot = {
    0:{
        0: numpy.array([-19.166287913526578, -19.138364878725337,  -1.433580667290669,
        -1.380397103680875,  -0.952004788812001]),
        1: numpy.array([-19.311603178866946, -19.28649377212775 ,  -1.441942685645377,
        -1.383227174808574,  -0.947659212581591]),
        2: numpy.array([-19.311603528861312, -19.286492702468035,  -1.441942352954055,
        -1.383225561888267,  -0.947658832030975])
        },
    1:{
        0: numpy.array([-19.16628791346259 , -19.138364878662415,  -1.433580667233645,
        -1.380397103626152,  -0.952004788753444]),
        1: numpy.array([-19.311603178796602, -19.28649377205813 ,  -1.441942685583543,
        -1.383227174748142,  -0.947659212517588]),
        2: numpy.array([-19.311603528948204, -19.286492702554483,  -1.441942352954244,
        -1.383225561884407,  -0.947658832019147])
        },
}

FLAG = 'Positive'
EPSILON = 1.0E-8

for idxT ,THRESHOLD in enumerate(THRESHOLD_list):
    for idx, xc in enumerate(functional_list):
        mftot = gksmc.GKSMC(mol)
        mftot.xc = xc
        mftot.ibp = IBP_list[idx]
        mftot.LIBXCT_factor = THRESHOLD[0]
        mftot.MSL_factor = THRESHOLD[1]
        mftot.Ndirect = Ndirect
        mftot.max_cycle = 50
        mftot.kernel()
        
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