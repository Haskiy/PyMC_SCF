#/usr/bin/env python
'''
Author: Li Hao
Date: 2022-04-18 20:53:25
LastEditTime: 2022-04-14 14:11:20
LastEditors: Li Hao
Description: A test and an example for collinear TDA of Multi-Collinear vs Locally Collinear 
             appraoch.
             Model system: H2O : Spin=2, Charge=0 (a collinear system).
             Test term: Excited energy.
             Benchmark: Multi-Collinear vs Locally Collinear appraoch.
             Content:
                1-3. TDAMC vs TDALC LDA : (SC,SFU,SFD), no numerical stable scheme.
                4-6. TDAMC vs TDALC LDA : (SC,SFU,SFD), with numerical stable scheme (LIBXCT_
                factor only).
            
             Note: the EPSILON of this test is (rtol=1e-3, atol=1e-2), mainly because of the ins-
             tability of spin_flip_down part.

FilePath: \pyMC\tests\test_input\mol_tda\nctdamc_CrSpin6.py
Motto: A + B = C!
'''
import os
import numpy
PATH_pyMC = os.path.abspath(os.path.dirname(__file__)).split('/pyMC')[0]
PATH_Bechdata = os.path.abspath(os.path.dirname(__file__)).split('/test_input')[0] + '/Benchmark_data/'
PATH_Predata = os.path.abspath(os.path.dirname(__file__)).split('/test_input')[0] + '/Pre_data/'

from pyscf import gto,dft
import sys
# ! NOTE: should add the path of the pyMC module.
sys.path.append(PATH_pyMC)
from pyMC.tdamc import tdamc_uks
from pyMC.tdamc import tdalc_uks

molcoords = """
 O                  0.00000000    0.00000000   -0.10983178 ;
 H                  0.00000000   -0.75754080    0.47724786 ;
 H                 -0.00000000    0.75754080    0.47724786 ;
"""

mol = gto.Mole()
mol.atom = molcoords
mol.spin = 2 # ^ Spin
mol.charge = 0
mol.basis = "cc-pvtz" # ^ Basis
mol.max_memory = 50000
mol.build()

THRESHOLD_list = [None,1e-10]
Extype_list = ['SPIN_CONSERVED','SPIN_FLIP_UP','SPIN_FLIP_DOWN']
BenchName_list = [['NLIBXCT_SC','NLIBXCT_SFU','NLIBXCT_SFD'],
                  ['LIBXCT_SC','LIBXCT_SFU','LIBXCT_SFD']]

FLAG = 'Positive'
EPSILON_r = 1.0E-3
EPSILON_a = 1.0E-2

mf_scf = dft.UKS(mol)
mf_scf.xc = 'SVWN'
mf_scf.grids.level = 6
mf_scf.kernel()

for idx_T,THRESHOLD in enumerate(THRESHOLD_list):
    for idx_Ex,Extype in enumerate(Extype_list):
        mf_tdamc = tdamc_uks.TDAMC_UKS(mf_scf)
        mf_tdamc.Ndirect = 200
        mf_tdamc.Extype = Extype
        mf_tdamc.LIBXCT_facrot = THRESHOLD
        mf_tdamc.kernel()

        mf_tdalc = tdalc_uks.TDALC_UKS(mf_scf)
        mf_tdalc.Extype = Extype
        mf_tdalc.LIBXCT_factor = THRESHOLD
        mf_tdalc.kernel()

        Diff = numpy.allclose(mf_tdamc.Extd,mf_tdalc.Extd,rtol=EPSILON_r,atol=EPSILON_a)
        if Diff:
            FLAG = 'Positive'
            print('NCTDA_mc_vs_lc_' + BenchName_list[idx_T][idx_Ex] + '_test : '+ FLAG)
        else:
            FLAG = 'Negative'
            print('NCTDA_mc_vs_lc_' + BenchName_list[idx_T][idx_Ex] + '_test : '+ FLAG)