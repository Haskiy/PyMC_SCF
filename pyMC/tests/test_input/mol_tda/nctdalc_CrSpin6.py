#/usr/bin/env python
'''
Author: Li Hao
Date: 2022-04-13 19:12:57
LastEditTime: 2022-04-14 14:10:02
LastEditors: Li Hao
Description: A test and an example for Locally Collinear noncollinear TDA for collinear spins.
             Model system: Cr : Spin=6, Charge=0 (a collinear system but with strong polarize).
             Test term: Excited energy.
             Benchmark: Results of pySD (old pyMC) codes.
             Content:
                1. TDALC LDA : no numerical stable scheme,
                2. TDALC LDA : with numerical stable scheme.
            
             Note: Numerical stable scheme of Locally Collinear approach switches the threshold
                   of derivatives on or off obtained from libxc only. And the threshold for s 
                   (magenization density norm), KST_factor, padding the instable case at s (as 
                   den) is 1e-10.
                    
             Note: Because of the instability of strong polarized systems, all tests use a same 
                   scf result of UKS. 

FilePath: \pyMC\tests\test_input\mol_tda\nctdalc_CrSpin6.py
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
from pyMC.tdamc import tdalc_gks

molcoords = """
Cr          1.1304262391426063     0.000000000000000     0.000000000000000 ;
"""

mol = gto.Mole()
mol.atom = molcoords
mol.spin = 6 # ^ Spin
mol.symmetry = True
mol.charge = 0
mol.basis = "cc-pvtz" # ^ Basis
mol.max_memory = 50000
mol.build()

Functional_list = ['SVWN']
THRESHOLD_list = [None,1e-10]

# 'nctdaLc_CrSpin6_Extd' is a dict which stores the Benchmark results,
# with form {0:{0:KNN,1:KLN}}, 
# where the first 0 -> ('SVWN'), and the second (0,1) -> (None,1e-10).
BENCHMARK_Extd = numpy.load(PATH_Bechdata + 'nctdalc_CrSpin6_Extd.npy',allow_pickle=True).item()
BENCHMARK_Name = numpy.load(PATH_Bechdata + 'nctdalc_CrSpin6_BechName.npy',allow_pickle=True).item()

FLAG = 'Positive'
EPSILON = 1.0E-8

for idx_xc,xc in enumerate(Functional_list):
    mf_scf = dft.UKS(mol)
    mf_scf.xc = xc
    mf_scf.grids.level = 6
    mf_scf.max_cycle = 0
    # mf_scf.kernel() obtains grids.coords and grids.weights.
    mf_scf.kernel()

    # Give mf_scf object the same scf result.
    mf_scf.mo_energy = numpy.load(PATH_Predata + 'CrSpin6_'+ xc + '_uks_mo_energy.npy')
    mf_scf.mo_occ = numpy.load(PATH_Predata + 'CrSpin6_'+ xc + '_uks_mo_occ.npy')
    mf_scf.mo_coeff = numpy.load(PATH_Predata + 'CrSpin6_'+ xc + '_uks_mo_coeff.npy')
    
    for idx_T,THRESHOLD in enumerate(THRESHOLD_list):
        # Though mf_scf -> UKS(), it will be transformed in to GKS() here.
        mf_tdalc = tdalc_gks.TDALC_GKS(mf_scf)
        mf_tdalc.LIBXCT_factor = THRESHOLD
        mf_tdalc.kernel()
        
        Diff = numpy.allclose(BENCHMARK_Extd[idx_xc][idx_T],mf_tdalc.Extd,atol=EPSILON)
        if Diff:
            FLAG = 'Positive'
            print(BENCHMARK_Name[idx_xc][idx_T] + ' : '+ FLAG)
            for i in range(len(mf_tdalc.Extd)):
                print(mf_tdalc.Extd[i])
        else:
            FLAG = 'Negative'
            print(BENCHMARK_Name[idx_xc][idx_T] + ' : '+ FLAG)
            for i in range(len(mf_tdalc.Extd)):
                print(mf_tdalc.Extd[i])