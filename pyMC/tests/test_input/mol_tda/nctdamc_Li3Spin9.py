#/usr/bin/env python
'''
Author: Li Hao
Date: 2022-04-13 19:32:07
LastEditTime: 2022-04-14 14:12:18
LastEditors: Li Hao
Description: A test and an example for Multi-Collinear noncollinear TDA for noncollinear spins.
             Model system: Li3 : Spin=9, Charge=0 (a noncollinear system with strong polarize).
             Test term: Excited energy.
             Benchmark: Results of pySD (old pyMC) codes.
             Content:
                1. TDAMC LDA : no numerical stable scheme,
                2. TDAMC GGA : no numerical stable scheme,
                3. TDAMC LDA : with numerical stable scheme,
                4. TDAMC GGA : with numerical stable scheme.
                    
             Note: Because of the instability of strong polarized systems, all tests use a same 
                   scf result of GKSMC.  

FilePath: \pyMC\tests\test_input\mol_tda\nctdamc_Li3Spin9.py
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
from pyMC.gksmc import gksmc
from pyMC import grids_mc
from pyMC.tdamc import tdamc_gks

molcoords = """
Li          2.309401076758503     0.000000000000000     0.000000000000000    ;
Li          -1.154700538379252    2.000000000000000     0.000000000000000    ;
Li          -1.154700538379252    -2.000000000000000    0.000000000000000    ;
"""

mol = gto.Mole()
mol.atom = molcoords
mol.spin = 9 # ^ Spin
mol.charge = 0
mol.basis = "cc-pvtz" # ^ Basis
mol.max_memory = 50000
mol.build()

Functional_list = ['SVWN','PBE']
THRESHOLD_list = [(None,None), (0.999,1e-10)]

BENCHMARK_Extd = numpy.load(PATH_Bechdata + 'nctdamc_Li3Spin9_Extd.npy',allow_pickle=True).item()
BENCHMARK_Name = numpy.load(PATH_Bechdata + 'nctdamc_Li3Spin9_BechName.npy',allow_pickle=True).item()

FLAG = 'Positive'
EPSILON = 1.0E-8

for idx_xc,xc in enumerate(Functional_list):
    mf_scf = gksmc.GKSMC(mol)
    mf_scf.parallel = True
    mf_scf.xc = xc
    # ! NOTE: that the following part is Gauss-Legendre scheme.
    # * By changing the phi angle, changing the symmetry.
    dft.Grids.gen_atomic_grids = grids_mc.Grids_hss.gen_atomic_grids_gauss_legendre
    mf_scf.grids.atom_grid = {"Li" : (50, 30, 24)} # ^ (A, B, C) : A--> radial grid; B--> theta; C--> phi
    mf_scf.Ndirect = 1454
    mf_scf.max_cycle = 0
    # mf_scf.kernel() obtains grids.coords and grids.weights.
    mf_scf.kernel()

    # Give mf_scf object the same scf result.
    mf_scf.mo_energy = numpy.load(PATH_Predata + 'Li3Spin9_' + xc + '_gksmc_mo_energy.npy')
    mf_scf.mo_occ = numpy.load(PATH_Predata + 'Li3Spin9_' + xc + '_gksmc_mo_occ.npy')
    mf_scf.mo_coeff = numpy.load(PATH_Predata + 'Li3Spin9_' + xc + '_gksmc_mo_coeff.npy')
    
    for idx_T,THRESHOLD in enumerate(THRESHOLD_list):
        mf_tdamc = tdamc_gks.TDAMC_GKS(mf_scf)
        mf_tdamc.Ndirect = 1454
        mf_tdamc.MSL_factor = THRESHOLD[0]
        mf_tdamc.Ndirect_lc = 2000
        mf_tdamc.LIBXCT_factor = THRESHOLD[1]
        mf_tdamc.kernel()
        
        Diff = numpy.allclose(BENCHMARK_Extd[idx_xc][idx_T],mf_tdamc.Extd,atol=EPSILON)
        if Diff:
            FLAG = 'Positive'
            print(BENCHMARK_Name[idx_xc][idx_T] + ' : '+ FLAG)
        else:
            FLAG = 'Negative'
            print(BENCHMARK_Name[idx_xc][idx_T] + ' : '+ FLAG)
            for i in range(len(mf_tdamc.Extd)):
                print(mf_tdamc.Extd[i])
      