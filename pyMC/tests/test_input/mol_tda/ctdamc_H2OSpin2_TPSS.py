#/usr/bin/env python
'''
Author: Li Hao
Date: 2022-04-14 12:48:40
LastEditTime: 2022-04-14 14:04:08
LastEditors: Li Hao
Description: A test and an example for Multi-Collinear collinear TDA.
             Model system: H2O : Spin=2, Charge=0 (a usual system).
             Test term: Types of excited energy: 
                            SFU : Spin_Flip_Up excited energy,
                            SFD : Spin_Flip_Down excited energy.
             Benchmark: Results of pySD (old pyMC) codes.
             Content:
                1-2. TDAMC MGGA : (SFU,SFD), no numerical stable scheme,
                3-4. TDAMC MGGA : (SFU,SFD), with numerical stable scheme.
            
             Note: Spin_Conserved excited energy hasn't implemented in MGGA.

FilePath: \pyMC\tests\test_input\mol_tda\ctdamc_H2OSpin2_TPSS.py
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

Functional_list = ['TPSS']
THRESHOLD_list = [None,1e-10]
Extype_list = ['SPIN_FLIP_UP','SPIN_FLIP_DOWN']

# 'ctda_H2OSpin2_Extd.npy' is a dict which stores the Benchmark results,
# with form {0:{0:{0:SFU,1:SFD},1:{0:SFU,1:SFD}}}
# where 0 -> (TPSS) and the first (0,1) -> (None, 1e-10).
BENCHMARK_Extd = numpy.load(PATH_Bechdata + 'ctdamc_H2OSpin2_TPSS_Extd.npy',allow_pickle=True).item()
BENCHMARK_Name = numpy.load(PATH_Bechdata + 'ctdamc_H2OSpin2_TPSS_BechName.npy',allow_pickle=True).item()

FLAG = 'Positive'
EPSILON = 1.0E-8

for idx_xc,xc in enumerate(Functional_list):
    mf_scf = dft.UKS(mol)
    mf_scf.xc = xc
    mf_scf.grids.level = 6
    mf_scf.kernel()
    
    for idx_T,THRESHOLD in enumerate(THRESHOLD_list):
        for idx_Ex, Extype in enumerate(Extype_list):
            mf_tdamc = tdamc_uks.TDAMC_UKS(mf_scf)
            mf_tdamc.Ndirect = 200
            mf_tdamc.LIBXCT_factor = THRESHOLD
            mf_tdamc.Extype = Extype
            mf_tdamc.kernel()
            
            Diff = numpy.allclose(BENCHMARK_Extd[idx_xc][idx_T][idx_Ex],mf_tdamc.Extd,atol=EPSILON)
            if Diff:
                FLAG = 'Positive'
                print(BENCHMARK_Name[idx_xc][idx_T][idx_Ex] + ' : '+ FLAG)
            else:
                FLAG = 'Negative'
                print(BENCHMARK_Name[idx_xc][idx_T][idx_Ex] + ' : '+ FLAG)
                for i in range(len(mf_tdamc.Extd)):
                    print(mf_tdamc.Extd[i])
                break

# Test_Results:
# SFD results 未通过，原因是TPSS的SCF也具有收敛性的问题，因为是MGGA libxc 在本机版本 1.7.6 与超算以及
# 服务器不同所导致的。如果用xcfun则不会。所以这里就先不纠结了。 