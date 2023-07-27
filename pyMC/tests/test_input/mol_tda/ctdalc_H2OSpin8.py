#/usr/bin/env python
'''
Author: Li Hao
Date: 2022-04-13 16:56:40
LastEditTime: 2022-04-14 14:13:05
LastEditors: Li Hao
Description: A test and an example for Locally Collinear collinear TDA.
             Model system: H2O : Spin=8, Charge=0 (a strong polarized system).
            
             Test term: Types of excited energy: 
                            SC  : Spin_Conserved excited energy, 
                            SFU : spin_Flip_Up excited energy,
                            SFD : spin_Flip_Down excited energy.
             Benchmark: Results of pySD (old pyMC) codes.
             Content:
                1-3. TDALC LDA : (SC,SFU,SFD), no numerical stable scheme,
                4-6. TDALC LDA : (SC,SFU,SFD), with numerical stable scheme.
             
             Note: Numerical stable scheme of Locally Collinear approach switches the threshold
                   of derivatives on or off obtained from libxc. And the threshold for s (mage-
                   nization density norm), KST_factor, padding the instable case at s (as den) 
                   is 1e-10. 
                
             Note: Because of the instability of strong polarized systems, all tests use a same 
                   scf result of UKS.

FilePath: \pyMC\tests\test_input\mol_tda\ctdalc_H2OSpin8.py
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
from pyMC.tdamc import tdalc_uks

molcoords = """
 O                  0.00000000    0.00000000   -0.10983178 ;
 H                  0.00000000   -0.75754080    0.47724786 ;
 H                 -0.00000000    0.75754080    0.47724786 ;
"""

mol = gto.Mole()
mol.atom = molcoords
mol.spin = 8 # ^ Spin
mol.charge = 0
mol.basis = "cc-pvtz" # ^ Basis
mol.max_memory = 50000
mol.build()

Functional_list = ['SVWN']
THRESHOLD_list = [None, 1e-10]
Extype_list = ['SPIN_CONSERVED','SPIN_FLIP_UP','SPIN_FLIP_DOWN']


# 'ctda_H2OSpin2_Extd.npy' is a dict which stores the Benchmark results,
# with form {0:{0:SC,1:SFU,:SFD},1:{0:SC,1:SFU,:SFD}}
# where the first (0,1) -> (None, 1e-10).
BENCHMARK_Extd = numpy.load(PATH_Bechdata + 'ctdalc_H2OSpin8_Extd.npy',allow_pickle=True).item()
BENCHMARK_Name = numpy.load(PATH_Bechdata + 'ctdalc_H2OSpin8_BechName.npy',allow_pickle=True).item()

FLAG = 'Positive'
EPSILON = 1.0E-8

# Just to keep the same test form written.
for idx_xc,xc in enumerate(Functional_list):
    mf_scf = dft.UKS(mol)
    mf_scf.xc = xc
    mf_scf.grids.level = 6
    mf_scf.max_cycle = 0
    # mf_scf.kernel() obtains grids.coords and grids.weights.
    mf_scf.kernel()
    
    # Give mf_scf object the same scf result.
    mf_scf.mo_energy = numpy.load(PATH_Predata + 'H2OSpin8_' + xc + '_uks_mo_energy.npy')
    mf_scf.mo_occ = numpy.load(PATH_Predata + 'H2OSpin8_' + xc + '_uks_mo_occ.npy')
    mf_scf.mo_coeff = numpy.load(PATH_Predata + 'H2OSpin8_' + xc + '_uks_mo_coeff.npy')
    
    for idx_T,THRESHOLD in enumerate(THRESHOLD_list):
        for idx_Ex, Extype in enumerate(Extype_list):
            mf_tdalc = tdalc_uks.TDALC_UKS(mf_scf)
            mf_tdalc.LIBXCT_factor = THRESHOLD
            mf_tdalc.Extype = Extype
            mf_tdalc.kernel()
            
            Diff = numpy.allclose(BENCHMARK_Extd[idx_xc][idx_T][idx_Ex],mf_tdalc.Extd,atol=EPSILON)
            if Diff:
                FLAG = 'Positive'
                print(BENCHMARK_Name[idx_xc][idx_T][idx_Ex] + ' : '+ FLAG)
            else:
                FLAG = 'Negative'
                print(BENCHMARK_Name[idx_xc][idx_T][idx_Ex] + ' : '+ FLAG)
                # for i in range(len(mf_tdalc.Extd)):
                #     print(mf_tdalc.Extd[i])
                # break