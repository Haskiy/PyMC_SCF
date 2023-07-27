'''
Author: Li Hao
Date: 2021-09-28 21:22:26
LastEditTime: 2021-11-22 12:18:46
LastEditors: Li Hao
Description: 
    This is a pre-verdion for giving birth to new tests for
    solid calculation.
FilePath: \pyMC\tools\input_pbc_file_generator.py

Motto: A + B = C!
'''

import numpy
# TO DO: ECP and PP parts  

# METHODDICT[2,2,3]
METHODSET = [
    [
        ['krhf', 'kuhf', 'kghf']    
       ,['krks', 'kuks', 'kgks', 'kgksm']   # # for instantiation name
        
     ],
    [ 
        ['pbcscf.KRHF', 'pbcscf.KUHF', 'pbcscf.KGHF'] 
       ,['pbcdft.KRKS', 'pbcdft.KUKS', 'pyMCpbc.KGKS', 'pySDpbc.KGKSM'] # for instantiation name
    ]
]

DF = ['DF', 'GDF', 'MDF']   # DF is used default in PySCF 

PYSDPATH = {"GAO" : "/public3/home/sc40234/Jobs_lh",
            "JIANG" : "/public3/home/sc51754/members/pzc", # no me of Jiang's chaosuan
            "LH" : "/home/lihao",
            "XIAO":"/public1/home/scg0213/Jobs_LH"}
STARS = "***************************************************************************"
NPROC = {"GAO" : 64,
         "JIANG" : 64,
         "LH" : 40,
         "XIAO": 64}

def prt_import(input_obj, filename):
    input_obj.path = PYSDPATH[input_obj.chaosuan.upper()]
    with open(filename, 'a+') as fn:
        fn.write('#!/usr/bin/env python\n')
        fn.write('import sys\n')
        fn.write('# ! NOTE: should add the path of the pyMC module.\n')
        fn.write('sys.path.append(r"' + input_obj.path +'")\n')
        fn.write('from pyscf import lib\n')
        fn.write('lib.misc.num_threads(n=' + str(NPROC[input_obj.chaosuan.upper()]) + ')\n')
        fn.write('from pyscf import gto,dft,scf,lib,tools\n')
        fn.write('from pyscf.pbc import gto as pbcgto\n')
        fn.write('from pyscf.pbc import scf as pbcscf\n')
        fn.write('from pyscf.pbc import dft as pbcdft\n')
        fn.write('from pyMC import pbc as pySDpbc\n')
        fn.write('from pyMC import tools as tools_hss\n')
        fn.write('from pyMC import grids_util\n')
        fn.write('from pyMC import gksm_util\n')
        fn.write('from pyMC import mole_sym\n')
        fn.write('from pyMC import lib as hsslib\n')
        fn.write('from pyMC.tools.tools_pbc import puwant\n')
        fn.write('import numpy\n')
        fn.write('\n')

def prt_molu(input_obj, filename):
    with open(filename, 'a+') as fn:
            fn.write('# ! NOTE: Single atom part to generate initial density matrixt\n')
            fn.write('### Atom bulid ###\n') 
            fn.write('molu_cell = pbcgto.Cell()\n') 
            fn.write('molu_cell.atom = "' + str(input_obj.atom) + ' 0 0 0" ###\n')
            fn.write('molu_cell.spin = ' + str(input_obj.spin)  + ' # ^ Spin\n')
            fn.write('molu_cell.basis = "'+ input_obj.basis +'" # ^ Basis\n')
            if input_obj.pseudo:
                fn.write('molu_cell.pseudo = ' '\'' + input_obj.pseudo + '\'' ' # ^ Pseudo\n')
            elif input_obj.ecp:
                fn.write('molu_cell.ecp = ' '\'' + input_obj.ecp + '\'' ' # ^ Ecp\n')
            if input_obj.exp_to_discard:
                fn.write('molu_cell.exp_to_discard = ' + str(input_obj.exp_to_discard) + '\n') # reduce diffuse
            fn.write('molu_cell.a = numpy.eye(3) \n') 
            fn.write('molu_cell.dimension = 0 \n')
            fn.write('molu_cell.max_memory = 50000\n')   # ^ can change the max_memory here
            fn.write('molu_cell.verbose = 6\n')
            fn.write('molu_cell.output = "./atom.txt"\n')
            fn.write('molu_cell.build() \n')
            fn.write('\n')
            fn.write('molu = molu_cell.to_mol()\n') # mol to cell
            fn.write('\n')
            fn.write('# ! NOTE: Single atom SCF calculation\n') # not sure
            fn.write('mf = scf.UHF(molu)\n')
            fn.write('# ! NOTE: Cell caculation part!\n')
            fn.write('mf.kernel()\n')
            fn.write('\n')      

def get_mol_coords(input_obj):
    input_obj.coords = []
    if input_obj.number == 3:
        input_obj.coords = [[ 0.0, input_obj.bond/numpy.sqrt(3), 0.0],
                            [-input_obj.bond*0.5, -input_obj.bond/numpy.sqrt(3)*0.5, 0.0],
                            [input_obj.bond*0.5,-input_obj.bond/numpy.sqrt(3)*0.5, 0.0]]
        
def prt_mol_coords(input_obj, filename):
    with open(filename, 'a+') as fn:
        fn.write('mol_coords = """\n')
        for iatm in range(input_obj.number):
            fn.write('{0:<10}  {1:<20.15f}  {2:<20.15f}  {3:<20.15f} ;\n'.\
                format(input_obj.atom, input_obj.coords[iatm][0],\
                    input_obj.coords[iatm][1], input_obj.coords[iatm][2]))
        fn.write('"""\n')
        fn.write('\n')

def prt_mol(input_obj, filename):
    with open(filename, 'a+') as fn:
            fn.write('# ! NOTE: Create pre-calculated molecule\n')
            fn.write('mol_cell = pbcgto.Cell()\n') 
            fn.write('mol_cell.atom = mol_coords\n')
            fn.write('mol_cell.spin=' + str(input_obj.spin*input_obj.number)  + ' # ^ Spin\n')
            fn.write('mol_cell.basis = "'+ input_obj.basis +'" # ^ Basis\n')
            if input_obj.pseudo:
                fn.write('mol_cell.pseudo = ' '\'' + input_obj.pseudo + '\'' ' # ^ Pseudo\n')
            if input_obj.ecp:
                fn.write('mol_cell.ecp = ' '\'' + input_obj.ecp + '\'' ' # ^ Ecp\n')
            if input_obj.exp_to_discard:
                fn.write('mol_cell.exp_to_discard = ' + str(input_obj.exp_to_discard) + '\n') # reduce diffuse
            fn.write('mol_cell.max_memory = 50000\n')   # ^ can change the max_memory here
            fn.write('mol_cell.a = numpy.eye(3)\n')
            fn.write('mol_cell.dimension = 0\n')
            fn.write('mol_cell.verbose = 6\n')
            fn.write('mol_cell.build() \n')
            fn.write('\n')
            fn.write('mol = mol_cell.to_mol()\n')
            fn.write('\n')

def prt_dm_initial(input_obj, filename):
    with open(filename, 'a+') as fn:
        fn.write('mo_coeffu = mf.mo_coeff\n')
        fn.write('noccu = mf.mo_occ\n')
        fn.write('nao = mo_coeffu.shape[-1]\n')
        fn.write('natom = mol.natm \n')
        fn.write('theta_dict = tools_hss.rotate_dm.get_init_guess_theta(mol_cell, rotatez_negative = '+str(input_obj.rotatez_negative)+',vortex = '+str(input_obj.vortex)+') \n')
        fn.write('\n')
        fn.write('mo_coeffu = tools_hss.rotate_dm.get_gks_dm_guess_mo(molu, mo_coeffu, natom\n')
        fn.write('                                , numpy.array([theta_dict[i][0] for i in range(3)]) \
            , rotatem = True, rotatel =  True)\n')
        fn.write('nocc = numpy.array((noccu[0].tolist()*natom+noccu[1].tolist()*natom))\n')
        fn.write('\n')
        # generate dm
        fn.write('mf = scf.GHF(mol)\n')
        fn.write('dm = mf.make_rdm1(mo_coeffu,nocc)\n')
        # save dm
        fn.write('numpy.save(\'' + input_obj.name + 'dm\', dm)\n')
        fn.write('\n')

def get_pbc_coords(input_obj):
    input_obj.coords = []
    if input_obj.number == 3:
        input_obj.coords = [[0.0, 0.0, input_obj.vacuum/2],
                            [input_obj.bond*1.0, 0.0, input_obj.vacuum/2],
                            [input_obj.bond*0.5, input_obj.bond*(3**.5)*.5, input_obj.vacuum/2],
                            [input_obj.bond*1.5, input_obj.bond*(3**.5)*.5, input_obj.vacuum/2],
                            [input_obj.bond*2.5, input_obj.bond*(3**.5)*.5, input_obj.vacuum/2],
                            [input_obj.bond*2.0, 0.0, input_obj.vacuum/2]]

        
def prt_pbc_coords(input_obj, filename):
    with open(filename, 'a+') as fn:
        fn.write('pbc_coords = """\n')
        for iatm in range(input_obj.number*2):
            fn.write('{0:<10}  {1:<20.15f}  {2:<20.15f}  {3:<20.15f} ;\n'.\
                format(input_obj.atom, input_obj.coords[iatm][0],\
                    input_obj.coords[iatm][1], input_obj.coords[iatm][2]))
        fn.write('"""\n')
        fn.write('\n')

def get_pbc_lattice(input_obj):
    input_obj.coords = []
    if input_obj.number == 3:
        input_obj.coords = [[input_obj.bond*3.0, 0, 0],
                            [0.0, input_obj.bond*(3.0**0.5), 0],
                            [0.0, 0.0, input_obj.vacuum]]

        
def prt_pbc_lattice(input_obj, filename):
    with open(filename, 'a+') as fn:
        fn.write('pbc_lattice = """\n')
        for iatm in range(input_obj.number):
            fn.write('{1:<20.15f}  {2:<20.15f}  {3:<20.15f} ;\n'.\
                format(input_obj.atom, input_obj.coords[iatm][0],\
                    input_obj.coords[iatm][1], input_obj.coords[iatm][2]))
        fn.write('"""\n')
        fn.write('\n')

def prt_cell(input_obj, filename):
    with open(filename, 'a+') as fn:
        fn.write('# ! NOTE: Information of original cell part (for monolayer materias)\n')
        fn.write('### Original cell bulid ###\n')
        fn.write(input_obj.name + ' = pbcgto.M()\n') # here is M, not Cell, mainly for xc attr
        fn.write(input_obj.name + '.atom = pbc_coords\n')
        fn.write(input_obj.name + '.basis = ' '\'' + input_obj.basis +  '\'' ' # ^ Basis\n')
        if input_obj.pseudo:
            fn.write(input_obj.name + '.pseudo = ' '\'' + input_obj.pseudo + '\'' ' # ^ Pseudo\n')
        if input_obj.ecp:
                fn.write(input_obj.name + '.ecp = ' '\'' + input_obj.ecp + '\'' ' # ^ Ecp\n')
        if input_obj.exp_to_discard:
                fn.write('molu_cell.exp_to_discard = ' + str(input_obj.exp_to_discard) + '\n') # reduce diffuse
        fn.write(input_obj.name + '.a = pbc_lattice # ^ Lattice\n')
        fn.write(input_obj.name + '.dimension = 2\n')
        fn.write(input_obj.name + '.output = ' '\'' + input_obj.cellout + '\'' '\n')
        fn.write(input_obj.name + '.max_memory = 50000\n')   # ^ can change the max_memory here
        fn.write(input_obj.name + '.verbose = ' + str(input_obj.verbose) + '\n')
        fn.write(input_obj.name + '.precision = ' + str(input_obj.precision) + '\n')
        fn.write(input_obj.name + '.build() \n')
        fn.write('\n')
        fn.write('# ! NOTE: Original cell is established\n') # not sure
        fn.write('\n')
        fn.write('# ! NOTE: Cell caculation part!\n')
        fn.write('\n')


def prt_make_kpts(input_obj, filename):
    with open(filename, 'a+') as fn:
        fn.write('kpts = ' + str(input_obj.name) + '.make_kpts('+ str(input_obj.make_kpts) + ')\n')
        # if input_obj.ndim_kpts == 3:
            
        # elif input_obj.ndim_kpts == 2:
        #     fn.write('kpts = ' + str(input_obj.name) + '.make_kpts(['+ str(input_obj.npd_kpts) + '] * 3)\n')
        #     fn.write('n = ' + str(input_obj.npd_kpts) + ' ** 3 \n')
        #     fn.write('kptslist = [] \n')
        #     fn.write('while(n > 0): \n')
        #     fn.write('    n -= ' + str(input_obj.npd_kpts) + '\n')
        #     fn.write('    kptslist.append(kpts[n])\n')
        #     fn.write('kpts = numpy.array(kptslist)\n')
        # elif input_obj.ndim_kpts == 1:
        #     fn.write('kpts = ' + str(input_obj.name) + '.make_kpts(['+ str(input_obj.npd_kpts) + '] * 3)\n')
        #     fn.write('n = ' + str(input_obj.npd_kpts) + ' ** 3 \n')
        #     fn.write('kptslist = [] \n')
        #     fn.write('while(n > 0): \n')
        #     fn.write('    n -= ' + str(input_obj.npd_kpts) + '**2 \n')
        #     fn.write('    kptslist.append(kpts[n])\n')
        #     fn.write('kpts = numpy.array(kptslist)\n')
        fn.write('\n')

def prt_dm_transform(input_obj, filename):
     with open(filename, 'a+') as fn:
        # load initial density matrix
        fn.write('dmi = numpy.load(\'' + str(input_obj.name) + 'dm.npy\')\n')
        fn.write('x = int(dmi.shape[-1])\n')
        fn.write('y = int(x/'+ str(input_obj.number) + ')\n')
        fn.write('z = int(y*.5)\n')
        fn.write('g = int(x*.5)\n')
        fn.write('dm1 = numpy.zeros((y,y), dtype = numpy.complex128) \n')
        fn.write('dm2 = numpy.zeros((y,y), dtype = numpy.complex128) \n')
        fn.write('dm3 = numpy.zeros((y,y), dtype = numpy.complex128) \n')
        # need to correct
        fn.write('dm1[:z,:z] = dmi[:z,:z] \n')
        fn.write('dm2[:z,:z] = dmi[z:y,z:y] \n')
        fn.write('dm3[:z,:z] = dmi[y:g,y:g] \n')        # aa block

        fn.write('dm1[z:,:z] = dmi[g:g+z,:z] \n')
        fn.write('dm2[z:,:z] = dmi[g+z:g+2*z,z:y] \n')
        fn.write('dm3[z:,:z] = dmi[g+2*z:g+3*z,y:g] \n')      # ba block

        fn.write('dm1[:z,z:] = dmi[:z,g:g+z] \n')
        fn.write('dm2[:z,z:] = dmi[z:y,g+z:g+2*z] \n')
        fn.write('dm3[:z,z:] = dmi[y:g,g+2*z:g+3*z] \n')  # ab block

        fn.write('dm1[z:,z:] = dmi[g:g+z,g:g+z] \n')
        fn.write('dm2[z:,z:] = dmi[g+z:g+2*z,g+z:g+2*z] \n')
        fn.write('dm3[z:,z:] = dmi[g+2*z:g+3*z,g+2*z:g+3*z] \n')  # bb block
        # initial new density matrix
        fn.write('dm = numpy.zeros((len(kpts),' + str(input_obj.number) + '*2*y, ' 
                                    + str(input_obj.number) + '*2*y) , dtype = numpy.complex128)\n')
        fn.write('dmlist = [dm1,dm2,dm3]\n')
        fn.write('idxlist = [0,1,2,0,1,2]\n')
        fn.write('for k in range(len(kpts)):\n')
        fn.write('    for i in range(len(idxlist)):\n')
        fn.write('        dm[k,i*z:(i+1)*z,i*z:(i+1)*z] = dmlist[idxlist[i]][:z,:z] \n')
        fn.write('        dm[k,i*z:(i+1)*z,i*z+x:(i+1)*z+x] = dmlist[idxlist[i]][:z,z:] \n')
        fn.write('        dm[k,i*z+x:(i+1)*z+x,i*z:(i+1)*z] = dmlist[idxlist[i]][z:,:z] \n')
        fn.write('        dm[k,i*z+x:(i+1)*z+x,i*z+x:(i+1)*z+x] = dmlist[idxlist[i]][z:,z:]\n')
        fn.write('\n')

def prt_method_and_kernel(input_obj, filename):
    if input_obj.df == None:
        input_obj.df = None
    elif input_obj.df.upper() == DF[0]:
        input_obj.df = None
    elif input_obj.df.upper() == DF[1]:
        input_obj.df = DF[1]
    elif input_obj.df.upper() == DF[2]:
        input_obj.df = DF[2]

    if input_obj.method.lower() in METHODSET[0][0]:
        forder = 0
        if input_obj.method.upper() == 'KRHF':
            sorder = 0
        if input_obj.method.upper() == 'KUHF':
            sorder = 1
        if input_obj.method.upper() == 'KGHF':
            sorder = 2
    elif input_obj.method.lower() in METHODSET[0][1]:
        forder = 1
        if input_obj.method.upper() == 'KRKS':
            sorder = 0
        if input_obj.method.upper() == 'KUKS':
            sorder = 1
        if input_obj.method.upper() == 'KGKS':
            sorder = 2
        if input_obj.method.upper() == 'KGKSM':
            sorder = 3
    else:
        print('You may give a wrong method name')
            
    method = METHODSET[0][forder][sorder]
    kmethod = METHODSET[1][forder][sorder]

    with open(filename, 'a+') as fn:
        if input_obj.df == DF[1]:
            fn.write('from pyscf.pbc import df as pbcdf\n')
            fn.write(input_obj.name + 'df = pbcdf.' + input_obj.df.upper() + '(' + input_obj.name
                     + ', kpts)\n')
            # if input_obj.basis = ???  TO DO
            #    input_obj.load_cdetri
            if input_obj.save_cderi:
                fn.write(input_obj.name + 'df._cderi_to_save = \'' + input_obj.save_cderi + '\'\n')
                fn.write(input_obj.name + 'df.build(j_only=True)\n') # j_only=True for DFT only !!!
                input_obj.load_cdetri = input_obj.save_cderi

            fn.write(method + '_' + input_obj.name + ' = ' + kmethod + '(' 
                    +  input_obj.name + ', kpts)\n')
            fn.write(method + '_' + input_obj.name + '.with_df = ' + input_obj.name + 'df\n')
            fn.write(method + '_' + input_obj.name + '.with_df._cderi = \'' + input_obj.load_cderi + '\'\n')
            if input_obj.B_grids: # or default uniform grids, not suitable for GDF 
                fn.write(method + '_' + input_obj.name + '.grids = pbcdft.BeckeGrids(' + input_obj.name + ')\n')
                if input_obj.grids_atom_grid:
                    fn.write(method + '_' + input_obj.name + '.grids.atom_grid = ' + input_obj.grids_atom_grid + '\n')
                else:
                    fn.write(method + '_' + input_obj.name + '.grids.level = ' + str(input_obj.grids_level) + '\n')
            else:
                pass
            

        elif input_obj.df == DF[2]:
            fn.write(input_obj.name + 'df = pbcmdf.' + input_obj.df.upper() + '(' + input_obj.name
                     + ', kpts)\n')
            # if input_obj.basis = ???  TO DO
            #    input_obj.load_cdetri
            if input_obj.save_cderi:
                fn.write(input_obj.name + 'df._cderi_to_save = \'' + input_obj.save_cderi + '\'\n')
                fn.write(input_obj.name + 'df.build(j_only=True)\n') # j_only=True for DFT only !!!
                input_obj.load_cdetri = input_obj.save_cderi

            fn.write(method + '_' + input_obj.name + ' = ' + kmethod + '(' 
                    +  input_obj.name + ', kpts)\n')
            fn.write(method + '_' + input_obj.name + '.with_df = ' + input_obj.name + 'df\n')
            fn.write(method + '_' + input_obj.name + '.with_df._cderi = \'' + input_obj.load_cdetri + '\'\n')
            if input_obj.B_grids: # or default uniform grids
                fn.write(method + '_' + input_obj.name + '.grids = pbcdft.BeckeGrids(' + input_obj.name + ')\n')
                if input_obj.grids_atom_grid:
                    fn.write(method + '_' + input_obj.name + '.grids.atom_grid = ' + input_obj.grids_atom_grid + '\n')
                else:
                    fn.write(method + '_' + input_obj.name + '.grids.level = ' + str(input_obj.grids_level) + '\n')
            else:
                pass

        else:
            fn.write(method + '_' + input_obj.name + ' = ' + kmethod + '(' 
                    +  input_obj.name + ', kpts)\n')

        if input_obj.method.lower() in METHODSET[0][0]:
            pass
        elif input_obj.method.lower() in METHODSET[0][1]:
            if input_obj.xc == None:
                xc = 'LDA,VWN'
            else:
                xc = input_obj.xc
            fn.write(method + '_' + input_obj.name +'.xc = ' '\'' + xc + '\'' ' # ^ Functional\n')
        if input_obj.Ndirect:
            fn.write(method + '_' + input_obj.name +'.Ndirect = '  + str(input_obj.Ndirect) +  '\n')
        if method == METHODSET[0][1][1]:
            fn.write('dma = dm[:,:x,:x]\n')
            fn.write('dmb = dm[:,x:,x:]\n')
            fn.write(method + '_' + input_obj.name + '.kernel(numpy.array([dma,dmb]))' '\n')
        elif method == METHODSET[0][1][3] or METHODSET[0][1][2]:
            fn.write(method + '_' + input_obj.name + '.kernel(dm)' '\n')
        fn.write('\n')
        
def get_filename(input_obj):
    chaosuan = input_obj.chaosuan.upper()
    xc = input_obj.xc
    # import pdb
    # pdb.set_trace()
    if input_obj.ecp:
        pp = input_obj.ecp # ecp name
    elif input_obj.pseudo:
        pp = input_obj.pseudo # pp name
    else:
        pp = 'None'
        
    if input_obj.method.lower() in METHODSET[0][0]:
        if xc == None:
            xc = 'Noxc'
        if input_obj.df == None:
            input_obj.df = DF[0]
        if input_obj.df.upper() == DF[0]:
            name = input_obj.name + '_' + input_obj.method + '_' + xc + '_B' + input_obj.basis  \
            + '_P' + str(pp) + '_DF' + '_N' + str(input_obj.Ndirect) + '_' + chaosuan + '.py'
            chkfile_name = input_obj.name + '_' + input_obj.method + '_' + xc + '_N' + str(input_obj.Ndirect)

        elif input_obj.df.upper() == 'GDF':
            name = input_obj.name + '_' + input_obj.method + '_' + xc + '_B' + input_obj.basis  \
            + '_P' + str(pp) + '_GDF' + '_N' + str(input_obj.Ndirect) + '_' + chaosuan + '.py'
            chkfile_name = input_obj.name + '_' + input_obj.method + '_' + xc + '_N' + str(input_obj.Ndirect)

        elif input_obj.df.upper() == 'MDF':
            name = input_obj.name + '_' + input_obj.method + '_' + xc + '_B' + input_obj.basis  \
            + '_P' + str(pp) + '_MDF' + '_N' + str(input_obj.Ndirect) + '_' + chaosuan + '.py'
            chkfile_name = input_obj.name + '_' + input_obj.method + '_' + xc + '_N' + str(input_obj.Ndirect)

    elif input_obj.method.lower() in METHODSET[0][1]:
        if xc == None:
            xc = 'LDA-VWN'
            if input_obj.df == None:
                input_obj.df = DF[0]

            if input_obj.df.upper()  == DF[0]:
                name = input_obj.name + '_' + input_obj.method + '_' + xc + '_B' + input_obj.basis  \
                      + '_P' + str(pp) + '_DF' + '_N' + str(input_obj.Ndirect) + '_' + chaosuan + '.py'
                chkfile_name = input_obj.name + '_' + input_obj.method + '_' + xc + '_N' + str(input_obj.Ndirect)

            elif input_obj.df.upper() == 'GDF':
                name = input_obj.name + '_' + input_obj.method + '_' + xc + '_B' + input_obj.basis  \
                      + '_P' + str(pp) + '_GDF' + '_N' + str(input_obj.Ndirect) + '_' + chaosuan + '.py'
                chkfile_name = input_obj.name + '_' + input_obj.method + '_' + xc + '_N' + str(input_obj.Ndirect)

            elif input_obj.df.upper() == 'MDF':
                name = input_obj.name + '_' + input_obj.method + '_' + xc + '_B' + input_obj.basis  \
                    + '_P' + str(pp) + '_MDF' + '_N' + str(input_obj.Ndirect) + '_' + chaosuan + '.py'
                chkfile_name = input_obj.name + '_' + input_obj.method + '_' + xc + '_N' + str(input_obj.Ndirect)
        else:
            if input_obj.df == None:
                input_obj.df = DF[0]
            if input_obj.df.upper() == DF[0]:
                name = input_obj.name + '_' + input_obj.method + '_' + xc + '_B' + input_obj.basis  \
                    + '_P' + str(pp) + '_DF' + '_N' + str(input_obj.Ndirect) + '_' + chaosuan + '.py'
                chkfile_name = input_obj.name + '_' + input_obj.method + '_' + xc + '_N' + str(input_obj.Ndirect)

            elif input_obj.df.upper() == 'GDF':
                name = input_obj.name + '_' + input_obj.method + '_' + xc + '_B' + input_obj.basis  \
                    + '_P' + str(pp) + '_GDF' + '_N' + str(input_obj.Ndirect) + '_' + chaosuan + '.py'
                chkfile_name = input_obj.name + '_' + input_obj.method + '_' + xc + '_N' + str(input_obj.Ndirect)

            elif input_obj.df.upper() == 'MDF':
                name = input_obj.name + '_' + input_obj.method + '_' + xc + '_B' + input_obj.basis  \
                + '_P' + str(pp) + '_MDF' + '_N' + str(input_obj.Ndirect) + '_' + chaosuan + '.py'
                chkfile_name = input_obj.name + '_' + input_obj.method + '_' + xc + '_N' + str(input_obj.Ndirect)
                
        chkfile_name = str(chkfile_name + '.chk')
    return name#, chkfile_name
    
def prt_final(input_obj, filename):           #, chkfile_name
    with open(filename, 'a+') as fn:
        fn.write('# ! Some post analysis\n')
        fn.write('\n')
        #fn.write('# ! Chkfile \n')
        #fn.write(str(input_obj.method.lower()) + '.chkfile = \'' + chkfile_name + '\'\n')
        fn.write('# ! Mo_energy analysis\n')
        fn.write('mo_energy = numpy.array(' + input_obj.method.lower() + '_' + input_obj.name + '.mo_energy) \n')
        fn.write('puwant.prt_mo_energy(mo_energy, kpts) \n' )
        fn.write('\n')

class input_file():
    def __init__(self):
        self.path = PYSDPATH['LH']
        self.name = 'diamond'
        self.atom =  'Li'
        self.number = 3
        self.spin = 0
        self.basis = 'cc-pvdz' 
        self.ecp = None
        self.pseudo = None
        self.bond = 2.88905 # Ag - 111 surface
        self.a = 'np.eye(3) * 3.5668'  # 'a' means lattice
        self.coords = None
        self.conv_tol = 1.0E-8
        self.max_cycle = 50
        self.verbose = 6
        self.grids_level = 3
        self.grids_atom_grid = False
        self.B_grids = True
        self.Ndirect = None
        self.xc = None  # it means 'LDA,VWN'
        self.exp_to_discard = 0.1
        self.save_cderi = False
        self.load_cderi = 'saved_cderi_h5' # There should be a file imported from a fixed location 
        self.make_kpts = [4, 4, 1]
        self.vacuum = 20
        self.rotatez_negative = False
        self.vortex = False
        self.precision = 1e-8
        self.method = 'KRHF'
        self.df = None
        self.cellout = './cell.txt'
        self.mo_energy_out = './mo_energy_out.txt'
        self.read = False
        self.chaosuan = 'LH'
        self.attr_general_dict = {
            'Name'                      : self.name,
            'Atom'                      : self.atom,
            'Bond'                      : self.bond,
            'Basis'                     : self.basis,
            'Pseudo'                    : self.pseudo,
            'Ecp'                       : self.ecp,
            'Functional'                : self.xc,
            'Grids_level'               : self.grids_level,
            'nKpts'                     : self.make_kpts,
            'Convergence tol'           : self.conv_tol,
            'Max scf cycle'             : self.max_cycle
        }
        self.attr_fix_dict = {
            'pyMC path'                 : self.path,
            'Verbose'                   : self.verbose,
            'Cell output file'          : self.cellout,
            #'Mo_energy output file'     : self.mo_energy_out
        }

    def create_attri_general_dict(self):
        self.attr_general_dict = {
            'Name'                      : self.name,
            'Atom'                      : self.atom,
            'Bond'                      : self.bond,
            'Basis'                     : self.basis,
            'Pseudo'                    : self.pseudo,
            'Ecp'                       : self.ecp,
            'Functional'                : self.xc,
            #'Lattice'                  : self.a,
            'Grids_level'               : self.grids_level,
            'nKpts'                     : self.make_kpts,
            'Convergence tol'           : self.conv_tol,
            'Max scf cycle'             : self.max_cycle
        }
        self.attr_fix_dict = {
            'pyMC path'             : self.path,
            'Verbose'               : self.verbose,
            'Cell output file'      : self.cellout,
            #'Mo_energy output file'     : self.mo_energy_out
        }
    
    def check_attributes(self):
        print(STARS)    
        self.prt_attr(self.attr_general_dict)
        print(STARS)
        self.prt_attr(self.attr_fix_dict)

    def prt_labels_attr(self, attr_general_dict):
        print('{0:<10} {1:<50}'.format('KeyNum.', 'Attr. Name'))
        for i, key in enumerate(attr_general_dict):
            print('{0:<10} {1:<50}'.format(i,key))
    
    def prt_attr(self, attr_dict):
        print('{0:<40} {0:<40}'.format('Attr. Name', 'Attr. Info.'))
        for i, key in enumerate(attr_dict):
            print('{0:<40} {1:<40}'.format(key, str(attr_dict[key])))
            

    def prt_input_file(self):
        self.create_attri_general_dict()
        self.check_attributes()
        filename = get_filename(self)  #, chkfile_name
        prt_import(self, filename)
        prt_molu(self, filename)
        get_mol_coords(self)
        prt_mol_coords(self, filename)
        prt_mol(self, filename)
        prt_dm_initial(self, filename)
        get_pbc_coords(self)
        prt_pbc_coords(self, filename)
        get_pbc_lattice(self)
        prt_pbc_lattice(self, filename)
        prt_cell(self, filename)
        prt_make_kpts(self, filename)
        prt_dm_transform(self, filename)
        prt_method_and_kernel(self, filename)
        prt_final(self,filename) #,chkfile_name)
    