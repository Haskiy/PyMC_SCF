#!/usr/bin/env python
'''
Author: Pu Zhichen
Date: 2021-05-24 10:45:19
LastEditTime: 2021-07-25 10:29:34
LastEditors: Pu Zhichen
Description: 
    This script is aiming at to generate the input file for Using in Chaosuan.
    Because there are many control parameters, a script to control, generate and CHECK the input file
    is of vital importance.
FilePath: \pyMC\tools\input_file_generator.py

 May the force be with you!
'''

import numpy

# METHODDICT saves all the methods that are implemented in pyMC.
# METHODDICT[2,2,4]
METHODDICT = [
    [
        ['scf.GHF', 'gksm_util.GKSM', 'gksm_util.GKSM_ibp'
         , 'gksm_util.GKSM_spha', 'gksm_util.GKSM_spha_ibp'] # GKS no symmetry
      , ['gksm_util.GHF_sym', 'gksm_util.GKSM', 'gksm_util.GKSM_ibp'
         , 'gksm_util.GKSM_spha', 'gksm_util.GKSM_spha_ibp'] # GKS symmetry
     ], # GKS
    [
        ['scf.dhf.DHF', 'gksm_util.GKSM_r', 'gksm_util.GKSM_r_ibp',
         'gksm_util.GKSM_r_spha', 'gksm_util.GKSM_r_ibp_spha'] # DKS no symmetry
      , ['gksm_util.gksm_r_debug.GKSM_r_symm', 'gksm_util.GKSM_r',
        'gksm_util.GKSM_r_ibp', 'gksm_util.GKSM_r_spha', 'gksm_util.GKSM_r_ibp_spha'] # DKS symmetry
     ]  # DKS
]
PYSDPATH = {"GAO" : "/public3/home/sc40234/Jobs_pzc",
            "JIANG" : "/public3/home/sc51754/members/pzc",
            "LH" : "/home/pzc"}
STARS = "***************************************************************************"
NPROC = {"GAO" : 64,
         "JIANG" : 64,
         "LH" : 40}

def prt_import(input_obj, filename):
    input_obj.path = PYSDPATH[input_obj.chaosuan.upper()]
    with open(filename, 'a+') as fn:
        fn.write('#!/usr/bin/env python\n')
        fn.write('from pyscf import gto,dft,scf,lib\n')
        fn.write('import numpy\n')
        fn.write('import sys\n')
        fn.write('# ! NOTE: should add the path of the pyMC module.\n')
        fn.write('sys.path.append(r"' + input_obj.path +'")\n')
        fn.write('from pyMC import tools as tools_hss\n')
        fn.write('from pyMC import grids_util\n')
        fn.write('from pyMC import gksm_util\n')
        fn.write('from pyMC import mole_sym\n')
        fn.write('lib.misc.num_threads(n=' + str(NPROC[input_obj.chaosuan.upper()]) + ')\n')
        fn.write('\n')

def prt_atom(input_obj, filename):
    with open(filename, 'a+') as fn:
        fn.write('# ! NOTE: Single atom part to generate initial density matrix\n')
        fn.write('### Atom bulid ###\n')
        fn.write('molu = gto.Mole()\n')
        fn.write('molu.verbose = ' + str(input_obj.verbose) + '\n')
        fn.write('molu.atom = "' + str(input_obj.atom) + ' 0 0 0" ###\n')
        fn.write('molu.spin=' + str(input_obj.spin)  + ' # ^ Spin\n')
        fn.write('molu.basis = "'+ input_obj.basis +'" # ^ Basis\n')
        fn.write('molu.symmetry=False ###\n')
        fn.write('molu.max_memory = 50000\n')   # ^ can change the max_memory here
        fn.write('molu.output = "'+ input_obj.atomout +'"\n')
        fn.write('molu.build()\n')
        fn.write('\n')
        fn.write('# ! NOTE: Single atom SCF calculation\n')
        if input_obj.Dirac:
            fn.write('mf = scf.dhf.DHF(molu)\n')
        else:
            fn.write('mf = scf.UHF(molu)\n')
        if input_obj.read:
            pass
        else:
            fn.write('mf.kernel()\n')
        fn.write('\n')
        
def get_filename(input_obj):
    chaosuan = input_obj.chaosuan.upper()
    if input_obj.Dirac:
        dirac = '4c'
    else:
        dirac = 'nr'
    if input_obj.Dsymmetry:
        symm = 'symm'
    else:
        symm = 'nosymm'
    if input_obj.vortex:
        vortex = 'vortex'
    else:
        vortex = 'no_vortex'
    xc = input_obj.xc
    if xc == None:
        xc = 'GHF'
    if input_obj.method.upper() == 'TRIDIRECTIONS':
        method = 'tri'
    elif input_obj.method.upper() == 'COLLINEAR':
        method = 'col'
    elif input_obj.method.upper() == 'MULTIDIRECTIONS':
        method = 'spha' + str(input_obj.Ndirect)
    if input_obj.fix:
        fix = 'fix'
    else:
        fix = 'nofix'
    name = input_obj.atom + str(input_obj.number) +'_' + method + '_' + xc.upper() + \
        '_' + dirac + '_' + symm + '_' + vortex + '_' + fix + '_' + chaosuan +\
        "{0:4.2f}".format(input_obj.bond) + '.py'
    
    return name

def get_coords(input_obj):
    input_obj.coords = []
    if input_obj.number == 3:
        input_obj.coords = [[ input_obj.bond/numpy.sqrt(3), 0.0, 0.0],
                            [-input_obj.bond/numpy.sqrt(3)*0.5, input_obj.bond*0.5, 0.0],
                            [-input_obj.bond/numpy.sqrt(3)*0.5,-input_obj.bond*0.5, 0.0]]

        
def prt_mol_coords(input_obj, filename):
    with open(filename, 'a+') as fn:
        fn.write('molcoords = """\n')
        for iatm in range(input_obj.number):
            fn.write('{0:<10}  {1:<20.15f}  {2:<20.15f}  {3:<20.15f} ;\n'.\
                format(input_obj.atom, input_obj.coords[iatm][0],\
                    input_obj.coords[iatm][1], input_obj.coords[iatm][2]))
        fn.write('"""\n')
        fn.write('\n')
        
        
def prt_mole(input_obj, filename):
    with open(filename, 'a+') as fn:
        fn.write('# ! NOTE: Create calculated clusters. \n')
        if input_obj.Dsymmetry:
            fn.write('mol = mole_sym.Mole_sym() \n')
            fn.write('mol.Dsymmetry = True\n')
            fn.write('mol.ovlp = S\n')
        else:
            fn.write('mol = gto.Mole() \n')
        fn.write('mol.verbose = ' + str(input_obj.verbose) + '\n')
        fn.write('mol.atom = molcoords \n')
        fn.write('mol.spin=' + str(input_obj.spin*input_obj.number)  + ' # ^ Spin\n')
        fn.write('mol.basis = "'+ input_obj.basis +'" # ^ Basis\n')
        fn.write('mol.output = "'+ input_obj.moleout +'"\n')
        fn.write('mol.max_memory = 50000 \n')  # ^ can change the max_memory here
        if input_obj.vortex:
            fn.write('mol.vortex = True \n')
        if input_obj.Dirac:
            fn.write('mol.dirac4c = True \n')
        if input_obj.Dsymmetry:
            fn.write('mol.build(singleatom = molu) \n')
        else:
            fn.write('mol.build() \n')
        fn.write('\n')
        
        
def prt_initial_dm(input_obj, filename):
    with open(filename, 'a+') as fn:
        if input_obj.read:
            fn.write('mo_coeffu = numpy.load("mo_coeff_init.npy")\n')
            fn.write('noccu = numpy.load("noccu.npy")\n')
        else:
            fn.write('mo_coeffu = mf.mo_coeff \n')
            fn.write('noccu = mf.mo_occ \n')
        fn.write('nao = mo_coeffu.shape[-1] \n')
        if input_obj.Dirac and input_obj.Mcorrect:
            fn.write('dm = mf.make_rdm1(mo_coeffu, noccu)\n')
            fn.write('mo_coeffu = tools_hss.rotate_utils2.get_z_oriented_atom(molu, mo_coeffu, dm)\n')
        fn.write('natom = mol.natm \n')
        fn.write('theta_dict = tools_hss.rotate_dm.get_init_guess_theta(mol, rotatez_negative = '+str(input_obj.rotatez_negative)+',vortex = '+str(input_obj.vortex)+') \n')
        fn.write('\n')
        
def prt_get_ini_mocoeff(input_obj, filename):
    with open(filename, 'a+') as fn:
        if not input_obj.Dirac:
            fn.write('mo_coeffu = tools_hss.rotate_dm.get_gks_dm_guess_mo(molu, mo_coeffu, natom\n')
            fn.write('                                , numpy.array([theta_dict[i][0] for i in range(3)]) \
                , rotatem = True, rotatel =  True)\n')
            fn.write('nocc = numpy.array((noccu[0].tolist()*natom+noccu[1].tolist()*natom))\n')
        else:
            fn.write('mo_coeffu = tools_hss.rotate_dm.get_gks_dm_guess_mo_4c(molu, mo_coeffu, natom, theta_dict)\n')
            fn.write('nocc = numpy.array((noccu[:molu.nao_2c()].tolist())*natom+(noccu[molu.nao_2c():].tolist())*natom)\n')
        fn.write('\n')    
        
def prt_cluster_S_mf(input_obj, filename):
    with open(filename, 'a+') as fn:
        fn.write('\n')
        fn.write('# ! NOTE: Create pre-calculated molecule to generate S matrix.\n')
        fn.write('molS = gto.Mole()\n')
        fn.write('molS.atom = molcoords\n')
        fn.write('molS.spin=' + str(input_obj.spin*input_obj.number) + ' # ^ Spin\n')
        fn.write('molS.basis = "' + input_obj.basis + '" # ^ Basis\n')
        fn.write('molS.max_memory = 50000\n')
        fn.write('molS.build()\n')
        if input_obj.Dirac:
            fn.write('mfs = scf.dhf.DHF(molS)\n')
        else:
            fn.write('mfs = scf.UHF(molS)\n')
        if input_obj.read:
            fn.write('S = numpy.load("ovlp.npy")\n')
        else:
            fn.write('mfs.kernel()\n')
            fn.write('mfs.max_cycle = 0\n')
            fn.write('S = mfs.get_ovlp()\n')
        fn.write('\n')

def prt_cluster_mf(input_obj, filename):
    if input_obj.Dirac:
        optDirac = 1
    else:
        optDirac = 0
    if input_obj.Dsymmetry:
        optsym = 1
    else:
        optsym = 0
    if input_obj.xc == None:
        optmethod = 0
    else:
        if input_obj.method.upper() == 'TRIDIRECTIONS':
            if input_obj.xc.upper() == 'SVWN' or input_obj.xc.upper() == 'LDA':
                optmethod = 1
            else:
                if input_obj.ibp:
                    optmethod = 2
                else:
                    optmethod = 1
        elif input_obj.method.upper() == 'COLLINEAR' or input_obj.method.upper() == 'MULTIDIRECTIONS':
            if input_obj.method.upper() == 'COLLINEAR':
                input_obj.Ndirect = 1
            if input_obj.xc.upper() == 'SVWN' or input_obj.xc.upper() == 'LDA':
                optmethod = 3
            else:
                if input_obj.ibp:
                    optmethod = 4
                else:
                    optmethod = 3
    method = METHODDICT[optDirac][optsym][optmethod]
        
    with open(filename, 'a+') as fn:
        fn.write('# ! NOTE: Molecule calculation part!\n')
        fn.write('# * NOTE: to get the Bxc and toque, IBP is a direct way.\n')
        if not input_obj.Dsymmetry:
            fn.write('mol.Dsymmetry = False\n')
        fn.write('mftot = ' + method + '(mol)\n')
        if not (input_obj.xc == None):
            fn.write('mftot.xc = "' + input_obj.xc + '" # ^ functional\n')
            fn.write('# ! NOTE: that the following part is Gauss-Legendre scheme.\n')
            fn.write('# * By changing the phi angle, changing the symmetry.\n')
            if isinstance(input_obj.grid, tuple):
                fn.write('dft.Grids.gen_atomic_grids = grids_util.Grids_hss.gen_atomic_grids_gauss_legendre\n')
                fn.write('mftot.grids.atom_grid = {"' + input_obj.atom + '" : ' + str(input_obj.grid) + '} # ^ (A, B, C) : A--> radial grid; B--> theta; C--> phi\n')
            else:
                fn.write('mftot.grids.level=' + str(input_obj.grid) + '\n')
        if input_obj.method.upper() != 'TRIDIRECTIONS': 
            fn.write('mftot.Ndirect=' + str(input_obj.Ndirect) + '\n')
        if input_obj.fix != None:
            if input_obj.vortex:
                fn.write('mftot.irrep_nelec = {\n')
                fn.write('    "1E12" : ' + str(input_obj.fix["1E12"]) + ',\n')
                fn.write('    "2E12" : ' + str(input_obj.fix["2E12"]) + ',\n')
                fn.write('    "A32" :' + str(input_obj.fix["A32"]) + '\n')
                fn.write('}\n')
            else:
                fn.write('mftot.irrep_nelec = {\n')
                fn.write('    "E12_1" : ' + str(input_obj.fix["E12_1"]) + ',\n')
                fn.write('    "E12_2" : ' + str(input_obj.fix["E12_2"]) + ',\n')
                fn.write('    "1E32" :' + str(input_obj.fix["1E32"]) + ',\n')
                fn.write('    "2E32" :' + str(input_obj.fix["2E32"]) + '\n')
                fn.write('}\n')
        fn.write('dm_tot  = mftot.make_rdm1(mo_coeff = mo_coeffu , mo_occ  = nocc)\n')
        fn.write('mftot.conv_tol = ' + str(input_obj.conv_tol) + ' # ^ tolerance\n')
        fn.write('mftot.max_cycle = ' + str(input_obj.max_cycle) + ' # ^ max cycle\n')
        fn.write('mftot.kernel(dm_tot)\n')
        fn.write('\n')
        
def prt_final(input_obj, filename):
    with open(filename, 'a+') as fn:
        fn.write('# ! Some post analysis\n')
        fn.write('mo_energy = mftot.mo_energy\n')
        fn.write('for i in range(mo_energy.shape[-1]):\n')
        fn.write('    print(mo_energy[i])\n')
        fn.write('dm_f = mftot.make_rdm1()\n')
        fn.write('numpy.savetxt("dm_f.txt",dm_f)\n')
        fn.write('mo_coeff = mftot.mo_coeff\n')
        fn.write('numpy.savetxt("mo_coeff.txt",mo_coeff)\n')
        fn.write('\n')
        
class input_file():
    def __init__(self):
        self.path = PYSDPATH['GAO']
        self.atom = 'Li'
        self.number = 3
        self.bond = 1.957966 # Angstrom
        self.xc = 'svwn'
        self.basis = 'cc-pvdz'
        self.spin = 1
        self.Dsymmetry = False
        self.Dirac = False
        self.vortex = False
        self.fix = None
        self.Mcorrect = True
        self.legendre = True
        self.grid = (80,60,60)
        self.conv_tol = 1.0E-8
        self.max_cycle = 50
        self.verbose = 6
        self.atomout = './atom.txt'
        self.moleout = './molecule.txt'
        self.method = 'Tridirections'
        self.Ndirect = 100
        self.read = False
        # ! ABORT ibp, default to be used.
        self.ibp = False
        self.chaosuan = 'Gao'
        self.rotatez_negative = False
        self.coords = None
        self.attr_general_dict = {
            'Atom'                  : self.atom,
            'Atom number'           : self.number,
            'Bond'                  : self.bond,
            'Functional'            : self.xc,
            'Basis'                 : self.basis,
            'Atom spin'             : self.spin,
            'Double group symmetry' : self.Dsymmetry,
            'Relativicity or not'   : self.Dirac,
            'Grids generator'       : self.legendre,
            'Grids level'           : self.grid,
            'Convergence tol'       : self.conv_tol,
            'rotatez_negative'      : self.rotatez_negative,
            'Max scf cycle'         : self.max_cycle
        }
        self.attr_4c = {
            'Vortex pattern'        : self.vortex
        }
        self.attr_GGA = {
            'Whether do ibp'        : self.ibp
        }
        self.attr_fix_dict = {
            'pyMC path'             : self.path,
            'Verbose'               : self.verbose,
            'Atom output file'      : self.atomout,
            'Molecule output file'  : self.moleout,
        }
        self.attr_dict = {
            'General'       : self.attr_general_dict,
            'Fix'           : self.attr_fix_dict
        } 
    
    def create_attri_dict(self):
        self.attr_general_dict = {
            'Atom'                  : self.atom,
            'Atom number'           : self.number,
            'Bond'                  : self.bond,
            'Functional'            : self.xc,
            'Basis'                 : self.basis,
            'Atom spin'             : self.spin,
            'Double group symmetry' : self.Dsymmetry,
            'Relativicity or not'   : self.Dirac,
            'Grids generator'       : self.legendre,
            'Grids level'           : self.grid,
            'Convergence tol'       : self.conv_tol,
            'rotatez_negative'      : self.rotatez_negative,
            'Max scf cycle'         : self.max_cycle
        }
        self.attr_fix_dict = {
            'pyMC path'             : self.path,
            'Verbose'               : self.verbose,
            'Atom output file'      : self.atomout,
            'Molecule output file'  : self.moleout,
        }
    
    def check_attributes(self):
        
        # * check Dirac4c
        if self.vortex == True:
            if self.Dirac == False:
                print('WARN: Vortex pattern can only be calculated from 4c calculations')
                print('WARN: DO 4C calculations.')
                # self.Dirac = True
            if self.Dsymmetry == False:
                print('WARN: Vortex pattern should be used with symmetry to ensures the correct pattern')
        if not self.Dirac:
            if self.rotatez_negative:
                print("WARN: Note that, in general, non-relativity calculations, atom has a z-pointed M vector.")
                
        # * check LDA and config
        if self.xc == None:
            print('GHF calculations')
        elif self.xc.upper() == 'SVWN' or self.xc.upper() == 'LDA':
            if self.ibp == True:
                print("WARN: LDA can't use ibp")
                self.ibp = False
        else:
            if self.ibp:
                print('WARN: GGA with ibp')
                if self.Dirac:
                    print('WARN: GGA with ibp')
                    # self.ibp = False
            else:
                print('WARN: GGA without ibp')
                if not self.Dirac:
                    print('WARN: For non-relative and relative calculations, ibp is recommended.\
                        Please change ibp to True, to get the Bxc.')
        print(STARS)        
        # * print all the settings
        self.prt_attr(self.attr_general_dict)
        print(STARS)
        self.prt_attr(self.attr_fix_dict)
        if self.method.upper() == 'MULTIDIRECTIONS':
            if self.xc == None:
                print("\033[7;31mERROR: GHF doesn't have Multidirections method.\033[1;31;40m")
            if self.Ndirect == 1:
                print("ERROR: Multidirections should have principle directions >1")
        elif self.method.upper() == 'COLLINEAR':
            if self.Ndirect != 1:
                print("ERROR: COLLINEAR should have principle directions =1")
                
        # * check Dirac4c settings.
        if self.Dirac:
            if not self.Mcorrect:
                print("WARN: It's strongly recommended that, correct the symmetry-broken M direction\
                    to the Z-axis. Thus, PLEASE SETTING THE Mcorrect to True !")
                
    
    def prt_general_infos(self):
        pass
    
    def get_coords(self):
        pass
    
    def prt_labels_attr(self,attr_dict):
        print('{0:<10} {1:<50}'.format('KeyNum.', 'Attr. Name'))
        for i, key in enumerate(attr_dict):
            print('{0:<10} {1:<50}'.format(i,key))
            
    def prt_attr(self, attr_dict):
        print('{0:<40} {0:<40}'.format('Attr. Name', 'Attr. Info.'))
        for i, key in enumerate(attr_dict):
            print('{0:<40} {1:<40}'.format(key, str(attr_dict[key])))
               
    def prt_default_info(self):
        pass

    def prt_input_file(self):
        self.create_attri_dict()
        self.check_attributes()
        filename = get_filename(self)
        get_coords(self)
        prt_import(self, filename)
        prt_atom(self, filename)
        prt_mol_coords(self, filename)
        if self.Dsymmetry:
            prt_cluster_S_mf(self, filename)
        prt_mole(self, filename)
        prt_initial_dm(self, filename)
        prt_get_ini_mocoeff(self, filename)
        prt_cluster_mf(self, filename)
        prt_final(self, filename)
        
    
    def change_one_attr(self, idx, attr):
        keys = list(attr.keys())
        print('Changing the ' + str(keys[idx]) + ' to')
        x = eval(input())
        attr[keys[idx]] = x
    
    
    def get_input_info(self):
        self.prt_general_infos()
        keep_change = True
        print("Fix some important infomations to be unchangable")
        self.prt_attr(self.attr_fix_dict)
        print("If you want to change above configurations? Please change the script.")
        print()
        print(STARS)
        print()
        print("Which part do you want to change?")
        self.prt_labels_attr(self.attr_general_dict)
        attr_len = self.attr_general_dict.__len__()
        while(keep_change):
            try:
                opt = int(input("Please input the index:   "))
                if opt > attr_len:
                    print('Invalid input!')
                    continue
            except:
                print("Not a number, reinput")
                print("Want to quit?")
                opt2 = input("press 0 to quit")
                if opt2 == 0:
                    exit()
                print('Invalid input!')
                continue
            self.change_one_attr(opt,self.attr_general_dict)   
        
            
            

    



