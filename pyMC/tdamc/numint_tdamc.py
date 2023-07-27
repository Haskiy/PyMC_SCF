#/usr/bin/env python
'''
Author: Li Hao
Date: 2022-03-14 21:17:55
LastEditTime: 2022-10-29 04:50:55
LastEditors: Li Hao
Description: This is the numerical integration file of Multi-Coliinear and Kubler TDA. 
             Define the calculation variables and functions.
FilePath: /pyMC/tdamc/numint_tdamc.py
Motto: A + B = C!
'''

import numpy
from pyscf.dft import numint 
from pyscf import lib
from pyMC.gksmc import numint_gksmc
from pyMC.lib import Spoints
import pylibxc

def excited_mag_structure(tdmf, idx, plot_ctrl=True, mulliken_ctrl=True):
    nao = tdmf.scf.mol.nao
    nso = 2*nao
    natm = tdmf.scf.mol.natm
    if tdmf.Extype.upper() == 'SPIN_FLIP_DOWN' or tdmf.Extype.upper() == 'SPIN_FLIP_UP':
        S = tdmf.scf.mol.intor('int1e_ovlp')
        mo_occ = tdmf.scf.mo_occ
        occidxa = numpy.where(mo_occ[0]>0)[0]
        occidxb = numpy.where(mo_occ[1]>0)[0]
        viridxa = numpy.where(mo_occ[0]==0)[0]
        viridxb = numpy.where(mo_occ[1]==0)[0]
        nocca = int(mo_occ[0].sum())
        noccb = int(mo_occ[1].sum())
        nvira = nao - nocca
        nvirb = nao - noccb
        Ca, Cb = tdmf.scf.mo_coeff
        Ca_vir = Ca[:,viridxa]
        Cb_occ = Cb[:,occidxb]
        Cb_vir = Cb[:,viridxb]
        Ca_occ = Ca[:,occidxa]
        
        dmaa,dmbb = tdmf.scf.make_rdm1()
        if tdmf.Extype.upper() == 'SPIN_FLIP_DOWN':
            Dhole = tdmf.U[:,idx].reshape(nocca,nvirb)@tdmf.U[:,idx].reshape(nocca,nvirb).conj().T ### just understand as a projection.
            Delec = tdmf.U[:,idx].reshape(nocca,nvirb).conj().T@tdmf.U[:,idx].reshape(nocca,nvirb)
            Delec = Delec.transpose(1,0)
            dmaa_new = dmaa - numpy.einsum('ij,ui,vj->uv',Dhole,Ca_occ,Ca_occ.conj())
            dmbb_new = dmbb + numpy.einsum('ab,ua,vb->uv',Delec,Cb_vir,Cb_vir.conj())
        elif tdmf.Extype.upper() == 'SPIN_FLIP_UP':
            Dhole = tdmf.U[:,idx].reshape(noccb,nvira)@tdmf.U[:,idx].reshape(noccb,nvira).conj().T
            Delec = tdmf.U[:,idx].reshape(noccb,nvira).conj().T@tdmf.U[:,idx].reshape(noccb,nvira)
            Delec = Delec.transpose(1,0)
            dmaa_new = dmaa + numpy.einsum('ij,ui,vj->uv',Delec,Ca_vir,Ca_vir.conj())
            dmbb_new = dmbb - numpy.einsum('ab,ua,vb->uv',Dhole,Cb_occ,Cb_occ.conj())
        
        if mulliken_ctrl:
            print("Mulliken popularization.")

            Na = numpy.zeros((natm))
            Nb = numpy.zeros((natm))
            Na_tot = dmaa_new@S # S = S.T
            Nb_tot = dmbb_new@S
            basis = tdmf.scf.mol.aoslice_by_atom()[:,-2:]
            for iatm in range(natm):
                idx0 = basis[iatm][0]
                idx1 = basis[iatm][1]
                Na[iatm] = numpy.trace(Na_tot[idx0:idx1,idx0:idx1])
                Nb[iatm] = numpy.trace(Nb_tot[idx0:idx1,idx0:idx1])
        else:
            Na = None
            Nb = None
        
        if plot_ctrl:
            print("Plot draw data.")
            ao = tdmf.scf._numint.eval_ao(tdmf.scf.mol, tdmf.scf.grids.coords, deriv=0)
            rhoa = tdmf.scf._numint.eval_rho(tdmf.scf.mol, ao, dmaa_new, xctype='LDA')
            rhob = tdmf.scf._numint.eval_rho(tdmf.scf.mol, ao, dmbb_new, xctype='LDA')
        else:
            rhoa = rhob = None
    
        return (dmaa_new, dmbb_new), (Na, Nb), (rhoa, rhob)
    
    elif tdmf.Extype.upper() == 'SPIN_CONSERVED':
        raise NotImplementedError("SPIN_CONSERVRD imlementation failed")
    
    elif tdmf.Extype.upper() == 'GKS' or tdmf.Extype == None:
        S = tdmf.scf.mol.intor('int1e_ovlp')
        mo_occ = tdmf.scf.mo_occ
        occidx = numpy.where(mo_occ>0)[0]
        viridx = numpy.where(mo_occ==0)[0]
        nocc = int(mo_occ.sum())
        nvir = nso - nocc
        
        C = tdmf.scf.mo_coeff
        C_vir = C[:,occidx]
        C_occ = C[:,viridx]
        
        # here may be a question because of the uks_to_gks.
        dms = tdmf.scf.make_rdm1()
        Dhole = tdmf.U[:,idx].reshape(nocc,nvir)@tdmf.U[:,idx].reshape(nocc,nvir).conj().T ### just understand as a projection.
        Delec = tdmf.U[:,idx].reshape(nocc,nvir).conj().T@tdmf.U[:,idx].reshape(nocc,nvir)
        Delec = Delec.transpose(1,0)
        dms_new = dms - numpy.einsum('ij,ui,vj->uv',Dhole,C_occ,C_occ.conj()) + numpy.einsum('ab,ua,vb->uv',Delec,C_vir,C_vir.conj())
         
        if mulliken_ctrl:
            print("Mulliken popularization.")
            N = numpy.zeros((natm))
            N_tot = dms_new@S # S = S.T
            basis = tdmf.scf.mol.aoslice_by_atom()[:,-2:]
            for iatm in range(natm):
                idx0 = basis[iatm][0]
                idx1 = basis[iatm][1]
                N[iatm] = numpy.trace(N_tot[idx0:idx1,idx0:idx1])
        else:
            N = None

        if plot_ctrl:
            print("Plot draw data.")
            ao = tdmf.scf._numint.eval_ao(tdmf.scf.mol, tdmf.scf.grids.coords, deriv=0)
            rho = tdmf.scf._numint.eval_rho(tdmf.scf.mol, ao, dmaa_new, xctype='LDA')
        else:
            rho = None
    
        return dms_new, N, rho

def spin_flip_deriv(self,xc_code,rhop,Ndirect,LIBXCT_factor=None,ncpu=None):
    r'''spin_flip_deriv: calculate the spin_flip_kernel for collinear TDA in Multi-Collinear approach, 
    with the gauss-legendre sample points and weights in the spin space. 
    
    Parameters
    ----------
    Args:
        xc_code : str
            Name of exchange-correlation functional.
        rhop : tuple
            Density and magenization density norm with form (rho,s), whrer rho, s with a shape (nvar,ngrid). 
            In LDA, GGA MGGA, nvar is 1,4,4 respectively, meaning 1, nabla_x, nabla_y, nabla_z.
        Ndirect : int
            The number of points in the spin space, for gauss-ledengre distribution.
    
    Kwargs:
        LIBXCT_factor : double or int: 
            The Threshold of for derivatives obatained from libxc.
            Deafult is None, which means no Threshold. 1e-10 is recommended.
        ncpu : int
            Number of cpu workers.
    
    Returns:
    ----------
        spin flip deriv : tuple
            For i_j in s_s,s_Ns,Ns_Nstmp,u_u,s_u,Ns_u, i_j means kernel \frac{\patial^2 f}{\partial i \partial j}.  
            Note Ns_Nstmp will be reshaped in collinear_tdamc_kernel().
    '''
    xctype = self._xc_type(xc_code)
    tarray, weight_array= numpy.polynomial.legendre.\
                    leggauss(Ndirect)
    weight_array *= 0.5
    tarray = tarray*0.5+0.5
    assert(numpy.abs(weight_array.sum()-1.0)<=1.0E-14)
    
    import multiprocessing
    if ncpu is None:
        ncpu = multiprocessing.cpu_count()
    import math
    nsbatch = math.ceil(Ndirect/ncpu)
    Ndirect_list = [(i, i+nsbatch) for i in range(0, Ndirect-nsbatch, nsbatch)]
    if Ndirect_list[-1][-1] < Ndirect:
        Ndirect_list.append((Ndirect_list[-1][-1], Ndirect))
    pool = multiprocessing.Pool()
    para_results = []
    
    if xctype == 'LDA':
        s_s = 0.0
        for para in Ndirect_list:
            # apply_async() only supports positional arguments.
            para_results.append(pool.apply_async(collinear_1d_kernel,
                                (self,xc_code,rhop,tarray[para[0]:para[1]],weight_array[para[0]:para[1]],
                                LIBXCT_factor)))
        pool.close()
        pool.join()
        
        for result_para in para_results:
            result = result_para.get()
            s_s += result
        return (s_s,)+(None,)*5
    
    elif xctype == 'GGA':    
        s_s = 0.0
        s_Ns = 0.0
        Ns_Nstmp = 0.0

        for para in Ndirect_list:
            para_results.append(pool.apply_async(collinear_1d_kernel,
                                (self,xc_code,rhop,tarray[para[0]:para[1]],weight_array[para[0]:para[1]],
                                LIBXCT_factor)))
        pool.close()
        pool.join()
        for result_para in para_results:
            result = result_para.get()
            s_s += result[0]
            s_Ns += result[1]
            Ns_Nstmp += result[2]
        return (s_s,s_Ns,Ns_Nstmp)+(None,)*3

    elif xctype == 'MGGA':
        s_s = 0.0
        s_Ns = 0.0
        Ns_Nstmp = 0.0
        u_u = 0.0
        s_u = 0.0
        Ns_u = 0.0
        
        for para in Ndirect_list:
            para_results.append(pool.apply_async(collinear_1d_kernel,
                                (self,xc_code,rhop,tarray[para[0]:para[1]],weight_array[para[0]:para[1]],
                                LIBXCT_factor)))

        pool.close()
        pool.join()
        for result_para in para_results:
            result = result_para.get()
            s_s += result[0]
            s_Ns += result[1]
            Ns_Nstmp += result[2]
            u_u += result[3]
            s_u += result[4]
            Ns_u += result[5]
        return s_s,s_Ns,Ns_Nstmp,u_u,s_u,Ns_u
    
def spin_conserved_deriv(self,xc_code,rhop,LIBXCT_factor=None):
    '''spin_conserved_deriv: calculate the spin_conserved_kernel for collinear TDA in Multi-Collinear
    approach.
    
    Parameters
    ----------
    Args:
        xc_code : str
            Name of xc functional.
        rhop : tuple 
            Density and magenization density norm.
    
    Kwargs:
        LIBXCT_factor : double or int)
            The Threshold of for derivatives obatained from libxc.
            Deafult is None, which means no Threshold. 1e-10 is recommended.
    
    Returns:
    ----------
        spin conserved deriv : tuple
            For i_j in n_n,n_s,s_s,n_Nn,n_Ns,s_Nn,s_Ns,Nn_Nntmp,Nn_Nstmp,Ns_Nstmp, i_j means kernel 
            \frac{\patial^2 f}{\partial i \partial j}.  
            Note Nn_Nntmp,Nn_Nstmp,Ns_Nstmp will be reshaped in collinear_tdamc_kernel().
    
    Raises:
    ----------
        ToDo : Spin conserved kernel in MGGA. 
    '''
    xctype = self._xc_type(xc_code)
    rho,s= rhop
    rhoa = 0.5*(rho + s)
    rhob = 0.5*(rho - s)
    if xctype == 'LDA':
        n_n = 0.0
        n_s = 0.0
        s_s = 0.0 
        
        LDA_kernel = self.eval_xc_collinear_kernel(xc_code,(rhoa, rhob),LIBXCT_factor=LIBXCT_factor)[0]
        n_n += LDA_kernel[0]
        n_s += LDA_kernel[1]
        s_s += LDA_kernel[2]
        return (n_n,n_s,s_s)+(None,)*7 
        
    elif xctype == 'GGA':
        n_n = 0.0
        n_s = 0.0
        n_Nn = 0.0
        n_Ns = 0.0
        s_s = 0.0
        s_Nn = 0.0
        s_Ns = 0.0
        Nn_Nntmp = 0.0
        Nn_Nstmp = 0.0
        Ns_Nstmp = 0.0
        
        GGA_kernel = self.eval_xc_collinear_kernel(xc_code,(rhoa, rhob),LIBXCT_factor=LIBXCT_factor)[0]
        n_n += GGA_kernel[0]
        n_s += GGA_kernel[1]
        n_Nn += GGA_kernel[2]
        n_Ns += GGA_kernel[3]
        s_s += GGA_kernel[4]
        s_Nn += GGA_kernel[5]
        s_Ns += GGA_kernel[6]
        Nn_Nntmp += GGA_kernel[7]
        Nn_Nstmp += GGA_kernel[8]
        Ns_Nstmp += GGA_kernel[9]
        return n_n,n_s,s_s,n_Nn,n_Ns,s_Nn,s_Ns,Nn_Nntmp,Nn_Nstmp,Ns_Nstmp
        
    elif xctype == 'MGGA':
        raise NotImplementedError("Spin-conserved scheme for collinear TDA not supports Meta-GGA")

def collinear_1d_kernel(self,xc_code,rhop,tarray,weight_array,LIBXCT_factor=None):
    '''collinear_1d_kernel: serves as a transformer to obtain spin flip part of kernel at strong polar points in
    numerical approach. 
    
    Parameters
    ----------
    Args:
        xc_code : str
            Name of exchange-correlation functional.
        rhop :tuple
            Density and magenization density norm with form (rho,s), whrer rho, s with a shape (nvar,ngrid). 
            In LDA, GGA MGGA, nvar is 1,4,4 respectively, meaning 1, nabla_x, nabla_y, nabla_z.
        tarray : numpy.array with shape (Ndirect,)
            Projection directions in gauss-legendre distribution.
        weight_array : numpy.array with shape (Ndirect,)
            Weights in gauss-legendre distribution.
    
    Kwargs:
        LIBXCT_factor : double or int
            The Threshold of for derivatives obatained from libxc.
            Deafult is None. Value Recommended is 1e-10.
            
    Returns:
    ----------
        spin flip part of kernel : tuple
            LDA: (s_s)
            GGA: (s_s,s_Ns,Ns_Nstmp)
            MGGA: (s_s,s_Ns,Ns_Nstmp,u_u,s_u,Ns_u)
    '''
    xctype = self._xc_type(xc_code)
    rho,s = rhop
    Ndirect = tarray.shape[-1]
    s_s = 0.0
    s_Ns = 0.0
    Ns_Nstmp = 0.0 
    u_u = 0.0
    s_u = 0.0
    Ns_u = 0.0
    
    if xctype == 'LDA':
        for idirect in range(Ndirect):
            rhoa = 0.5*(rho + s*tarray[idirect])
            rhob = 0.5*(rho - s*tarray[idirect])
            s_s += self.eval_xc_collinear_kernel(xc_code,(rhoa,rhob),LIBXCT_factor=LIBXCT_factor)[1]*\
                weight_array[idirect]
        return s_s
    
    elif xctype == 'GGA':
        for idirect in range(Ndirect):
            rhoa = 0.5*(rho + s*tarray[idirect])
            rhob = 0.5*(rho - s*tarray[idirect])
            GGA_kernel = self.eval_xc_collinear_kernel(xc_code,(rhoa,rhob),LIBXCT_factor=LIBXCT_factor)[1]
            s_s += GGA_kernel[0]*weight_array[idirect]
            s_Ns += GGA_kernel[1]*weight_array[idirect]
            Ns_Nstmp += GGA_kernel[2]*weight_array[idirect]
        return s_s,s_Ns,Ns_Nstmp
    
    elif xctype == 'MGGA':
        for idirect in range(Ndirect):
            rhoa = 0.5*(rho + s*tarray[idirect])
            rhob = 0.5*(rho - s*tarray[idirect])
            MGGA_kernel = self.eval_xc_collinear_kernel(xc_code,(rhoa,rhob),LIBXCT_factor=LIBXCT_factor)[1]
            s_s += MGGA_kernel[0]*weight_array[idirect]
            s_Ns += MGGA_kernel[1]*weight_array[idirect]
            Ns_Nstmp += MGGA_kernel[2]*weight_array[idirect]
            u_u += MGGA_kernel[3]*weight_array[idirect]
            s_u += MGGA_kernel[4]*weight_array[idirect]
            Ns_u += MGGA_kernel[5]*weight_array[idirect]
        return s_s,s_Ns,Ns_Nstmp,u_u,s_u,Ns_u

def Kubler_spin_flip_deriv(self,xc_code,rhop,LIBXCT_factor=None,KST_factor=1e-10):
    '''Kubler_spin_flip_deriv: calculate the spin_flip_kernel for collinear TDA in Locally Collinear
    approach. Only LDA is implemented.
    
    Parameters
    ----------
    Args:
        xc_code : str 
            Name of xc functional.
        rhop : tuole
            Density and magenization density norm.
    
    Kwargs:
        LIBXCT_factor : double or int
            The Threshold of for derivatives obatained from libxc.
        KST_factor : double or int
            The Threshold for s (magenization density norm), padding the instable case at s (as den) is 
            small. Deafult is 1e-10. Recommended is 1e-10.
    
    Returns:
    ----------
        Kubler_spin_flip_deriv : numpy.array
            Note vs uses the first derivative only.
    
    Raises:
    ----------
        ToDo : Kubler_spin_flip_deriv in GGA and MGGA.
    '''
    xctype = self._xc_type(xc_code)
    rho,Mz = rhop
    s = numpy.abs(Mz)
    if xctype == 'LDA':
        rhoa = 0.5*(rho + s)
        rhob = 0.5*(rho - s)
        vs,s_s = self.eval_xc_Kubler_kernel(xc_code,(rhoa, rhob),LIBXCT_factor=LIBXCT_factor)[1]
        idx_zero = s <= KST_factor
        with numpy.errstate(divide='ignore',invalid='ignore'):
            vs = vs/s
        # Substituted by collinear kernel.
        vs[idx_zero] = s_s[idx_zero]
        # Abandon instable term.
        # vs[idx_zero] = 0.0
        return vs
    else:
        raise NotImplementedError("Kubler scheme only supports for LDA")
    
def Kubler_spin_conserved_deriv(self,xc_code,rhop,LIBXCT_factor=None):
    '''Kubler_spin_conserved_deriv: calculate the spin_conserved_kernel for collinear TDA in Locally Collinear
    approach. Only LDA is implemented.
    
    Parameters
    ----------
    Args:
        xc_code : str
            Name of xc functional.
        rhop : tuple
            Density and magenization density norm.
    
    Kwargs:
        LIBXCT_factor : double or int
            The Threshold of for derivatives obatained from libxc.
    
    Returns:
    ----------
        Kubler_spin_conserved_deriv : tuple
            For i_j in n_n,n_s,s_s, i_j means kernel \frac{\patial^2 f}{\partial i \partial j}.  
    
    Raises:
    ----------
        ToDo : Kubler_spin_conserved_deriv in GGA and MGGA. 
    '''
    xctype = self._xc_type(xc_code)
    rho,Mz = rhop
    s = numpy.abs(Mz)
    if xctype == 'LDA':
        rhoa = 0.5*(rho + s)
        rhob = 0.5*(rho - s)
        n_n,n_s,s_s,vs = self.eval_xc_Kubler_kernel(xc_code,(rhoa, rhob),LIBXCT_factor=LIBXCT_factor)[0]
        n_s *= Mz/s
        # s_s *=  (Mz/s)**2
        return n_n,n_s,s_s
    else:
        raise NotImplementedError("Kubler scheme only supports for LDA")

def collinear_tdalc_kernel(self,xc_code,rhop,Extype='SPIN_CONSERVED',LIBXCT_factor=None,KST_factor=1e-10):
    '''collinear_tdalc_kernel: serves as a branch to obtain spin_flip_deriv and spin_conserved_deriv in Locally
    Collinear approach.
    '''
    # import pdb
    # pdb.set_trace()
    xctype = self._xc_type(xc_code)
    if xctype != 'LDA':
        raise ValueError("Kubler method can only calculate!")
    if Extype=='SPIN_FLIP_UP' or Extype=='SPIN_FLIP_DOWN':
        s_s = Kubler_spin_flip_deriv(self,xc_code,rhop,LIBXCT_factor=LIBXCT_factor,KST_factor=KST_factor)
        return s_s
    
    elif Extype=='SPIN_CONSERVED':
        n_n,n_s,s_s = Kubler_spin_conserved_deriv(self,xc_code,rhop,LIBXCT_factor=LIBXCT_factor)
        return n_n,n_s,s_s
            
def collinear_tdamc_kernel(self,xc_code,rhop,Ndirect,Extype='SPIN_CONSERVED',LIBXCT_factor=None,ncpu=None):
    '''collinear_tdamc_kernel: serves as a branch to obtain spin_flip_deriv and spin_conserved_deriv in Multi-
    Collinear approach. Nn_Nntmp,Nn_Nstmp,Ns_Nstmp is reshaped here.
    '''
    xctype = self._xc_type(xc_code)
    if Extype=='SPIN_FLIP_UP' or Extype=='SPIN_FLIP_DOWN':
        s_s,s_Ns,Ns_Nstmp,u_u,s_u,Ns_u = \
            spin_flip_deriv(self,xc_code,rhop,Ndirect,LIBXCT_factor=LIBXCT_factor,ncpu=ncpu)
            
        if xctype == 'LDA':
            return s_s
        elif xctype == 'GGA' or xctype == 'MGGA':
            ngrid = s_s.shape[-1]
            Ns_Ns = numpy.zeros((3,3,ngrid))
            Ns_Ns[0,0] = Ns_Nstmp[0]
            Ns_Ns[0,1:3] = Ns_Nstmp[1:3]
            Ns_Ns[1:3,0] = Ns_Nstmp[1:3]
            Ns_Ns[1,1] = Ns_Nstmp[3]
            Ns_Ns[1,2] = Ns_Nstmp[4]
            Ns_Ns[2,1] = Ns_Nstmp[4]
            Ns_Ns[2,2] = Ns_Nstmp[5]
            if xctype == 'GGA':
                return s_s, s_Ns, Ns_Ns
            elif xctype == 'MGGA':
                return s_s, s_Ns, Ns_Ns, u_u, s_u, Ns_u
        
    elif Extype=='SPIN_CONSERVED':
        n_n,n_s,s_s,n_Nn,n_Ns,s_Nn,s_Ns,Nn_Nntmp,Nn_Nstmp,Ns_Nstmp = \
            spin_conserved_deriv(self,xc_code,rhop,LIBXCT_factor=LIBXCT_factor)
        if xctype == 'LDA':
            return n_n,n_s,s_s
        elif xctype == 'GGA':
            ngrid = s_s.shape[-1]
            Nn_Nn = numpy.zeros((3,3,ngrid))
            Ns_Ns = numpy.zeros((3,3,ngrid))
            Nn_Nn[0,0] = Nn_Nntmp[0]
            Nn_Nn[0,1:3] = Nn_Nntmp[1:3]
            Nn_Nn[1:3,0] = Nn_Nntmp[1:3]
            Nn_Nn[1,1] = Nn_Nntmp[3]
            Nn_Nn[1,2] = Nn_Nntmp[4]
            Nn_Nn[2,1] = Nn_Nntmp[4]
            Nn_Nn[2,2] = Nn_Nntmp[5]
            # Nn_Ns may be stored as Nn_Nn, just need (6,3,ngrid), but
            # (3,3,3,ngrid) is more mathametical.
            Nn_Ns = Nn_Nstmp
            Ns_Ns[0,0] = Ns_Nstmp[0]
            Ns_Ns[0,1:3] = Ns_Nstmp[1:3]
            Ns_Ns[1:3,0] = Ns_Nstmp[1:3]
            Ns_Ns[1,1] = Ns_Nstmp[3]
            Ns_Ns[1,2] = Ns_Nstmp[4]
            Ns_Ns[2,1] = Ns_Nstmp[4]
            Ns_Ns[2,2] = Ns_Nstmp[5]
            return n_n,n_s,n_Nn,n_Ns,s_s,s_Nn,s_Ns,Nn_Nn,Nn_Ns,Ns_Ns
        
        elif xctype == 'MGGA':
            raise NotImplementedError("Spin-conserve scheme not suppports Meta-GGA")
    
def nr_collinear_tdamc(self,xc_code,rhop,grids,Ndirect,C_mo,Extype='SPIN_CONSERVED',LIBXCT_factor=None,ncpu=None):
    '''nr_collinear_tda: calculates the K_aibj for Multi-Collinear TDA for collinear system. 
       
    Parameters
    ----------
    Args:
        xc_code : str
            Name of exchange-correlation functional.
        rhop : tuple
            Density and magenization density norm with form (rho,s), whrer rho,s with a shape (nvar,ngrid). 
            In LDA, GGA MGGA, nvar is 1,4,4 respectively, meaning 1, nabla_x, nabla_y, nabla_z.
        grids : mf.object
            mf is an object for DFT class and grids.coords and grids.weights are used here, supplying the sample 
            points and weights in real space.
        Ndirect : int
            The number of sample points in spin space, for gauss-ledengre distribution.
        C_mo : tuple
            Molecular orbital cofficience.
    
    Kwargs:
        Extype : str
            Three excited energy types -> SPIN_FLIP_UP, SPIN_FLIP_DOWN, SPIN_CONSERVED.
        LIBXCT_factor : double or int
            The Threshold of for derivatives obatained from libxc.
            Deafult is None, which means no Threshold. 1e-10 is recommended.
        ncpu : 
            Number of cpu workers.
    
    Returns:
    ----------
        K_aibj : numpy.array for SPIN_FLIP excited type and tuple for SPIN_CONSERVED excited type.
            aibj means related orbitals. a,b are virtual orbitals, and i,j are occupied orbitals.
                
    Raises:
    ----------
        ToDo : SPIN_CONSERVED MGGA collinear TDA.
    '''
    xctype = self._xc_type(xc_code)
    ngrid = grids.coords.shape[0]
    weights = grids.weights
    
    import multiprocessing
    if ncpu is None:
        ncpu = multiprocessing.cpu_count()
    import math
    nsbatch = math.ceil(ngrid/ncpu)
    NX_list = [(i, i+nsbatch) for i in range(0, ngrid-nsbatch, nsbatch)]
    if NX_list[-1][-1] < ngrid:
        NX_list.append((NX_list[-1][-1], ngrid))
    pool = multiprocessing.Pool()
    para_results = []
    
    # import pdb
    # pdb.set_trace()
    if Extype=='SPIN_FLIP_UP' or Extype=='SPIN_FLIP_DOWN':
        mo_vir,mo_occ = C_mo
        K_aibj = 0.0

        kernel = collinear_tdamc_kernel(self,xc_code,rhop,Ndirect,Extype=Extype,
                                  LIBXCT_factor=LIBXCT_factor,ncpu=ncpu)
        
        if xctype == 'LDA':
            for para in NX_list:
                idxi,idxf = para
                para_results.append(pool.apply_async(K_aibj_collinear_generator_flip,
                                    (xctype, mo_vir[idxi:idxf], mo_occ[idxi:idxf], 
                                    kernel[idxi:idxf], weights[idxi:idxf])))
            pool.close()
            pool.join()
            
        elif xctype == 'GGA' or xctype == 'MGGA':
            for para in NX_list:
                idxi,idxf = para
                kernel_para = []
                for i in range(len(kernel)):
                    kernel_para.append(kernel[i][...,idxi:idxf])
                para_results.append(pool.apply_async(K_aibj_collinear_generator_flip,
                                    (xctype, mo_vir[:,idxi:idxf], mo_occ[:,idxi:idxf], 
                                    kernel_para, weights[idxi:idxf])))
            pool.close()
            pool.join()
        
        for result_para in para_results:
            result = result_para.get()
            K_aibj += result
        return K_aibj
                 
    elif Extype=='SPIN_CONSERVED':
        mo_a_vir,mo_a_occ,mo_b_vir,mo_b_occ = C_mo 
        K_aibj_aaaa = 0.0
        K_aibj_aabb = 0.0
        K_aibj_bbaa = 0.0
        K_aibj_bbbb = 0.0
        kernel = collinear_tdamc_kernel(self,xc_code,rhop,Ndirect,Extype=Extype,
                                        LIBXCT_factor=LIBXCT_factor,ncpu=ncpu)
        # import pdb
        # pdb.set_trace()
        if xctype == 'LDA':
            for para in NX_list:
                idxi,idxf = para
                kernel_para = []
                for i in range(len(kernel)):
                    kernel_para.append(kernel[i][...,idxi:idxf])
                para_results.append(pool.apply_async(K_aibj_collinear_generator_conserved,
                                    (xctype, mo_a_vir[idxi:idxf],mo_a_occ[idxi:idxf],
                                    mo_b_vir[idxi:idxf],mo_b_occ[idxi:idxf], 
                                    kernel_para, weights[idxi:idxf])))
            pool.close()
            pool.join()
        
        elif xctype == 'GGA':
            for para in NX_list:
                idxi,idxf = para
                kernel_para = []
                for i in range(len(kernel)):
                    kernel_para.append(kernel[i][...,idxi:idxf])
                para_results.append(pool.apply_async(K_aibj_collinear_generator_conserved,
                                    (xctype, mo_a_vir[:,idxi:idxf],mo_a_occ[:,idxi:idxf],
                                    mo_b_vir[:,idxi:idxf],mo_b_occ[:,idxi:idxf], 
                                    kernel_para, weights[idxi:idxf])))    
            pool.close()
            pool.join()
        
        elif xctype == 'MGGA':
            raise NotImplementedError("Spin-conserved scheme isn't implemented in Meta-GGA")
        
        # import pdb
        # pdb.set_trace()
        for result_para in para_results:
            result = result_para.get()
            K_aibj_aaaa += result[0]
            K_aibj_aabb += result[1]
            K_aibj_bbaa += result[2]
            K_aibj_bbbb += result[3]
        # import pdb
        # pdb.set_trace()
            
        return K_aibj_aaaa,K_aibj_aabb,K_aibj_bbaa,K_aibj_bbbb

def nr_collinear_tdalc(self,xc_code,rhop,grids,C_mo,Extype='SPIN_CONSERVED',LIBXCT_factor=None,
                       KST_factor=1e-10,ncpu=None):
    '''nr_collinear_tdalc: calculates the K_aibj for Locally Collinear TDA for collinear system.
    
    Parameters
    ----------
    Args:
        xc_code : str
            Name of exchange-correlation functional.
        rhop : _tuple
            Density and magenization density norm with form (rho,s), whrer rho, s with a shape (nvar,ngrid). 
            In LDA, GGA MGGA, nvar is 1,4,4 respectively, meaning 1, nabla_x, nabla_y, nabla_z.
        grids : mf.object
            mf is an object for DFT class and grids.coords and grids.weights are used here, supplying the sample 
            points and weights in real space.
        C_mo : tuple
            Molecular orbital cofficience. 
            C_mo = (mo_a_vir,mo_a_occ,mo_b_vir,mo_b_occ), which means:
                Virtual alpha molecular orbitals, occupied alpha molecular orbitals, 
                Virtual beta molecular orbitals, occupied beta molecular orbitals, respectively.
    
    Kwargs:
        Extype : str
            Three excited energy types -> SPIN_FLIP_UP, SPIN_FLIP_DOWN, SPIN_CONSERVED.
        LIBXCT_factor : double or int
            The Threshold of for derivatives obatained from libxc.
            Deafult is None, which means no Threshold. 1e-10 is recommended.
        KST_factor : double or int: 
            The Threshold of for derivatives obatained from libxc.
            Deafult is 1e-10. Value recommended is 1e-10.
        ncpu : int
            Number of cpu workers.
    
    Returns:
    ----------
       K_aibj : numpy.array for SPIN_FLIP excited type and tuple for SPIN_CONSERVED excited type.
            aibj means related orbitals. a,b are virtual orbitals, and i,j are occupied orbitals.
    '''
    xctype = self._xc_type(xc_code)
    ngrid = grids.coords.shape[0]
    weights = grids.weights
    
    import multiprocessing
    if ncpu is None:
        ncpu = multiprocessing.cpu_count()
    import math
    nsbatch = math.ceil(ngrid/ncpu)
    NX_list = [(i, i+nsbatch) for i in range(0, ngrid-nsbatch, nsbatch)]
    if NX_list[-1][-1] < ngrid:
        NX_list.append((NX_list[-1][-1], ngrid))
    pool = multiprocessing.Pool()
    para_results = []
    
    if Extype=='SPIN_FLIP_UP' or Extype=='SPIN_FLIP_DOWN':
        mo_vir,mo_occ = C_mo
        K_aibj = 0.0
        kernel = collinear_tdalc_kernel(self,xc_code,rhop,Extype=Extype,LIBXCT_factor=LIBXCT_factor,
                                        KST_factor=KST_factor)
        
        if xctype == 'LDA':
            for para in NX_list:
                idxi,idxf = para
                para_results.append(pool.apply_async(K_aibj_collinear_generator_flip,
                                    (xctype, mo_vir[idxi:idxf], mo_occ[idxi:idxf], 
                                    kernel[idxi:idxf], weights[idxi:idxf])))
            pool.close()
            pool.join()
            
        for result_para in para_results:
            result = result_para.get()
            K_aibj += result
        return K_aibj

    elif Extype=='SPIN_CONSERVED':
        mo_a_vir,mo_a_occ,mo_b_vir,mo_b_occ = C_mo 
        K_aibj_aaaa = 0.0
        K_aibj_aabb = 0.0
        K_aibj_bbaa = 0.0
        K_aibj_bbbb = 0.0
        kernel = collinear_tdalc_kernel(self,xc_code,rhop,Extype=Extype,LIBXCT_factor=LIBXCT_factor)
        if xctype == 'LDA':
            for para in NX_list:
                idxi,idxf = para
                kernel_para = []
                for i in range(len(kernel)):
                    kernel_para.append(kernel[i][...,idxi:idxf])
                para_results.append(pool.apply_async(K_aibj_collinear_generator_conserved,
                                    (xctype, mo_a_vir[idxi:idxf],mo_a_occ[idxi:idxf],
                                    mo_b_vir[idxi:idxf],mo_b_occ[idxi:idxf], 
                                    kernel_para, weights[idxi:idxf])))
            pool.close()
            pool.join()
    #     import pdb
    #     pdb.set_trace()
        for result_para in para_results:
            result = result_para.get()
            K_aibj_aaaa += result[0]
            K_aibj_aabb += result[1]
            K_aibj_bbaa += result[2]
            K_aibj_bbbb += result[3]
        return K_aibj_aaaa,K_aibj_aabb,K_aibj_bbaa,K_aibj_bbbb  
            
def K_aibj_collinear_generator_flip(xctype, mo_a_vir, mo_b_occ, kernel, weights):
    '''K_aibj_collinear_generator_flip: calculates <ai|kernel|bj> for collinear spin flip term. 
    
    Parameters
    ----------
    Args:
        xctype : str
            xctype -> LDA, GGA, MGGA. 
        mo_a_vir : numpy.array
            Virtual alpha molecular orbitals. 
        mo_b_occ : numpy.array
            Occupied beta molecular orbitals.
        kernel : tuple
            Spin_flip_kernel. In LDA, GGA and MGGA, len(kernel) = 1, 3, 6, respectively.
        weights : numpy.array
            Weights of sample points in real space.
  
    Returns:
    ----------
        K_aibj : numpy.array
    '''
    if xctype == 'LDA':
        # construct gks ab(ba) blocks, ai means orbital a to orbital i
        ai_ab = numpy.einsum('na,ni->nai',mo_a_vir.conj(),mo_b_occ,optimize=True)
        
        s_s = kernel
        s_s *= weights
        
        # s_s part
        K_aibj = numpy.einsum('n,nai,nbj->aibj',s_s, ai_ab, ai_ab.conj(),optimize=True)
        
    elif xctype == 'GGA':
        # construct gks ab(ba) blocks, ai means orbital a to orbital i
        ai_ab = numpy.einsum('na,ni->nai',mo_a_vir[0].conj(),mo_b_occ[0],optimize=True)
        
        # construct gradient terms
        ai_na_b = numpy.einsum('gna,ni->gnai',mo_a_vir[1:4].conj(),mo_b_occ[0],optimize=True)
        ai_a_nb = numpy.einsum('na,gni->gnai',mo_a_vir[0].conj(),mo_b_occ[1:4],optimize=True)
        
        # construct nrho,nMz blocks to calculate K_aibj
        ai_nMx = ai_na_b + ai_a_nb
        
        s_s, s_Ns, Ns_Ns =  kernel
        s_s *= weights
        s_Ns *= weights
        Ns_Ns *= weights
        
        # s_s part
        K_aibj = numpy.einsum('n,nai,nbj->aibj',s_s, ai_ab, ai_ab.conj(),optimize=True)
        
        # Ns_s part
        K_aibj+= numpy.einsum('gn,gnai,nbj->aibj',
                             s_Ns, ai_nMx.conj(), 
                              ai_ab.conj(),optimize=True)
        # s_Ns part
        K_aibj+= numpy.einsum('gn,nai,gnbj->aibj',s_Ns, ai_ab.conj(), 
                              ai_nMx.conj(),optimize=True)
        
        # Ns_Ns part
        K_aibj+= numpy.einsum('ghn,gnai,hnbj->aibj',Ns_Ns, ai_nMx.conj(), 
                              ai_nMx.conj(),optimize=True)

    elif xctype == 'MGGA':
#         print('aaa')
        # construct gks ab(ba) blocks, ai means orbital a to orbital i
        ai_ab = numpy.einsum('na,ni->nai',mo_a_vir[0].conj(),mo_b_occ[0],optimize=True)
        
        # construct gradient terms
        ai_na_b = numpy.einsum('gna,ni->gnai',mo_a_vir[1:4].conj(),mo_b_occ[0],optimize=True)
        ai_a_nb = numpy.einsum('na,gni->gnai',mo_a_vir[0].conj(),mo_b_occ[1:4],optimize=True)
        ai_na_nb = 0.5*numpy.einsum('gna,gni->nai',mo_a_vir[1:4].conj(),mo_b_occ[1:4],optimize=True)
        
        # construct nMx,tau blocks to calculate K_aibj
        ai_nMx = ai_na_b + ai_a_nb
          
        s_s, s_Ns, Ns_Ns,u_u,s_u, Ns_u = kernel
        s_s *= weights
        s_Ns *= weights
        Ns_Ns *= weights
        u_u *= weights
        s_u *= weights 
        Ns_u *= weights
        
        # s_s part
        K_aibj = numpy.einsum('n,nai,nbj->aibj',s_s,ai_ab,ai_ab.conj(),optimize=True)
        
        # Ns_s part
        K_aibj+= numpy.einsum('gn,gnai,nbj->aibj',s_Ns,ai_nMx,ai_ab.conj(),optimize=True)
        
        # s_Ns part
        K_aibj+= numpy.einsum('gn,nai,gnbj->aibj',s_Ns,ai_ab, ai_nMx.conj(),optimize=True)
        
        # Ns_Ns part
        K_aibj+= numpy.einsum('ghn,gnai,hnbj->aibj',Ns_Ns,ai_nMx,ai_nMx.conj(),optimize=True)

        # u_u part
        K_aibj += numpy.einsum('n,nai,nbj->aibj',u_u, ai_na_nb,ai_na_nb.conj(),optimize=True)

        # s_u part
        K_aibj += numpy.einsum('n,nai,nbj->aibj',s_u, ai_ab, ai_na_nb.conj(),optimize=True)
        
        # u_s part
        K_aibj += numpy.einsum('n,nai,nbj->aibj',s_u, ai_na_nb,ai_ab.conj(),optimize=True)
        
        # Ns_u part
        K_aibj+= numpy.einsum('gn,gnai,nbj->aibj',Ns_u, ai_nMx, ai_na_nb.conj(),optimize=True)
        
        # u_Ns part
        K_aibj+= numpy.einsum('gn,nai,gnbj->aibj',Ns_u, ai_na_nb, ai_nMx.conj(),optimize=True)
        
    else:
        raise NotImplementedError("Please check the xc_code keyword")
        
    return K_aibj

def K_aibj_collinear_generator_conserved(xctype, mo_a_vir,mo_a_occ,mo_b_vir,mo_b_occ, kernel, weights):
    '''K_aibj_collinear_generator_conserved: calculates <ai|kernel|bj> for collinear spin conserved term.
    
    Parameters
    ----------
    Args:
        xctype : str
            xctype -> LDA, GGA, MGGA. 
        mo_a_vir,mo_a_occ,mo_b_vir,mo_b_occ :  numpy.array
            Virtual alpha molecular orbitals, occupied alpha molecular orbitals, 
            Virtual beta molecular orbitals, occupied beta molecular orbitals, respectively.
        kernel : tuple
            Spin_flip_kernel. In LDA and GGA, len(kernel) = 3, 10, respectively.
        weights : numpy.array
            Weights of sample points in real space.
    
    Returns:
    ----------
        K_aibj : numpy.array
            K_aibj.len()=4, including K_aibj_aaaa,K_aibj_aabb,K_aibj_bbaa,K_aibj_bbbb.
    
    Raises:
    ----------
        ToDo : Spin_conserved in MGGA.
    '''
    if xctype == 'LDA':
        n_n,n_s,s_s = kernel
        n_n *= weights
        n_s *= weights
        s_s *= weights
        
        # construct gks aa,ab,ba,bb blocks, ai means orbital a to orbital i
        # Substitute ai_rho with ai_aa and ai_bb
        ai_aa = numpy.einsum('na,ni->nai',mo_a_vir.conj(),mo_a_occ,optimize=True)
        ai_bb = numpy.einsum('na,ni->nai',mo_b_vir.conj(),mo_b_occ,optimize=True)
    
        # calculate K_aibj
        K_aibj_aaaa = 0.0
        K_aibj_aabb = 0.0
        K_aibj_bbaa = 0.0
        K_aibj_bbbb = 0.0
        
        # nn
        K_aibj_aaaa += numpy.einsum('n,nai,nbj->aibj',n_n,ai_aa,ai_aa.conj(),optimize=True)   
        K_aibj_aabb += numpy.einsum('n,nai,nbj->aibj',n_n,ai_aa,ai_bb.conj(),optimize=True)
        K_aibj_bbaa += numpy.einsum('n,nai,nbj->aibj',n_n,ai_bb,ai_aa.conj(),optimize=True)
        K_aibj_bbbb += numpy.einsum('n,nai,nbj->aibj',n_n,ai_bb,ai_bb.conj(),optimize=True)
        
        # ns
        K_aibj_aaaa += numpy.einsum('n,nai,nbj->aibj',n_s,ai_aa,ai_aa.conj(),optimize=True)  
        K_aibj_aabb += -1*numpy.einsum('n,nai,nbj->aibj',n_s,ai_aa,ai_bb.conj(),optimize=True)
        K_aibj_bbaa += numpy.einsum('n,nai,nbj->aibj',n_s,ai_bb,ai_aa.conj(),optimize=True)
        K_aibj_bbbb += -1*numpy.einsum('n,nai,nbj->aibj',n_s,ai_bb,ai_bb.conj(),optimize=True)
        
        # sn
        K_aibj_aaaa += numpy.einsum('n,nai,nbj->aibj',n_s,ai_aa,ai_aa.conj(),optimize=True)  
        K_aibj_aabb += numpy.einsum('n,nai,nbj->aibj',n_s,ai_aa,ai_bb.conj(),optimize=True)
        K_aibj_bbaa += -1*numpy.einsum('n,nai,nbj->aibj',n_s,ai_bb,ai_aa.conj(),optimize=True)
        K_aibj_bbbb += -1*numpy.einsum('n,nai,nbj->aibj',n_s,ai_bb,ai_bb.conj(),optimize=True)
 
        # ss
        K_aibj_aaaa += numpy.einsum('n,nai,nbj->aibj',s_s,ai_aa,ai_aa.conj(),optimize=True)  
        K_aibj_aabb += -1*numpy.einsum('n,nai,nbj->aibj',s_s,ai_aa,ai_bb.conj(),optimize=True)
        K_aibj_bbaa += -1*numpy.einsum('n,nai,nbj->aibj',s_s,ai_bb,ai_aa.conj(),optimize=True)
        K_aibj_bbbb += numpy.einsum('n,nai,nbj->aibj',s_s,ai_bb,ai_bb.conj(),optimize=True)
        
    elif xctype == 'GGA':
        n_n,n_s,n_Nn,n_Ns,s_s,s_Nn,s_Ns,Nn_Nn,Nn_Ns,Ns_Ns = kernel
        n_n *= weights
        n_s *= weights
        n_Nn *= weights
        n_Ns *= weights
        s_s *= weights
        s_Nn *= weights
        s_Ns *= weights
        Nn_Nn *= weights
        Nn_Ns *= weights
        Ns_Ns *= weights

        # construct gks aa,ab,ba,bb blocks, ai means orbital a to orbital i
        ai_aa = numpy.einsum('na,ni->nai',mo_a_vir[0].conj(),mo_a_occ[0],optimize=True)
        ai_bb = numpy.einsum('na,ni->nai',mo_b_vir[0].conj(),mo_b_occ[0],optimize=True)
        
        # construct gradient terms
        ai_na_a = numpy.einsum('gna,ni->gnai',mo_a_vir[1:4].conj(),mo_a_occ[0],optimize=True)
        ai_a_na = numpy.einsum('na,gni->gnai',mo_a_vir[0].conj(),mo_a_occ[1:4],optimize=True)
        ai_nb_b = numpy.einsum('gna,ni->gnai',mo_b_vir[1:4].conj(),mo_b_occ[0],optimize=True)
        ai_b_nb = numpy.einsum('na,gni->gnai',mo_b_vir[0].conj(),mo_b_occ[1:4],optimize=True)
        
        # construct nrho,nMz blocks to calculate K_aibj
        ai_Naa = ai_na_a + ai_a_na 
        ai_Nbb = ai_nb_b + ai_b_nb
        
        # calculate K_aibj
        K_aibj_aaaa = 0.0
        K_aibj_aabb = 0.0
        K_aibj_bbaa = 0.0
        K_aibj_bbbb = 0.0
        
        # nn
        K_aibj_aaaa += numpy.einsum('n,nai,nbj->aibj',n_n,ai_aa,ai_aa.conj(),optimize=True)  
        K_aibj_aabb += numpy.einsum('n,nai,nbj->aibj',n_n,ai_aa,ai_bb.conj(),optimize=True)
        K_aibj_bbaa += numpy.einsum('n,nai,nbj->aibj',n_n,ai_bb,ai_aa.conj(),optimize=True)
        K_aibj_bbbb += numpy.einsum('n,nai,nbj->aibj',n_n,ai_bb,ai_bb.conj(),optimize=True)
        
        # ns
        K_aibj_aaaa += numpy.einsum('n,nai,nbj->aibj',n_s,ai_aa,ai_aa.conj(),optimize=True)  
        K_aibj_aabb += -1*numpy.einsum('n,nai,nbj->aibj',n_s,ai_aa,ai_bb.conj(),optimize=True)
        K_aibj_bbaa += numpy.einsum('n,nai,nbj->aibj',n_s,ai_bb,ai_aa.conj(),optimize=True)
        K_aibj_bbbb += -1*numpy.einsum('n,nai,nbj->aibj',n_s,ai_bb,ai_bb.conj(),optimize=True)
        
        # sn
        K_aibj_aaaa += numpy.einsum('n,nai,nbj->aibj',n_s,ai_aa,ai_aa.conj(),optimize=True)  
        K_aibj_aabb += numpy.einsum('n,nai,nbj->aibj',n_s,ai_aa,ai_bb.conj(),optimize=True)
        K_aibj_bbaa += -1*numpy.einsum('n,nai,nbj->aibj',n_s,ai_bb,ai_aa.conj(),optimize=True)
        K_aibj_bbbb += -1*numpy.einsum('n,nai,nbj->aibj',n_s,ai_bb,ai_bb.conj(),optimize=True)
 
        # n_Nn
        K_aibj_aaaa += numpy.einsum('gn,nai,gnbj->aibj',n_Nn,ai_aa,ai_Naa.conj(),optimize=True)
        K_aibj_aabb += numpy.einsum('gn,nai,gnbj->aibj',n_Nn,ai_aa,ai_Nbb.conj(),optimize=True)
        K_aibj_bbaa += numpy.einsum('gn,nai,gnbj->aibj',n_Nn,ai_bb,ai_Naa.conj(),optimize=True)
        K_aibj_bbbb += numpy.einsum('gn,nai,gnbj->aibj',n_Nn,ai_bb,ai_Nbb.conj(),optimize=True)
        
        # Nn_n
        K_aibj_aaaa += numpy.einsum('gn,gnai,nbj->aibj',n_Nn,ai_Naa,ai_aa.conj(),optimize=True)
        K_aibj_aabb += numpy.einsum('gn,gnai,nbj->aibj',n_Nn,ai_Naa,ai_bb.conj(),optimize=True)
        K_aibj_bbaa += numpy.einsum('gn,gnai,nbj->aibj',n_Nn,ai_Nbb,ai_aa.conj(),optimize=True)
        K_aibj_bbbb += numpy.einsum('gn,gnai,nbj->aibj',n_Nn,ai_Nbb,ai_bb.conj(),optimize=True)
        
        # n_Ns
        K_aibj_aaaa += numpy.einsum('gn,nai,gnbj->aibj',n_Ns,ai_aa,ai_Naa.conj(),optimize=True)
        K_aibj_aabb += -1.0*numpy.einsum('gn,nai,gnbj->aibj',n_Ns,ai_aa,ai_Nbb.conj(),optimize=True)
        K_aibj_bbaa += numpy.einsum('gn,nai,gnbj->aibj',n_Ns,ai_bb,ai_Naa.conj(),optimize=True)
        K_aibj_bbbb += -1.0*numpy.einsum('gn,nai,gnbj->aibj',n_Ns,ai_bb,ai_Nbb.conj(),optimize=True)
        
        # Ns_n
        K_aibj_aaaa += numpy.einsum('gn,gnai,nbj->aibj',n_Ns,ai_Naa,ai_aa.conj(),optimize=True)
        K_aibj_aabb += numpy.einsum('gn,gnai,nbj->aibj',n_Ns,ai_Naa,ai_bb.conj(),optimize=True)
        K_aibj_bbaa += -1.0*numpy.einsum('gn,gnai,nbj->aibj',n_Ns,ai_Nbb,ai_aa.conj(),optimize=True)
        K_aibj_bbbb += -1.0*numpy.einsum('gn,gnai,nbj->aibj',n_Ns,ai_Nbb,ai_bb.conj(),optimize=True)

        # ss
        K_aibj_aaaa += numpy.einsum('n,nai,nbj->aibj',s_s,ai_aa,ai_aa.conj(),optimize=True)  
        K_aibj_aabb += -1*numpy.einsum('n,nai,nbj->aibj',s_s,ai_aa,ai_bb.conj(),optimize=True)
        K_aibj_bbaa += -1*numpy.einsum('n,nai,nbj->aibj',s_s,ai_bb,ai_aa.conj(),optimize=True)
        K_aibj_bbbb += numpy.einsum('n,nai,nbj->aibj',s_s,ai_bb,ai_bb.conj(),optimize=True)

        # s_Nn
        K_aibj_aaaa += numpy.einsum('gn,nai,gnbj->aibj',s_Nn,ai_aa,ai_Naa.conj(),optimize=True)
        K_aibj_aabb += numpy.einsum('gn,nai,gnbj->aibj',s_Nn,ai_aa,ai_Nbb.conj(),optimize=True)
        K_aibj_bbaa += -1*numpy.einsum('gn,nai,gnbj->aibj',s_Nn,ai_bb,ai_Naa.conj(),optimize=True)
        K_aibj_bbbb += -1*numpy.einsum('gn,nai,gnbj->aibj',s_Nn,ai_bb,ai_Nbb.conj(),optimize=True)
        
        # Nn_s
        K_aibj_aaaa += numpy.einsum('gn,gnai,nbj->aibj',s_Nn,ai_Naa,ai_aa.conj(),optimize=True)
        K_aibj_aabb += -1*numpy.einsum('gn,gnai,nbj->aibj',s_Nn,ai_Naa,ai_bb.conj(),optimize=True)
        K_aibj_bbaa += numpy.einsum('gn,gnai,nbj->aibj',s_Nn,ai_Nbb,ai_aa.conj(),optimize=True)
        K_aibj_bbbb += -1*numpy.einsum('gn,gnai,nbj->aibj',s_Nn,ai_Nbb,ai_bb.conj(),optimize=True)

        # s_Ns
        K_aibj_aaaa += numpy.einsum('gn,nai,gnbj->aibj',s_Ns,ai_aa,ai_Naa.conj(),optimize=True)
        K_aibj_aabb += -1*numpy.einsum('gn,nai,gnbj->aibj',s_Ns,ai_aa,ai_Nbb.conj(),optimize=True)
        K_aibj_bbaa += -1*numpy.einsum('gn,nai,gnbj->aibj',s_Ns,ai_bb,ai_Naa.conj(),optimize=True)
        K_aibj_bbbb += numpy.einsum('gn,nai,gnbj->aibj',s_Ns,ai_bb,ai_Nbb.conj(),optimize=True)
        
        # Ns_s
        K_aibj_aaaa += numpy.einsum('gn,gnai,nbj->aibj',s_Ns,ai_Naa,ai_aa.conj(),optimize=True)
        K_aibj_aabb += -1*numpy.einsum('gn,gnai,nbj->aibj',s_Ns,ai_Naa,ai_bb.conj(),optimize=True)
        K_aibj_bbaa += -1*numpy.einsum('gn,gnai,nbj->aibj',s_Ns,ai_Nbb,ai_aa.conj(),optimize=True)
        K_aibj_bbbb += numpy.einsum('gn,gnai,nbj->aibj',s_Ns,ai_Nbb,ai_bb.conj(),optimize=True)
        
        # Nn_Nn part
        K_aibj_aaaa += numpy.einsum('ghn,gnai,hnbj->aibj',Nn_Nn,ai_Naa,ai_Naa.conj(),optimize=True)
        K_aibj_aabb += numpy.einsum('ghn,gnai,hnbj->aibj',Nn_Nn,ai_Naa,ai_Nbb.conj(),optimize=True)
        K_aibj_bbaa += numpy.einsum('ghn,gnai,hnbj->aibj',Nn_Nn,ai_Nbb,ai_Naa.conj(),optimize=True)
        K_aibj_bbbb += numpy.einsum('ghn,gnai,hnbj->aibj',Nn_Nn,ai_Nbb,ai_Nbb.conj(),optimize=True)

        # Nn_Ns part
        K_aibj_aaaa += numpy.einsum('ghn,gnai,hnbj->aibj',Nn_Ns,ai_Naa,ai_Naa.conj(),optimize=True)
        K_aibj_aabb += -1*numpy.einsum('ghn,gnai,hnbj->aibj',Nn_Ns,ai_Naa,ai_Nbb.conj(),optimize=True)
        K_aibj_bbaa += numpy.einsum('ghn,gnai,hnbj->aibj',Nn_Ns,ai_Nbb,ai_Naa.conj(),optimize=True)
        K_aibj_bbbb += -1*numpy.einsum('ghn,gnai,hnbj->aibj',Nn_Ns,ai_Nbb,ai_Nbb.conj(),optimize=True)
        
        # Ns_Nn part
        K_aibj_aaaa += numpy.einsum('ghn,gnai,hnbj->aibj',Nn_Ns,ai_Naa,ai_Naa.conj(),optimize=True)
        K_aibj_aabb += numpy.einsum('ghn,gnai,hnbj->aibj',Nn_Ns,ai_Naa,ai_Nbb.conj(),optimize=True)
        K_aibj_bbaa += -1*numpy.einsum('ghn,gnai,hnbj->aibj',Nn_Ns,ai_Nbb,ai_Naa.conj(),optimize=True)
        K_aibj_bbbb += -1*numpy.einsum('ghn,gnai,hnbj->aibj',Nn_Ns,ai_Nbb,ai_Nbb.conj(),optimize=True)

        # Ns_Ns part
        K_aibj_aaaa += numpy.einsum('ghn,gnai,hnbj->aibj',Ns_Ns,ai_Naa,ai_Naa.conj(),optimize=True)
        K_aibj_aabb += -1*numpy.einsum('ghn,gnai,hnbj->aibj',Ns_Ns,ai_Naa,ai_Nbb.conj(),optimize=True)
        K_aibj_bbaa += -1*numpy.einsum('ghn,gnai,hnbj->aibj',Ns_Ns,ai_Nbb,ai_Naa.conj(),optimize=True)
        K_aibj_bbbb += numpy.einsum('ghn,gnai,hnbj->aibj',Ns_Ns,ai_Nbb,ai_Nbb.conj(),optimize=True)

    elif xctype == 'MGGA':
            raise NotImplementedError("Spin-conserved schem isn't implemented in Meta-GGA")

    return K_aibj_aaaa,K_aibj_aabb,K_aibj_bbaa,K_aibj_bbbb
    
def noncollinear_tdamc_kernel(self, mol, xc_code, grids, dms, Ndirect=None,Ndirect_lc=None, 
                              MSL_factor=None, LIBXCT_factor=None,ncpu=None): 
    r'''noncollinear_tdamc_kernel: serves as a calculator to obtain noncollinear tda kernel in Multi-Collinear approach. 
    Numerical stable approach is introduced here, which is used to deal with the numerical problem at strong polar points,
    controled by MSL_factor. 
        In Numerical stable approach, noncollinear case is transformed onto collinear case, with the direction of spin ma-
    genization density vector (\boldsymbol{m} = (m_x,m_y,m_z)) projection direction, to substitute kernel at these strong 
    polar points. New kernel is composed of two parts, spin conserved part and spin flip part. Note spin conserved kernel 
    points at the principle direction, while spin flip part points at the two directions perpendicular to the principle d-
    irection.
    
    Parameters
    ----------
    Args:
        mol : an instance of :class:`Mole` in pySCF
        
        xc_code : str
            Name of exchange-correlation functional.
        grids : mf.object
            mf is an object for DFT class and grids.coords and grids.weights are used here, supplying the sample 
            points and weights in real space.
        dms : tuple
            (dmaa,dmab,dmba,dmbb), density matrix.
    
    Kwargs:
        Ndirect : int
            The number of sample points in spin space, for lebedev distribution.
        Ndirect_lc : int
            The number of sample points in spin space, for gauss-legendre distribution.
        MSL_factor : double or int
            The factor to determine the strong polar points.
            Deafult is None. Value Recommended is 0.999.
        LIBXCT_factor : double or int
            The Threshold of for derivatives obatained from libxc.
            Deafult is None. Value Recommended is 1e-10.
        ncpu : int
            Number of cpu workers.
        
    Returns:
    ----------
        Noncollinear tda kernel : tuple
            LDA: kxc_nn,kxc_ns,kxc_ss.
            GGA: kxc_nn, kxc_ns, kxc_n_Nn, kxc_n_Ns, kxc_ss, kxc_s_Nn, kxc_s_Ns,\
                 kxc_Nn_Nn, kxc_Nn_Ns, kxc_Ns_Ns.
                 
    Raises:
    ----------
        ToDo : Noncollinear TDA kernel in MGGA.
    '''
    xctype = self._xc_type(xc_code)
    # import pdb
    # pdb.set_trace()
    if Ndirect is None:
        Ndirect = 1
    NX,factor = self.Spoints.make_sph_samples(Ndirect)
    
    coords = grids.coords
    weights = grids.weights
    numpy.save('coords',grids.coords)
    numpy.save('weights',grids.weights)

    dmaa, dmab, dmba, dmbb = dms
    
    if xctype == 'LDA':
        ao_deriv = 0
        ao = self.eval_ao(mol, coords, deriv=ao_deriv)
        
        rho_aa = self.eval_rho(mol, ao, dmaa.real, xctype=xctype)
        rho_bb = self.eval_rho(mol, ao, dmbb.real, xctype=xctype)
        Mx = self.eval_rho(mol, ao, (dmba+dmab).real, xctype=xctype)
        My = self.eval_rho(mol, ao, (-dmba*1.0j+dmab*1.0j).real, xctype=xctype) 
        Mz = rho_aa - rho_bb
        
        ngrid = rho_aa.shape[0]
        rhop = rho_aa + rho_bb
    
        kxc_nn = numpy.zeros((ngrid),dtype=numpy.complex128)
        kxc_ns = numpy.zeros((3,ngrid),dtype=numpy.complex128)
        kxc_ss = numpy.zeros((3,3,ngrid),dtype=numpy.complex128)
        
    
        import multiprocessing
        import math
        # ~ init some parameters in parallel.
        if ncpu is None:
            ncpu = multiprocessing.cpu_count()
        nsbatch = math.ceil(Ndirect/ncpu)
        NX_list = [(i, i+nsbatch) for i in range(0, Ndirect-nsbatch, nsbatch)]
        if NX_list[-1][-1] < Ndirect:
            NX_list.append((NX_list[-1][-1], Ndirect))
            
        pool = multiprocessing.Pool()
        para_results = []
        for index in NX_list:
            para_results.append(pool.apply_async(LDA_tdamc_kernel,(self, xc_code, rhop, Mx, My, Mz, NX, index,
                                                                    factor,LIBXCT_factor)))
            
        pool.close()
        pool.join()
        
        for para_result in para_results:
            result = para_result.get()
            kxc_nn += result[0]
            kxc_ns += result[1]
            kxc_ss += result[2]
        
        # import pdb
        # pdb.set_trace()
        if MSL_factor:
            """
            Numerical stable approach MC_to_Col
            """
            # Put numerical instable grids into a list.
            M_norm = numpy.sqrt(Mx*Mx + My*My + Mz*Mz)
            idx_instable = M_norm>=MSL_factor*rhop
            N_instable = idx_instable.sum()        
            NX_instable = numpy.zeros((N_instable,3))

            NX_instable[:,0] = Mx[idx_instable]/M_norm[idx_instable]
            NX_instable[:,1] = My[idx_instable]/M_norm[idx_instable]
            NX_instable[:,2] = Mz[idx_instable]/M_norm[idx_instable]

            # SVG to get another two components perpendicul.
            NX_tmp = numpy.zeros((N_instable,3,3))
            NX_tmp[:,:,0] = NX_instable
            NX_instable_orth = numpy.linalg.svd(NX_tmp[...,:,:])[0]

            # spin-conserve like part!
            rho_aa_c = 0.5*(rhop[idx_instable] + M_norm[idx_instable])
            rho_bb_c = 0.5*(rhop[idx_instable] - M_norm[idx_instable])

            kxc_nn_stable,kxc_ns_stable,kxc_ss_stable = \
            self.eval_xc_collinear_kernel(xc_code,(rho_aa_c,rho_bb_c),LIBXCT_factor=LIBXCT_factor)[0]
            # self.eval_xc_noncollinear_kernel(xc_code,(rho_aa_c,rho_bb_c),LIBXCT_factor=LIBXCT_factor)

            kxc_nn[idx_instable] = kxc_nn_stable
            for i in range(3):
                 kxc_ns[i,idx_instable] = kxc_ns_stable*NX_instable[:,i]
            for i in range(3):
                for j in range(3):
                    kxc_ss[i,j,idx_instable] = kxc_ss_stable*NX_instable[:,i]*NX_instable[:,j]

            # spin-flip like part!
            kxc_ss_stable = 0.0
            # Special 1d picking method
            tarray, weight_array_st= numpy.polynomial.legendre.leggauss(Ndirect_lc)
            # correct the integral area from [-1,1] to [0,1] 
            weight_array_st *= 0.5
            tarray = tarray*0.5+0.5
            assert(numpy.abs(weight_array_st.sum()-1.0)<=1.0E-14)

            import multiprocessing
            import math
            # ~ init some parameters in parallel.
            ncpu = multiprocessing.cpu_count()
            nsbatch = math.ceil(Ndirect_lc/ncpu)
            Ndirect_lc_list = [(i, i+nsbatch) for i in range(0, Ndirect_lc-nsbatch, nsbatch)]
            if Ndirect_lc_list[-1][-1] < Ndirect_lc:
                Ndirect_lc_list.append((Ndirect_lc_list[-1][-1], Ndirect_lc))

            pool = multiprocessing.Pool()
            para_results = []
            for index in Ndirect_lc_list:
                para_results.append(pool.apply_async(collinear_1d_kernel,(self,xc_code,(rhop[idx_instable],M_norm[idx_instable]),
                                                                          tarray[index[0]:index[1]],weight_array_st[index[0]:index[1]],LIBXCT_factor)))
            pool.close()
            pool.join()

            for para_result in para_results:
                result = para_result.get()
                kxc_ss_stable += result

            for iaxis in range(1,3):
                for i in range(3):
                    for j in range(3):
                        kxc_ss[i,j,idx_instable]+= kxc_ss_stable* \
                            NX_instable_orth[:,i,iaxis]* \
                            NX_instable_orth[:,j,iaxis]        
            """
            End of numerical instabilities
            """
        
        kxc_nn *= weights
        kxc_ns *= weights
        kxc_ss *= weights
        
        return kxc_nn,kxc_ns,kxc_ss

    if xctype == 'GGA':
        ao_deriv = 1
        ao = self.eval_ao(mol, coords, deriv=ao_deriv)
        
        rho_aa = self.eval_rho(mol, ao, dmaa.real, xctype=xctype)
        rho_bb = self.eval_rho(mol, ao, dmbb.real, xctype=xctype)
        Mx = self.eval_rho(mol, ao, (dmba+dmab).real, xctype=xctype)
        My = self.eval_rho(mol, ao, (-dmba*1.0j+dmab*1.0j).real, xctype=xctype)
        Mz = rho_aa - rho_bb
        rhop = rho_aa + rho_bb
        ngrid = rho_aa.shape[1]
    
        kxc_nn = numpy.zeros((ngrid),dtype=numpy.complex128)
        kxc_ns = numpy.zeros((3,ngrid),dtype=numpy.complex128)
        kxc_n_Nn = numpy.zeros((3,ngrid),dtype=numpy.complex128)
        kxc_n_Ns = numpy.zeros((3,3,ngrid),dtype=numpy.complex128)
        
        kxc_ss = numpy.zeros((3,3,ngrid),dtype=numpy.complex128)
        kxc_s_Nn = numpy.zeros((3,3,ngrid),dtype=numpy.complex128)
        kxc_s_Ns = numpy.zeros((3,3,3,ngrid),dtype=numpy.complex128)
        
        kxc_Nn_Nn = numpy.zeros((6,ngrid),dtype=numpy.complex128)
        kxc_Nn_Ns = numpy.zeros((3,3,3,ngrid),dtype=numpy.complex128)
        kxc_Ns_Ns = numpy.zeros((6,3,3,ngrid),dtype=numpy.complex128)
 
        import multiprocessing
        import math
        if ncpu is None:
            ncpu = multiprocessing.cpu_count()
        nsbatch = math.ceil(Ndirect/ncpu)
        NX_list = [(i, i+nsbatch) for i in range(0, Ndirect-nsbatch, nsbatch)]
        if NX_list[-1][-1] < Ndirect:
            NX_list.append((NX_list[-1][-1], Ndirect))
        
        # import pdb
        # pdb.set_trace()
        pool = multiprocessing.Pool()
        para_results = []
        for index in NX_list:
            para_results.append(pool.apply_async(GGA_tdamc_kernel,(self, xc_code, rhop, Mx, My, Mz, NX, index,
                                                 factor, LIBXCT_factor)))
        pool.close()
        pool.join()
        
        for para_result in para_results:
            result = para_result.get()
            kxc_nn += result[0]
            kxc_ns += result[1]
            kxc_n_Nn += result[2]
            kxc_n_Ns += result[3]
            kxc_ss += result[4]
            kxc_s_Nn += result[5]
            kxc_s_Ns += result[6]
            kxc_Nn_Nn += result[7]
            kxc_Nn_Ns += result[8]
            kxc_Ns_Ns += result[9]
        
        # import pdb
        # pdb.set_trace()
        if MSL_factor:
            """
            Numerical stable approach MC_to_Col
            """
            # Put numerical instable grids into a list.
    #         import pdb
    #         pdb.set_trace()
            M_norm = numpy.sqrt(Mx[0]**2 + My[0]**2 + Mz[0]**2)
            idx_instable = M_norm>=MSL_factor*rhop[0]
            N_instable = idx_instable.sum()        
            NX_instable = numpy.zeros((N_instable,3))

            NX_instable[:,0] = Mx[0,idx_instable]/M_norm[idx_instable]
            NX_instable[:,1] = My[0,idx_instable]/M_norm[idx_instable]
            NX_instable[:,2] = Mz[0,idx_instable]/M_norm[idx_instable]

            # SVG to get another two components perpendicul.
            NX_tmp = numpy.zeros((N_instable,3,3))
            NX_tmp[:,:,0] = NX_instable
            NX_instable_orth = numpy.linalg.svd(NX_tmp[...,:,:])[0]

            s_stable = numpy.zeros((4,N_instable))
            s_stable[0] = M_norm[idx_instable]
            s_stable[1:] = Mx[1:,idx_instable]*NX_instable[:,0] \
                            +My[1:,idx_instable]*NX_instable[:,1] \
                            +Mz[1:,idx_instable]*NX_instable[:,2] 

            rho_aa_c = 0.5*(rhop[:,idx_instable] + s_stable[:,:])
            rho_bb_c = 0.5*(rhop[:,idx_instable] - s_stable[:,:])

            # spin-conserve like part!
            kxc_nn_stable, kxc_ns_stable, kxc_n_Nn_stable, kxc_n_Ns_stable, kxc_ss_stable,\
            kxc_s_Nn_stable, kxc_s_Ns_stable,kxc_Nn_Nn_stable, kxc_Nn_Ns_stable, kxc_Ns_Ns_stable\
            = self.eval_xc_collinear_kernel(xc_code,(rho_aa_c,rho_bb_c),LIBXCT_factor=LIBXCT_factor)[0]
            # = self.eval_xc_noncollinear_kernel(xc_code,(rho_aa_c,rho_bb_c),LIBXCT_factor=LIBXCT_factor)

            kxc_nn[idx_instable] = kxc_nn_stable
            for i in range(3):
                 kxc_ns[i,idx_instable] = kxc_ns_stable*NX_instable[:,i]

            kxc_n_Nn[:,idx_instable] = kxc_n_Nn_stable

            for i in range(3):
                for j in range(3):
                    kxc_n_Ns[i,j,idx_instable] = kxc_n_Ns_stable[i]*NX_instable[:,j]

            for i in range(3):
                for j in range(3):
                    kxc_ss[i,j,idx_instable] = kxc_ss_stable*NX_instable[:,i]*NX_instable[:,j]

            for i in range(3):
                for j in range(3):
                    kxc_s_Nn[i,j,idx_instable] = kxc_s_Nn_stable[i]*NX_instable[:,j]

            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        kxc_s_Ns[i,j,k,idx_instable] = kxc_s_Ns_stable[i]*NX_instable[:,j]*NX_instable[:,k]

            # This term has six subindices given by nabla
            kxc_Nn_Nn[:,idx_instable] = kxc_Nn_Nn_stable

            # Indices i,j mean two nabla, k means projection directions
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        kxc_Nn_Ns[i,j,k,idx_instable] = kxc_Nn_Ns_stable[i,j]*NX_instable[:,k]

            # Indices i mean the indice combined by two nablas, k means projection directions 
            for i in range(6):
                for j in range(3):
                    for k in range(3):
                         kxc_Ns_Ns[i,j,k,idx_instable] = kxc_Ns_Ns_stable[i]*NX_instable[:,j]*NX_instable[:,k]

    #         # spin-flip like part!
            kxc_ss_stable = 0.0
            kxc_s_Ns_stable = 0.0
            kxc_Ns_Ns_stable =0.0

    #         import pdb 
    #         pdb.set_trace()
            # Special 1d picking method
            tarray, weight_array_st= numpy.polynomial.legendre.leggauss(Ndirect_lc)
            # correct the integral area from [-1,1] to [0,1] 
            weight_array_st *= 0.5
            tarray = tarray*0.5+0.5
            assert(numpy.abs(weight_array_st.sum()-1.0)<=1.0E-14)

            import multiprocessing
            import math
            # ~ init some parameters in parallel.
    #         import pdb
    #         pdb.set_trace()
            if ncpu is None:
                ncpu = multiprocessing.cpu_count()
            nsbatch = math.ceil(Ndirect_lc/ncpu)
            Ndirect_lc_list = [(i, i+nsbatch) for i in range(0, Ndirect_lc-nsbatch, nsbatch)]
            if Ndirect_lc_list[-1][-1] < Ndirect_lc:
                Ndirect_lc_list.append((Ndirect_lc_list[-1][-1], Ndirect_lc))

            pool = multiprocessing.Pool()
            para_results = []
            for index in Ndirect_lc_list:
                para_results.append(pool.apply_async(collinear_1d_kernel,(self,xc_code,(rhop[:,idx_instable],s_stable[:,:]),
                                                                          tarray[index[0]:index[1]],weight_array_st[index[0]:index[1]],LIBXCT_factor)))
            # ~ finisht the parallel part.
            pool.close()
            pool.join()

            # import pdb
            # pdb.set_trace()
            for para_result in para_results:
                result = para_result.get()
                kxc_ss_stable +=result[0]
                kxc_s_Ns_stable +=result[1]
                kxc_Ns_Ns_stable +=result[2]

            for iaxis in range(1,3):
                for i in range(3):
                    for j in range(3):
                        kxc_ss[i,j,idx_instable]+= kxc_ss_stable*\
                            NX_instable_orth[:,i,iaxis]*\
                            NX_instable_orth[:,j,iaxis]  

            for iaxis in range(1,3):
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            kxc_s_Ns[i,j,k,idx_instable] += kxc_s_Ns_stable[i]*\
                            NX_instable_orth[:,j,iaxis]*NX_instable_orth[:,k,iaxis]

            for iaxis in range(1,3):                
                for i in range(6):
                    for j in range(3):
                        for k in range(3):
                             kxc_Ns_Ns[i,j,k,idx_instable] += kxc_Ns_Ns_stable[i]*\
                                NX_instable_orth[:,j,iaxis]*NX_instable_orth[:,k,iaxis]
            """
            End of numerical instabilities
            """
#         import pdb
#         pdb.set_trace()
        kxc_nn *= weights
        kxc_ns *= weights
        kxc_n_Nn *= weights
        kxc_n_Ns *= weights
        kxc_ss *= weights
        kxc_s_Nn *= weights
        kxc_s_Ns *= weights
        kxc_Nn_Nn *= weights
        kxc_Nn_Ns *= weights
        kxc_Ns_Ns *= weights
#         numpy.save('kxc_Nn_Ns0313', kxc_Nn_Ns)
        return kxc_nn, kxc_ns, kxc_n_Nn, kxc_n_Ns, kxc_ss, kxc_s_Nn, kxc_s_Ns,\
                kxc_Nn_Nn, kxc_Nn_Ns, kxc_Ns_Ns
    if xctype == 'MGGA':
        raise NotImplementedError("Noncollinear TDA isn't implemented in Meta-GGA")

def LDA_tdamc_kernel(self, xc_code, rhop, Mx, My, Mz, NX, index, factor,LIBXCT_factor=None):
    '''LDA_tdamc_kernel: calculates Multi-Collinear TDA kernel in LDA.
    
    Parameters
    ----------
    Args:
        xc_code : str
            Name of exchange-correlation functional.
        rhop :tuple
            Density and magenization density norm with form (rho,s), whrer rho, s with a shape (grid,). 
        Mx,My,Mz : numpy.array with shape (ngrid,)
            Magenization density vector.
        NX : numpy.array with shape (ngrid,3)
            Projection directions in lebedev distribution.
        index : tuple
            index = (init,finish), index means one of  parallel parts.
        factor : numpy.array with shape (ngrid,)
            Weights in lebedev distribution.
    
    Kwargs:
        LIBXCT_factor : double or int
            The Threshold of for derivatives obatained from libxc.
            Deafult is None. Value Recommended is 1e-10.
            
    Returns:
    ----------
        LDA collinear TDA kernel : tuple
            kxc_nn_total,kxc_ns_total,kxc_ss_total.
    '''
    init, finish = index
    ngrid = rhop.shape[0]
    kxc_nn_total = numpy.zeros((ngrid),dtype=numpy.complex128)
    kxc_ns_total = numpy.zeros((3,ngrid),dtype=numpy.complex128)
    kxc_ss_total = numpy.zeros((3,3,ngrid),dtype=numpy.complex128)
    
    for idirect in range(init,finish):
        s = Mx*NX[idirect,0]+ My*NX[idirect,1]+ Mz*NX[idirect,2]
        rho_a = 0.5*(rhop + s)
        rho_b = 0.5*(rhop - s)
        kxc_nn_drct,kxc_ns_drct,kxc_ss_drct = self.eval_xc_noncollinear_kernel(xc_code, (rho_a,rho_b), spin=1, relativity=0, deriv=3, omega=None,
                                                                                verbose=None, LIBXCT_factor=LIBXCT_factor)
                                  
        kxc_nn_total += kxc_nn_drct*factor[idirect]
        
        for i in range(3):
            kxc_ns_total[i] += kxc_ns_drct*NX[idirect,i]*factor[idirect]
            
        for i in range(3):
            for j in range(3):
                kxc_ss_total[i,j] += kxc_ss_drct*NX[idirect,i]*NX[idirect,j]*factor[idirect]
                
    return kxc_nn_total,kxc_ns_total,kxc_ss_total

def GGA_tdamc_kernel(self, xc_code, rhop, Mx, My, Mz, NX, index, factor,LIBXCT_factor=None):
    '''GGA_tdamc_kernel: calculates Multi-Collinear TDA kernel in GGA.
    
    Parameters
    ----------
    Args:
        Args:
        xc_code : str
            Name of exchange-correlation functional.
        rhop : tuple
            Density and magenization density norm with form (rho,s), whrer rho, s with a shape (4,grid). 
            4 mean 1, nabla_x, nabla_y, nabla_z.
        Mx,My,Mz : numpy.array withs shape (4,ngrid)
            Magenization density vector and its gradient.
        NX : numpy.array with shape (ngrid,3)
            Projection directions in lebedev distribution.
        index : tuple
            index = (init,finish), index means one of  parallel parts.
        factor : numpy.array with shape (ngrid,)
            Weights in lebedev distribution.
    
    Kwargs:
        LIBXCT_factor : double or int
            The Threshold of for derivatives obatained from libxc.
            Deafult is None. Value Recommended is 1e-10.
    
    Returns:
    ----------
        GGA collinear TDA kernel : tuple
            kxc_nn_total, kxc_ns_total, kxc_n_Nn_total, kxc_n_Ns_total, kxc_ss_total, kxc_s_Nn_total, kxc_s_Ns_total,\
            kxc_Nn_Nn_total, kxc_Nn_Ns_total, kxc_Ns_Ns_total
    '''
    ngrid = rhop.shape[1]
    init, finish = index
#     ndirect = NX.shape[0]
    kxc_nn_total = numpy.zeros((ngrid),dtype=numpy.complex128)
    kxc_ns_total = numpy.zeros((3,ngrid),dtype=numpy.complex128)
    kxc_n_Nn_total = numpy.zeros((3,ngrid),dtype=numpy.complex128)
    kxc_n_Ns_total = numpy.zeros((3,3,ngrid),dtype=numpy.complex128)
    
    kxc_ss_total = numpy.zeros((3,3,ngrid),dtype=numpy.complex128)
    kxc_s_Nn_total = numpy.zeros((3,3,ngrid),dtype=numpy.complex128)
    kxc_s_Ns_total = numpy.zeros((3,3,3,ngrid),dtype=numpy.complex128)
    
    kxc_Nn_Nn_total = numpy.zeros((6,ngrid),dtype=numpy.complex128)
    kxc_Nn_Ns_total = numpy.zeros((3,3,3,ngrid),dtype=numpy.complex128)
    kxc_Ns_Ns_total = numpy.zeros((6,3,3,ngrid),dtype=numpy.complex128)
    
#     import pdb
#     pdb.set_trace()
#     for idrct in range(ndirect):
    for idrct in range(init,finish):
        s = Mx*NX[idrct,0]+ My*NX[idrct,1]+ Mz*NX[idrct,2]
        rho_a = 0.5*(rhop + s)
        rho_b = 0.5*(rhop - s)
        
        kxc_nn_drct, kxc_ns_drct, kxc_n_Nn_drct, kxc_n_Ns_drct, kxc_ss_drct, kxc_s_Nn_drct, kxc_s_Ns_drct, \
        kxc_Nn_Nn_drct, kxc_Nn_Ns_drct, kxc_Ns_Ns_drct = self.eval_xc_noncollinear_kernel(xc_code, (rho_a,rho_b), spin=1,relativity=0, deriv=3, 
                                                                                          omega=None, verbose=None, LIBXCT_factor=LIBXCT_factor)
    
        kxc_nn_total += kxc_nn_drct*factor[idrct]
        
        for i in range(3):
            kxc_ns_total[i] += kxc_ns_drct*NX[idrct,i]*factor[idrct]
        
        for i in range(3):
            kxc_n_Nn_total[i] += kxc_n_Nn_drct[i]*factor[idrct]
            
        # The first indice is given by nabla, another is given by projection directions
        for i in range(3):
            for j in range(3):
                kxc_n_Ns_total[i,j] += kxc_n_Ns_drct[i]*NX[idrct,j]*factor[idrct]
        
        # The two indices are both given by projection directions
        for i in range(3):
            for j in range(3):
                kxc_ss_total[i,j] += kxc_ss_drct*NX[idrct,i]*NX[idrct,j]*factor[idrct]
        
        # The first indice is given by nabla, another is given by projection directions
        for i in range(3):
            for j in range(3):
                kxc_s_Nn_total[i,j] += kxc_s_Nn_drct[i]*NX[idrct,j]*factor[idrct]
        
        # The first indice is given by nabla, another two are given by projection directions
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    kxc_s_Ns_total[i,j,k] += kxc_s_Ns_drct[i]*NX[idrct,j]*NX[idrct,k]*factor[idrct]
        
        # This term has six subindices given by nabla
        kxc_Nn_Nn_total += kxc_Nn_Nn_drct*factor[idrct]
        
        # Indices i,j mean two nabla, k means projection directions
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    kxc_Nn_Ns_total[i,j,k] += kxc_Nn_Ns_drct[i,j]*NX[idrct,k]*factor[idrct]
                    
        # Indices i mean the indice combined by two nablas, j,k means projection directions 
        for i in range(6):
            for j in range(3):
                for k in range(3):
                     kxc_Ns_Ns_total[i,j,k] += kxc_Ns_Ns_drct[i]*NX[idrct,j]*NX[idrct,k]*factor[idrct]
         
                
    return kxc_nn_total, kxc_ns_total, kxc_n_Nn_total, kxc_n_Ns_total, kxc_ss_total, kxc_s_Nn_total, kxc_s_Ns_total,\
            kxc_Nn_Nn_total, kxc_Nn_Ns_total, kxc_Ns_Ns_total
    
def nr_noncollinear_tdamc(self, mol, xc_code, grids, dms, C_mo, Ndirect=None,Ndirect_lc=None, 
                          MSL_factor=None, LIBXCT_factor=None,ncpu =None):
    '''nr_noncollinear_tdamc: calculates the K_aibj for Multi-Collinear TDA for noncollinear system.
    
    Parameters
    ----------
    Args:
        mol : an instance of :class:`Mole` in pySCF
           
        xc_code : str
            Name of exchange-correlation functional.
        grids : mf.object
            mf is an object for DFT class and grids.coords and grids.weights are used here, supplying the sample 
            points and weights in real space.
        dms : tuple
            (dmaa,dmab,dmba,dmbb), density matrix.
        C_mo : tuple
            Molecular orbital cofficience. 
            C_mo = (mo_a_vir,mo_a_occ,mo_b_vir,mo_b_occ), which means:
                Virtual alpha molecular orbitals, occupied alpha molecular orbitals, 
                Virtual beta molecular orbitals, occupied beta molecular orbitals, respectively.
    
    Kwargs:
        Ndirect : int
            The number of sample points in spin space, for lebedev distribution.
        Ndirect_lc : int
            The number of sample points in spin space, for gauss-legendre distribution.
        MSL_factor : double or int
            The factor to determine the strong polar points.
            Deafult is None. Value Recommended is 0.999.
        LIBXCT_factor : double or int
            The Threshold of for derivatives obatained from libxc.
            Deafult is None. Value Recommended is 1e-10.
        ncpu : int
            Number of cpu workers.
    Returns:
    ----------
        K_aibj : numpy.array
            aibj means related orbitals. a,b are virtual orbitals, and i,j are occupied orbitals.
    '''
    xctype = self._xc_type(xc_code) 
    mo_a_vir,mo_a_occ, mo_b_vir, mo_b_occ = C_mo
    
    # import pdb
    # pdb.set_trace()
    
    kernel = self.noncollinear_tdamc_kernel(mol, xc_code, grids, dms, Ndirect=Ndirect, Ndirect_lc=Ndirect_lc, 
                                            MSL_factor=MSL_factor, LIBXCT_factor=LIBXCT_factor,ncpu=ncpu)
    K_aibj = 0.0
    
    import multiprocessing
    import math
    # ~ init some parameters in parallel.
    ngrid = grids.coords.shape[0]
    
    if ncpu is None:
        ncpu = multiprocessing.cpu_count()
    nsbatch = math.ceil(ngrid/ncpu)
    NX_list = [(i, i+nsbatch) for i in range(0, ngrid-nsbatch, nsbatch)]
    if NX_list[-1][-1] < ngrid:
        NX_list.append((NX_list[-1][-1], ngrid))
        
    pool = multiprocessing.Pool()
    para_results = []

    # import pdb
    # pdb.set_trace()
    if xctype == 'LDA':
        for index in NX_list:
            idxi,idxf = index
            kernel_para = []
            for i in range(len(kernel)):
                kernel_para.append(kernel[i][...,idxi:idxf])
            para_results.append(pool.apply_async(K_aibj_noncollinear_generator,(xctype, 
                                                            mo_a_occ[idxi:idxf],mo_b_occ[idxi:idxf],
                                                            mo_a_vir[idxi:idxf],mo_b_vir[idxi:idxf],kernel_para)))
    elif xctype == 'GGA':   
        for index in NX_list:
            idxi,idxf = index
            kernel_para = []
            for i in range(len(kernel)):
                kernel_para.append(kernel[i][...,idxi:idxf])
            para_results.append(pool.apply_async(K_aibj_noncollinear_generator,(xctype, 
                                                        mo_a_occ[:,idxi:idxf],mo_b_occ[:,idxi:idxf],
                                                        mo_a_vir[:,idxi:idxf],mo_b_vir[:,idxi:idxf],kernel_para)))
    pool.close()
    pool.join()

    # ~ get the final result
    for para_result in para_results:
    #     import pdb
    #     pdb.set_trace()

        result = para_result.get()
        K_aibj += result
    return K_aibj

def nr_noncollinear_tdalc(self, mol, xc_code, grids,dms,C_mo,LIBXCT_factor=None,KST_factor=1e-10,ncpu=None):
    '''nr_Kubler_noncollinear_tdalc: calculates the K_aibj for Lcoally Collinear TDA for noncollinear system.
    
    Parameters
    ----------
    Args:
        mol : an instance of :class:`Mole` in pySCF
           
        xc_code : str
            Name of exchange-correlation functional.
        grids : mf.object
            mf is an object for DFT class and grids.coords and grids.weights are used here, supplying the sample 
            points and weights in real space.
        dms : tuple
            (dmaa,dmab,dmba,dmbb), density matrix.
        C_mo : tuple
            Molecular orbital cofficience. 
            C_mo = (mo_a_vir,mo_a_occ,mo_b_vir,mo_b_occ), which means:
                Virtual alpha molecular orbitals, occupied alpha molecular orbitals, 
                Virtual beta molecular orbitals, occupied beta molecular orbitals, respectively.
    
    Kwargs:
        LIBXCT_factor : double or int
            The Threshold of for derivatives obatained from libxc.
            Deafult is None. Value Recommended is 1e-10.
        KST_factor : double or int
            The Threshold for s (magenization density norm), padding the instable case at s (as den) too small. 
            Deafult is 1e-10. Value Recommended is 1e-10.
        ncpu : int
            Number of cpu workers.
    Returns:
    ----------
        K_aibj : numpy.array
           aibj means related orbitals. a,b are virtual orbitals, and i,j are occupied orbitals.
    
    Raises:
    ----------
        ToDo : K_aibj for noncollinear system in GGA and MGGA.
    '''
    xctype = self._xc_type(xc_code) 
    mo_a_vir,mo_a_occ, mo_b_vir, mo_b_occ = C_mo
    K_aibj = 0.0
    
    # import pdb
    # pdb.set_trace()
    
    kernel = self.noncollinear_tdalc_kernel(mol, xc_code, grids, dms, LIBXCT_factor=LIBXCT_factor,KST_factor=KST_factor)
    
    import multiprocessing
    import math
    # ~ init some parameters in parallel.
    Ngrid = grids.coords.shape[0]
    if ncpu is None:
        ncpu = multiprocessing.cpu_count()
    nsbatch = math.ceil(Ngrid/ncpu)
    NX_list = [(i, i+nsbatch) for i in range(0, Ngrid-nsbatch, nsbatch)]
    if NX_list[-1][-1] < Ngrid:
        NX_list.append((NX_list[-1][-1], Ngrid))
        
    pool = multiprocessing.Pool()
    para_results = []

    if xctype == 'LDA':
        for index in NX_list:
            idxi,idxf = index
            kernel_para = []
            for i in range(len(kernel)):
                kernel_para.append(kernel[i][...,idxi:idxf])
            para_results.append(pool.apply_async(K_aibj_noncollinear_generator,(xctype, 
                                                            mo_a_occ[idxi:idxf],mo_b_occ[idxi:idxf],
                                                            mo_a_vir[idxi:idxf],mo_b_vir[idxi:idxf],kernel_para)))
        
    else:
        raise NotImplementedError("GGA and MGGA is not implemented")
    
    pool.close()
    pool.join()

    # ~ get the final result
    for para_result in para_results:
    #     import pdb
    #     pdb.set_trace()

        result = para_result.get()
        K_aibj += result

    return K_aibj
    
def noncollinear_tdalc_kernel(self,mol, xc_code, grids, dms, LIBXCT_factor=None,KST_factor=1e-10):
    '''noncollinear_tdalc_kernel: serves as a calculator to obtain noncollinear tda kernel in Locally Collinear approach.
    
    Parameters
    ----------
    Args:
        mol : an instance of :class:`Mole` in pySCF
           
        xc_code : str
            Name of exchange-correlation functional.
        grids : mf.object
            mf is an object for DFT class and grids.coords and grids.weights are used here, supplying the sample 
            points and weights in real space.
        dms : tuple
            (dmaa,dmab,dmba,dmbb), density matrix.
    
    Kwargs:
        LIBXCT_factor : double or int
            The Threshold of for derivatives obatained from libxc.
            Deafult is None. Value Recommended is 1e-10.
        KST_factor : double or int
            The Threshold for s (magenization density norm), padding the instable case at s (as den) too small. 
            Deafult is 1e-10. Value Recommended is 1e-10.
    
    Returns:
    ----------
        Noncollinear tda kernel : tuple
            LDA: kxc_nn,kxc_ns,kxc_ss.
            
    Raises:
    ----------
        ToDo : Lcoally Collinear TDA kernel for noncollinear system in GGA and MGGA.
    '''
    xctype = self._xc_type(xc_code) 
    coords = grids.coords
    weights = grids.weights
    numpy.save('coords',grids.coords)
    numpy.save('weights',grids.weights)
    dmaa, dmab, dmba, dmbb = dms
    
    if xctype == 'LDA':
        ao_deriv = 0
        ao = self.eval_ao(mol, coords, deriv=ao_deriv)
        
        rho_aa = self.eval_rho(mol, ao, dmaa.real, xctype=xctype)
        rho_bb = self.eval_rho(mol, ao, dmbb.real, xctype=xctype)
        Mx = self.eval_rho(mol, ao, (dmba+dmab).real, xctype=xctype)
        My = self.eval_rho(mol, ao, (-dmba*1.0j+dmab*1.0j).real, xctype=xctype) 
        Mz = rho_aa - rho_bb
        
        ngrid = rho_aa.shape[0]
        rhop = rho_aa + rho_bb
    
        kxc_nn = numpy.zeros((ngrid),dtype=numpy.complex128)
        kxc_ns = numpy.zeros((3,ngrid),dtype=numpy.complex128)
        kxc_ss = numpy.zeros((3,3,ngrid),dtype=numpy.complex128)

        s = numpy.sqrt(Mx**2 + My**2 + Mz**2)
        idx_zero = numpy.abs(s) <= KST_factor
        NX = numpy.zeros((ngrid,3))
        with numpy.errstate(divide='ignore',invalid='ignore'):
            NX[:,0] = Mx/s
            NX[:,1] = My/s
            NX[:,2] = Mz/s
        NX[idx_zero,:] = 0.0
        rhoa = 0.5*(rhop + s)
        rhob = 0.5*(rhop - s)
        n_n,n_s,s_s,vs = self.eval_xc_Kubler_kernel(xc_code,(rhoa, rhob),LIBXCT_factor=LIBXCT_factor)[0]
        
        with numpy.errstate(divide='ignore',invalid='ignore'):
            vs = vs/s
        
        vs[idx_zero] = s_s[idx_zero]
        kxc_nn += n_n
        for i in range(3):
            kxc_ns[i] += n_s*NX[:,i]

        for i in range(3):
            kxc_ss[i][i] += vs

        for i in range(3):
            for j in range(3):
                kxc_ss[i][j] += (s_s-vs)*NX[:,i]*NX[:,j]
                # if i != j:
                #     kxc_ss[i,j] -= s_s*NX[:,i]*NX[:,j]
                # elif i==j:
                #     kxc_ss[i,j] += vs

        kxc_nn *= weights
        kxc_ns *= weights
        kxc_ss *= weights
        return kxc_nn,kxc_ns,kxc_ss
    else:
        raise NotImplementedError("GGA and MGGA is not implemented")

def K_aibj_noncollinear_generator(xctype, mo_a_occ,mo_b_occ,mo_a_vir,mo_b_vir,kernel):
    '''K_aibj_noncollinear_generator: calculates <ai|kernel|bj> for noncollinear TDA term.
    
    Parameters
    ----------
    Args:
       xctype : str
            xctype -> LDA, GGA, MGGA. 
        mo_a_vir,mo_a_occ,mo_b_vir,mo_b_occ :  numpy.array
            Virtual alpha molecular orbitals, occupied alpha molecular orbitals, 
            Virtual beta molecular orbitals, occupied beta molecular orbitals, respectively.
        kernel : tuple
            Noncollinear TDA kernel. In LDA and GGA, len(kernel) = 3, 10, respectively.
     
    Returns:
    ----------
       K_aibj : numpy.array 
    
    Raises:
    ----------
        ToDo : K_aibj for noncollinear TDA in MGGA.
    '''
    if xctype == 'LDA':
        kxc_nn,kxc_ns,kxc_ss = kernel 
        # construct gks aa,ab,ba,bb blocks, ai means orbital a to orbital i
        ai_aa = numpy.einsum('na,ni->nai',mo_a_vir.conj(),mo_a_occ,optimize=True)
        ai_ab = numpy.einsum('na,ni->nai',mo_a_vir.conj(),mo_b_occ,optimize=True)
        ai_ba = numpy.einsum('na,ni->nai',mo_b_vir.conj(),mo_a_occ,optimize=True)
        ai_bb = numpy.einsum('na,ni->nai',mo_b_vir.conj(),mo_b_occ,optimize=True)
        
        # construct rho,Mx,My,Mz blocks to calculate K_aibj
        ai_rho = ai_aa + ai_bb
        ai_Mx = ai_ab + ai_ba
        ai_My = -1.0j*ai_ab + 1.0j*ai_ba
        ai_Mz = ai_aa - ai_bb
        
        ai_s = [ai_Mx,ai_My,ai_Mz]
        
        # calculate K_aibj
        K_aibj = 0.0

        # kxc_nn
        K_aibj += numpy.einsum('n,nai,nbj->aibj',kxc_nn,
                              ai_rho,ai_rho.conj(),optimize=True)  
        # kxc_ns
        for i in range(3):
            K_aibj += numpy.einsum('n,nai,nbj->aibj',kxc_ns[i],
                                  ai_rho,ai_s[i].conj(),optimize=True)
        
        # kxc_sn
        for i in range(3):
            K_aibj += numpy.einsum('n,nai,nbj->aibj',kxc_ns[i],
                                  ai_s[i],ai_rho.conj(),optimize=True)
            
        # kxc_ss
        for i in range(3):
            for j in range(3):
                K_aibj += numpy.einsum('n,nai,nbj->aibj',kxc_ss[i][j],
                                  ai_s[i],ai_s[j].conj(),optimize=True)
                
    elif xctype == 'GGA':
        kxc_nn, kxc_ns, kxc_n_Nn, kxc_n_Ns, kxc_ss, kxc_s_Nn, kxc_s_Ns, \
        kxc_Nn_Nn, kxc_Nn_Ns, kxc_Ns_Ns = kernel 
        
        # Prepareing work
        # construct gks aa,ab,ba,bb blocks, ai means orbital a to orbital i
        ai_aa = numpy.einsum('na,ni->nai',mo_a_vir[0].conj(),mo_a_occ[0],optimize=True)
        ai_ab = numpy.einsum('na,ni->nai',mo_a_vir[0].conj(),mo_b_occ[0],optimize=True)
        ai_ba = numpy.einsum('na,ni->nai',mo_b_vir[0].conj(),mo_a_occ[0],optimize=True)
        ai_bb = numpy.einsum('na,ni->nai',mo_b_vir[0].conj(),mo_b_occ[0],optimize=True)
        
        # construct rho,Mx,My,Mz blocks to calculate K_aibj
        ai_rho = ai_aa + ai_bb
        ai_Mx = ai_ab + ai_ba
        ai_My = -1.0j*ai_ab + 1.0j*ai_ba
        ai_Mz = ai_aa - ai_bb
        ai_s = [ai_Mx,ai_My,ai_Mz]
        
        # construct gradient terms
        ai_na_a = numpy.einsum('gna,ni->gnai',mo_a_vir[1:4].conj(),mo_a_occ[0],optimize=True)
        ai_a_na = numpy.einsum('na,gni->gnai',mo_a_vir[0].conj(),mo_a_occ[1:4],optimize=True)
        ai_na_b = numpy.einsum('gna,ni->gnai',mo_a_vir[1:4].conj(),mo_b_occ[0],optimize=True)
        ai_a_nb = numpy.einsum('na,gni->gnai',mo_a_vir[0].conj(),mo_b_occ[1:4],optimize=True)
        ai_nb_a = numpy.einsum('gna,ni->gnai',mo_b_vir[1:4].conj(),mo_a_occ[0],optimize=True)
        ai_b_na = numpy.einsum('na,gni->gnai',mo_b_vir[0].conj(),mo_a_occ[1:4],optimize=True)
        ai_nb_b = numpy.einsum('gna,ni->gnai',mo_b_vir[1:4].conj(),mo_b_occ[0],optimize=True)
        ai_b_nb = numpy.einsum('na,gni->gnai',mo_b_vir[0].conj(),mo_b_occ[1:4],optimize=True)
        
        ai_nrho = ai_na_a + ai_a_na + ai_nb_b + ai_b_nb
        ai_nMx = ai_na_b + ai_a_nb + ai_nb_a + ai_b_na
        ai_nMy = -1.0j*(ai_na_b + ai_a_nb) + 1.0j*(ai_nb_a + ai_b_na)
        ai_nMz = ai_na_a + ai_a_na - ai_nb_b - ai_b_nb
        ai_ns = [ai_nMx,ai_nMy,ai_nMz]
        # calculate K_aibj
        K_aibj = 0.0

        # kxc_nn
        K_aibj += numpy.einsum('n,nai,nbj->aibj',kxc_nn,
                              ai_rho,ai_rho.conj(),optimize=True)  
        # kxc_ns
        for i in range(3):
            K_aibj += numpy.einsum('n,nai,nbj->aibj',kxc_ns[i],
                                  ai_rho,ai_s[i].conj(),optimize=True)
        
        # kxc_sn
        for i in range(3):
            K_aibj += numpy.einsum('n,nai,nbj->aibj',kxc_ns[i],
                                  ai_s[i],ai_rho.conj(),optimize=True)
        # kxc_n_Nn
        K_aibj += numpy.einsum('gn,nai,gnbj->aibj',kxc_n_Nn,
                               ai_rho,ai_nrho.conj(),optimize=True)
        
        # kxc_Nn_n
        K_aibj += numpy.einsum('gn,gnai,nbj->aibj',kxc_n_Nn,
                               ai_nrho,ai_rho.conj(),optimize=True)
        
        # kxc_n_Ns
        for i in range(3):
            K_aibj += numpy.einsum('gn,nai,gnbj->aibj',kxc_n_Ns[:,i],
                                    ai_rho,ai_ns[i].conj(),optimize=True)
        
        # kxc_Ns_n
        for i in range(3):
            K_aibj += numpy.einsum('gn,gnai,nbj->aibj',kxc_n_Ns[:,i],
                                    ai_ns[i],ai_rho.conj(),optimize=True)
                
        # kxc_ss
        for i in range(3):
            for j in range(3):
                K_aibj += numpy.einsum('n,nai,nbj->aibj',kxc_ss[i][j],
                                  ai_s[i],ai_s[j].conj(),optimize=True)
        # kxc_s_Nn
        for i in range(3):
            K_aibj += numpy.einsum('gn,nai,gnbj->aibj',kxc_s_Nn[:,i],
                                   ai_s[i],ai_nrho.conj(),optimize=True)
            
        # kxc_Nn_s
        for i in range(3):
            K_aibj += numpy.einsum('gn,gnai,nbj->aibj',kxc_s_Nn[:,i],
                                   ai_nrho,ai_s[i].conj(),optimize=True)
        
        # kxc_s_Ns
        for i in range(3):
            for j in range(3):
                K_aibj += numpy.einsum('gn,nai,gnbj->aibj',kxc_s_Ns[:,i,j],
                                       ai_s[i],ai_ns[j].conj(),optimize=True)
                    
        # kxc_Ns_s
        for i in range(3):
            for j in range(3):
                K_aibj += numpy.einsum('gn,gnai,nbj->aibj',kxc_s_Ns[:,i,j],
                                        ai_ns[i],ai_s[j].conj(),optimize=True)
                      
        offset2 = numint_gksmc.get_2d_offset()
        # kxc_Nn_Nn

        for i in range(3):
            for j in range(3):
                K_aibj += numpy.einsum('n,nai,nbj->aibj',kxc_Nn_Nn[offset2[i,j]],
                                        ai_nrho[i],ai_nrho[j].conj(),optimize=True)

        # kxc_Nn_Ns
        # Note ai_ns[k][j] ! This is due to the storation form of coresponding kernel. 
        for i in range(3):
            K_aibj += numpy.einsum('ghn,gnai,hnbj->aibj',kxc_Nn_Ns[:,:,i],
                                ai_nrho,ai_ns[i].conj(),optimize=True)
                    
        # kxc_Ns_Nn
        for i in range(3):
            K_aibj += numpy.einsum('ghn,hnai,gnbj->aibj',kxc_Nn_Ns[:,:,i],
                                ai_ns[i],ai_nrho.conj(),optimize=True)
    
        # kxc_Ns_Ns
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        K_aibj += numpy.einsum('n,nai,nbj->aibj',kxc_Ns_Ns[offset2[i,j],k,l],
                                        ai_ns[k][i],ai_ns[l][j].conj(),optimize=True)
                        
    elif xctype == 'MGGA':
        raise NotImplementedError("Meta-GGA is not implemented")
        
    return K_aibj

def r_noncollinear_tdamc(self, nir, mol, xc_code, grids, dms, mo, Ndirect=None,Ndirect_lc=None, 
                          MSL_factor=None, LIBXCT_factor=None,ncpu =None):
    '''nr_noncollinear_tdamc: calculates the K_aibj for Multi-Collinear TDA for noncollinear system.
    
    Parameters
    ----------
    Args:
        mol : an instance of :class:`Mole` in pySCF
           
        xc_code : str
            Name of exchange-correlation functional.
        grids : mf.object
            mf is an object for DFT class and grids.coords and grids.weights are used here, supplying the sample 
            points and weights in real space.
        dms : tuple
            (dmaa,dmab,dmba,dmbb), density matrix.
        C_mo : tuple
            Molecular orbital cofficience. 
            C_mo = (mo_a_vir,mo_a_occ,mo_b_vir,mo_b_occ), which means:
                Virtual alpha molecular orbitals, occupied alpha molecular orbitals, 
                Virtual beta molecular orbitals, occupied beta molecular orbitals, respectively.
    
    Kwargs:
        Ndirect : int
            The number of sample points in spin space, for lebedev distribution.
        Ndirect_lc : int
            The number of sample points in spin space, for gauss-legendre distribution.
        MSL_factor : double or int
            The factor to determine the strong polar points.
            Deafult is None. Value Recommended is 0.999.
        LIBXCT_factor : double or int
            The Threshold of for derivatives obatained from libxc.
            Deafult is None. Value Recommended is 1e-10.
        ncpu : int
            Number of cpu workers.
    Returns:
    ----------
        K_aibj : numpy.array
            aibj means related orbitals. a,b are virtual orbitals, and i,j are occupied orbitals.
    '''
    xctype = self._xc_type(xc_code) 
    mo_vir_L, mo_vir_S,mo_occ_L, mo_occ_S, = mo 
    kernel = self.r_noncollinear_tdamc_kernel(mol, nir, xc_code, grids, dms, Ndirect=Ndirect, Ndirect_lc=Ndirect_lc, 
                                            MSL_factor=MSL_factor, LIBXCT_factor=LIBXCT_factor,ncpu=ncpu)
    K_aibj = 0.0
    
    # import pdb
    # pdb.set_trace()
    
    import multiprocessing
    import math
    # ~ init some parameters in parallel.
    ngrid = grids.coords.shape[0]
    
    if ncpu is None:
        ncpu = multiprocessing.cpu_count()
    nsbatch = math.ceil(ngrid/ncpu)
    NX_list = [(i, i+nsbatch) for i in range(0, ngrid-nsbatch, nsbatch)]
    if NX_list[-1][-1] < ngrid:
        NX_list.append((NX_list[-1][-1], ngrid))
        
    pool = multiprocessing.Pool()
    para_results = []

    # import pdb
    # pdb.set_trace()
    if xctype == 'LDA':
        for index in NX_list:
            idxi,idxf = index
            kernel_para = []
            for i in range(len(kernel)):
                kernel_para.append(kernel[i][...,idxi:idxf])
                
            para_results.append(pool.apply_async(K_aibj_noncollinear_generator_r,(xctype, 
                                                            mo_vir_L[:,:,idxi:idxf],mo_vir_S[:,:,idxi:idxf],
                                                            mo_occ_L[:,:,idxi:idxf],mo_occ_S[:,:,idxi:idxf],
                                                            kernel_para)))
    elif xctype == 'GGA':
        for index in NX_list:
            idxi,idxf = index
            kernel_para = []
            for i in range(len(kernel)):
                kernel_para.append(kernel[i][...,idxi:idxf])
                
            para_results.append(pool.apply_async(K_aibj_noncollinear_generator_r,(xctype, 
                                                            mo_vir_L[:,:,idxi:idxf],mo_vir_S[:,:,idxi:idxf],
                                                            mo_occ_L[:,:,idxi:idxf],mo_occ_S[:,:,idxi:idxf],
                                                            kernel_para)))
    else:
        raise NotImplementedError("")
    pool.close()
    pool.join()

    # ~ get the final result
    for para_result in para_results:
    #     import pdb
    #     pdb.set_trace()

        result = para_result.get()
        K_aibj += result
    return K_aibj

def r_noncollinear_tdamc_kernel(self, mol, nir, xc_code, grids, dms, Ndirect=None,Ndirect_lc=None, 
                              MSL_factor=None, LIBXCT_factor=None,ncpu=None): 
    r'''noncollinear_tdamc_kernel: serves as a calculator to obtain noncollinear tda kernel in Multi-Collinear approach. 
    Numerical stable approach is introduced here, which is used to deal with the numerical problem at strong polar points,
    controled by MSL_factor. 
        In Numerical stable approach, noncollinear case is transformed onto collinear case, with the direction of spin ma-
    genization density vector (\boldsymbol{m} = (m_x,m_y,m_z)) projection direction, to substitute kernel at these strong 
    polar points. New kernel is composed of two parts, spin conserved part and spin flip part. Note spin conserved kernel 
    points at the principle direction, while spin flip part points at the two directions perpendicular to the principle d-
    irection.
    
    Parameters
    ----------
    Args:
        mol : an instance of :class:`Mole` in pySCF
        
        xc_code : str
            Name of exchange-correlation functional.
        grids : mf.object
            mf is an object for DFT class and grids.coords and grids.weights are used here, supplying the sample 
            points and weights in real space.
        dms : tuple
            (dmaa,dmab,dmba,dmbb), density matrix.
    
    Kwargs:
        Ndirect : int
            The number of sample points in spin space, for lebedev distribution.
        Ndirect_lc : int
            The number of sample points in spin space, for gauss-legendre distribution.
        MSL_factor : double or int
            The factor to determine the strong polar points.
            Deafult is None. Value Recommended is 0.999.
        LIBXCT_factor : double or int
            The Threshold of for derivatives obatained from libxc.
            Deafult is None. Value Recommended is 1e-10.
        ncpu : int
            Number of cpu workers.
        
    Returns:
    ----------
        Noncollinear tda kernel : tuple
            LDA: kxc_nn,kxc_ns,kxc_ss.
            GGA: kxc_nn, kxc_ns, kxc_n_Nn, kxc_n_Ns, kxc_ss, kxc_s_Nn, kxc_s_Ns,\
                 kxc_Nn_Nn, kxc_Nn_Ns, kxc_Ns_Ns.
                 
    Raises:
    ----------
        ToDo : Noncollinear TDA kernel in MGGA.
    '''
    xctype = self._xc_type(xc_code)
    # import pdb
    # pdb.set_trace()
    if Ndirect is None:
        Ndirect = 1
    NX,factor = self.Spoints.make_sph_samples(Ndirect)
    
    coords = grids.coords
    weights = grids.weights
    numpy.save('coords',grids.coords)
    numpy.save('weights',grids.weights)

    
    if xctype == 'LDA':
        ao_deriv = 0

        ao = nir.eval_ao(mol, coords, deriv=ao_deriv) ### not same
        rho = nir.eval_rho(mol, ao, dms) ### not same
 
        ngrid = rho.shape[-1]
        rho = rho.astype(numpy.double) ### not same
    
        kxc_nn = numpy.zeros((ngrid),dtype=numpy.complex128)
        kxc_ns = numpy.zeros((3,ngrid),dtype=numpy.complex128)
        kxc_ss = numpy.zeros((3,3,ngrid),dtype=numpy.complex128)
        
    
        import multiprocessing
        import math
        # ~ init some parameters in parallel.
        if ncpu is None:
            ncpu = multiprocessing.cpu_count()
        nsbatch = math.ceil(Ndirect/ncpu)
        NX_list = [(i, i+nsbatch) for i in range(0, Ndirect-nsbatch, nsbatch)]
        if NX_list[-1][-1] < Ndirect:
            NX_list.append((NX_list[-1][-1], Ndirect))
            
        pool = multiprocessing.Pool()
        para_results = []
        for index in NX_list:
            para_results.append(pool.apply_async(LDA_tdamc_kernel,(self, xc_code, rho[0], 
            rho[1], rho[2], rho[3], NX, index, factor,LIBXCT_factor)))
            
        pool.close()
        pool.join()
        
        for para_result in para_results:
            result = para_result.get()
            kxc_nn += result[0]
            kxc_ns += result[1]
            kxc_ss += result[2]
        
        # import pdb
        # pdb.set_trace()
        if MSL_factor:
            """
            Numerical stable approach MC_to_Col
            """
            # Put numerical instable grids into a list.
            M_norm = numpy.sqrt(rho[1]*rho[1] + rho[2]*rho[2] + rho[3]*rho[3])
            idx_instable = M_norm>=MSL_factor*rho[0]
            N_instable = idx_instable.sum()        
            NX_instable = numpy.zeros((N_instable,3))

            NX_instable[:,0] = rho[1,idx_instable]/M_norm[idx_instable]
            NX_instable[:,1] = rho[2,idx_instable]/M_norm[idx_instable]
            NX_instable[:,2] = rho[3,idx_instable]/M_norm[idx_instable]

            # SVG to get another two components perpendicul.
            NX_tmp = numpy.zeros((N_instable,3,3))
            NX_tmp[:,:,0] = NX_instable
            NX_instable_orth = numpy.linalg.svd(NX_tmp[...,:,:])[0]

            # spin-conserve like part!
            rho_aa_c = 0.5*(rho[0,idx_instable] + M_norm[idx_instable])
            rho_bb_c = 0.5*(rho[0,idx_instable] - M_norm[idx_instable])

            kxc_nn_stable,kxc_ns_stable,kxc_ss_stable = \
            self.eval_xc_collinear_kernel(xc_code,(rho_aa_c,rho_bb_c),LIBXCT_factor=LIBXCT_factor)[0]
            # self.eval_xc_noncollinear_kernel(xc_code,(rho_aa_c,rho_bb_c),LIBXCT_factor=LIBXCT_factor)

            kxc_nn[idx_instable] = kxc_nn_stable
            for i in range(3):
                 kxc_ns[i,idx_instable] = kxc_ns_stable*NX_instable[:,i]
            for i in range(3):
                for j in range(3):
                    kxc_ss[i,j,idx_instable] = kxc_ss_stable*NX_instable[:,i]*NX_instable[:,j]

            # spin-flip like part!
            kxc_ss_stable = 0.0
            # Special 1d picking method
            tarray, weight_array_st= numpy.polynomial.legendre.leggauss(Ndirect_lc)
            # correct the integral area from [-1,1] to [0,1] 
            weight_array_st *= 0.5
            tarray = tarray*0.5+0.5
            assert(numpy.abs(weight_array_st.sum()-1.0)<=1.0E-14)

            import multiprocessing
            import math
            # ~ init some parameters in parallel.
            ncpu = multiprocessing.cpu_count()
            nsbatch = math.ceil(Ndirect_lc/ncpu)
            Ndirect_lc_list = [(i, i+nsbatch) for i in range(0, Ndirect_lc-nsbatch, nsbatch)]
            if Ndirect_lc_list[-1][-1] < Ndirect_lc:
                Ndirect_lc_list.append((Ndirect_lc_list[-1][-1], Ndirect_lc))

            pool = multiprocessing.Pool()
            para_results = []
            for index in Ndirect_lc_list:
                para_results.append(pool.apply_async(collinear_1d_kernel,(self,xc_code,(rho[0][idx_instable],M_norm[idx_instable]),
                                                                          tarray[index[0]:index[1]],weight_array_st[index[0]:index[1]],LIBXCT_factor)))
            pool.close()
            pool.join()

            for para_result in para_results:
                result = para_result.get()
                kxc_ss_stable += result

            for iaxis in range(1,3):
                for i in range(3):
                    for j in range(3):
                        kxc_ss[i,j,idx_instable]+= kxc_ss_stable* \
                            NX_instable_orth[:,i,iaxis]* \
                            NX_instable_orth[:,j,iaxis]        
            """
            End of numerical instabilities
            """
        
        kxc_nn *= weights
        kxc_ns *= weights
        kxc_ss *= weights
        
        return kxc_nn,kxc_ns,kxc_ss
    elif xctype == 'GGA':
        ao_deriv = 1
        ao = nir.eval_ao(mol, coords, deriv=ao_deriv) ### not same
        rho = nir.eval_rho(mol, ao, dms, xctype=xctype) ### not same
        
        ngrid = rho.shape[-1]
        rho = rho.astype(numpy.double)
    
        kxc_nn = numpy.zeros((ngrid),dtype=numpy.complex128)
        kxc_ns = numpy.zeros((3,ngrid),dtype=numpy.complex128)
        kxc_n_Nn = numpy.zeros((3,ngrid),dtype=numpy.complex128)
        kxc_n_Ns = numpy.zeros((3,3,ngrid),dtype=numpy.complex128)
        
        kxc_ss = numpy.zeros((3,3,ngrid),dtype=numpy.complex128)
        kxc_s_Nn = numpy.zeros((3,3,ngrid),dtype=numpy.complex128)
        kxc_s_Ns = numpy.zeros((3,3,3,ngrid),dtype=numpy.complex128)
        
        kxc_Nn_Nn = numpy.zeros((6,ngrid),dtype=numpy.complex128)
        kxc_Nn_Ns = numpy.zeros((3,3,3,ngrid),dtype=numpy.complex128)
        kxc_Ns_Ns = numpy.zeros((6,3,3,ngrid),dtype=numpy.complex128)
 
        import multiprocessing
        import math
        if ncpu is None:
            ncpu = multiprocessing.cpu_count()
        nsbatch = math.ceil(Ndirect/ncpu)
        NX_list = [(i, i+nsbatch) for i in range(0, Ndirect-nsbatch, nsbatch)]
        if NX_list[-1][-1] < Ndirect:
            NX_list.append((NX_list[-1][-1], Ndirect))
        
        # import pdb
        # pdb.set_trace()
        pool = multiprocessing.Pool()
        para_results = []
        for index in NX_list:
            para_results.append(pool.apply_async(GGA_tdamc_kernel,(self, xc_code, rho[0], 
            rho[1], rho[2], rho[3], NX, index, factor, LIBXCT_factor)))
        pool.close()
        pool.join()
        
        for para_result in para_results:
            result = para_result.get()
            kxc_nn += result[0]
            kxc_ns += result[1]
            kxc_n_Nn += result[2]
            kxc_n_Ns += result[3]
            kxc_ss += result[4]
            kxc_s_Nn += result[5]
            kxc_s_Ns += result[6]
            kxc_Nn_Nn += result[7]
            kxc_Nn_Ns += result[8]
            kxc_Ns_Ns += result[9]
        
        # import pdb
        # pdb.set_trace()
        if MSL_factor:
            """
            Numerical stable approach MC_to_Col
            """
            # Put numerical instable grids into a list.
    #         import pdb
    #         pdb.set_trace()
            M_norm = numpy.sqrt(rho[1,0]**2 + rho[2,0]**2 + rho[3,0]**2)
            idx_instable = M_norm>=MSL_factor*rho[0,0]
            N_instable = idx_instable.sum()        
            NX_instable = numpy.zeros((N_instable,3))

            NX_instable[:,0] = rho[1,0,idx_instable]/M_norm[idx_instable]
            NX_instable[:,1] = rho[2,0,idx_instable]/M_norm[idx_instable]
            NX_instable[:,2] = rho[3,0,idx_instable]/M_norm[idx_instable]

            # SVG to get another two components perpendicul.
            NX_tmp = numpy.zeros((N_instable,3,3))
            NX_tmp[:,:,0] = NX_instable
            NX_instable_orth = numpy.linalg.svd(NX_tmp[...,:,:])[0]

            s_stable = numpy.zeros((4,N_instable))
            s_stable[0] = M_norm[idx_instable]
            s_stable[1:] = rho[1,1:,idx_instable]*NX_instable[:,0] \
                            +rho[2,1:,idx_instable]*NX_instable[:,1] \
                            +rho[3,1:,idx_instable]*NX_instable[:,2] 

            rho_aa_c = 0.5*(rho[0,:,idx_instable] + s_stable[:,:])
            rho_bb_c = 0.5*(rho[0,:,idx_instable] - s_stable[:,:])

            # spin-conserve like part!
            kxc_nn_stable, kxc_ns_stable, kxc_n_Nn_stable, kxc_n_Ns_stable, kxc_ss_stable,\
            kxc_s_Nn_stable, kxc_s_Ns_stable,kxc_Nn_Nn_stable, kxc_Nn_Ns_stable, kxc_Ns_Ns_stable\
            = self.eval_xc_collinear_kernel(xc_code,(rho_aa_c,rho_bb_c),LIBXCT_factor=LIBXCT_factor)[0]
            # = self.eval_xc_noncollinear_kernel(xc_code,(rho_aa_c,rho_bb_c),LIBXCT_factor=LIBXCT_factor)

            kxc_nn[idx_instable] = kxc_nn_stable
            for i in range(3):
                 kxc_ns[i,idx_instable] = kxc_ns_stable*NX_instable[:,i]

            kxc_n_Nn[:,idx_instable] = kxc_n_Nn_stable

            for i in range(3):
                for j in range(3):
                    kxc_n_Ns[i,j,idx_instable] = kxc_n_Ns_stable[i]*NX_instable[:,j]

            for i in range(3):
                for j in range(3):
                    kxc_ss[i,j,idx_instable] = kxc_ss_stable*NX_instable[:,i]*NX_instable[:,j]

            for i in range(3):
                for j in range(3):
                    kxc_s_Nn[i,j,idx_instable] = kxc_s_Nn_stable[i]*NX_instable[:,j]

            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        kxc_s_Ns[i,j,k,idx_instable] = kxc_s_Ns_stable[i]*NX_instable[:,j]*NX_instable[:,k]

            # This term has six subindices given by nabla
            kxc_Nn_Nn[:,idx_instable] = kxc_Nn_Nn_stable

            # Indices i,j mean two nabla, k means projection directions
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        kxc_Nn_Ns[i,j,k,idx_instable] = kxc_Nn_Ns_stable[i,j]*NX_instable[:,k]

            # Indices i mean the indice combined by two nablas, k means projection directions 
            for i in range(6):
                for j in range(3):
                    for k in range(3):
                         kxc_Ns_Ns[i,j,k,idx_instable] = kxc_Ns_Ns_stable[i]*NX_instable[:,j]*NX_instable[:,k]

    #         # spin-flip like part!
            kxc_ss_stable = 0.0
            kxc_s_Ns_stable = 0.0
            kxc_Ns_Ns_stable =0.0

    #         import pdb 
    #         pdb.set_trace()
            # Special 1d picking method
            tarray, weight_array_st= numpy.polynomial.legendre.leggauss(Ndirect_lc)
            # correct the integral area from [-1,1] to [0,1] 
            weight_array_st *= 0.5
            tarray = tarray*0.5+0.5
            assert(numpy.abs(weight_array_st.sum()-1.0)<=1.0E-14)

            import multiprocessing
            import math
            # ~ init some parameters in parallel.
    #         import pdb
    #         pdb.set_trace()
            if ncpu is None:
                ncpu = multiprocessing.cpu_count()
            nsbatch = math.ceil(Ndirect_lc/ncpu)
            Ndirect_lc_list = [(i, i+nsbatch) for i in range(0, Ndirect_lc-nsbatch, nsbatch)]
            if Ndirect_lc_list[-1][-1] < Ndirect_lc:
                Ndirect_lc_list.append((Ndirect_lc_list[-1][-1], Ndirect_lc))

            pool = multiprocessing.Pool()
            para_results = []
            for index in Ndirect_lc_list:
                para_results.append(pool.apply_async(collinear_1d_kernel,(self,xc_code,(rho[:,idx_instable],s_stable[:,:]),
                                                                          tarray[index[0]:index[1]],weight_array_st[index[0]:index[1]],LIBXCT_factor)))
            # ~ finisht the parallel part.
            pool.close()
            pool.join()

            # import pdb
            # pdb.set_trace()
            for para_result in para_results:
                result = para_result.get()
                kxc_ss_stable +=result[0]
                kxc_s_Ns_stable +=result[1]
                kxc_Ns_Ns_stable +=result[2]

            for iaxis in range(1,3):
                for i in range(3):
                    for j in range(3):
                        kxc_ss[i,j,idx_instable]+= kxc_ss_stable*\
                            NX_instable_orth[:,i,iaxis]*\
                            NX_instable_orth[:,j,iaxis]  

            for iaxis in range(1,3):
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            kxc_s_Ns[i,j,k,idx_instable] += kxc_s_Ns_stable[i]*\
                            NX_instable_orth[:,j,iaxis]*NX_instable_orth[:,k,iaxis]

            for iaxis in range(1,3):                
                for i in range(6):
                    for j in range(3):
                        for k in range(3):
                             kxc_Ns_Ns[i,j,k,idx_instable] += kxc_Ns_Ns_stable[i]*\
                                NX_instable_orth[:,j,iaxis]*NX_instable_orth[:,k,iaxis]
            """
            End of numerical instabilities
            """
#         import pdb
#         pdb.set_trace()
        kxc_nn *= weights
        kxc_ns *= weights
        kxc_n_Nn *= weights
        kxc_n_Ns *= weights
        kxc_ss *= weights
        kxc_s_Nn *= weights
        kxc_s_Ns *= weights
        kxc_Nn_Nn *= weights
        kxc_Nn_Ns *= weights
        kxc_Ns_Ns *= weights
#         numpy.save('kxc_Nn_Ns0313', kxc_Nn_Ns)
        return kxc_nn, kxc_ns, kxc_n_Nn, kxc_n_Ns, kxc_ss, kxc_s_Nn, kxc_s_Ns,\
                kxc_Nn_Nn, kxc_Nn_Ns, kxc_Ns_Ns
    else:
        raise NotImplementedError("Relativistic noncollinear TDA isn't implemented in MGGA")

def K_aibj_noncollinear_generator_r(xctype, mo_vir_L, mo_vir_S, mo_occ_L, mo_occ_S, kernel):
    '''K_aibj_noncollinear_generator: calculates <ai|kernel|bj> for noncollinear TDA term.
    
    Parameters
    ----------
    Args:
       xctype : str
            xctype -> LDA, GGA, MGGA. 
        mo_a_vir,mo_a_occ,mo_b_vir,mo_b_occ :  numpy.array
            Virtual alpha molecular orbitals, occupied alpha molecular orbitals, 
            Virtual beta molecular orbitals, occupied beta molecular orbitals, respectively.
        kernel : tuple
            Noncollinear TDA kernel. In LDA and GGA, len(kernel) = 3, 10, respectively.
     
    Returns:
    ----------
       K_aibj : numpy.array 
    
    Raises:
    ----------
        ToDo : K_aibj for noncollinear TDA in MGGA.
    '''
    if xctype == 'LDA':
        kxc_nn,kxc_ns,kxc_ss = kernel 
        # construct gks aa,ab,ba,bb blocks, ai means orbital a to orbital i       
        ai_rho = numpy.einsum('cxpa,cxpi->pai', mo_vir_L.conj(), mo_occ_L, optimize=True)
        ai_rho+= numpy.einsum('cxpa,cxpi->pai', mo_vir_S.conj(), mo_occ_S, optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        betasigma_x = numpy.array(
            [[0,1,0,0],
             [1,0,0,0],
             [0,0,0,-1],
             [0,0,-1,0]]
        )
        betasigma_y = numpy.array(
            [[0,-1.0j,0,0],
             [1.0j,0,0,0],
             [0,0,0,1.0j],
             [0,0,-1.0j,0]]
        )
        betasigma_z = numpy.array(
            [[1,0,0,0],
             [0,-1,0,0],
             [0,0,-1,0],
             [0,0,0,1]]
        )
        ai_Mx = numpy.einsum('cxpa,cd,dxpi->pai', mo_vir_L.conj(), betasigma_x[:2,:2], mo_occ_L, optimize=True)
        ai_Mx+= numpy.einsum('cxpa,cd,dxpi->pai', mo_vir_S.conj(), betasigma_x[2:,2:], mo_occ_S, optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        ai_My = numpy.einsum('cxpa,cd,dxpi->pai', mo_vir_L.conj(), betasigma_y[:2,:2], mo_occ_L, optimize=True)
        ai_My+= numpy.einsum('cxpa,cd,dxpi->pai', mo_vir_S.conj(), betasigma_y[2:,2:], mo_occ_S, optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        ai_Mz = numpy.einsum('cxpa,cd,dxpi->pai', mo_vir_L.conj(), betasigma_z[:2,:2], mo_occ_L, optimize=True)
        ai_Mz+= numpy.einsum('cxpa,cd,dxpi->pai', mo_vir_S.conj(), betasigma_z[2:,2:], mo_occ_S, optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        # construct rho,Mx,My,Mz blocks to calculate K_aibj
        ai_s = [ai_Mx,ai_My,ai_Mz]
        
        # calculate K_aibj
        K_aibj = 0.0
        # print(ai_rho.shape,kxc_nn.shape)
        # kxc_nn
        K_aibj += numpy.einsum('n,nai,nbj->aibj',kxc_nn,
                              ai_rho,ai_rho.conj(),optimize=True)  
        # kxc_ns
        for i in range(3):
            K_aibj += numpy.einsum('n,nai,nbj->aibj',kxc_ns[i],
                                  ai_rho,ai_s[i].conj(),optimize=True)
        
        # kxc_sn
        for i in range(3):
            K_aibj += numpy.einsum('n,nai,nbj->aibj',kxc_ns[i],
                                  ai_s[i],ai_rho.conj(),optimize=True)
            
        # kxc_ss
        for i in range(3):
            for j in range(3):
                K_aibj += numpy.einsum('n,nai,nbj->aibj',kxc_ss[i][j],
                                  ai_s[i],ai_s[j].conj(),optimize=True)
    elif xctype == 'GGA':
        kxc_nn, kxc_ns, kxc_n_Nn, kxc_n_Ns, kxc_ss, kxc_s_Nn, kxc_s_Ns, \
            kxc_Nn_Nn, kxc_Nn_Ns, kxc_Ns_Ns = kernel
        ai_rho = numpy.einsum('cxpa,cpi->xpai', mo_vir_L.conj(), mo_occ_L[:,0], optimize=True)
        ai_rho+= numpy.einsum('cpa,cxpi->xpai', mo_vir_L[:,0].conj(), mo_occ_L, optimize=True)
        ai_rho+= numpy.einsum('cxpa,cpi->xpai', mo_vir_S.conj(), mo_occ_S[:,0], optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        ai_rho+= numpy.einsum('cpa,cxpi->xpai', mo_vir_S[:,0].conj(), mo_occ_S, optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        ai_rho[0]*= 0.5
        betasigma_x = numpy.array(
            [[0,1,0,0],
             [1,0,0,0],
             [0,0,0,-1],
             [0,0,-1,0]]
        )
        betasigma_y = numpy.array(
            [[0,-1.0j,0,0],
             [1.0j,0,0,0],
             [0,0,0,1.0j],
             [0,0,-1.0j,0]]
        )
        betasigma_z = numpy.array(
            [[1,0,0,0],
             [0,-1,0,0],
             [0,0,-1,0],
             [0,0,0,1]]
        )
        ai_Mx = numpy.einsum('cxpa,cd,dpi->xpai', mo_vir_L.conj(), betasigma_x[:2,:2], mo_occ_L[:,0], optimize=True)
        ai_Mx+= numpy.einsum('cpa,cd,dxpi->xpai', mo_vir_L[:,0].conj(), betasigma_x[:2,:2], mo_occ_L, optimize=True)
        ai_Mx+= numpy.einsum('cxpa,cd,dpi->xpai', mo_vir_S.conj(), betasigma_x[2:,2:], mo_occ_S[:,0], optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        ai_Mx+= numpy.einsum('cpa,cd,dxpi->xpai', mo_vir_S[:,0].conj(), betasigma_x[2:,2:], mo_occ_S, optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        ai_My = numpy.einsum('cxpa,cd,dpi->xpai', mo_vir_L.conj(), betasigma_y[:2,:2], mo_occ_L[:,0], optimize=True)
        ai_My+= numpy.einsum('cpa,cd,dxpi->xpai', mo_vir_L[:,0].conj(), betasigma_y[:2,:2], mo_occ_L, optimize=True)
        ai_My+= numpy.einsum('cxpa,cd,dpi->xpai', mo_vir_S.conj(), betasigma_y[2:,2:], mo_occ_S[:,0], optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        ai_My+= numpy.einsum('cpa,cd,dxpi->xpai', mo_vir_S[:,0].conj(), betasigma_y[2:,2:], mo_occ_S, optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        ai_Mz = numpy.einsum('cxpa,cd,dpi->xpai', mo_vir_L.conj(), betasigma_z[:2,:2], mo_occ_L[:,0], optimize=True)
        ai_Mz+= numpy.einsum('cpa,cd,dxpi->xpai', mo_vir_L[:,0].conj(), betasigma_z[:2,:2], mo_occ_L, optimize=True)
        ai_Mz+= numpy.einsum('cxpa,cd,dpi->xpai', mo_vir_S.conj(), betasigma_z[2:,2:], mo_occ_S[:,0], optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        ai_Mz+= numpy.einsum('cpa,cd,dxpi->xpai', mo_vir_S[:,0].conj(), betasigma_z[2:,2:], mo_occ_S, optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        
        ai_Mx[0]*=0.5
        ai_My[0]*=0.5
        ai_Mz[0]*=0.5
        ai_s = [ai_Mx,ai_My,ai_Mz]
        
        K_aibj = 0.0

        # kxc_nn
        K_aibj += numpy.einsum('n,nai,nbj->aibj',kxc_nn,
                              ai_rho[0],ai_rho[0].conj(),optimize=True)
        # kxc_ns
        for i in range(3):
            K_aibj += numpy.einsum('n,nai,nbj->aibj',kxc_ns[i],
                                  ai_rho[0],ai_s[i][0].conj(),optimize=True)
        
        # kxc_sn
        for i in range(3):
            K_aibj += numpy.einsum('n,nai,nbj->aibj',kxc_ns[i],
                                  ai_s[i][0],ai_rho[0].conj(),optimize=True)
        # kxc_n_Nn
        K_aibj += numpy.einsum('gn,nai,gnbj->aibj',kxc_n_Nn,
                               ai_rho[0],ai_rho[1:].conj(),optimize=True)
        
        # kxc_Nn_n
        K_aibj += numpy.einsum('gn,gnai,nbj->aibj',kxc_n_Nn,
                               ai_rho[1:],ai_rho[0].conj(),optimize=True)
        
        # kxc_n_Ns
        for i in range(3):
            K_aibj += numpy.einsum('gn,nai,gnbj->aibj',kxc_n_Ns[:,i],
                                    ai_rho[0],ai_s[i][1:].conj(),optimize=True)
        
        # kxc_Ns_n
        for i in range(3):
            K_aibj += numpy.einsum('gn,gnai,nbj->aibj',kxc_n_Ns[:,i],
                                    ai_s[i][1:],ai_rho[0].conj(),optimize=True)
                
        # kxc_ss
        for i in range(3):
            for j in range(3):
                K_aibj += numpy.einsum('n,nai,nbj->aibj',kxc_ss[i][j],
                                  ai_s[i][0],ai_s[j][0].conj(),optimize=True)
        # kxc_s_Nn
        for i in range(3):
            K_aibj += numpy.einsum('gn,nai,gnbj->aibj',kxc_s_Nn[:,i],
                                   ai_s[i][0],ai_rho[1:].conj(),optimize=True)
            
        # kxc_Nn_s
        for i in range(3):
            K_aibj += numpy.einsum('gn,gnai,nbj->aibj',kxc_s_Nn[:,i],
                                   ai_rho[1:],ai_s[i][0].conj(),optimize=True)
        
        # kxc_s_Ns
        for i in range(3):
            for j in range(3):
                K_aibj += numpy.einsum('gn,nai,gnbj->aibj',kxc_s_Ns[:,i,j],
                                       ai_s[i][0],ai_s[j][1:].conj(),optimize=True)
                    
        # kxc_Ns_s
        for i in range(3):
            for j in range(3):
                K_aibj += numpy.einsum('gn,gnai,nbj->aibj',kxc_s_Ns[:,i,j],
                                        ai_s[i][1:],ai_s[j][0].conj(),optimize=True)
                      
        offset2 = numint_gksmc.get_2d_offset()
        # kxc_Nn_Nn

        for i in range(3):
            for j in range(3):
                K_aibj += numpy.einsum('n,nai,nbj->aibj',kxc_Nn_Nn[offset2[i,j]],
                                        ai_rho[i+1],ai_rho[j+1].conj(),optimize=True)

        # kxc_Nn_Ns
        # Note ai_ns[k][j] ! This is due to the storation form of coresponding kernel. 
        for i in range(3):
            K_aibj += numpy.einsum('ghn,gnai,hnbj->aibj',kxc_Nn_Ns[:,:,i],
                                ai_rho[1:],ai_s[i][1:].conj(),optimize=True)
                    
        # kxc_Ns_Nn
        for i in range(3):
            K_aibj += numpy.einsum('ghn,hnai,gnbj->aibj',kxc_Nn_Ns[:,:,i],
                                ai_s[i][1:],ai_rho[1:].conj(),optimize=True)
    
        # kxc_Ns_Ns
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        K_aibj += numpy.einsum('n,nai,nbj->aibj',kxc_Ns_Ns[offset2[i,j],k,l],
                                        ai_s[k][i+1],ai_s[l][j+1].conj(),optimize=True)

    else:
        raise NotImplementedError("Only LDA is implemented.")
    
    return K_aibj 

def nr_collinear_tddft_mc(self,xc_code,rhop,grids,Ndirect,C_mo,Extype='SPIN_CONSERVED',LIBXCT_factor=None,ncpu=None):
    '''nr_collinear_tda: calculates the K_aibj for Multi-Collinear TDA for collinear system. 
       
    Parameters
    ----------
    Args:
        xc_code : str
            Name of exchange-correlation functional.
        rhop : tuple
            Density and magenization density norm with form (rho,s), whrer rho,s with a shape (nvar,ngrid). 
            In LDA, GGA MGGA, nvar is 1,4,4 respectively, meaning 1, nabla_x, nabla_y, nabla_z.
        grids : mf.object
            mf is an object for DFT class and grids.coords and grids.weights are used here, supplying the sample 
            points and weights in real space.
        Ndirect : int
            The number of sample points in spin space, for gauss-ledengre distribution.
        C_mo : tuple
            Molecular orbital cofficience.
    
    Kwargs:
        Extype : str
            Three excited energy types -> SPIN_FLIP_UP, SPIN_FLIP_DOWN, SPIN_CONSERVED.
        LIBXCT_factor : double or int
            The Threshold of for derivatives obatained from libxc.
            Deafult is None, which means no Threshold. 1e-10 is recommended.
        ncpu : 
            Number of cpu workers.
    
    Returns:
    ----------
        K_aibj : numpy.array for SPIN_FLIP excited type and tuple for SPIN_CONSERVED excited type.
            aibj means related orbitals. a,b are virtual orbitals, and i,j are occupied orbitals.
                
    Raises:
    ----------
        ToDo : SPIN_CONSERVED MGGA collinear TDA.
    '''
    xctype = self._xc_type(xc_code)
    ngrid = grids.coords.shape[0]
    weights = grids.weights
    
    import multiprocessing
    if ncpu is None:
        ncpu = multiprocessing.cpu_count()
    import math
    nsbatch = math.ceil(ngrid/ncpu)
    NX_list = [(i, i+nsbatch) for i in range(0, ngrid-nsbatch, nsbatch)]
    if NX_list[-1][-1] < ngrid:
        NX_list.append((NX_list[-1][-1], ngrid))
    pool = multiprocessing.Pool()
    para_results = []
    
    mo_a_vir,mo_a_occ,mo_b_vir,mo_b_occ = C_mo
    
    K_aibj_A_sfd = 0.0
    K_aibj_B_sfd = 0.0
    K_aibj_A_sfu = 0.0
    K_aibj_B_sfu = 0.0

    if 'SPIN_FLIP' in Extype:
    # if Extype=='SPIN_FLIP_UP' or Extype=='SPIN_FLIP_DOWN':
        kernel = collinear_tdamc_kernel(self,xc_code,rhop,Ndirect,Extype=Extype,
                                  LIBXCT_factor=LIBXCT_factor,ncpu=ncpu)
        
        if xctype == 'LDA':
            for para in NX_list:
                idxi,idxf = para
                para_results.append(pool.apply_async(K_aibj_collinear_generator_flip_tddft,
                                    (xctype, mo_a_vir[idxi:idxf],mo_a_occ[idxi:idxf],
                                             mo_b_vir[idxi:idxf],mo_b_occ[idxi:idxf], 
                                             kernel[idxi:idxf], weights[idxi:idxf])))
            pool.close()
            pool.join()
            
        elif xctype == 'GGA' or xctype == 'MGGA':
            for para in NX_list:
                idxi,idxf = para
                kernel_para = []
                for i in range(len(kernel)):
                    kernel_para.append(kernel[i][...,idxi:idxf])
                para_results.append(pool.apply_async(K_aibj_collinear_generator_flip_tddft,
                                    (xctype, mo_a_vir[:,idxi:idxf], mo_a_occ[:,idxi:idxf], 
                                             mo_b_vir[:,idxi:idxf], mo_b_occ[:,idxi:idxf], 
                                    kernel_para, weights[idxi:idxf])))
            pool.close()
            pool.join()
        
        for result_para in para_results:
            result = result_para.get()
            K_aibj_A_sfd += result[0]
            K_aibj_B_sfd += result[1]
            K_aibj_A_sfu += result[2]
            K_aibj_B_sfu += result[3]
           
        return K_aibj_A_sfd,K_aibj_B_sfd,K_aibj_A_sfu,K_aibj_B_sfu
    
    elif Extype=='SPIN_CONSERVED':
        mo_a_vir,mo_a_occ,mo_b_vir,mo_b_occ = C_mo 
        K_aibj_A_aaaa = 0.0
        K_aibj_A_aabb = 0.0
        K_aibj_A_bbaa = 0.0
        K_aibj_A_bbbb = 0.0
        
        K_aibj_B_aaaa = 0.0
        K_aibj_B_aabb = 0.0
        K_aibj_B_bbaa = 0.0
        K_aibj_B_bbbb = 0.0
        
        kernel = collinear_tdamc_kernel(self,xc_code,rhop,Ndirect,Extype=Extype,
                                        LIBXCT_factor=LIBXCT_factor,ncpu=ncpu)
        # import pdb
        # pdb.set_trace()
        if xctype == 'LDA':
            for para in NX_list:
                idxi,idxf = para
                kernel_para = []
                for i in range(len(kernel)):
                    kernel_para.append(kernel[i][...,idxi:idxf])
                para_results.append(pool.apply_async(K_aibj_collinear_generator_conserved_tddft,
                                    (xctype, mo_a_vir[idxi:idxf],mo_a_occ[idxi:idxf],
                                    mo_b_vir[idxi:idxf],mo_b_occ[idxi:idxf], 
                                    kernel_para, weights[idxi:idxf])))
            pool.close()
            pool.join()
        
        elif xctype == 'GGA':
            for para in NX_list:
                idxi,idxf = para
                kernel_para = []
                for i in range(len(kernel)):
                    kernel_para.append(kernel[i][...,idxi:idxf])
                para_results.append(pool.apply_async(K_aibj_collinear_generator_conserved_tddft,
                                    (xctype, mo_a_vir[:,idxi:idxf],mo_a_occ[:,idxi:idxf],
                                    mo_b_vir[:,idxi:idxf],mo_b_occ[:,idxi:idxf], 
                                    kernel_para, weights[idxi:idxf])))    
            pool.close()
            pool.join()
        
        elif xctype == 'MGGA':
            raise NotImplementedError("Spin-conserved scheme isn't implemented in Meta-GGA")
        
        # import pdb
        # pdb.set_trace()
        for result_para in para_results:
            result = result_para.get()
            K_aibj_A_aaaa += result[0][0]
            K_aibj_A_aabb += result[0][1]
            K_aibj_A_bbaa += result[0][2]
            K_aibj_A_bbbb += result[0][3]
            
            K_aibj_B_aaaa += result[1][0]
            K_aibj_B_aabb += result[1][1]
            K_aibj_B_bbaa += result[1][2]
            K_aibj_B_bbbb += result[1][3]

        return ((K_aibj_A_aaaa,K_aibj_A_aabb,K_aibj_A_bbaa,K_aibj_A_bbbb),
                (K_aibj_B_aaaa,K_aibj_B_aabb,K_aibj_B_bbaa,K_aibj_B_bbbb))


def K_aibj_collinear_generator_flip_tddft(xctype, mo_a_vir,mo_a_occ,mo_b_vir,mo_b_occ, kernel, weights):
    '''K_aibj_collinear_generator_flip: calculates <ai|kernel|bj> for collinear spin flip term. 
    
    Parameters
    ----------
    Args:
        xctype : str
            xctype -> LDA, GGA, MGGA. 
        mo_a_vir : numpy.array
            Virtual alpha molecular orbitals. 
        mo_b_occ : numpy.array
            Occupied beta molecular orbitals.
        kernel : tuple
            Spin_flip_kernel. In LDA, GGA and MGGA, len(kernel) = 1, 3, 6, respectively.
        weights : numpy.array
            Weights of sample points in real space.
  
    Returns:
    ----------
        K_aibj : numpy.array
    '''
    if xctype == 'LDA':
        # construct gks ab(ba) blocks, ai means orbital a to orbital i
        ai_ab = numpy.einsum('na,ni->nai',mo_a_vir.conj(),mo_b_occ,optimize=True)
        ai_ba = numpy.einsum('nb,nj->nbj',mo_b_vir.conj(),mo_a_occ,optimize=True)
        
        s_s = kernel
        s_s *= weights
        
        # s_s part
        # Note: .conj()
        K_aibj_A_sfd = numpy.einsum('n,nai,nbj->aibj',s_s, ai_ba, ai_ba.conj(),optimize=True)
        K_aibj_B_sfd = numpy.einsum('n,nai,nbj->aibj',s_s, ai_ba.conj(), ai_ab.conj(),optimize=True)
        K_aibj_A_sfu = numpy.einsum('n,nai,nbj->aibj',s_s, ai_ab, ai_ab.conj(),optimize=True)
        K_aibj_B_sfu = numpy.einsum('n,nai,nbj->aibj',s_s, ai_ab.conj(), ai_ba.conj(),optimize=True)
        
    elif xctype == 'GGA':
        # construct gks ab(ba) blocks, ai means orbital a to orbital i
        ai_ab = numpy.einsum('na,ni->nai',mo_a_vir[0].conj(),mo_b_occ[0],optimize=True)
        ai_ba = numpy.einsum('na,ni->nai',mo_b_vir[0].conj(),mo_a_occ[0],optimize=True)
        
        # construct gradient terms
        ai_na_b = numpy.einsum('gna,ni->gnai',mo_a_vir[1:4].conj(),mo_b_occ[0],optimize=True)
        ai_a_nb = numpy.einsum('na,gni->gnai',mo_a_vir[0].conj(),mo_b_occ[1:4],optimize=True)
        ai_nb_a = numpy.einsum('gnb,nj->gnbj',mo_b_vir[1:4].conj(),mo_a_occ[0],optimize=True)
        ai_b_na = numpy.einsum('nb,gnj->gnbj',mo_b_vir[0].conj(),mo_a_occ[1:4],optimize=True)
        
        # construct nrho,nMz blocks to calculate K_aibj
        ai_nMx_ab = ai_na_b + ai_a_nb
        ai_nMx_ba = ai_nb_a + ai_b_na
        
        s_s, s_Ns, Ns_Ns =  kernel
        s_s *= weights
        s_Ns *= weights
        Ns_Ns *= weights
        
        # s_s part
        K_aibj_A_sfd = numpy.einsum('n,nai,nbj->aibj',s_s, ai_ba, ai_ba.conj(),optimize=True)
        K_aibj_B_sfd = numpy.einsum('n,nai,nbj->aibj',s_s, ai_ba, ai_ab,optimize=True)
        K_aibj_A_sfu = numpy.einsum('n,nai,nbj->aibj',s_s, ai_ab, ai_ab.conj(),optimize=True)
        K_aibj_B_sfu = numpy.einsum('n,nai,nbj->aibj',s_s, ai_ab, ai_ba,optimize=True)
        
        # Ns_s part
        K_aibj_A_sfd += numpy.einsum('gn,gnai,nbj->aibj',
                             s_Ns, ai_nMx_ba.conj(), 
                              ai_ba.conj(),optimize=True)
        K_aibj_B_sfd += numpy.einsum('gn,gnai,nbj->aibj',
                             s_Ns, ai_nMx_ba.conj(), 
                              ai_ab.conj(),optimize=True)
        K_aibj_A_sfu += numpy.einsum('gn,gnai,nbj->aibj',
                             s_Ns, ai_nMx_ab.conj(), 
                              ai_ab.conj(),optimize=True)
        K_aibj_B_sfu += numpy.einsum('gn,gnai,nbj->aibj',
                             s_Ns, ai_nMx_ab.conj(), 
                              ai_ba.conj(),optimize=True)
        
        # s_Ns part
        K_aibj_A_sfd += numpy.einsum('gn,nai,gnbj->aibj',s_Ns, ai_ba.conj(), 
                              ai_nMx_ba.conj(),optimize=True)
        K_aibj_B_sfd += numpy.einsum('gn,nai,gnbj->aibj',s_Ns, ai_ba.conj(), 
                              ai_nMx_ab.conj(),optimize=True)
        K_aibj_A_sfu += numpy.einsum('gn,nai,gnbj->aibj',s_Ns, ai_ab.conj(), 
                              ai_nMx_ab.conj(),optimize=True)
        K_aibj_B_sfu += numpy.einsum('gn,nai,gnbj->aibj',s_Ns, ai_ab.conj(), 
                              ai_nMx_ba.conj(),optimize=True)
        
        # Ns_Ns part
        K_aibj_A_sfd += numpy.einsum('ghn,gnai,hnbj->aibj',Ns_Ns, ai_nMx_ba.conj(), 
                              ai_nMx_ba.conj(),optimize=True)
        K_aibj_B_sfd += numpy.einsum('ghn,gnai,hnbj->aibj',Ns_Ns, ai_nMx_ba.conj(), 
                              ai_nMx_ab.conj(),optimize=True)
        K_aibj_A_sfu += numpy.einsum('ghn,gnai,hnbj->aibj',Ns_Ns, ai_nMx_ab.conj(), 
                              ai_nMx_ab.conj(),optimize=True)
        K_aibj_B_sfu += numpy.einsum('ghn,gnai,hnbj->aibj',Ns_Ns, ai_nMx_ab.conj(), 
                              ai_nMx_ba.conj(),optimize=True)
        

    elif xctype == 'MGGA':
        # construct gks ab(ba) blocks, ai means orbital a to orbital i
        ai_ab = numpy.einsum('na,ni->nai',mo_a_vir[0].conj(),mo_b_occ[0],optimize=True)
        ai_ba = numpy.einsum('na,ni->nai',mo_b_vir[0].conj(),mo_a_occ[0],optimize=True)
        
        # construct gradient terms
        ai_na_b = numpy.einsum('gna,ni->gnai',mo_a_vir[1:4].conj(),mo_b_occ[0],optimize=True)
        ai_a_nb = numpy.einsum('na,gni->gnai',mo_a_vir[0].conj(),mo_b_occ[1:4],optimize=True)
        ai_na_nb = 0.5*numpy.einsum('gna,gni->nai',mo_a_vir[1:4].conj(),mo_b_occ[1:4],optimize=True)
        
        ai_nb_a = numpy.einsum('gnb,nj->gnbj',mo_b_vir[1:4].conj(),mo_a_occ[0],optimize=True)
        ai_b_na = numpy.einsum('nb,gnj->gnbj',mo_b_vir[0].conj(),mo_a_occ[1:4],optimize=True)
        ai_nb_na = 0.5*numpy.einsum('gnb,gnj->nbj',mo_b_vir[1:4].conj(),mo_a_occ[1:4],optimize=True)
        
        # construct nMx,tau blocks to calculate K_aibj
        ai_nMx_ab = ai_na_b + ai_a_nb
        ai_nMx_ba = ai_nb_a + ai_b_na
          
        s_s, s_Ns, Ns_Ns,u_u,s_u, Ns_u = kernel
        s_s *= weights
        s_Ns *= weights
        Ns_Ns *= weights
        u_u *= weights
        s_u *= weights 
        Ns_u *= weights
        
        # s_s part
        K_aibj_A_sfd = numpy.einsum('n,nai,nbj->aibj',s_s, ai_ba, ai_ba.conj(),optimize=True)
        K_aibj_B_sfd = numpy.einsum('n,nai,nbj->aibj',s_s, ai_ba, ai_ab,optimize=True)
        K_aibj_A_sfu = numpy.einsum('n,nai,nbj->aibj',s_s, ai_ab, ai_ab.conj(),optimize=True)
        K_aibj_B_sfu = numpy.einsum('n,nai,nbj->aibj',s_s, ai_ab, ai_ba,optimize=True)
        
        # Ns_s part
        K_aibj_A_sfd += numpy.einsum('gn,gnai,nbj->aibj',
                             s_Ns, ai_nMx_ba.conj(), 
                              ai_ba.conj(),optimize=True)
        K_aibj_B_sfd += numpy.einsum('gn,gnai,nbj->aibj',
                             s_Ns, ai_nMx_ba.conj(), 
                              ai_ab.conj(),optimize=True)
        K_aibj_A_sfu += numpy.einsum('gn,gnai,nbj->aibj',
                             s_Ns, ai_nMx_ab.conj(), 
                              ai_ab.conj(),optimize=True)
        K_aibj_B_sfu += numpy.einsum('gn,gnai,nbj->aibj',
                             s_Ns, ai_nMx_ab.conj(), 
                              ai_ba.conj(),optimize=True)
        
        # s_Ns part
        K_aibj_A_sfd += numpy.einsum('gn,nai,gnbj->aibj',s_Ns, ai_ba.conj(), 
                              ai_nMx_ba.conj(),optimize=True)
        K_aibj_B_sfd += numpy.einsum('gn,nai,gnbj->aibj',s_Ns, ai_ba.conj(), 
                              ai_nMx_ab.conj(),optimize=True)
        K_aibj_A_sfu += numpy.einsum('gn,nai,gnbj->aibj',s_Ns, ai_ab.conj(), 
                              ai_nMx_ab.conj(),optimize=True)
        K_aibj_B_sfu += numpy.einsum('gn,nai,gnbj->aibj',s_Ns, ai_ab.conj(), 
                              ai_nMx_ba.conj(),optimize=True)
        
        # Ns_Ns part
        K_aibj_A_sfd += numpy.einsum('ghn,gnai,hnbj->aibj',Ns_Ns, ai_nMx_ba.conj(), 
                              ai_nMx_ba.conj(),optimize=True)
        K_aibj_B_sfd += numpy.einsum('ghn,gnai,hnbj->aibj',Ns_Ns, ai_nMx_ba.conj(), 
                              ai_nMx_ab.conj(),optimize=True)
        K_aibj_A_sfu += numpy.einsum('ghn,gnai,hnbj->aibj',Ns_Ns, ai_nMx_ab.conj(), 
                              ai_nMx_ab.conj(),optimize=True)
        K_aibj_B_sfu += numpy.einsum('ghn,gnai,hnbj->aibj',Ns_Ns, ai_nMx_ab.conj(), 
                              ai_nMx_ba.conj(),optimize=True)

        # u_u part
        K_aibj_A_sfd += numpy.einsum('n,nai,nbj->aibj',u_u, ai_nb_na,ai_nb_na.conj(),optimize=True)
        K_aibj_B_sfd += numpy.einsum('n,nai,nbj->aibj',u_u, ai_nb_na,ai_na_nb.conj(),optimize=True)
        K_aibj_A_sfu += numpy.einsum('n,nai,nbj->aibj',u_u, ai_na_nb,ai_na_nb.conj(),optimize=True)
        K_aibj_B_sfu += numpy.einsum('n,nai,nbj->aibj',u_u, ai_na_nb,ai_nb_na.conj(),optimize=True)

        # s_u part
        K_aibj_A_sfd += numpy.einsum('n,nai,nbj->aibj',s_u, ai_ba, ai_nb_na.conj(),optimize=True)
        K_aibj_B_sfd += numpy.einsum('n,nai,nbj->aibj',s_u, ai_ba, ai_na_nb.conj(),optimize=True)
        K_aibj_A_sfu += numpy.einsum('n,nai,nbj->aibj',s_u, ai_ab, ai_na_nb.conj(),optimize=True)
        K_aibj_B_sfu += numpy.einsum('n,nai,nbj->aibj',s_u, ai_ab, ai_nb_na.conj(),optimize=True)
        
        # u_s part
        K_aibj_A_sfd += numpy.einsum('n,nai,nbj->aibj',s_u, ai_nb_na,ai_ba.conj(),optimize=True)
        K_aibj_B_sfd += numpy.einsum('n,nai,nbj->aibj',s_u, ai_nb_na,ai_ab.conj(),optimize=True)
        K_aibj_A_sfu += numpy.einsum('n,nai,nbj->aibj',s_u, ai_na_nb,ai_ab.conj(),optimize=True)
        K_aibj_B_sfu += numpy.einsum('n,nai,nbj->aibj',s_u, ai_na_nb,ai_ba.conj(),optimize=True)
        
        # Ns_u part
        K_aibj_A_sfd += numpy.einsum('gn,gnai,nbj->aibj',Ns_u, ai_nMx_ba, ai_nb_na.conj(),optimize=True)
        K_aibj_B_sfd += numpy.einsum('gn,gnai,nbj->aibj',Ns_u, ai_nMx_ba, ai_na_nb.conj(),optimize=True)
        K_aibj_A_sfu += numpy.einsum('gn,gnai,nbj->aibj',Ns_u, ai_nMx_ab, ai_na_nb.conj(),optimize=True)
        K_aibj_B_sfu += numpy.einsum('gn,gnai,nbj->aibj',Ns_u, ai_nMx_ab, ai_nb_na.conj(),optimize=True)
        
        # u_Ns part
        K_aibj_A_sfd += numpy.einsum('gn,nai,gnbj->aibj',Ns_u, ai_nb_na, ai_nMx_ba.conj(),optimize=True)
        K_aibj_B_sfd += numpy.einsum('gn,nai,gnbj->aibj',Ns_u, ai_nb_na, ai_nMx_ab.conj(),optimize=True)
        K_aibj_A_sfu += numpy.einsum('gn,nai,gnbj->aibj',Ns_u, ai_na_nb, ai_nMx_ab.conj(),optimize=True)
        K_aibj_B_sfu += numpy.einsum('gn,nai,gnbj->aibj',Ns_u, ai_na_nb, ai_nMx_ba.conj(),optimize=True)
    else:
        raise NotImplementedError("Please check the xc_code keyword")
        
    return K_aibj_A_sfd,K_aibj_B_sfd,K_aibj_A_sfu,K_aibj_B_sfu

def K_aibj_collinear_generator_conserved_tddft(xctype, mo_a_vir,mo_a_occ,mo_b_vir,mo_b_occ, kernel, weights):
    '''K_aibj_collinear_generator_conserved: calculates <ai|kernel|bj> for collinear spin conserved term.
    
    Parameters
    ----------
    Args:
        xctype : str
            xctype -> LDA, GGA, MGGA. 
        mo_a_vir,mo_a_occ,mo_b_vir,mo_b_occ :  numpy.array
            Virtual alpha molecular orbitals, occupied alpha molecular orbitals, 
            Virtual beta molecular orbitals, occupied beta molecular orbitals, respectively.
        kernel : tuple
            Spin_flip_kernel. In LDA and GGA, len(kernel) = 3, 10, respectively.
        weights : numpy.array
            Weights of sample points in real space.
    
    Returns:
    ----------
        K_aibj : numpy.array
            K_aibj.len()=4, including K_aibj_aaaa,K_aibj_aabb,K_aibj_bbaa,K_aibj_bbbb.
    
    Raises:
    ----------
        ToDo : Spin_conserved in MGGA.
    '''
    # calculate K_aibj
    K_aibj_A_aaaa = 0.0
    K_aibj_A_aabb = 0.0
    K_aibj_A_bbaa = 0.0
    K_aibj_A_bbbb = 0.0
    
    K_aibj_B_aaaa = 0.0
    K_aibj_B_aabb = 0.0
    K_aibj_B_bbaa = 0.0
    K_aibj_B_bbbb = 0.0
    
    if xctype == 'LDA':
        n_n,n_s,s_s = kernel
        # This is a yinhuan !!!
        n_n *= weights
        n_s *= weights
        s_s *= weights
        
        # construct gks aa,ab,ba,bb blocks, ai means orbital a to orbital i
        # Substitute ai_rho with ai_aa and ai_bb
        ai_aa = numpy.einsum('na,ni->nai',mo_a_vir.conj(),mo_a_occ,optimize=True)
        ai_bb = numpy.einsum('na,ni->nai',mo_b_vir.conj(),mo_b_occ,optimize=True)
        
        # nn
        K_aibj_A_aaaa += numpy.einsum('n,nai,nbj->aibj',n_n,ai_aa,ai_aa.conj(),optimize=True)   
        K_aibj_A_aabb += numpy.einsum('n,nai,nbj->aibj',n_n,ai_aa,ai_bb.conj(),optimize=True)
        K_aibj_A_bbaa += numpy.einsum('n,nai,nbj->aibj',n_n,ai_bb,ai_aa.conj(),optimize=True)
        K_aibj_A_bbbb += numpy.einsum('n,nai,nbj->aibj',n_n,ai_bb,ai_bb.conj(),optimize=True)
        
        # ns
        K_aibj_A_aaaa += numpy.einsum('n,nai,nbj->aibj',n_s,ai_aa,ai_aa.conj(),optimize=True)  
        K_aibj_A_aabb += -1*numpy.einsum('n,nai,nbj->aibj',n_s,ai_aa,ai_bb.conj(),optimize=True)
        K_aibj_A_bbaa += numpy.einsum('n,nai,nbj->aibj',n_s,ai_bb,ai_aa.conj(),optimize=True)
        K_aibj_A_bbbb += -1*numpy.einsum('n,nai,nbj->aibj',n_s,ai_bb,ai_bb.conj(),optimize=True)
        
        # sn
        K_aibj_A_aaaa += numpy.einsum('n,nai,nbj->aibj',n_s,ai_aa,ai_aa.conj(),optimize=True)  
        K_aibj_A_aabb += numpy.einsum('n,nai,nbj->aibj',n_s,ai_aa,ai_bb.conj(),optimize=True)
        K_aibj_A_bbaa += -1*numpy.einsum('n,nai,nbj->aibj',n_s,ai_bb,ai_aa.conj(),optimize=True)
        K_aibj_A_bbbb += -1*numpy.einsum('n,nai,nbj->aibj',n_s,ai_bb,ai_bb.conj(),optimize=True)
 
        # ss
        K_aibj_A_aaaa += numpy.einsum('n,nai,nbj->aibj',s_s,ai_aa,ai_aa.conj(),optimize=True)  
        K_aibj_A_aabb += -1*numpy.einsum('n,nai,nbj->aibj',s_s,ai_aa,ai_bb.conj(),optimize=True)
        K_aibj_A_bbaa += -1*numpy.einsum('n,nai,nbj->aibj',s_s,ai_bb,ai_aa.conj(),optimize=True)
        K_aibj_A_bbbb += numpy.einsum('n,nai,nbj->aibj',s_s,ai_bb,ai_bb.conj(),optimize=True)
        
        
        # nn
        K_aibj_B_aaaa += numpy.einsum('n,nai,nbj->aibj',n_n,ai_aa,ai_aa,optimize=True)   
        K_aibj_B_aabb += numpy.einsum('n,nai,nbj->aibj',n_n,ai_aa,ai_bb,optimize=True)
        K_aibj_B_bbaa += numpy.einsum('n,nai,nbj->aibj',n_n,ai_bb,ai_aa,optimize=True)
        K_aibj_B_bbbb += numpy.einsum('n,nai,nbj->aibj',n_n,ai_bb,ai_bb,optimize=True)
        
        # ns
        K_aibj_B_aaaa    += numpy.einsum('n,nai,nbj->aibj',n_s,ai_aa,ai_aa,optimize=True) 
        K_aibj_B_aabb += -1*numpy.einsum('n,nai,nbj->aibj',n_s,ai_aa,ai_bb,optimize=True)
        K_aibj_B_bbaa    += numpy.einsum('n,nai,nbj->aibj',n_s,ai_bb,ai_aa,optimize=True)
        K_aibj_B_bbbb += -1*numpy.einsum('n,nai,nbj->aibj',n_s,ai_bb,ai_bb,optimize=True)
        
        # sn
        K_aibj_B_aaaa    += numpy.einsum('n,nai,nbj->aibj',n_s,ai_aa,ai_aa,optimize=True)  
        K_aibj_B_aabb    += numpy.einsum('n,nai,nbj->aibj',n_s,ai_aa,ai_bb,optimize=True)
        K_aibj_B_bbaa += -1*numpy.einsum('n,nai,nbj->aibj',n_s,ai_bb,ai_aa,optimize=True)
        K_aibj_B_bbbb += -1*numpy.einsum('n,nai,nbj->aibj',n_s,ai_bb,ai_bb,optimize=True)
 
        # ss
        K_aibj_B_aaaa    += numpy.einsum('n,nai,nbj->aibj',s_s,ai_aa,ai_aa,optimize=True)  
        K_aibj_B_aabb += -1*numpy.einsum('n,nai,nbj->aibj',s_s,ai_aa,ai_bb,optimize=True)
        K_aibj_B_bbaa += -1*numpy.einsum('n,nai,nbj->aibj',s_s,ai_bb,ai_aa,optimize=True)
        K_aibj_B_bbbb    += numpy.einsum('n,nai,nbj->aibj',s_s,ai_bb,ai_bb,optimize=True)
        
    elif xctype == 'GGA':
        n_n,n_s,n_Nn,n_Ns,s_s,s_Nn,s_Ns,Nn_Nn,Nn_Ns,Ns_Ns = kernel
        n_n *= weights
        n_s *= weights
        n_Nn *= weights
        n_Ns *= weights
        s_s *= weights
        s_Nn *= weights
        s_Ns *= weights
        Nn_Nn *= weights
        Nn_Ns *= weights
        Ns_Ns *= weights

        # construct gks aa,ab,ba,bb blocks, ai means orbital a to orbital i
        ai_aa = numpy.einsum('na,ni->nai',mo_a_vir[0].conj(),mo_a_occ[0],optimize=True)
        ai_bb = numpy.einsum('na,ni->nai',mo_b_vir[0].conj(),mo_b_occ[0],optimize=True)
        
        # construct gradient terms
        ai_na_a = numpy.einsum('gna,ni->gnai',mo_a_vir[1:4].conj(),mo_a_occ[0],optimize=True)
        ai_a_na = numpy.einsum('na,gni->gnai',mo_a_vir[0].conj(),mo_a_occ[1:4],optimize=True)
        ai_nb_b = numpy.einsum('gna,ni->gnai',mo_b_vir[1:4].conj(),mo_b_occ[0],optimize=True)
        ai_b_nb = numpy.einsum('na,gni->gnai',mo_b_vir[0].conj(),mo_b_occ[1:4],optimize=True)
        
        # construct nrho,nMz blocks to calculate K_aibj
        ai_Naa = ai_na_a + ai_a_na 
        ai_Nbb = ai_nb_b + ai_b_nb
        
        # nn
        K_aibj_A_aaaa += numpy.einsum('n,nai,nbj->aibj',n_n,ai_aa,ai_aa.conj(),optimize=True)  
        K_aibj_A_aabb += numpy.einsum('n,nai,nbj->aibj',n_n,ai_aa,ai_bb.conj(),optimize=True)
        K_aibj_A_bbaa += numpy.einsum('n,nai,nbj->aibj',n_n,ai_bb,ai_aa.conj(),optimize=True)
        K_aibj_A_bbbb += numpy.einsum('n,nai,nbj->aibj',n_n,ai_bb,ai_bb.conj(),optimize=True)
        
        # ns
        K_aibj_A_aaaa += numpy.einsum('n,nai,nbj->aibj',n_s,ai_aa,ai_aa.conj(),optimize=True)  
        K_aibj_A_aabb += -1*numpy.einsum('n,nai,nbj->aibj',n_s,ai_aa,ai_bb.conj(),optimize=True)
        K_aibj_A_bbaa += numpy.einsum('n,nai,nbj->aibj',n_s,ai_bb,ai_aa.conj(),optimize=True)
        K_aibj_A_bbbb += -1*numpy.einsum('n,nai,nbj->aibj',n_s,ai_bb,ai_bb.conj(),optimize=True)
        
        # sn
        K_aibj_A_aaaa += numpy.einsum('n,nai,nbj->aibj',n_s,ai_aa,ai_aa.conj(),optimize=True)  
        K_aibj_A_aabb += numpy.einsum('n,nai,nbj->aibj',n_s,ai_aa,ai_bb.conj(),optimize=True)
        K_aibj_A_bbaa += -1*numpy.einsum('n,nai,nbj->aibj',n_s,ai_bb,ai_aa.conj(),optimize=True)
        K_aibj_A_bbbb += -1*numpy.einsum('n,nai,nbj->aibj',n_s,ai_bb,ai_bb.conj(),optimize=True)
 
        # n_Nn
        K_aibj_A_aaaa += numpy.einsum('gn,nai,gnbj->aibj',n_Nn,ai_aa,ai_Naa.conj(),optimize=True)
        K_aibj_A_aabb += numpy.einsum('gn,nai,gnbj->aibj',n_Nn,ai_aa,ai_Nbb.conj(),optimize=True)
        K_aibj_A_bbaa += numpy.einsum('gn,nai,gnbj->aibj',n_Nn,ai_bb,ai_Naa.conj(),optimize=True)
        K_aibj_A_bbbb += numpy.einsum('gn,nai,gnbj->aibj',n_Nn,ai_bb,ai_Nbb.conj(),optimize=True)
        
        # Nn_n
        K_aibj_A_aaaa += numpy.einsum('gn,gnai,nbj->aibj',n_Nn,ai_Naa,ai_aa.conj(),optimize=True)
        K_aibj_A_aabb += numpy.einsum('gn,gnai,nbj->aibj',n_Nn,ai_Naa,ai_bb.conj(),optimize=True)
        K_aibj_A_bbaa += numpy.einsum('gn,gnai,nbj->aibj',n_Nn,ai_Nbb,ai_aa.conj(),optimize=True)
        K_aibj_A_bbbb += numpy.einsum('gn,gnai,nbj->aibj',n_Nn,ai_Nbb,ai_bb.conj(),optimize=True)
        
        # n_Ns
        K_aibj_A_aaaa += numpy.einsum('gn,nai,gnbj->aibj',n_Ns,ai_aa,ai_Naa.conj(),optimize=True)
        K_aibj_A_aabb += -1.0*numpy.einsum('gn,nai,gnbj->aibj',n_Ns,ai_aa,ai_Nbb.conj(),optimize=True)
        K_aibj_A_bbaa += numpy.einsum('gn,nai,gnbj->aibj',n_Ns,ai_bb,ai_Naa.conj(),optimize=True)
        K_aibj_A_bbbb += -1.0*numpy.einsum('gn,nai,gnbj->aibj',n_Ns,ai_bb,ai_Nbb.conj(),optimize=True)
        
        # Ns_n
        K_aibj_A_aaaa += numpy.einsum('gn,gnai,nbj->aibj',n_Ns,ai_Naa,ai_aa.conj(),optimize=True)
        K_aibj_A_aabb += numpy.einsum('gn,gnai,nbj->aibj',n_Ns,ai_Naa,ai_bb.conj(),optimize=True)
        K_aibj_A_bbaa += -1.0*numpy.einsum('gn,gnai,nbj->aibj',n_Ns,ai_Nbb,ai_aa.conj(),optimize=True)
        K_aibj_A_bbbb += -1.0*numpy.einsum('gn,gnai,nbj->aibj',n_Ns,ai_Nbb,ai_bb.conj(),optimize=True)

        # ss
        K_aibj_A_aaaa += numpy.einsum('n,nai,nbj->aibj',s_s,ai_aa,ai_aa.conj(),optimize=True)  
        K_aibj_A_aabb += -1*numpy.einsum('n,nai,nbj->aibj',s_s,ai_aa,ai_bb.conj(),optimize=True)
        K_aibj_A_bbaa += -1*numpy.einsum('n,nai,nbj->aibj',s_s,ai_bb,ai_aa.conj(),optimize=True)
        K_aibj_A_bbbb += numpy.einsum('n,nai,nbj->aibj',s_s,ai_bb,ai_bb.conj(),optimize=True)

        # s_Nn
        K_aibj_A_aaaa += numpy.einsum('gn,nai,gnbj->aibj',s_Nn,ai_aa,ai_Naa.conj(),optimize=True)
        K_aibj_A_aabb += numpy.einsum('gn,nai,gnbj->aibj',s_Nn,ai_aa,ai_Nbb.conj(),optimize=True)
        K_aibj_A_bbaa += -1*numpy.einsum('gn,nai,gnbj->aibj',s_Nn,ai_bb,ai_Naa.conj(),optimize=True)
        K_aibj_A_bbbb += -1*numpy.einsum('gn,nai,gnbj->aibj',s_Nn,ai_bb,ai_Nbb.conj(),optimize=True)
        
        # Nn_s
        K_aibj_A_aaaa += numpy.einsum('gn,gnai,nbj->aibj',s_Nn,ai_Naa,ai_aa.conj(),optimize=True)
        K_aibj_A_aabb += -1*numpy.einsum('gn,gnai,nbj->aibj',s_Nn,ai_Naa,ai_bb.conj(),optimize=True)
        K_aibj_A_bbaa += numpy.einsum('gn,gnai,nbj->aibj',s_Nn,ai_Nbb,ai_aa.conj(),optimize=True)
        K_aibj_A_bbbb += -1*numpy.einsum('gn,gnai,nbj->aibj',s_Nn,ai_Nbb,ai_bb.conj(),optimize=True)

        # s_Ns
        K_aibj_A_aaaa += numpy.einsum('gn,nai,gnbj->aibj',s_Ns,ai_aa,ai_Naa.conj(),optimize=True)
        K_aibj_A_aabb += -1*numpy.einsum('gn,nai,gnbj->aibj',s_Ns,ai_aa,ai_Nbb.conj(),optimize=True)
        K_aibj_A_bbaa += -1*numpy.einsum('gn,nai,gnbj->aibj',s_Ns,ai_bb,ai_Naa.conj(),optimize=True)
        K_aibj_A_bbbb += numpy.einsum('gn,nai,gnbj->aibj',s_Ns,ai_bb,ai_Nbb.conj(),optimize=True)
        
        # Ns_s
        K_aibj_A_aaaa += numpy.einsum('gn,gnai,nbj->aibj',s_Ns,ai_Naa,ai_aa.conj(),optimize=True)
        K_aibj_A_aabb += -1*numpy.einsum('gn,gnai,nbj->aibj',s_Ns,ai_Naa,ai_bb.conj(),optimize=True)
        K_aibj_A_bbaa += -1*numpy.einsum('gn,gnai,nbj->aibj',s_Ns,ai_Nbb,ai_aa.conj(),optimize=True)
        K_aibj_A_bbbb += numpy.einsum('gn,gnai,nbj->aibj',s_Ns,ai_Nbb,ai_bb.conj(),optimize=True)
        
        # Nn_Nn part
        K_aibj_A_aaaa += numpy.einsum('ghn,gnai,hnbj->aibj',Nn_Nn,ai_Naa,ai_Naa.conj(),optimize=True)
        K_aibj_A_aabb += numpy.einsum('ghn,gnai,hnbj->aibj',Nn_Nn,ai_Naa,ai_Nbb.conj(),optimize=True)
        K_aibj_A_bbaa += numpy.einsum('ghn,gnai,hnbj->aibj',Nn_Nn,ai_Nbb,ai_Naa.conj(),optimize=True)
        K_aibj_A_bbbb += numpy.einsum('ghn,gnai,hnbj->aibj',Nn_Nn,ai_Nbb,ai_Nbb.conj(),optimize=True)

        # Nn_Ns part
        K_aibj_A_aaaa += numpy.einsum('ghn,gnai,hnbj->aibj',Nn_Ns,ai_Naa,ai_Naa.conj(),optimize=True)
        K_aibj_A_aabb += -1*numpy.einsum('ghn,gnai,hnbj->aibj',Nn_Ns,ai_Naa,ai_Nbb.conj(),optimize=True)
        K_aibj_A_bbaa += numpy.einsum('ghn,gnai,hnbj->aibj',Nn_Ns,ai_Nbb,ai_Naa.conj(),optimize=True)
        K_aibj_A_bbbb += -1*numpy.einsum('ghn,gnai,hnbj->aibj',Nn_Ns,ai_Nbb,ai_Nbb.conj(),optimize=True)
        
        # Ns_Nn part
        K_aibj_A_aaaa += numpy.einsum('ghn,gnai,hnbj->aibj',Nn_Ns,ai_Naa,ai_Naa.conj(),optimize=True)
        K_aibj_A_aabb += numpy.einsum('ghn,gnai,hnbj->aibj',Nn_Ns,ai_Naa,ai_Nbb.conj(),optimize=True)
        K_aibj_A_bbaa += -1*numpy.einsum('ghn,gnai,hnbj->aibj',Nn_Ns,ai_Nbb,ai_Naa.conj(),optimize=True)
        K_aibj_A_bbbb += -1*numpy.einsum('ghn,gnai,hnbj->aibj',Nn_Ns,ai_Nbb,ai_Nbb.conj(),optimize=True)

        # Ns_Ns part
        K_aibj_A_aaaa += numpy.einsum('ghn,gnai,hnbj->aibj',Ns_Ns,ai_Naa,ai_Naa.conj(),optimize=True)
        K_aibj_A_aabb += -1*numpy.einsum('ghn,gnai,hnbj->aibj',Ns_Ns,ai_Naa,ai_Nbb.conj(),optimize=True)
        K_aibj_A_bbaa += -1*numpy.einsum('ghn,gnai,hnbj->aibj',Ns_Ns,ai_Nbb,ai_Naa.conj(),optimize=True)
        K_aibj_A_bbbb += numpy.einsum('ghn,gnai,hnbj->aibj',Ns_Ns,ai_Nbb,ai_Nbb.conj(),optimize=True)
        
        
        # nn
        K_aibj_B_aaaa += numpy.einsum('n,nai,nbj->aibj',n_n,ai_aa,ai_aa,optimize=True)  
        K_aibj_B_aabb += numpy.einsum('n,nai,nbj->aibj',n_n,ai_aa,ai_bb,optimize=True)
        K_aibj_B_bbaa += numpy.einsum('n,nai,nbj->aibj',n_n,ai_bb,ai_aa,optimize=True)
        K_aibj_B_bbbb += numpy.einsum('n,nai,nbj->aibj',n_n,ai_bb,ai_bb,optimize=True)
        
        # ns
        K_aibj_B_aaaa += numpy.einsum('n,nai,nbj->aibj',n_s,ai_aa,ai_aa,optimize=True)  
        K_aibj_B_aabb += -1*numpy.einsum('n,nai,nbj->aibj',n_s,ai_aa,ai_bb,optimize=True)
        K_aibj_B_bbaa += numpy.einsum('n,nai,nbj->aibj',n_s,ai_bb,ai_aa,optimize=True)
        K_aibj_B_bbbb += -1*numpy.einsum('n,nai,nbj->aibj',n_s,ai_bb,ai_bb,optimize=True)
        
        # sn
        K_aibj_B_aaaa += numpy.einsum('n,nai,nbj->aibj',n_s,ai_aa,ai_aa,optimize=True)  
        K_aibj_B_aabb += numpy.einsum('n,nai,nbj->aibj',n_s,ai_aa,ai_bb,optimize=True)
        K_aibj_B_bbaa += -1*numpy.einsum('n,nai,nbj->aibj',n_s,ai_bb,ai_aa,optimize=True)
        K_aibj_B_bbbb += -1*numpy.einsum('n,nai,nbj->aibj',n_s,ai_bb,ai_bb,optimize=True)
 
        # n_Nn
        K_aibj_B_aaaa += numpy.einsum('gn,nai,gnbj->aibj',n_Nn,ai_aa,ai_Naa,optimize=True)
        K_aibj_B_aabb += numpy.einsum('gn,nai,gnbj->aibj',n_Nn,ai_aa,ai_Nbb,optimize=True)
        K_aibj_B_bbaa += numpy.einsum('gn,nai,gnbj->aibj',n_Nn,ai_bb,ai_Naa,optimize=True)
        K_aibj_B_bbbb += numpy.einsum('gn,nai,gnbj->aibj',n_Nn,ai_bb,ai_Nbb,optimize=True)
        
        # Nn_n
        K_aibj_B_aaaa += numpy.einsum('gn,gnai,nbj->aibj',n_Nn,ai_Naa,ai_aa,optimize=True)
        K_aibj_B_aabb += numpy.einsum('gn,gnai,nbj->aibj',n_Nn,ai_Naa,ai_bb,optimize=True)
        K_aibj_B_bbaa += numpy.einsum('gn,gnai,nbj->aibj',n_Nn,ai_Nbb,ai_aa,optimize=True)
        K_aibj_B_bbbb += numpy.einsum('gn,gnai,nbj->aibj',n_Nn,ai_Nbb,ai_bb,optimize=True)
        
        # n_Ns
        K_aibj_B_aaaa += numpy.einsum('gn,nai,gnbj->aibj',n_Ns,ai_aa,ai_Naa,optimize=True)
        K_aibj_B_aabb += -1.0*numpy.einsum('gn,nai,gnbj->aibj',n_Ns,ai_aa,ai_Nbb,optimize=True)
        K_aibj_B_bbaa += numpy.einsum('gn,nai,gnbj->aibj',n_Ns,ai_bb,ai_Naa,optimize=True)
        K_aibj_B_bbbb += -1.0*numpy.einsum('gn,nai,gnbj->aibj',n_Ns,ai_bb,ai_Nbb,optimize=True)
        
        # Ns_n
        K_aibj_B_aaaa += numpy.einsum('gn,gnai,nbj->aibj',n_Ns,ai_Naa,ai_aa,optimize=True)
        K_aibj_B_aabb += numpy.einsum('gn,gnai,nbj->aibj',n_Ns,ai_Naa,ai_bb,optimize=True)
        K_aibj_B_bbaa += -1.0*numpy.einsum('gn,gnai,nbj->aibj',n_Ns,ai_Nbb,ai_aa,optimize=True)
        K_aibj_B_bbbb += -1.0*numpy.einsum('gn,gnai,nbj->aibj',n_Ns,ai_Nbb,ai_bb,optimize=True)

        # ss
        K_aibj_B_aaaa += numpy.einsum('n,nai,nbj->aibj',s_s,ai_aa,ai_aa,optimize=True)  
        K_aibj_B_aabb += -1*numpy.einsum('n,nai,nbj->aibj',s_s,ai_aa,ai_bb,optimize=True)
        K_aibj_B_bbaa += -1*numpy.einsum('n,nai,nbj->aibj',s_s,ai_bb,ai_aa,optimize=True)
        K_aibj_B_bbbb += numpy.einsum('n,nai,nbj->aibj',s_s,ai_bb,ai_bb,optimize=True)

        # s_Nn
        K_aibj_B_aaaa += numpy.einsum('gn,nai,gnbj->aibj',s_Nn,ai_aa,ai_Naa,optimize=True)
        K_aibj_B_aabb += numpy.einsum('gn,nai,gnbj->aibj',s_Nn,ai_aa,ai_Nbb,optimize=True)
        K_aibj_B_bbaa += -1*numpy.einsum('gn,nai,gnbj->aibj',s_Nn,ai_bb,ai_Naa,optimize=True)
        K_aibj_B_bbbb += -1*numpy.einsum('gn,nai,gnbj->aibj',s_Nn,ai_bb,ai_Nbb,optimize=True)
        
        # Nn_s
        K_aibj_B_aaaa += numpy.einsum('gn,gnai,nbj->aibj',s_Nn,ai_Naa,ai_aa,optimize=True)
        K_aibj_B_aabb += -1*numpy.einsum('gn,gnai,nbj->aibj',s_Nn,ai_Naa,ai_bb,optimize=True)
        K_aibj_B_bbaa += numpy.einsum('gn,gnai,nbj->aibj',s_Nn,ai_Nbb,ai_aa,optimize=True)
        K_aibj_B_bbbb += -1*numpy.einsum('gn,gnai,nbj->aibj',s_Nn,ai_Nbb,ai_bb,optimize=True)

        # s_Ns
        K_aibj_B_aaaa += numpy.einsum('gn,nai,gnbj->aibj',s_Ns,ai_aa,ai_Naa,optimize=True)
        K_aibj_B_aabb += -1*numpy.einsum('gn,nai,gnbj->aibj',s_Ns,ai_aa,ai_Nbb,optimize=True)
        K_aibj_B_bbaa += -1*numpy.einsum('gn,nai,gnbj->aibj',s_Ns,ai_bb,ai_Naa,optimize=True)
        K_aibj_B_bbbb += numpy.einsum('gn,nai,gnbj->aibj',s_Ns,ai_bb,ai_Nbb,optimize=True)
        
        # Ns_s
        K_aibj_B_aaaa += numpy.einsum('gn,gnai,nbj->aibj',s_Ns,ai_Naa,ai_aa,optimize=True)
        K_aibj_B_aabb += -1*numpy.einsum('gn,gnai,nbj->aibj',s_Ns,ai_Naa,ai_bb,optimize=True)
        K_aibj_B_bbaa += -1*numpy.einsum('gn,gnai,nbj->aibj',s_Ns,ai_Nbb,ai_aa,optimize=True)
        K_aibj_B_bbbb += numpy.einsum('gn,gnai,nbj->aibj',s_Ns,ai_Nbb,ai_bb,optimize=True)
        
        # Nn_Nn part
        K_aibj_B_aaaa += numpy.einsum('ghn,gnai,hnbj->aibj',Nn_Nn,ai_Naa,ai_Naa,optimize=True)
        K_aibj_B_aabb += numpy.einsum('ghn,gnai,hnbj->aibj',Nn_Nn,ai_Naa,ai_Nbb,optimize=True)
        K_aibj_B_bbaa += numpy.einsum('ghn,gnai,hnbj->aibj',Nn_Nn,ai_Nbb,ai_Naa,optimize=True)
        K_aibj_B_bbbb += numpy.einsum('ghn,gnai,hnbj->aibj',Nn_Nn,ai_Nbb,ai_Nbb,optimize=True)

        # Nn_Ns part
        K_aibj_B_aaaa += numpy.einsum('ghn,gnai,hnbj->aibj',Nn_Ns,ai_Naa,ai_Naa,optimize=True)
        K_aibj_B_aabb += -1*numpy.einsum('ghn,gnai,hnbj->aibj',Nn_Ns,ai_Naa,ai_Nbb,optimize=True)
        K_aibj_B_bbaa += numpy.einsum('ghn,gnai,hnbj->aibj',Nn_Ns,ai_Nbb,ai_Naa,optimize=True)
        K_aibj_B_bbbb += -1*numpy.einsum('ghn,gnai,hnbj->aibj',Nn_Ns,ai_Nbb,ai_Nbb,optimize=True)
        
        # Ns_Nn part
        K_aibj_B_aaaa += numpy.einsum('ghn,gnai,hnbj->aibj',Nn_Ns,ai_Naa,ai_Naa,optimize=True)
        K_aibj_B_aabb += numpy.einsum('ghn,gnai,hnbj->aibj',Nn_Ns,ai_Naa,ai_Nbb,optimize=True)
        K_aibj_B_bbaa += -1*numpy.einsum('ghn,gnai,hnbj->aibj',Nn_Ns,ai_Nbb,ai_Naa,optimize=True)
        K_aibj_B_bbbb += -1*numpy.einsum('ghn,gnai,hnbj->aibj',Nn_Ns,ai_Nbb,ai_Nbb,optimize=True)

        # Ns_Ns part
        K_aibj_B_aaaa += numpy.einsum('ghn,gnai,hnbj->aibj',Ns_Ns,ai_Naa,ai_Naa,optimize=True)
        K_aibj_B_aabb += -1*numpy.einsum('ghn,gnai,hnbj->aibj',Ns_Ns,ai_Naa,ai_Nbb,optimize=True)
        K_aibj_B_bbaa += -1*numpy.einsum('ghn,gnai,hnbj->aibj',Ns_Ns,ai_Nbb,ai_Naa,optimize=True)
        K_aibj_B_bbbb += numpy.einsum('ghn,gnai,hnbj->aibj',Ns_Ns,ai_Nbb,ai_Nbb,optimize=True)

    elif xctype == 'MGGA':
        raise NotImplementedError("Spin-conserved schem isn't implemented in Meta-GGA")
    
    return ((K_aibj_A_aaaa,K_aibj_A_aabb,K_aibj_A_bbaa,K_aibj_A_bbbb),
            (K_aibj_B_aaaa,K_aibj_B_aabb,K_aibj_B_bbaa,K_aibj_B_bbbb))

def nr_noncollinear_tddft_mc(self, mol, xc_code, grids, dms, C_mo, Ndirect=None,Ndirect_lc=None, 
                          MSL_factor=None, LIBXCT_factor=None,ncpu =None):
    '''nr_noncollinear_tdamc: calculates the K_aibj for Multi-Collinear TDA for noncollinear system.
    
    Parameters
    ----------
    Args:
        mol : an instance of :class:`Mole` in pySCF
           
        xc_code : str
            Name of exchange-correlation functional.
        grids : mf.object
            mf is an object for DFT class and grids.coords and grids.weights are used here, supplying the sample 
            points and weights in real space.
        dms : tuple
            (dmaa,dmab,dmba,dmbb), density matrix.
        C_mo : tuple
            Molecular orbital cofficience. 
            C_mo = (mo_a_vir,mo_a_occ,mo_b_vir,mo_b_occ), which means:
                Virtual alpha molecular orbitals, occupied alpha molecular orbitals, 
                Virtual beta molecular orbitals, occupied beta molecular orbitals, respectively.
    
    Kwargs:
        Ndirect : int
            The number of sample points in spin space, for lebedev distribution.
        Ndirect_lc : int
            The number of sample points in spin space, for gauss-legendre distribution.
        MSL_factor : double or int
            The factor to determine the strong polar points.
            Deafult is None. Value Recommended is 0.999.
        LIBXCT_factor : double or int
            The Threshold of for derivatives obatained from libxc.
            Deafult is None. Value Recommended is 1e-10.
        ncpu : int
            Number of cpu workers.
    Returns:
    ----------
        K_aibj : numpy.array
            aibj means related orbitals. a,b are virtual orbitals, and i,j are occupied orbitals.
    '''
    xctype = self._xc_type(xc_code) 
    mo_a_vir,mo_a_occ, mo_b_vir, mo_b_occ = C_mo
    kernel = self.noncollinear_tdamc_kernel(mol, xc_code, grids, dms, Ndirect=Ndirect, Ndirect_lc=Ndirect_lc, 
                                            MSL_factor=MSL_factor, LIBXCT_factor=LIBXCT_factor,ncpu=ncpu)
    K_aibj_A = 0.0
    K_aibj_B = 0.0
    
    import multiprocessing
    import math
    # ~ init some parameters in parallel.
    ngrid = grids.coords.shape[0]
    
    if ncpu is None:
        ncpu = multiprocessing.cpu_count()
    nsbatch = math.ceil(ngrid/ncpu)
    NX_list = [(i, i+nsbatch) for i in range(0, ngrid-nsbatch, nsbatch)]
    if NX_list[-1][-1] < ngrid:
        NX_list.append((NX_list[-1][-1], ngrid))
        
    pool = multiprocessing.Pool()
    para_results = []

    # import pdb
    # pdb.set_trace()
    if xctype == 'LDA':
        for index in NX_list:
            idxi,idxf = index
            kernel_para = []
            for i in range(len(kernel)):
                kernel_para.append(kernel[i][...,idxi:idxf])
            para_results.append(pool.apply_async(K_aibj_noncollinear_generator_tddft,(xctype, 
                                                 mo_a_occ[idxi:idxf],mo_b_occ[idxi:idxf],
                                                 mo_a_vir[idxi:idxf],mo_b_vir[idxi:idxf],kernel_para)))
    elif xctype == 'GGA':   
        for index in NX_list:
            idxi,idxf = index
            kernel_para = []
            for i in range(len(kernel)):
                kernel_para.append(kernel[i][...,idxi:idxf])
            para_results.append(pool.apply_async(K_aibj_noncollinear_generator_tddft,(xctype, 
                                                 mo_a_occ[:,idxi:idxf],mo_b_occ[:,idxi:idxf],
                                                 mo_a_vir[:,idxi:idxf],mo_b_vir[:,idxi:idxf],kernel_para)))
    pool.close()
    pool.join()

    # ~ get the final result
    for para_result in para_results:
        result = para_result.get()
        K_aibj_A += result[0]
        K_aibj_B += result[1]
        
    return K_aibj_A,K_aibj_B

def K_aibj_noncollinear_generator_tddft(xctype, mo_a_occ,mo_b_occ,mo_a_vir,mo_b_vir,kernel):
    '''K_aibj_noncollinear_generator: calculates <ai|kernel|bj> for noncollinear TDA term.
    
    Parameters
    ----------
    Args:
       xctype : str
            xctype -> LDA, GGA, MGGA. 
        mo_a_vir,mo_a_occ,mo_b_vir,mo_b_occ :  numpy.array
            Virtual alpha molecular orbitals, occupied alpha molecular orbitals, 
            Virtual beta molecular orbitals, occupied beta molecular orbitals, respectively.
        kernel : tuple
            Noncollinear TDA kernel. In LDA and GGA, len(kernel) = 3, 10, respectively.
     
    Returns:
    ----------
       K_aibj : numpy.array 
    
    Raises:
    ----------
        ToDo : K_aibj for noncollinear TDA in MGGA.
    '''
    if xctype == 'LDA':
        kxc_nn,kxc_ns,kxc_ss = kernel 
        # construct gks aa,ab,ba,bb blocks, ai means orbital a to orbital i
        ai_aa = numpy.einsum('na,ni->nai',mo_a_vir.conj(),mo_a_occ,optimize=True)
        ai_ab = numpy.einsum('na,ni->nai',mo_a_vir.conj(),mo_b_occ,optimize=True)
        ai_ba = numpy.einsum('na,ni->nai',mo_b_vir.conj(),mo_a_occ,optimize=True)
        ai_bb = numpy.einsum('na,ni->nai',mo_b_vir.conj(),mo_b_occ,optimize=True)
        
        # construct rho,Mx,My,Mz blocks to calculate K_aibj
        ai_rho = ai_aa + ai_bb
        ai_Mx = ai_ab + ai_ba
        ai_My = -1.0j*ai_ab + 1.0j*ai_ba
        ai_Mz = ai_aa - ai_bb
        
        ai_s = [ai_Mx,ai_My,ai_Mz]
        
        # calculate K_aibj
        K_aibj_A,K_aibj_B = (0.0,0.0)

        # kxc_nn
        K_aibj_A += numpy.einsum('n,nai,nbj->aibj',kxc_nn,
                              ai_rho,ai_rho.conj(),optimize=True)
        K_aibj_B += numpy.einsum('n,nai,nbj->aibj',kxc_nn,
                              ai_rho,ai_rho,optimize=True)
        
        # kxc_ns
        for i in range(3):
            K_aibj_A += numpy.einsum('n,nai,nbj->aibj',kxc_ns[i],
                                  ai_rho,ai_s[i].conj(),optimize=True)
            K_aibj_B += numpy.einsum('n,nai,nbj->aibj',kxc_ns[i],
                                  ai_rho,ai_s[i],optimize=True)
        
        # kxc_sn
        for i in range(3):
            K_aibj_A += numpy.einsum('n,nai,nbj->aibj',kxc_ns[i],
                                  ai_s[i],ai_rho.conj(),optimize=True)
            K_aibj_B += numpy.einsum('n,nai,nbj->aibj',kxc_ns[i],
                                  ai_s[i],ai_rho,optimize=True)
            
        # kxc_ss
        for i in range(3):
            for j in range(3):
                K_aibj_A += numpy.einsum('n,nai,nbj->aibj',kxc_ss[i][j],
                                  ai_s[i],ai_s[j].conj(),optimize=True)
                K_aibj_B += numpy.einsum('n,nai,nbj->aibj',kxc_ss[i][j],
                                  ai_s[i],ai_s[j],optimize=True)
                
    elif xctype == 'GGA':
        kxc_nn, kxc_ns, kxc_n_Nn, kxc_n_Ns, kxc_ss, kxc_s_Nn, kxc_s_Ns, \
        kxc_Nn_Nn, kxc_Nn_Ns, kxc_Ns_Ns = kernel 
        
        # Prepareing work
        # construct gks aa,ab,ba,bb blocks, ai means orbital a to orbital i
        ai_aa = numpy.einsum('na,ni->nai',mo_a_vir[0].conj(),mo_a_occ[0],optimize=True)
        ai_ab = numpy.einsum('na,ni->nai',mo_a_vir[0].conj(),mo_b_occ[0],optimize=True)
        ai_ba = numpy.einsum('na,ni->nai',mo_b_vir[0].conj(),mo_a_occ[0],optimize=True)
        ai_bb = numpy.einsum('na,ni->nai',mo_b_vir[0].conj(),mo_b_occ[0],optimize=True)
        
        # construct rho,Mx,My,Mz blocks to calculate K_aibj
        ai_rho = ai_aa + ai_bb
        ai_Mx = ai_ab + ai_ba
        ai_My = -1.0j*ai_ab + 1.0j*ai_ba
        ai_Mz = ai_aa - ai_bb
        ai_s = [ai_Mx,ai_My,ai_Mz]
        
        # construct gradient terms
        ai_na_a = numpy.einsum('gna,ni->gnai',mo_a_vir[1:4].conj(),mo_a_occ[0],optimize=True)
        ai_a_na = numpy.einsum('na,gni->gnai',mo_a_vir[0].conj(),mo_a_occ[1:4],optimize=True)
        ai_na_b = numpy.einsum('gna,ni->gnai',mo_a_vir[1:4].conj(),mo_b_occ[0],optimize=True)
        ai_a_nb = numpy.einsum('na,gni->gnai',mo_a_vir[0].conj(),mo_b_occ[1:4],optimize=True)
        ai_nb_a = numpy.einsum('gna,ni->gnai',mo_b_vir[1:4].conj(),mo_a_occ[0],optimize=True)
        ai_b_na = numpy.einsum('na,gni->gnai',mo_b_vir[0].conj(),mo_a_occ[1:4],optimize=True)
        ai_nb_b = numpy.einsum('gna,ni->gnai',mo_b_vir[1:4].conj(),mo_b_occ[0],optimize=True)
        ai_b_nb = numpy.einsum('na,gni->gnai',mo_b_vir[0].conj(),mo_b_occ[1:4],optimize=True)
        
        ai_nrho = ai_na_a + ai_a_na + ai_nb_b + ai_b_nb
        ai_nMx = ai_na_b + ai_a_nb + ai_nb_a + ai_b_na
        ai_nMy = -1.0j*(ai_na_b + ai_a_nb) + 1.0j*(ai_nb_a + ai_b_na)
        ai_nMz = ai_na_a + ai_a_na - ai_nb_b - ai_b_nb
        ai_ns = [ai_nMx,ai_nMy,ai_nMz]
        # calculate K_aibj
        K_aibj_A,K_aibj_B = (0.0,0.0)

        # kxc_nn
        K_aibj_A += numpy.einsum('n,nai,nbj->aibj',kxc_nn,
                              ai_rho,ai_rho.conj(),optimize=True)
        K_aibj_B += numpy.einsum('n,nai,nbj->aibj',kxc_nn,
                              ai_rho,ai_rho,optimize=True)
        # kxc_ns
        for i in range(3):
            K_aibj_A += numpy.einsum('n,nai,nbj->aibj',kxc_ns[i],
                                  ai_rho,ai_s[i].conj(),optimize=True)
            K_aibj_B += numpy.einsum('n,nai,nbj->aibj',kxc_ns[i],
                                  ai_rho,ai_s[i],optimize=True)
        
        # kxc_sn
        for i in range(3):
            K_aibj_A += numpy.einsum('n,nai,nbj->aibj',kxc_ns[i],
                                  ai_s[i],ai_rho.conj(),optimize=True)
            K_aibj_B += numpy.einsum('n,nai,nbj->aibj',kxc_ns[i],
                                  ai_s[i],ai_rho,optimize=True)
        # kxc_n_Nn
        K_aibj_A += numpy.einsum('gn,nai,gnbj->aibj',kxc_n_Nn,
                               ai_rho,ai_nrho.conj(),optimize=True)
        K_aibj_B += numpy.einsum('gn,nai,gnbj->aibj',kxc_n_Nn,
                               ai_rho,ai_nrho.conj(),optimize=True)
        
        # kxc_Nn_n
        K_aibj_A += numpy.einsum('gn,gnai,nbj->aibj',kxc_n_Nn,
                               ai_nrho,ai_rho.conj(),optimize=True)
        K_aibj_B += numpy.einsum('gn,gnai,nbj->aibj',kxc_n_Nn,
                               ai_nrho,ai_rho,optimize=True)
        
        # kxc_n_Ns
        for i in range(3):
            K_aibj_A += numpy.einsum('gn,nai,gnbj->aibj',kxc_n_Ns[:,i],
                                    ai_rho,ai_ns[i].conj(),optimize=True)
            K_aibj_B += numpy.einsum('gn,nai,gnbj->aibj',kxc_n_Ns[:,i],
                                    ai_rho,ai_ns[i],optimize=True)
        
        # kxc_Ns_n
        for i in range(3):
            K_aibj_A += numpy.einsum('gn,gnai,nbj->aibj',kxc_n_Ns[:,i],
                                    ai_ns[i],ai_rho.conj(),optimize=True)
            K_aibj_B += numpy.einsum('gn,gnai,nbj->aibj',kxc_n_Ns[:,i],
                                    ai_ns[i],ai_rho,optimize=True)
                
        # kxc_ss
        for i in range(3):
            for j in range(3):
                K_aibj_A += numpy.einsum('n,nai,nbj->aibj',kxc_ss[i][j],
                                  ai_s[i],ai_s[j].conj(),optimize=True)
                K_aibj_B += numpy.einsum('n,nai,nbj->aibj',kxc_ss[i][j],
                                  ai_s[i],ai_s[j],optimize=True)
        # kxc_s_Nn
        for i in range(3):
            K_aibj_A += numpy.einsum('gn,nai,gnbj->aibj',kxc_s_Nn[:,i],
                                   ai_s[i],ai_nrho.conj(),optimize=True)
            K_aibj_B += numpy.einsum('gn,nai,gnbj->aibj',kxc_s_Nn[:,i],
                                   ai_s[i],ai_nrho,optimize=True)
            
        # kxc_Nn_s
        for i in range(3):
            K_aibj_A += numpy.einsum('gn,gnai,nbj->aibj',kxc_s_Nn[:,i],
                                   ai_nrho,ai_s[i].conj(),optimize=True)
            K_aibj_B += numpy.einsum('gn,gnai,nbj->aibj',kxc_s_Nn[:,i],
                                   ai_nrho,ai_s[i],optimize=True)
        
        # kxc_s_Ns
        for i in range(3):
            for j in range(3):
                K_aibj_A += numpy.einsum('gn,nai,gnbj->aibj',kxc_s_Ns[:,i,j],
                                       ai_s[i],ai_ns[j].conj(),optimize=True)
                K_aibj_B += numpy.einsum('gn,nai,gnbj->aibj',kxc_s_Ns[:,i,j],
                                       ai_s[i],ai_ns[j],optimize=True)
                    
        # kxc_Ns_s
        for i in range(3):
            for j in range(3):
                K_aibj_A += numpy.einsum('gn,gnai,nbj->aibj',kxc_s_Ns[:,i,j],
                                        ai_ns[i],ai_s[j].conj(),optimize=True)
                K_aibj_B += numpy.einsum('gn,gnai,nbj->aibj',kxc_s_Ns[:,i,j],
                                        ai_ns[i],ai_s[j],optimize=True)
                      
        offset2 = numint_gksmc.get_2d_offset()
        # kxc_Nn_Nn

        for i in range(3):
            for j in range(3):
                K_aibj_A += numpy.einsum('n,nai,nbj->aibj',kxc_Nn_Nn[offset2[i,j]],
                                        ai_nrho[i],ai_nrho[j].conj(),optimize=True)
                K_aibj_B += numpy.einsum('n,nai,nbj->aibj',kxc_Nn_Nn[offset2[i,j]],
                                        ai_nrho[i],ai_nrho[j],optimize=True)

        # kxc_Nn_Ns
        # Note ai_ns[k][j] ! This is due to the storation form of coresponding kernel. 
        for i in range(3):
            K_aibj_A += numpy.einsum('ghn,gnai,hnbj->aibj',kxc_Nn_Ns[:,:,i],
                                ai_nrho,ai_ns[i].conj(),optimize=True)
            K_aibj_B += numpy.einsum('ghn,gnai,hnbj->aibj',kxc_Nn_Ns[:,:,i],
                                ai_nrho,ai_ns[i],optimize=True)
                    
        # kxc_Ns_Nn
        for i in range(3):
            K_aibj_A += numpy.einsum('ghn,hnai,gnbj->aibj',kxc_Nn_Ns[:,:,i],
                                ai_ns[i],ai_nrho.conj(),optimize=True)
            K_aibj_B += numpy.einsum('ghn,hnai,gnbj->aibj',kxc_Nn_Ns[:,:,i],
                                ai_ns[i],ai_nrho,optimize=True)
    
        # kxc_Ns_Ns
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        K_aibj_A += numpy.einsum('n,nai,nbj->aibj',kxc_Ns_Ns[offset2[i,j],k,l],
                                        ai_ns[k][i],ai_ns[l][j].conj(),optimize=True)
                        K_aibj_B += numpy.einsum('n,nai,nbj->aibj',kxc_Ns_Ns[offset2[i,j],k,l],
                                        ai_ns[k][i],ai_ns[l][j],optimize=True)
                        
    elif xctype == 'MGGA':
        raise NotImplementedError("Meta-GGA is not implemented")
        
    return K_aibj_A,K_aibj_B

def r_noncollinear_tddft_mc(self, nir, mol, xc_code, grids, dms, mo, Ndirect=None,Ndirect_lc=None, 
                          MSL_factor=None, LIBXCT_factor=None,ncpu =None):
    '''nr_noncollinear_tddft: calculates the K_aibj for Multi-Collinear TDDFT for noncollinear system.
    
    Parameters
    ----------
    Args:
        mol : an instance of :class:`Mole` in pySCF
           
        xc_code : str
            Name of exchange-correlation functional.
        grids : mf.object
            mf is an object for DFT class and grids.coords and grids.weights are used here, supplying the sample 
            points and weights in real space.
        dms : tuple
            (dmaa,dmab,dmba,dmbb), density matrix.
        C_mo : tuple
            Molecular orbital cofficience. 
            C_mo = (mo_a_vir,mo_a_occ,mo_b_vir,mo_b_occ), which means:
                Virtual alpha molecular orbitals, occupied alpha molecular orbitals, 
                Virtual beta molecular orbitals, occupied beta molecular orbitals, respectively.
    
    Kwargs:
        Ndirect : int
            The number of sample points in spin space, for lebedev distribution.
        Ndirect_lc : int
            The number of sample points in spin space, for gauss-legendre distribution.
        MSL_factor : double or int
            The factor to determine the strong polar points.
            Deafult is None. Value Recommended is 0.999.
        LIBXCT_factor : double or int
            The Threshold of for derivatives obatained from libxc.
            Deafult is None. Value Recommended is 1e-10.
        ncpu : int
            Number of cpu workers.
    Returns:
    ----------
        K_aibj : numpy.array
            aibj means related orbitals. a,b are virtual orbitals, and i,j are occupied orbitals.
    '''
    xctype = self._xc_type(xc_code) 
    mo_vir_L, mo_vir_S,mo_occ_L, mo_occ_S = mo 
    kernel = self.r_noncollinear_tdamc_kernel(mol, nir, xc_code, grids, dms, Ndirect=Ndirect, Ndirect_lc=Ndirect_lc, 
                                            MSL_factor=MSL_factor, LIBXCT_factor=LIBXCT_factor,ncpu=ncpu)
    K_aibj_A,K_aibj_B = (0.0,0.0)
    
    # import pdb
    # pdb.set_trace()
    
    import multiprocessing
    import math
    # ~ init some parameters in parallel.
    ngrid = grids.coords.shape[0]
    
    if ncpu is None:
        ncpu = multiprocessing.cpu_count()
    nsbatch = math.ceil(ngrid/ncpu)
    NX_list = [(i, i+nsbatch) for i in range(0, ngrid-nsbatch, nsbatch)]
    if NX_list[-1][-1] < ngrid:
        NX_list.append((NX_list[-1][-1], ngrid))
        
    pool = multiprocessing.Pool()
    para_results = []

    # import pdb
    # pdb.set_trace()
    if xctype == 'LDA':
        for index in NX_list:
            idxi,idxf = index
            kernel_para = []
            for i in range(len(kernel)):
                kernel_para.append(kernel[i][...,idxi:idxf])
                
            para_results.append(pool.apply_async(K_aibj_noncollinear_generator_r_tddft,(xctype, 
                                                            mo_vir_L[:,:,idxi:idxf],mo_vir_S[:,:,idxi:idxf],
                                                            mo_occ_L[:,:,idxi:idxf],mo_occ_S[:,:,idxi:idxf],
                                                            kernel_para)))
    elif xctype == 'GGA':
        for index in NX_list:
            idxi,idxf = index
            kernel_para = []
            for i in range(len(kernel)):
                kernel_para.append(kernel[i][...,idxi:idxf])
                
            para_results.append(pool.apply_async(K_aibj_noncollinear_generator_r_tddft,(xctype, 
                                                            mo_vir_L[:,:,idxi:idxf],mo_vir_S[:,:,idxi:idxf],
                                                            mo_occ_L[:,:,idxi:idxf],mo_occ_S[:,:,idxi:idxf],
                                                            kernel_para)))
    else:
        raise NotImplementedError("")
    pool.close()
    pool.join()
    import pdb
    pdb.set_trace()
    # ~ get the final result
    for para_result in para_results:
        result = para_result.get()
        K_aibj_A += result[0]
        K_aibj_B += result[1]
    return K_aibj_A,K_aibj_B

def K_aibj_noncollinear_generator_r_tddft(xctype, mo_vir_L, mo_vir_S, mo_occ_L, mo_occ_S, kernel):
    '''K_aibj_noncollinear_generator: calculates <ai|kernel|bj> for noncollinear TDA term.
    
    Parameters
    ----------
    Args:
       xctype : str
            xctype -> LDA, GGA, MGGA. 
        mo_a_vir,mo_a_occ,mo_b_vir,mo_b_occ :  numpy.array
            Virtual alpha molecular orbitals, occupied alpha molecular orbitals, 
            Virtual beta molecular orbitals, occupied beta molecular orbitals, respectively.
        kernel : tuple
            Noncollinear kernel. In LDA and GGA, len(kernel) = 3, 10, respectively.
     
    Returns:
    ----------
       K_aibj : numpy.array 
    
    Raises:
    ----------
        ToDo : K_aibj for noncollinear TDDFT in MGGA.
    '''
    # calculate K_aibj
    K_aibj_A,K_aibj_B = (0.0,0.0)
    
    if xctype == 'LDA':
        kxc_nn,kxc_ns,kxc_ss = kernel 
        # construct gks aa,ab,ba,bb blocks, ai means orbital a to orbital i       
        ai_rho = numpy.einsum('cxpa,cxpi->pai', mo_vir_L.conj(), mo_occ_L, optimize=True)
        ai_rho+= numpy.einsum('cxpa,cxpi->pai', mo_vir_S.conj(), mo_occ_S, optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        betasigma_x = numpy.array(
            [[0,1,0,0],
             [1,0,0,0],
             [0,0,0,-1],
             [0,0,-1,0]]
        )
        betasigma_y = numpy.array(
            [[0,-1.0j,0,0],
             [1.0j,0,0,0],
             [0,0,0,1.0j],
             [0,0,-1.0j,0]]
        )
        betasigma_z = numpy.array(
            [[1,0,0,0],
             [0,-1,0,0],
             [0,0,-1,0],
             [0,0,0,1]]
        )
        ai_Mx = numpy.einsum('cxpa,cd,dxpi->pai', mo_vir_L.conj(), betasigma_x[:2,:2], mo_occ_L, optimize=True)
        ai_Mx+= numpy.einsum('cxpa,cd,dxpi->pai', mo_vir_S.conj(), betasigma_x[2:,2:], mo_occ_S, optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        ai_My = numpy.einsum('cxpa,cd,dxpi->pai', mo_vir_L.conj(), betasigma_y[:2,:2], mo_occ_L, optimize=True)
        ai_My+= numpy.einsum('cxpa,cd,dxpi->pai', mo_vir_S.conj(), betasigma_y[2:,2:], mo_occ_S, optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        ai_Mz = numpy.einsum('cxpa,cd,dxpi->pai', mo_vir_L.conj(), betasigma_z[:2,:2], mo_occ_L, optimize=True)
        ai_Mz+= numpy.einsum('cxpa,cd,dxpi->pai', mo_vir_S.conj(), betasigma_z[2:,2:], mo_occ_S, optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        # construct rho,Mx,My,Mz blocks to calculate K_aibj
        ai_s = [ai_Mx,ai_My,ai_Mz]
       
        # kxc_nn
        K_aibj_A += numpy.einsum('n,nai,nbj->aibj',kxc_nn,
                              ai_rho,ai_rho.conj(),optimize=True)
        K_aibj_B += numpy.einsum('n,nai,nbj->aibj',kxc_nn,
                              ai_rho,ai_rho,optimize=True)
        # kxc_ns
        for i in range(3):
            K_aibj_A += numpy.einsum('n,nai,nbj->aibj',kxc_ns[i],
                                  ai_rho,ai_s[i].conj(),optimize=True)
            K_aibj_B += numpy.einsum('n,nai,nbj->aibj',kxc_ns[i],
                                  ai_rho,ai_s[i],optimize=True)
        # # kxc_sn
        for i in range(3):
            K_aibj_A += numpy.einsum('n,nai,nbj->aibj',kxc_ns[i],
                                  ai_s[i],ai_rho.conj(),optimize=True)
            K_aibj_B += numpy.einsum('n,nai,nbj->aibj',kxc_ns[i],
                                  ai_s[i],ai_rho,optimize=True)  
        # kxc_ss
        for i in range(3):
            for j in range(3):
                K_aibj_A += numpy.einsum('n,nai,nbj->aibj',kxc_ss[i][j],
                                  ai_s[i],ai_s[j].conj(),optimize=True)
                K_aibj_B += numpy.einsum('n,nai,nbj->aibj',kxc_ss[i][j],
                                  ai_s[i],ai_s[j],optimize=True)
                
    elif xctype == 'GGA':
        kxc_nn, kxc_ns, kxc_n_Nn, kxc_n_Ns, kxc_ss, kxc_s_Nn, kxc_s_Ns, \
            kxc_Nn_Nn, kxc_Nn_Ns, kxc_Ns_Ns = kernel
        ai_rho = numpy.einsum('cxpa,cpi->xpai', mo_vir_L.conj(), mo_occ_L[:,0], optimize=True)
        ai_rho+= numpy.einsum('cpa,cxpi->xpai', mo_vir_L[:,0].conj(), mo_occ_L, optimize=True)
        ai_rho+= numpy.einsum('cxpa,cpi->xpai', mo_vir_S.conj(), mo_occ_S[:,0], optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        ai_rho+= numpy.einsum('cpa,cxpi->xpai', mo_vir_S[:,0].conj(), mo_occ_S, optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        ai_rho[0]*= 0.5
        betasigma_x = numpy.array(
            [[0,1,0,0],
             [1,0,0,0],
             [0,0,0,-1],
             [0,0,-1,0]]
        )
        betasigma_y = numpy.array(
            [[0,-1.0j,0,0],
             [1.0j,0,0,0],
             [0,0,0,1.0j],
             [0,0,-1.0j,0]]
        )
        betasigma_z = numpy.array(
            [[1,0,0,0],
             [0,-1,0,0],
             [0,0,-1,0],
             [0,0,0,1]]
        )
        ai_Mx = numpy.einsum('cxpa,cd,dpi->xpai', mo_vir_L.conj(), betasigma_x[:2,:2], mo_occ_L[:,0], optimize=True)
        ai_Mx+= numpy.einsum('cpa,cd,dxpi->xpai', mo_vir_L[:,0].conj(), betasigma_x[:2,:2], mo_occ_L, optimize=True)
        ai_Mx+= numpy.einsum('cxpa,cd,dpi->xpai', mo_vir_S.conj(), betasigma_x[2:,2:], mo_occ_S[:,0], optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        ai_Mx+= numpy.einsum('cpa,cd,dxpi->xpai', mo_vir_S[:,0].conj(), betasigma_x[2:,2:], mo_occ_S, optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        ai_My = numpy.einsum('cxpa,cd,dpi->xpai', mo_vir_L.conj(), betasigma_y[:2,:2], mo_occ_L[:,0], optimize=True)
        ai_My+= numpy.einsum('cpa,cd,dxpi->xpai', mo_vir_L[:,0].conj(), betasigma_y[:2,:2], mo_occ_L, optimize=True)
        ai_My+= numpy.einsum('cxpa,cd,dpi->xpai', mo_vir_S.conj(), betasigma_y[2:,2:], mo_occ_S[:,0], optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        ai_My+= numpy.einsum('cpa,cd,dxpi->xpai', mo_vir_S[:,0].conj(), betasigma_y[2:,2:], mo_occ_S, optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        ai_Mz = numpy.einsum('cxpa,cd,dpi->xpai', mo_vir_L.conj(), betasigma_z[:2,:2], mo_occ_L[:,0], optimize=True)
        ai_Mz+= numpy.einsum('cpa,cd,dxpi->xpai', mo_vir_L[:,0].conj(), betasigma_z[:2,:2], mo_occ_L, optimize=True)
        ai_Mz+= numpy.einsum('cxpa,cd,dpi->xpai', mo_vir_S.conj(), betasigma_z[2:,2:], mo_occ_S[:,0], optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        ai_Mz+= numpy.einsum('cpa,cd,dxpi->xpai', mo_vir_S[:,0].conj(), betasigma_z[2:,2:], mo_occ_S, optimize=True)*(0.5/lib.param.LIGHT_SPEED)**2
        
        ai_Mx[0]*=0.5
        ai_My[0]*=0.5
        ai_Mz[0]*=0.5
        ai_s = [ai_Mx,ai_My,ai_Mz]

        # kxc_nn
        K_aibj_A += numpy.einsum('n,nai,nbj->aibj',kxc_nn,
                              ai_rho[0],ai_rho[0].conj(),optimize=True)
        K_aibj_B += numpy.einsum('n,nai,nbj->aibj',kxc_nn,
                              ai_rho[0],ai_rho[0],optimize=True)
        # kxc_ns
        for i in range(3):
            K_aibj_A += numpy.einsum('n,nai,nbj->aibj',kxc_ns[i],
                                  ai_rho[0],ai_s[i][0].conj(),optimize=True)
            K_aibj_B += numpy.einsum('n,nai,nbj->aibj',kxc_ns[i],
                                  ai_rho[0],ai_s[i][0],optimize=True)
        # kxc_sn
        for i in range(3):
            K_aibj_A += numpy.einsum('n,nai,nbj->aibj',kxc_ns[i],
                                  ai_s[i][0],ai_rho[0].conj(),optimize=True)
            K_aibj_B += numpy.einsum('n,nai,nbj->aibj',kxc_ns[i],
                                  ai_s[i][0],ai_rho[0],optimize=True)
        # kxc_n_Nn
        K_aibj_A += numpy.einsum('gn,nai,gnbj->aibj',kxc_n_Nn,
                               ai_rho[0],ai_rho[1:].conj(),optimize=True)
        K_aibj_B += numpy.einsum('gn,nai,gnbj->aibj',kxc_n_Nn,
                               ai_rho[0],ai_rho[1:],optimize=True)
        
        # kxc_Nn_n
        K_aibj_A += numpy.einsum('gn,gnai,nbj->aibj',kxc_n_Nn,
                               ai_rho[1:],ai_rho[0].conj(),optimize=True)
        K_aibj_B += numpy.einsum('gn,gnai,nbj->aibj',kxc_n_Nn,
                               ai_rho[1:],ai_rho[0],optimize=True)
        
        # kxc_n_Ns
        for i in range(3):
            K_aibj_A += numpy.einsum('gn,nai,gnbj->aibj',kxc_n_Ns[:,i],
                                    ai_rho[0],ai_s[i][1:].conj(),optimize=True)
            K_aibj_B += numpy.einsum('gn,nai,gnbj->aibj',kxc_n_Ns[:,i],
                                    ai_rho[0],ai_s[i][1:],optimize=True)
        # kxc_Ns_n
        for i in range(3):
            K_aibj_A += numpy.einsum('gn,gnai,nbj->aibj',kxc_n_Ns[:,i],
                                    ai_s[i][1:],ai_rho[0].conj(),optimize=True)
            K_aibj_B += numpy.einsum('gn,gnai,nbj->aibj',kxc_n_Ns[:,i],
                                    ai_s[i][1:],ai_rho[0],optimize=True)
        # kxc_ss
        for i in range(3):
            for j in range(3):
                K_aibj_A += numpy.einsum('n,nai,nbj->aibj',kxc_ss[i][j],
                                  ai_s[i][0],ai_s[j][0].conj(),optimize=True)
                K_aibj_B += numpy.einsum('n,nai,nbj->aibj',kxc_ss[i][j],
                                  ai_s[i][0],ai_s[j][0],optimize=True)
        # kxc_s_Nn
        for i in range(3):
            K_aibj_A += numpy.einsum('gn,nai,gnbj->aibj',kxc_s_Nn[:,i],
                                   ai_s[i][0],ai_rho[1:].conj(),optimize=True)
            K_aibj_B += numpy.einsum('gn,nai,gnbj->aibj',kxc_s_Nn[:,i],
                                   ai_s[i][0],ai_rho[1:],optimize=True)
            
        # kxc_Nn_s
        for i in range(3):
            K_aibj_A += numpy.einsum('gn,gnai,nbj->aibj',kxc_s_Nn[:,i],
                                   ai_rho[1:],ai_s[i][0].conj(),optimize=True)
            K_aibj_B += numpy.einsum('gn,gnai,nbj->aibj',kxc_s_Nn[:,i],
                                   ai_rho[1:],ai_s[i][0],optimize=True)
        
        # kxc_s_Ns
        for i in range(3):
            for j in range(3):
                K_aibj_A += numpy.einsum('gn,nai,gnbj->aibj',kxc_s_Ns[:,i,j],
                                       ai_s[i][0],ai_s[j][1:].conj(),optimize=True)
                K_aibj_B += numpy.einsum('gn,nai,gnbj->aibj',kxc_s_Ns[:,i,j],
                                       ai_s[i][0],ai_s[j][1:],optimize=True)
                
        # kxc_Ns_s
        for i in range(3):
            for j in range(3):
                K_aibj_A += numpy.einsum('gn,gnai,nbj->aibj',kxc_s_Ns[:,i,j],
                                        ai_s[i][1:],ai_s[j][0].conj(),optimize=True)
                K_aibj_B += numpy.einsum('gn,gnai,nbj->aibj',kxc_s_Ns[:,i,j],
                                        ai_s[i][1:],ai_s[j][0],optimize=True)
                      
        offset2 = numint_gksmc.get_2d_offset()
        # kxc_Nn_Nn
        for i in range(3):
            for j in range(3):
                K_aibj_A += numpy.einsum('n,nai,nbj->aibj',kxc_Nn_Nn[offset2[i,j]],
                                        ai_rho[i+1],ai_rho[j+1].conj(),optimize=True)
                K_aibj_B += numpy.einsum('n,nai,nbj->aibj',kxc_Nn_Nn[offset2[i,j]],
                                        ai_rho[i+1],ai_rho[j+1],optimize=True)

        # kxc_Nn_Ns
        # Note ai_ns[k][j] ! This is due to the storation form of coresponding kernel. 
        for i in range(3):
            K_aibj_A += numpy.einsum('ghn,gnai,hnbj->aibj',kxc_Nn_Ns[:,:,i],
                                ai_rho[1:],ai_s[i][1:].conj(),optimize=True)
            K_aibj_B += numpy.einsum('ghn,gnai,hnbj->aibj',kxc_Nn_Ns[:,:,i],
                                ai_rho[1:],ai_s[i][1:],optimize=True)
                    
        # kxc_Ns_Nn
        for i in range(3):
            K_aibj_A += numpy.einsum('ghn,hnai,gnbj->aibj',kxc_Nn_Ns[:,:,i],
                                ai_s[i][1:],ai_rho[1:].conj(),optimize=True)
            K_aibj_B += numpy.einsum('ghn,hnai,gnbj->aibj',kxc_Nn_Ns[:,:,i],
                                ai_s[i][1:],ai_rho[1:],optimize=True)
    
        # kxc_Ns_Ns
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        K_aibj_A += numpy.einsum('n,nai,nbj->aibj',kxc_Ns_Ns[offset2[i,j],k,l],
                                        ai_s[k][i+1],ai_s[l][j+1].conj(),optimize=True)
                        K_aibj_B += numpy.einsum('n,nai,nbj->aibj',kxc_Ns_Ns[offset2[i,j],k,l],
                                        ai_s[k][i+1],ai_s[l][j+1],optimize=True)

    else:
        raise NotImplementedError("Only LDA is implemented.")
    
    return K_aibj_A,K_aibj_B


class numint_tdamc(numint_gksmc.numint_gksmc,numint.NumInt):
    '''numint_tdamc'''
    def __init__(self):
        numint.NumInt.__init__(self)
        self.Spoints = Spoints.Spoints()

    numint_gksmc.uks_gga_wv0_intbypart_noweight = numint_gksmc.uks_gga_wv0_intbypart_noweight
    
    spin_flip_deriv = spin_flip_deriv
    spin_conserved_deriv = spin_conserved_deriv
    Kubler_spin_flip_deriv = Kubler_spin_flip_deriv
    
    nr_collinear_tdamc = nr_collinear_tdamc
    nr_collinear_tdalc = nr_collinear_tdalc
    collinear_tdamc_kernel = collinear_tdamc_kernel
    collinear_tdalc_kernel = collinear_tdalc_kernel
    collinear_1d_kernel = collinear_1d_kernel
    
    nr_noncollinear_tdamc = nr_noncollinear_tdamc
    nr_noncollinear_tdalc = nr_noncollinear_tdalc
    LDA_tda_kernel = LDA_tdamc_kernel
    GGA_tda_kernel = GGA_tdamc_kernel
    noncollinear_tdamc_kernel = noncollinear_tdamc_kernel
    noncollinear_tdalc_kernel = noncollinear_tdalc_kernel
    r_noncollinear_tdamc_kernel = r_noncollinear_tdamc_kernel
    r_noncollinear_tdamc = r_noncollinear_tdamc
    
    nr_collinear_tddft_mc = nr_collinear_tddft_mc
    nr_noncollinear_tddft_mc = nr_noncollinear_tddft_mc
    r_noncollinear_tddft_mc = r_noncollinear_tddft_mc
    
    def eval_xc_collinear_kernel(self, xc_code, rho, spin=1, relativity=0, deriv=2, omega=None,
                    verbose=None, LIBXCT_factor=None):
        '''eval_xc_collinear_kernel: serves as a calculator and transformer to obtain kernel for collinear case.
        
        Parameters
        ----------
        Args:
            xc_code : str
                Name of exchange-correlation functional.
            rho : tuple
                Density and magenization density norm with form (rhoa,rhob), whrer rhoa, rhob with a shape (nvar,grid). 
                For LDA, GGA and MGGA, nvar is 1, 4, 4, respectively. 4 means 1, nabla_x, nabla_y, nabla_z.
        
        Kwargs:
            spin : int
                spin = 1, to get derivatives based on rhoa and rhob.
            LIBXCT_factor : double or int
                The Threshold of for derivatives obatained from libxc.
                Deafult is None. Value Recommended is 1e-10.
        
        Returns:
        ----------
            kernel : tuple
                kernel[0], kernel[1] gives spin conserved and flip related derivatives, respectively.
        '''
        
        xctype = self._xc_type(xc_code)
        if omega is None: omega = self.omega
        
        if xctype != 'MGGA':
            vxc, fxc = self.libxc.eval_xc(xc_code, rho, spin, relativity, deriv,
                                    omega, verbose)[1:3]
        else:
            vxc, fxc = self.eval_xc_dlibxc(xc_code,rho)[1:3]
        
        vrho = vxc[0]
        v2rho2 = fxc[0]
        u_u, u_d, d_d = v2rho2.T 
        
        if LIBXCT_factor is None:
            LIBXCT_factor = -1
            
        if xctype == 'LDA':
            # transform to s 
            rhoa = rho[0]  
            rhob = rho[1] 
            # construct variables
            idxa_instable = rhoa <= LIBXCT_factor
            u_u[idxa_instable] *= 0.0
            u_d[idxa_instable] *= 0.0

            idxb_instable = rhob <= LIBXCT_factor
            d_d[idxb_instable] *= 0.0
            u_d[idxb_instable] *= 0.0

            pn_n = 0.25*(u_u + 2*u_d + d_d)
            pn_s = 0.25*(u_u - d_d)
            ps_s = 0.25*(u_u - 2*u_d + d_d)
            return (pn_n, pn_s, ps_s), ps_s
            
        elif xctype == 'GGA':
            # transform to s 
            rhoa = rho[0]  
            rhob = rho[1] 
            idxa_instable = rhoa[0] <= LIBXCT_factor
            u_u[idxa_instable] *= 0.0
            u_d[idxa_instable] *= 0.0

            idxb_instable = rhob[0] <= LIBXCT_factor
            d_d[idxb_instable] *= 0.0
            u_d[idxb_instable] *= 0.0

            pn_n = 0.25*(u_u + 2*u_d + d_d)
            pn_s = 0.25*(u_u - d_d)
            ps_s = 0.25*(u_u - 2*u_d + d_d)

            vsigma = vxc[1]
            v2rhosigma = fxc[1]
            v2sigma2 = fxc[2]
            ngrid = rho[0][0].shape[-1]  # rho.nidm = [2,4,ngrid]
            # calculate part of the derivatives
            # wva, wvb (2D numpy array): 
            # wvrho_nrho (2D numpy array):rhoa_nablarhoa,rhoa_nablarhob,rhob_nablarhoa,rhob_nablarhob
            # wvnrho_nrho (2D numpy array):ax_ax,ax_ay,ax_az,ay_ay,ay_az,az_az --> 0:6
            #                               0      1    2      3     4     5
            # ax_bx,ax_by,ax_bz, ay_bx,ay_by,ay_bz, az_bx,az_by,az_bz --> 6:15
            #   6     7     8      9    10   11      12    13    14   
            # bx_bx,bx_by,bx_bz,by_by,by_bz,bz_bz --> 15:21
            #  15    16    17    18    19    20   
            wva, wvb, wvrho_nrho, wvnrho_nrho =\
                numint_gksmc.uks_gga_wv0_intbypart_noweight(rho, (vrho, vsigma), (v2rho2, v2rhosigma, v2sigma2))

            wvrho_nrho[:9,idxa_instable] *= 0.0
            wvrho_nrho[3:,idxb_instable] *= 0.0

            wvnrho_nrho[:15,idxa_instable] *= 0.0
            wvnrho_nrho[6:,idxb_instable] *= 0.0

            # initiate some temperate variables.
            pn_Nn = numpy.zeros((3,ngrid))
            pn_Ns = numpy.zeros((3,ngrid))
            ps_Nn = numpy.zeros((3,ngrid))
            ps_Ns = numpy.zeros((3,ngrid))
            pNn_Nn = numpy.zeros((6,ngrid))
            pNn_Ns = numpy.zeros((3,3,ngrid))
            pNs_Ns = numpy.zeros((6,ngrid))

            pn_Nn[0] = 0.25*(wvrho_nrho[0] + wvrho_nrho[3] + wvrho_nrho[6] + wvrho_nrho[9] )
            pn_Nn[1] = 0.25*(wvrho_nrho[1] + wvrho_nrho[4] + wvrho_nrho[7] + wvrho_nrho[10])
            pn_Nn[2] = 0.25*(wvrho_nrho[2] + wvrho_nrho[5] + wvrho_nrho[8] + wvrho_nrho[11])

            pn_Ns[0] = 0.25*(wvrho_nrho[0] - wvrho_nrho[3] + wvrho_nrho[6] - wvrho_nrho[9] )
            pn_Ns[1] = 0.25*(wvrho_nrho[1] - wvrho_nrho[4] + wvrho_nrho[7] - wvrho_nrho[10])
            pn_Ns[2] = 0.25*(wvrho_nrho[2] - wvrho_nrho[5] + wvrho_nrho[8] - wvrho_nrho[11])

            ps_Nn[0] = 0.25*(wvrho_nrho[0] + wvrho_nrho[3] - wvrho_nrho[6] - wvrho_nrho[9] )
            ps_Nn[1] = 0.25*(wvrho_nrho[1] + wvrho_nrho[4] - wvrho_nrho[7] - wvrho_nrho[10])
            ps_Nn[2] = 0.25*(wvrho_nrho[2] + wvrho_nrho[5] - wvrho_nrho[8] - wvrho_nrho[11])
                
            ps_Ns[0] = 0.25*(wvrho_nrho[0] - wvrho_nrho[3] - wvrho_nrho[6] + wvrho_nrho[9] )
            ps_Ns[1] = 0.25*(wvrho_nrho[1] - wvrho_nrho[4] - wvrho_nrho[7] + wvrho_nrho[10])
            ps_Ns[2] = 0.25*(wvrho_nrho[2] - wvrho_nrho[5] - wvrho_nrho[8] + wvrho_nrho[11])

            pNn_Nn[0] = 0.25*(wvnrho_nrho[0] + wvnrho_nrho[6 ] + wvnrho_nrho[6 ] + wvnrho_nrho[15]) # xx
            pNn_Nn[1] = 0.25*(wvnrho_nrho[1] + wvnrho_nrho[7 ] + wvnrho_nrho[9 ] + wvnrho_nrho[16]) # xy
            pNn_Nn[2] = 0.25*(wvnrho_nrho[2] + wvnrho_nrho[8 ] + wvnrho_nrho[12] + wvnrho_nrho[17]) # xz
            pNn_Nn[3] = 0.25*(wvnrho_nrho[3] + wvnrho_nrho[10] + wvnrho_nrho[10] + wvnrho_nrho[18]) # yy
            pNn_Nn[4] = 0.25*(wvnrho_nrho[4] + wvnrho_nrho[11] + wvnrho_nrho[13] + wvnrho_nrho[19]) # yz
            pNn_Nn[5] = 0.25*(wvnrho_nrho[5] + wvnrho_nrho[14] + wvnrho_nrho[14] + wvnrho_nrho[20]) # zz

            pNn_Ns[0,0] = (wvnrho_nrho[0] - wvnrho_nrho[15])*0.25
            pNn_Ns[0,1] = (wvnrho_nrho[1] - wvnrho_nrho[7] + wvnrho_nrho[9]  - wvnrho_nrho[16])*0.25
            pNn_Ns[0,2] = (wvnrho_nrho[2] - wvnrho_nrho[8] + wvnrho_nrho[12] - wvnrho_nrho[17])*0.25
            pNn_Ns[1,0] = (wvnrho_nrho[1] - wvnrho_nrho[9] + wvnrho_nrho[7]  - wvnrho_nrho[16])*0.25
            pNn_Ns[1,1] = (wvnrho_nrho[3] - wvnrho_nrho[18])*0.25
            pNn_Ns[1,2] = (wvnrho_nrho[4] - wvnrho_nrho[11] + wvnrho_nrho[13] - wvnrho_nrho[19])*0.25
            pNn_Ns[2,0] = (wvnrho_nrho[2] - wvnrho_nrho[12] + wvnrho_nrho[8]  - wvnrho_nrho[17])*0.25
            pNn_Ns[2,1] = (wvnrho_nrho[4] - wvnrho_nrho[13] + wvnrho_nrho[11] - wvnrho_nrho[19])*0.25
            pNn_Ns[2,2] = (wvnrho_nrho[5] - wvnrho_nrho[20])*0.25
                
            pNs_Ns[0] = 0.25*(wvnrho_nrho[0] - wvnrho_nrho[6 ] - wvnrho_nrho[6 ] + wvnrho_nrho[15]) # xx
            pNs_Ns[1] = 0.25*(wvnrho_nrho[1] - wvnrho_nrho[7 ] - wvnrho_nrho[9 ] + wvnrho_nrho[16]) # xy
            pNs_Ns[2] = 0.25*(wvnrho_nrho[2] - wvnrho_nrho[8 ] - wvnrho_nrho[12] + wvnrho_nrho[17]) # xz
            pNs_Ns[3] = 0.25*(wvnrho_nrho[3] - wvnrho_nrho[10] - wvnrho_nrho[10] + wvnrho_nrho[18]) # yy
            pNs_Ns[4] = 0.25*(wvnrho_nrho[4] - wvnrho_nrho[11] - wvnrho_nrho[13] + wvnrho_nrho[19]) # yz
            pNs_Ns[5] = 0.25*(wvnrho_nrho[5] - wvnrho_nrho[14] - wvnrho_nrho[14] + wvnrho_nrho[20]) # zz
                
            return (pn_n,pn_s,pn_Nn,pn_Ns,ps_s,ps_Nn,ps_Ns,pNn_Nn,pNn_Ns,pNs_Ns),(ps_s, ps_Ns, pNs_Ns)
            
        elif xctype == 'MGGA':
            rhoa = rho[0]  
            rhob = rho[1] 
            idxa_instable = rhoa[0] <= LIBXCT_factor
            idxb_instable = rhob[0] <= LIBXCT_factor
            # vxc = (vrho, vsigma, vlapl, vtau)
            # fxc = (v2rho2, v2rhosigma, v2rholapl, v2rhotau, v2sigma2, v2sigmalapl, v2sigmatau, v2lapl2, v2lapltau, v2tau2)
            v2tau2 = fxc[9]
            tu_tu, tu_td, td_td = v2tau2.T
            vsigma = vxc[1]
            vtau = vxc[3]
            v2rhosigma = fxc[1]
            v2rhotau = fxc[3]
            v2sigma2 = fxc[4]
            v2sigmatau = fxc[6]
                
            u_tu, u_td,d_tu, d_td = v2rhotau.T
            ngrid = rho[0][0].shape[-1]
            # calculate part of the derivatives
            # wva, wvb (2D numpy array): 
            # wvrho_nrho (2D numpy array):rhoa_nablarhoa,rhoa_nablarhob,rhob_nablarhoa,rhob_nablarhob
            # wvnrho_nrho (2D numpy array):ax_ax,ax_ay,ax_az,ay_ay,ay_az,az_az --> 0:6
            #                               0      1    2      3     4     5
            # ax_bx,ax_by,ax_bz, ay_bx,ay_by,ay_bz, az_bx,az_by,az_bz --> 6:15
            #   6     7     8      9    10   11      12    13    14   
            # bx_bx,bx_by,bx_bz,by_by,by_bz,bz_bz --> 15:21
            #  15    16    17    18    19    20   
            wva, wvb, wvrho_nrho, wvnrho_nrho, wvtaua,wvtaub, wvnrho_tau =\
                numint_gksmc.uks_gga_wv0_intbypart_noweight(rho, (vrho, vsigma,vtau), (v2rho2, v2rhosigma, v2sigma2, v2sigmatau))
            # initiate some temperate variables.
            u_u[idxa_instable] *= 0.0
            u_d[idxa_instable] *= 0.0
            tu_tu[idxa_instable] *= 0.0
            tu_td[idxa_instable] *= 0.0
            u_tu[idxa_instable] *= 0.0
            u_td[idxa_instable] *= 0.0
            d_tu[idxa_instable] *= 0.0
            
            d_d[idxb_instable] *= 0.0
            u_d[idxb_instable] *= 0.0
            td_td[idxb_instable] *= 0.0
            td_td[idxb_instable] *= 0.0
            d_td[idxb_instable] *= 0.0
            u_td[idxb_instable] *= 0.0
            d_tu[idxb_instable] *= 0.0
            
            ps_s = 0.25*(u_u - 2*u_d + d_d)
            pu_u = 0.25*(tu_tu - 2*tu_td + td_td)
            ps_u = 0.25*(u_tu - u_td - d_tu + d_td)
                
            ps_Ns = numpy.zeros((3,ngrid))
            pNs_Ns = numpy.zeros((6,ngrid))
            pNs_u = numpy.zeros((3,ngrid))
            
            wvrho_nrho[:9,idxa_instable] *= 0.0
            wvrho_nrho[3:,idxb_instable] *= 0.0

            wvnrho_nrho[:15,idxa_instable] *= 0.0
            wvnrho_nrho[6:,idxb_instable] *= 0.0
            wvnrho_tau[:9,idxa_instable] *= 0.0
            wvnrho_tau[3:,idxb_instable] *= 0.0
            
            ps_Ns[0] = 0.25*(wvrho_nrho[0] - wvrho_nrho[3] - wvrho_nrho[6] + wvrho_nrho[9] )
            ps_Ns[1] = 0.25*(wvrho_nrho[1] - wvrho_nrho[4] - wvrho_nrho[7] + wvrho_nrho[10])
            ps_Ns[2] = 0.25*(wvrho_nrho[2] - wvrho_nrho[5] - wvrho_nrho[8] + wvrho_nrho[11])
            
            pNs_u[0] = 0.25*(wvnrho_tau[0] - wvnrho_tau[3] - wvnrho_tau[6] + wvnrho_tau[9] )
            pNs_u[1] = 0.25*(wvnrho_tau[1] - wvnrho_tau[4] - wvnrho_tau[7] + wvnrho_tau[10])
            pNs_u[2] = 0.25*(wvnrho_tau[2] - wvnrho_tau[5] - wvnrho_tau[8] + wvnrho_tau[11])
            
            pNs_Ns[0] = 0.25*(wvnrho_nrho[0] - wvnrho_nrho[6 ] - wvnrho_nrho[6 ] + wvnrho_nrho[15]) # xx
            pNs_Ns[1] = 0.25*(wvnrho_nrho[1] - wvnrho_nrho[7 ] - wvnrho_nrho[9 ] + wvnrho_nrho[16]) # xy
            pNs_Ns[2] = 0.25*(wvnrho_nrho[2] - wvnrho_nrho[8 ] - wvnrho_nrho[12] + wvnrho_nrho[17]) # xz
            pNs_Ns[3] = 0.25*(wvnrho_nrho[3] - wvnrho_nrho[10] - wvnrho_nrho[10] + wvnrho_nrho[18]) # yy
            pNs_Ns[4] = 0.25*(wvnrho_nrho[4] - wvnrho_nrho[11] - wvnrho_nrho[13] + wvnrho_nrho[19]) # yz
            pNs_Ns[5] = 0.25*(wvnrho_nrho[5] - wvnrho_nrho[14] - wvnrho_nrho[14] + wvnrho_nrho[20]) # zz

            # ToDo: Spin-conserved part of MGGA is a long work!
            return (None),(ps_s, ps_Ns, pNs_Ns, pu_u, ps_u, pNs_u)
            # raise NotImplementedError("Meta-GGA is not implemented")
             
    def eval_xc_Kubler_kernel(self, xc_code, rho, spin=1, relativity=0, deriv=2, omega=None,
                    verbose=None, LIBXCT_factor=None):
        '''eval_xc_Kubler_kernel: serves as a calculator and transformer to obtain kernel for collinear case in
        Locally Collinear approach.
        
        Parameters
        ----------
        Args:
            xc_code : str
                Name of exchange-correlation functional.
            rho : tuple
                Density and magenization density norm with form (rhoa,rhob), whrer rhoa, rhob with a shape (nvar,grid). 
                For LDA, GGA and MGGA, nvar is 1, 4, 4, respectively. 4 means 1, nabla_x, nabla_y, nabla_z.
        
        Kwargs:
            spin : int
                spin = 1, to get derivatives based on rhoa and rhob.
            LIBXCT_factor : double or int
                The Threshold of for derivatives obatained from libxc.
                Deafult is None. Value Recommended is 1e-10.
        
        Returns:
        ----------
           kernel : tuple
                kernel[0], kernel[1] gives spin conserved and flip related derivatives, respectively.
        
        Raises:
        ----------
            ToDo : derivatives for collinear case in Locally Collinear approach in GGA and MGGA.
        '''
        xctype = self._xc_type(xc_code)
        if omega is None: omega = self.omega
        
        vxc,fxc = self.libxc.eval_xc(xc_code, rho, spin, relativity, deriv,
                                    omega, verbose)[1:3]

        # vxc[0] -- vrho ; fxc[0] -- v2rho2
        u,d = vxc[0].T
        u_u, u_d, d_d = fxc[0].T 
        if LIBXCT_factor is None:
            LIBXCT_factor = -1
        
        if xctype == 'LDA':
            # transform to s 
            rhoa = rho[0]  
            rhob = rho[1] 
            # construct variables
            idxa_instable = rhoa <= LIBXCT_factor
            idxb_instable = rhob <= LIBXCT_factor
            u[idxa_instable] *= 0.0
            u_u[idxa_instable] *= 0.0
            u_d[idxa_instable] *= 0.0
            
            d[idxb_instable] *= 0.0
            d_d[idxb_instable] *= 0.0
            u_d[idxb_instable] *= 0.0
            
            # Spin-Flip term  
            pvs =  0.5*(u - d)
            # Spin-Conserved term  
            pn_n = 0.25*(u_u + 2*u_d + d_d)
            pn_s = 0.25*(u_u - d_d)   
            ps_s = 0.25*(u_u - 2*u_d + d_d)
            return (pn_n, pn_s, ps_s,pvs), (pvs,ps_s)
        else:
            raise NotImplementedError("GGA and MGGA are not implemented.")
            
    def eval_xc_noncollinear_kernel(self, xc_code, rho, spin=1, relativity=0, deriv=3, omega=None,
                    verbose=None,LIBXCT_factor=None):
        '''eval_xc_noncollinear_kernel: serves as a calculator and transformer to obtain effective kernel for noncollinear case in
        MUlti-Collinear approach.
        
        Parameters
        ----------
        Args:
            xc_code : str
                Name of exchange-correlation functional.
            rho : tuple
                Density and magenization density norm with form (rhoa,rhob), whrer rhoa, rhob with a shape (nvar,grid). 
                For LDA, GGA and MGGA, nvar is 1, 4, 4, respectively. 4 means 1, nabla_x, nabla_y, nabla_z.
        
        Kwargs:
            spin : int
                spin = 1, to get derivatives based on rhoa and rhob.
            deriv : int
                In Multi-Collinear approach deriv = deriv+1 to get the higher derivatives to construct effective kernel. 
            LIBXCT_factor : double or int
                The Threshold of for derivatives obatained from libxc.
                Deafult is None. Value Recommended is 1e-10.
        
        Returns:
        ----------
            effective kernel : tuple
                
        Raises:
        ----------
            ToDo : effective kernel for MGGA.
        '''
        xctype = self._xc_type(xc_code)
        if omega is None: omega = self.omega
        vxc,fxc, kxc= self.libxc.eval_xc(xc_code, rho, spin, relativity, deriv,
                                    omega, verbose)[1:4]
            # fxc = (v2rho2, v2rhosigma, v2rholapl, v2rhotau, v2sigma2, v2sigmalapl, v2sigmatau, v2lapl2, v2lapltau, v2tau2)
            # kxc = (v3rho3, v3rho2sigma, v3rhosigma2, v3sigma3)
        if LIBXCT_factor is None:
            LIBXCT_factor = -1
        
        if xctype == 'LDA':
            # import pdb
            # pdb.set_trace()
            # transform to ss
            rhoa = rho[0]    
            rhob = rho[1]
            s = rhoa - rhob

            # obtain basic kernel in pyscf-libxc 
            u_u,u_d,d_d = fxc[0].T
            u_u_u, u_u_d, u_d_d, d_d_d = kxc[0].T

            idxa_instable = rhoa <= LIBXCT_factor 
            u_u[idxa_instable] *= 0.0 
            u_d[idxa_instable] *= 0.0 
            u_u_u[idxa_instable] *= 0.0 
            u_u_d[idxa_instable] *= 0.0
            u_d_d[idxa_instable] *= 0.0

            idxb_instable = rhob <= LIBXCT_factor  
            d_d[idxb_instable] *= 0.0 
            u_d[idxb_instable] *= 0.0 
            d_d_d[idxb_instable] *= 0.0 
            u_u_d[idxb_instable] *= 0.0
            u_d_d[idxb_instable] *= 0.0 

            pn_n = 0.25*(u_u + 2*u_d + d_d)
            pn_s = 0.25*(u_u - d_d)
            ps_s = 0.25*(u_u -2*u_d + d_d)
            pn_n_s = 0.125*(u_u_u + u_u_d - u_d_d - d_d_d)
            pn_s_s = 0.125*(u_u_u - u_u_d - u_d_d + d_d_d) 
            ps_s_s = 0.125*(u_u_u - 3*u_u_d + 3*u_d_d - d_d_d) 

            # construct multi-collinear approach kernel k^eff
            kxc_nn = pn_n + s*pn_n_s
            kxc_ns = 2*pn_s + s*pn_s_s
            # kxc_ns = kxc_sn
            kxc_ss = 3*ps_s + s*ps_s_s

            return kxc_nn, kxc_ns, kxc_ss

        elif xctype == 'GGA':
            # import pdb
            # pdb.set_trace()
            rhoa = rho[0]    
            rhob = rho[1]
            s = rhoa - rhob
            ngrid = rhoa[0].shape[-1] # rho.nidm = [2,4,ngrid]

            # fxc = (v2rho2, v2rhosigma, v2rholapl, v2rhotau, v2sigma2, v2sigmalapl, v2sigmatau, v2lapl2, v2lapltau, v2tau2)
            # kxc = (v3rho3, v3rho2sigma, v3rhosigma2, v3sigma3)
            vrho = vxc[0]    
            v2rho2 = fxc[0]
            vsigma = vxc[1]
            v2rhosigma = fxc[1]
            v2sigma2 = fxc[2]
             
            u_u,u_d,d_d = fxc[0].T
            u_u_u, u_u_d, u_d_d, d_d_d = kxc[0].T

            
            idxa_instable = rhoa[0] <= LIBXCT_factor 
            idxb_instable = rhob[0] <= LIBXCT_factor 

            u_u[idxa_instable] *= 0.0 
            u_d[idxa_instable] *= 0.0 
            d_d[idxb_instable] *= 0.0 
            u_d[idxb_instable] *= 0.0 
            
            pn_n = 0.25*(u_u + 2*u_d + d_d)
            pn_s = 0.25*(u_u - d_d)
            ps_s = 0.25*(u_u -2*u_d + d_d)

            # ~ Second order part BEGIND
            # calculate part of the derivatives
            # wva, wvb (2D numpy array): 
            # wvrho_nrho (2D numpy array):rhoa_nablarhoa,rhoa_nablarhob,rhob_nablarhoa,rhob_nablarhob
            # wvnrho_nrho (2D numpy array):ax_ax,ax_ay,ax_az,ay_ay,ay_az,az_az --> 0:6
            #                               0      1    2      3     4     5
            # ax_bx,ax_by,ax_bz, ay_bx,ay_by,ay_bz, az_bx,az_by,az_bz --> 6:15
            #   6     7     8      9    10   11      12    13    14   
            # bx_bx,bx_by,bx_bz,by_by,by_bz,bz_bz --> 15:21
            #  15    16    17    18    19    20   
            wva, wvb, wvrho_nrho, wvnrho_nrho =\
                numint_gksmc.uks_gga_wv0_intbypart_noweight(rho, (vrho, vsigma), (v2rho2, v2rhosigma, v2sigma2))
            
            wvrho_nrho[:9,idxa_instable] *= 0.0
            wvrho_nrho[3:,idxb_instable] *= 0.0
            wvnrho_nrho[:15,idxa_instable] *= 0.0
            wvnrho_nrho[6:,idxb_instable] *= 0.0

            pn_Nn = numpy.zeros((3,ngrid))
            pn_Ns = numpy.zeros((3,ngrid))
            ps_Nn = numpy.zeros((3,ngrid))
            ps_Ns = numpy.zeros((3,ngrid))
            pNn_Nn = numpy.zeros((6,ngrid))
            pNn_Ns = numpy.zeros((3,3,ngrid))
            pNs_Ns = numpy.zeros((6,ngrid))

            pn_Nn[0] = 0.25*(wvrho_nrho[0] + wvrho_nrho[3] + wvrho_nrho[6] + wvrho_nrho[9] )
            pn_Nn[1] = 0.25*(wvrho_nrho[1] + wvrho_nrho[4] + wvrho_nrho[7] + wvrho_nrho[10])
            pn_Nn[2] = 0.25*(wvrho_nrho[2] + wvrho_nrho[5] + wvrho_nrho[8] + wvrho_nrho[11])

            pn_Ns[0] = 0.25*(wvrho_nrho[0] - wvrho_nrho[3] + wvrho_nrho[6] - wvrho_nrho[9] )
            pn_Ns[1] = 0.25*(wvrho_nrho[1] - wvrho_nrho[4] + wvrho_nrho[7] - wvrho_nrho[10])
            pn_Ns[2] = 0.25*(wvrho_nrho[2] - wvrho_nrho[5] + wvrho_nrho[8] - wvrho_nrho[11])

            ps_Nn[0] = 0.25*(wvrho_nrho[0] + wvrho_nrho[3] - wvrho_nrho[6] - wvrho_nrho[9] )
            ps_Nn[1] = 0.25*(wvrho_nrho[1] + wvrho_nrho[4] - wvrho_nrho[7] - wvrho_nrho[10])
            ps_Nn[2] = 0.25*(wvrho_nrho[2] + wvrho_nrho[5] - wvrho_nrho[8] - wvrho_nrho[11])

            ps_Ns[0] = 0.25*(wvrho_nrho[0] - wvrho_nrho[3] - wvrho_nrho[6] + wvrho_nrho[9] )
            ps_Ns[1] = 0.25*(wvrho_nrho[1] - wvrho_nrho[4] - wvrho_nrho[7] + wvrho_nrho[10])
            ps_Ns[2] = 0.25*(wvrho_nrho[2] - wvrho_nrho[5] - wvrho_nrho[8] + wvrho_nrho[11])
            
            pNn_Nn[0] = 0.25*(wvnrho_nrho[0] + wvnrho_nrho[6 ] + wvnrho_nrho[6 ] + wvnrho_nrho[15]) # xx
            pNn_Nn[1] = 0.25*(wvnrho_nrho[1] + wvnrho_nrho[7 ] + wvnrho_nrho[9 ] + wvnrho_nrho[16]) # xy
            pNn_Nn[2] = 0.25*(wvnrho_nrho[2] + wvnrho_nrho[8 ] + wvnrho_nrho[12] + wvnrho_nrho[17]) # xz
            pNn_Nn[3] = 0.25*(wvnrho_nrho[3] + wvnrho_nrho[10] + wvnrho_nrho[10] + wvnrho_nrho[18]) # yy
            pNn_Nn[4] = 0.25*(wvnrho_nrho[4] + wvnrho_nrho[11] + wvnrho_nrho[13] + wvnrho_nrho[19]) # yz
            pNn_Nn[5] = 0.25*(wvnrho_nrho[5] + wvnrho_nrho[14] + wvnrho_nrho[14] + wvnrho_nrho[20]) # zz

            pNn_Ns[0,0] = (wvnrho_nrho[0] - wvnrho_nrho[15])*0.25
            pNn_Ns[0,1] = (wvnrho_nrho[1] - wvnrho_nrho[7] + wvnrho_nrho[9]  - wvnrho_nrho[16])*0.25
            pNn_Ns[0,2] = (wvnrho_nrho[2] - wvnrho_nrho[8] + wvnrho_nrho[12] - wvnrho_nrho[17])*0.25
            pNn_Ns[1,0] = (wvnrho_nrho[1] - wvnrho_nrho[9] + wvnrho_nrho[7]  - wvnrho_nrho[16])*0.25
            pNn_Ns[1,1] = (wvnrho_nrho[3] - wvnrho_nrho[18])*0.25
            pNn_Ns[1,2] = (wvnrho_nrho[4] - wvnrho_nrho[11] + wvnrho_nrho[13] - wvnrho_nrho[19])*0.25
            pNn_Ns[2,0] = (wvnrho_nrho[2] - wvnrho_nrho[12] + wvnrho_nrho[8]  - wvnrho_nrho[17])*0.25
            pNn_Ns[2,1] = (wvnrho_nrho[4] - wvnrho_nrho[13] + wvnrho_nrho[11] - wvnrho_nrho[19])*0.25
            pNn_Ns[2,2] = (wvnrho_nrho[5] - wvnrho_nrho[20])*0.25

            pNs_Ns[0] = 0.25*(wvnrho_nrho[0] - wvnrho_nrho[6 ] - wvnrho_nrho[6 ] + wvnrho_nrho[15]) # xx
            pNs_Ns[1] = 0.25*(wvnrho_nrho[1] - wvnrho_nrho[7 ] - wvnrho_nrho[9 ] + wvnrho_nrho[16]) # xy
            pNs_Ns[2] = 0.25*(wvnrho_nrho[2] - wvnrho_nrho[8 ] - wvnrho_nrho[12] + wvnrho_nrho[17]) # xz
            pNs_Ns[3] = 0.25*(wvnrho_nrho[3] - wvnrho_nrho[10] - wvnrho_nrho[10] + wvnrho_nrho[18]) # yy
            pNs_Ns[4] = 0.25*(wvnrho_nrho[4] - wvnrho_nrho[11] - wvnrho_nrho[13] + wvnrho_nrho[19]) # yz
            pNs_Ns[5] = 0.25*(wvnrho_nrho[5] - wvnrho_nrho[14] - wvnrho_nrho[14] + wvnrho_nrho[20]) # zz


            # ~ Third order part BEGIND
            u_u_u[idxa_instable] *= 0.0 
            u_u_d[idxa_instable] *= 0.0
            u_d_d[idxa_instable] *= 0.0

            d_d_d[idxb_instable] *= 0.0 
            u_u_d[idxb_instable] *= 0.0
            u_d_d[idxb_instable] *= 0.0 

            pn_n_s = 0.125*(u_u_u + u_u_d - u_d_d - d_d_d)
            pn_s_s = 0.125*(u_u_u - u_u_d - u_d_d + d_d_d) 
            ps_s_s = 0.125*(u_u_u - 3*u_u_d + 3*u_d_d - d_d_d) 
            
            # One of the most troblesome codes are done in get_kxc_in_s_n
            # n_s_Ns : (3, ngrid) x y z
            # s_s_Ns : (3, ngrid) x y z
            # n_Ns_Ns : (6, ngrid) xx xy xz yy yz zz
            # s_Ns_Ns : (6, ngrid) xx xy xz yy yz zz
            # s_Nn_Ns : (3, 3, ngrid) (x y z) times (x y z)
            # Nn_Ns_Ns : (3,6,ngrid) (x y z) times (xx xy xz yy yz zz)
            # Ns_Ns_Ns : (10, ngrid) xxx xxy xxz xyy xyz xzz yyy yyz yzz zzz
            # Need n_n_Ns
            pn_n_Ns, pn_s_Ns, ps_s_Ns, pn_s_Nn, ps_s_Nn, pn_Ns_Ns, ps_Ns_Ns, ps_Nn_Ns, \
                pn_Nn_Ns, ps_Nn_Nn, pNn_Ns_Ns, pNs_Ns_Ns, pNn_Nn_Ns = numint_gksmc.get_kxc_in_s_n_kernel(rho, v2rhosigma, v2sigma2, kxc,
                                                                      LIBXCT_factor) 
 
            # construct multi-collinear approach kernel k^eff
            # The term 1,2
            kxc_nn = pn_n + s[0]*pn_n_s + s[1]*pn_n_Ns[0] + s[2]*pn_n_Ns[1] + s[3]*pn_n_Ns[2]
            kxc_ns = 2*pn_s + s[0]*pn_s_s + s[1]*pn_s_Ns[0] + s[2]*pn_s_Ns[1] + s[3]*pn_s_Ns[2]
 
            # The term 3 
            kxc_n_Nn = numpy.zeros((3,ngrid))
            kxc_n_Nn[0]  = pn_Nn[0] + s[0]*pn_s_Nn[0] 
            kxc_n_Nn[0] += s[1]*pn_Nn_Ns[0][0]
            kxc_n_Nn[0] += s[2]*pn_Nn_Ns[0][1]
            kxc_n_Nn[0] += s[3]*pn_Nn_Ns[0][2]

            kxc_n_Nn[1]  = pn_Nn[1] + s[0]*pn_s_Nn[1] 
            kxc_n_Nn[1] += s[1]*pn_Nn_Ns[1][0]
            kxc_n_Nn[1] += s[2]*pn_Nn_Ns[1][1]
            kxc_n_Nn[1] += s[3]*pn_Nn_Ns[1][2]

            kxc_n_Nn[2]  = pn_Nn[2] + s[0]*pn_s_Nn[2] 
            kxc_n_Nn[2] += s[1]*pn_Nn_Ns[2][0]
            kxc_n_Nn[2] += s[2]*pn_Nn_Ns[2][1]
            kxc_n_Nn[2] += s[3]*pn_Nn_Ns[2][2]

            # The term 4 
            kxc_n_Ns = numpy.zeros((3,ngrid))
            kxc_n_Ns[0]  = 2*pn_Ns[0] + s[0]*pn_s_Ns[0] 
            kxc_n_Ns[0] += s[1]*pn_Ns_Ns[0]
            kxc_n_Ns[0] += s[2]*pn_Ns_Ns[1]
            kxc_n_Ns[0] += s[3]*pn_Ns_Ns[2]

            kxc_n_Ns[1]  = 2*pn_Ns[1] + s[0]*pn_s_Ns[1] 
            kxc_n_Ns[1] += s[1]*pn_Ns_Ns[1]
            kxc_n_Ns[1] += s[2]*pn_Ns_Ns[3]
            kxc_n_Ns[1] += s[3]*pn_Ns_Ns[4]
           
            kxc_n_Ns[2]  = 2*pn_Ns[2] + s[0]*pn_s_Ns[2] 
            kxc_n_Ns[2] += s[1]*pn_Ns_Ns[2]
            kxc_n_Ns[2] += s[2]*pn_Ns_Ns[4]
            kxc_n_Ns[2] += s[3]*pn_Ns_Ns[5]

            # The term 5
            kxc_ss = 3*ps_s + s[0]*ps_s_s + s[1]*ps_s_Ns[0] + s[2]*ps_s_Ns[1] + s[3]*ps_s_Ns[2]

            # The term 6
            kxc_s_Nn = numpy.zeros((3,ngrid))
            kxc_s_Nn[0]  = 2*ps_Nn[0] + s[0]*ps_s_Nn[0] 
            kxc_s_Nn[0] += s[1]*ps_Nn_Ns[0][0]
            kxc_s_Nn[0] += s[2]*ps_Nn_Ns[0][1]
            kxc_s_Nn[0] += s[3]*ps_Nn_Ns[0][2]

            kxc_s_Nn[1]  = 2*ps_Nn[1] + s[0]*ps_s_Nn[1] 
            kxc_s_Nn[1] += s[1]*ps_Nn_Ns[1][0]
            kxc_s_Nn[1] += s[2]*ps_Nn_Ns[1][1]
            kxc_s_Nn[1] += s[3]*ps_Nn_Ns[1][2]

            kxc_s_Nn[2]  = 2*ps_Nn[2] + s[0]*ps_s_Nn[2] 
            kxc_s_Nn[2] += s[1]*ps_Nn_Ns[2][0]
            kxc_s_Nn[2] += s[2]*ps_Nn_Ns[2][1]
            kxc_s_Nn[2] += s[3]*ps_Nn_Ns[2][2]

            # The term 7
            kxc_s_Ns = numpy.zeros((3,ngrid))
            kxc_s_Ns[0]  = 3*ps_Ns[0] + s[0]*ps_s_Ns[0] 
            kxc_s_Ns[0] += s[1]*ps_Ns_Ns[0]
            kxc_s_Ns[0] += s[2]*ps_Ns_Ns[1]
            kxc_s_Ns[0] += s[3]*ps_Ns_Ns[2]

            kxc_s_Ns[1]  = 3*ps_Ns[1] + s[0]*ps_s_Ns[1] 
            kxc_s_Ns[1] += s[1]*ps_Ns_Ns[1]
            kxc_s_Ns[1] += s[2]*ps_Ns_Ns[3]
            kxc_s_Ns[1] += s[3]*ps_Ns_Ns[4]
           
            kxc_s_Ns[2]  = 3*ps_Ns[2] + s[0]*ps_s_Ns[2] 
            kxc_s_Ns[2] += s[1]*ps_Ns_Ns[2]
            kxc_s_Ns[2] += s[2]*ps_Ns_Ns[4]
            kxc_s_Ns[2] += s[3]*ps_Ns_Ns[5]

            # The term 8
            kxc_Nn_Nn = numpy.zeros((6,ngrid))
            kxc_Nn_Nn[0]  = pNn_Nn[0] + s[0]*ps_Nn_Nn[0] + s[1]*pNn_Nn_Ns[0][0]
            kxc_Nn_Nn[0] += s[2]*pNn_Nn_Ns[0][1]
            kxc_Nn_Nn[0] += s[3]*pNn_Nn_Ns[0][2]

            kxc_Nn_Nn[1]  = pNn_Nn[1] + s[0]*ps_Nn_Nn[1] + s[1]*pNn_Nn_Ns[1][0]
            kxc_Nn_Nn[1] += s[2]*pNn_Nn_Ns[1][1]
            kxc_Nn_Nn[1] += s[3]*pNn_Nn_Ns[1][2]    

            kxc_Nn_Nn[2]  = pNn_Nn[2] + s[0]*ps_Nn_Nn[2] + s[1]*pNn_Nn_Ns[2][0]
            kxc_Nn_Nn[2] += s[2]*pNn_Nn_Ns[2][1]
            kxc_Nn_Nn[2] += s[3]*pNn_Nn_Ns[2][2]

            kxc_Nn_Nn[3]  = pNn_Nn[3] + s[0]*ps_Nn_Nn[3] + s[1]*pNn_Nn_Ns[3][0]
            kxc_Nn_Nn[3] += s[2]*pNn_Nn_Ns[3][1]
            kxc_Nn_Nn[3] += s[3]*pNn_Nn_Ns[3][2]

            kxc_Nn_Nn[4]  = pNn_Nn[4] + s[0]*ps_Nn_Nn[4] + s[1]*pNn_Nn_Ns[4][0]
            kxc_Nn_Nn[4] += s[2]*pNn_Nn_Ns[4][1]
            kxc_Nn_Nn[4] += s[3]*pNn_Nn_Ns[4][2]    

            kxc_Nn_Nn[5]  = pNn_Nn[5] + s[0]*ps_Nn_Nn[5] + s[1]*pNn_Nn_Ns[5][0]
            kxc_Nn_Nn[5] += s[2]*pNn_Nn_Ns[5][1]
            kxc_Nn_Nn[5] += s[3]*pNn_Nn_Ns[5][2]

            # The term 9
            kxc_Nn_Ns = numpy.zeros((3,3,ngrid))
            kxc_Nn_Ns[0][0]  = 2*pNn_Ns[0,0] + s[0]*ps_Nn_Ns[0][0] + s[1]*pNn_Ns_Ns[0][0]
            kxc_Nn_Ns[0][0] += s[2]*pNn_Ns_Ns[0][1]
            kxc_Nn_Ns[0][0] += s[3]*pNn_Ns_Ns[0][2]

            kxc_Nn_Ns[0][1]  = 2*pNn_Ns[0,1] + s[0]*ps_Nn_Ns[0][1] + s[1]*pNn_Ns_Ns[0][1]
            kxc_Nn_Ns[0][1] += s[2]*pNn_Ns_Ns[0][3]
            kxc_Nn_Ns[0][1] += s[3]*pNn_Ns_Ns[0][4]

            kxc_Nn_Ns[0][2]  = 2*pNn_Ns[0,2] + s[0]*ps_Nn_Ns[0][2] + s[1]*pNn_Ns_Ns[0][2]
            kxc_Nn_Ns[0][2] += s[2]*pNn_Ns_Ns[0][4]
            kxc_Nn_Ns[0][2] += s[3]*pNn_Ns_Ns[0][5]

            kxc_Nn_Ns[1][0]  = 2*pNn_Ns[1,0] + s[0]*ps_Nn_Ns[1][0] + s[1]*pNn_Ns_Ns[1][0]
            kxc_Nn_Ns[1][0] += s[2]*pNn_Ns_Ns[1][1]
            kxc_Nn_Ns[1][0] += s[3]*pNn_Ns_Ns[1][2]

            kxc_Nn_Ns[1][1]  = 2*pNn_Ns[1,1] + s[0]*ps_Nn_Ns[1][1] + s[1]*pNn_Ns_Ns[1][1]
            kxc_Nn_Ns[1][1] += s[2]*pNn_Ns_Ns[1][3]
            kxc_Nn_Ns[1][1] += s[3]*pNn_Ns_Ns[1][4]
           
            kxc_Nn_Ns[1][2]  = 2*pNn_Ns[1,2] + s[0]*ps_Nn_Ns[1][2] + s[1]*pNn_Ns_Ns[1][2]
            kxc_Nn_Ns[1][2] += s[2]*pNn_Ns_Ns[1][4]
            kxc_Nn_Ns[1][2] += s[3]*pNn_Ns_Ns[1][5]

            kxc_Nn_Ns[2][0]  = 2*pNn_Ns[2,0] + s[0]*ps_Nn_Ns[2][0] + s[1]*pNn_Ns_Ns[2][0]
            kxc_Nn_Ns[2][0] += s[2]*pNn_Ns_Ns[2][1]
            kxc_Nn_Ns[2][0] += s[3]*pNn_Ns_Ns[2][2]

            kxc_Nn_Ns[2][1]  = 2*pNn_Ns[2,1] + s[0]*ps_Nn_Ns[2][1] + s[1]*pNn_Ns_Ns[2][1]
            kxc_Nn_Ns[2][1] += s[2]*pNn_Ns_Ns[2][3]
            kxc_Nn_Ns[2][1] += s[3]*pNn_Ns_Ns[2][4]

            kxc_Nn_Ns[2][2]  = 2*pNn_Ns[2,2] + s[0]*ps_Nn_Ns[2][2] + s[1]*pNn_Ns_Ns[2][2]
            kxc_Nn_Ns[2][2] += s[2]*pNn_Ns_Ns[2][4]
            kxc_Nn_Ns[2][2] += s[3]*pNn_Ns_Ns[2][5]

           # The term 10
            kxc_Ns_Ns = numpy.zeros((6,ngrid))
            kxc_Ns_Ns[0]  = 3*pNs_Ns[0] + s[0]*ps_Ns_Ns[0] + s[1]*pNs_Ns_Ns[0]
            kxc_Ns_Ns[0] += s[2]*pNs_Ns_Ns[1]
            kxc_Ns_Ns[0] += s[3]*pNs_Ns_Ns[2]

            kxc_Ns_Ns[1]  = 3*pNs_Ns[1] + s[0]*ps_Ns_Ns[1] + s[1]*pNs_Ns_Ns[1]
            kxc_Ns_Ns[1] += s[2]*pNs_Ns_Ns[3]
            kxc_Ns_Ns[1] += s[3]*pNs_Ns_Ns[4]

            kxc_Ns_Ns[2]  = 3*pNs_Ns[2] + s[0]*ps_Ns_Ns[2] + s[1]*pNs_Ns_Ns[2]
            kxc_Ns_Ns[2] += s[2]*pNs_Ns_Ns[4]
            kxc_Ns_Ns[2] += s[3]*pNs_Ns_Ns[5]

            kxc_Ns_Ns[3]  = 3*pNs_Ns[3] + s[0]*ps_Ns_Ns[3] + s[1]*pNs_Ns_Ns[3]
            kxc_Ns_Ns[3] += s[2]*pNs_Ns_Ns[6]
            kxc_Ns_Ns[3] += s[3]*pNs_Ns_Ns[7]

            kxc_Ns_Ns[4]  = 3*pNs_Ns[4] + s[0]*ps_Ns_Ns[4] + s[1]*pNs_Ns_Ns[4]
            kxc_Ns_Ns[4] += s[2]*pNs_Ns_Ns[7]
            kxc_Ns_Ns[4] += s[3]*pNs_Ns_Ns[8]

            kxc_Ns_Ns[5]  = 3*pNs_Ns[5] + s[0]*ps_Ns_Ns[5] + s[1]*pNs_Ns_Ns[5]
            kxc_Ns_Ns[5] += s[2]*pNs_Ns_Ns[8]
            kxc_Ns_Ns[5] += s[3]*pNs_Ns_Ns[9]


            return kxc_nn, kxc_ns, kxc_n_Nn, kxc_n_Ns, kxc_ss, kxc_s_Nn, kxc_s_Ns, \
                kxc_Nn_Nn, kxc_Nn_Ns, kxc_Ns_Ns
                
        if xctype == 'MGGA':
            raise NotImplementedError("Meta-GGA is not implemented")
        
    def eval_xc_dlibxc(self,xc_code,rho):
        # ToDo: this function will be abandoned after pySCF debugging the corresponding part. 
        r'''eval_xc_dlibxc: serves as a temporary transformer to obtain MGGA derivatives from libxc.
        
        Parameters
        ----------
        Args:
            xc_code : str
                Name of exchange-correlation functional.
            rho : tuple
                Density and magenization density norm with form (rhoa,rhob), whrer rhoa, rhob with a shape (6,grid). 
                6 means 1, nabla_x, nabla_y, nabla_z, laplacian=\nabla^2 den, tau = 1/2(\nabla f)^2
        
        Returns:
        ----------
            derivatives : tuple
                exc, vxc, fxc related with MGGA.
        '''
        func_x = 'MGGA_X_'+ str(xc_code)
        func_c = 'MGGA_C_'+ str(xc_code)
        func_x = pylibxc.LibXCFunctional(func_x,'polarized')
        func_c = pylibxc.LibXCFunctional(func_c,'polarized')
        inp = {}
        # transform dim of variables for pylibxc
        rho_u = rho[0][0].reshape(1,-1)
        rho_d = rho[1][0].reshape(1,-1)
        ngrid = rho_u.shape[-1] 

        sigma_uu = numpy.sum((rho[0][1:4] * rho[0][1:4]).T, axis = 1).reshape(1,-1)
        sigma_ud = numpy.sum((rho[0][1:4] * rho[1][1:4]).T, axis = 1).reshape(1,-1)
        sigma_dd = numpy.sum((rho[1][1:4] * rho[1][1:4]).T, axis = 1).reshape(1,-1)
        
        lapl_u = rho[0][4].reshape(1,-1)
        lapl_d = rho[1][4].reshape(1,-1)
        tau_u = rho[0][5].reshape(1,-1)
        tau_d = rho[1][5].reshape(1,-1)

        inp["rho"] = numpy.empty((ngrid,2))
        inp["rho"][:,0] = rho_u
        inp["rho"][:,1] = rho_d

        inp["sigma"] = numpy.empty((ngrid,3))
        inp["sigma"][:,0] = sigma_uu
        inp["sigma"][:,1] = sigma_ud
        inp["sigma"][:,2] = sigma_dd

        inp["lapl"] = numpy.empty((ngrid,2))
        inp["lapl"][:,0] = lapl_u
        inp["lapl"][:,1] = lapl_d

        inp["tau"] = numpy.empty((ngrid,2))
        inp["tau"][:,0] = tau_u
        inp["tau"][:,1] = tau_d
        # MGGA:
        #                     EXC: zk
        #                     VXC: vrho, vsigma, vlapl (optional), vtau
        #                     FXC: v2rho2, v2rhosigma, v2rholapl, v2rhotau, v2sigma2,
        #                          v2sigmalapl, v2sigmatau, v2lapl2, v2lapltau, v2tau2
        #                     KXC: v3rho3, v3rho2sigma, v3rho2lapl, v3rho2tau, v3rhosigma2,
        #                          v3rhosigmalapl, v3rhosigmatau, v3rholapl2, v3rholapltau,
        #                          v3rhotau2, v3sigma3, v3sigma2lapl, v3sigma2tau,
        #                          v3sigmalapl2, v3sigmalapltau, v3sigmatau2, v3lapl3,
        #                          v3lapl2tau, v3lapltau2, v3tau3
        #                     LXC: v4rho4, v4rho3sigma, v4rho3lapl, v4rho3tau, v4rho2sigma2,
        #                          v4rho2sigmalapl, v4rho2sigmatau, v4rho2lapl2, v4rho2lapltau,
        #                          v4rho2tau2, v4rhosigma3, v4rhosigma2lapl, v4rhosigma2tau,
        #                          v4rhosigmalapl2, v4rhosigmalapltau, v4rhosigmatau2,
        #                          v4rholapl3, v4rholapl2tau, v4rholapltau2, v4rhotau3,
        #                          v4sigma4, v4sigma3lapl, v4sigma3tau, v4sigma2lapl2,
        #                          v4sigma2lapltau, v4sigma2tau2, v4sigmalapl3, v4sigmalapl2tau,
        #                          v4sigmalapltau2, v4sigmatau3, v4lapl4, v4lapl3tau,
        #                          v4lapl2tau2, v4lapltau3, v4tau4

        ret_X = func_x.compute(inp,do_fxc = True) 
        ret_C = func_c.compute(inp,do_fxc = True)
        for key in ret_X.keys():
            ret_X[key] += ret_C[key]
        # for k_X, v_X in ret_X.items():
        #     for k_C, v_C in ret_C.items():
        #         if k_X == k_C:
        #                 v_X += v_C
        # import pdb
        # pdb.set_trace()
        ret = ret_X
        exc = ret["zk"]
        vrho = ret["vrho"]
        vsigma = ret["vsigma"]
        vlapl = ret["vlapl"]
        vtau = ret["vtau"]
        vxc = (vrho, vsigma, vlapl, vtau)

        v2rho2 = ret["v2rho2"]
        v2rhosigma = ret["v2rhosigma"]
        v2rholapl = ret["v2rholapl"]
        v2rhotau = ret["v2rhotau"]
        v2sigma2 = ret["v2sigma2"]
        v2sigmalapl = ret["v2sigmalapl"]
        v2sigmatau = ret["v2sigmatau"]
        v2lapl2 = ret["v2lapl2"]
        v2lapltau = ret["v2lapltau"]
        v2tau2 = ret["v2tau2"]
        fxc = (v2rho2, v2rhosigma, v2rholapl, v2rhotau, v2sigma2, v2sigmalapl, v2sigmatau, v2lapl2, v2lapltau, v2tau2)
        return exc, vxc, fxc