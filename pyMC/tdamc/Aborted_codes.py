import numpy
from pyscf import lib
from pyMC.tdamc import tddft_mc_uks 
from pyMC.tdamc import tddft_mc_gks 
from pyMC.gksmc import numint_gksmc 

def non_collinear_ApB_matx_r(e_ia, kernel, x0, xctype, weights, ais, uvs, mf,*args):
    fxc,hyec = kernel 
    mo_vir_L, mo_vir_S, mo_occ_L , mo_occ_S = ais
    nstates = x0.shape[-1]
    C_vir, C_occ = uvs
    nocc = C_occ.shape[-1]
    nvir = C_vir.shape[-1]
    # import pdb
    # pdb.set_trace()
    if xctype == 'LDA':
        n_n,n_s,s_s = fxc 
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
        ai_s = numpy.array([ai_Mx,ai_My,ai_Mz])
        
        rho1 = numpy.einsum('pbj,jbn->pn', ai_rho.conj(), x0.reshape(nocc,nvir,nstates), optimize=True)
        M1 = numpy.einsum('xpbj,jbn->xpn', ai_s.conj(), x0.reshape(nocc,nvir,nstates), optimize=True)
        
        # A_ia = 0.0
        A_ia  = numpy.einsum('pai,p,pn->ian', ai_rho, n_n, rho1, optimize=True) # n_n
        A_ia += numpy.einsum('pai,xp,xpn->ian', ai_rho, n_s, M1, optimize=True) # n_s
        A_ia += numpy.einsum('xpai,xp,pn->ian', ai_s, n_s, rho1, optimize=True) # s_n
        A_ia += numpy.einsum('xpai,xyp,ypn->ian', ai_s, s_s, M1, optimize=True) # s_s
        
        B_ia  = numpy.einsum('pai,p,pn->ian', ai_rho, n_n, rho1.conj(), optimize=True) # n_n
        B_ia += numpy.einsum('pai,xp,xpn->ian', ai_rho, n_s, M1.conj(), optimize=True) # n_s
        B_ia += numpy.einsum('xpai,xp,pn->ian', ai_s, n_s, rho1.conj(), optimize=True) # s_n
        B_ia += numpy.einsum('xpai,xyp,ypn->ian', ai_s, s_s, M1.conj(), optimize=True) # s_s
        # B_ia *= 0.0
        # The orbital energy difference is calculated here
        A_ia += numpy.einsum('ia,ian->ian', e_ia.reshape(nocc,nvir), x0.reshape(nocc,nvir,nstates), optimize=True)
        
    elif xctype == 'GGA':
        n_n, n_s, n_Nn, n_Ns, s_s, s_Nn, s_Ns, Nn_Nntmp, Nn_Ns, Ns_Nstmp = fxc
        
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
        
        ai_s = numpy.array([ai_Mx,ai_My,ai_Mz])
        ngrid = Nn_Nntmp.shape[-1]
        rho1 = numpy.einsum('pbj,jbn->pn', ai_rho[0].conj(), x0.reshape(nocc,nvir,nstates), optimize=True)
        M1 = numpy.einsum('xpbj,jbn->xpn', ai_s[:,0].conj(), x0.reshape(nocc,nvir,nstates), optimize=True)
        nrho1 = numpy.einsum('xpbj,jbn->xpn', ai_rho[1:].conj(), x0.reshape(nocc,nvir,nstates), optimize=True)
        # y means nabla_i, x means m_i
        nM1 = numpy.einsum('xypbj,jbn->yxpn', ai_s[:,1:].conj(), x0.reshape(nocc,nvir,nstates), optimize=True)
        
        # import pdb
        # pdb.set_trace()
        A_ia = numpy.einsum('pai,p,pn->ian', ai_rho[0], n_n, rho1, optimize=True).astype(numpy.complex128) # n_n
        
        A_ia+= numpy.einsum('pai,xp,xpn->ian', ai_rho[0], n_s, M1, optimize=True) # n_s
        A_ia+= numpy.einsum('xpai,xp,pn->ian', ai_s[:,0], n_s, rho1, optimize=True) # s_n
        
        A_ia+= numpy.einsum('pai,xp,xpn->ian', ai_rho[0], n_Nn, nrho1, optimize=True) # n_Nn
        A_ia+= numpy.einsum('xpai,xp,pn->ian', ai_rho[1:], n_Nn, rho1, optimize=True) # Nn_n
        
        A_ia+= numpy.einsum('pai,yxp,yxpn->ian', ai_rho[0], n_Ns, nM1, optimize=True) # n_Ns
        A_ia+= numpy.einsum('xypai,yxp,pn->ian', ai_s[:,1:], n_Ns, rho1, optimize=True) # Ns_n
        
        A_ia+= numpy.einsum('xpai,xyp,ypn->ian', ai_s[:,0], s_s, M1, optimize=True) # s_s
        
        A_ia+= numpy.einsum('xpai,yxp,ypn->ian', ai_s[:,0], s_Nn, nrho1, optimize=True) # s_Nn
        A_ia+= numpy.einsum('ypai,yxp,xpn->ian', ai_rho[1:], s_Nn, M1, optimize=True) # Nn_s
        
        A_ia+= numpy.einsum('xpai,zyxp,zypn->ian', ai_s[:,0], s_Ns, nM1, optimize=True) # s_Ns
        A_ia+= numpy.einsum('xzpai,zyxp,ypn->ian', ai_s[:,1:], s_Ns, M1, optimize=True) # Ns_s
        
        offset2 = numint_gksmc.get_2d_offset()
        Nn_Nn = numpy.zeros((3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Nn_Nn[i,j] = Nn_Nntmp[offset2[i,j]] 
        A_ia+= numpy.einsum('xpai,yxp,ypn->ian', ai_rho[1:], Nn_Nn, nrho1, optimize=True) # Nn_Nn
        
        A_ia+= numpy.einsum('zpai,zyxp,yxpn->ian', ai_rho[1:], Nn_Ns, nM1, optimize=True) # Nn_Ns
        A_ia+= numpy.einsum('xzpai,zyxp,ypn->ian', ai_s[:,1:], Nn_Ns, nrho1, optimize=True) # Ns_Nn
        
        Ns_Ns = numpy.zeros((3,3,3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Ns_Ns[i,j] = Ns_Nstmp[offset2[i,j]] 
        
        A_ia+= numpy.einsum('wzpai,zyxwp,yxpn->ian', ai_s[:,1:], Ns_Ns, nM1, optimize=True) # Ns_Ns
        
        B_ia = numpy.einsum('pai,p,pn->ian', ai_rho[0], n_n, rho1.conj(), optimize=True).astype(numpy.complex128) # n_n
        
        B_ia+= numpy.einsum('pai,xp,xpn->ian', ai_rho[0], n_s, M1.conj(), optimize=True) # n_s
        B_ia+= numpy.einsum('xpai,xp,pn->ian', ai_s[:,0], n_s, rho1.conj(), optimize=True) # s_n
        
        B_ia+= numpy.einsum('pai,xp,xpn->ian', ai_rho[0], n_Nn, nrho1.conj(), optimize=True) # n_Nn
        B_ia+= numpy.einsum('xpai,xp,pn->ian', ai_rho[1:], n_Nn, rho1.conj(), optimize=True) # Nn_n
        
        B_ia+= numpy.einsum('pai,yxp,yxpn->ian', ai_rho[0], n_Ns, nM1.conj(), optimize=True) # n_Ns
        B_ia+= numpy.einsum('xypai,yxp,pn->ian', ai_s[:,1:], n_Ns, rho1.conj(), optimize=True) # Ns_n
        
        B_ia+= numpy.einsum('xpai,xyp,ypn->ian', ai_s[:,0], s_s, M1.conj(), optimize=True) # s_s
        
        B_ia+= numpy.einsum('xpai,yxp,ypn->ian', ai_s[:,0], s_Nn, nrho1.conj(), optimize=True) # s_Nn
        B_ia+= numpy.einsum('ypai,yxp,xpn->ian', ai_rho[1:], s_Nn, M1.conj(), optimize=True) # Nn_s
        
        B_ia+= numpy.einsum('xpai,zyxp,zypn->ian', ai_s[:,0], s_Ns, nM1.conj(), optimize=True) # s_Ns
        B_ia+= numpy.einsum('xzpai,zyxp,ypn->ian', ai_s[:,1:], s_Ns, M1.conj(), optimize=True) # Ns_s
        
        offset2 = numint_gksmc.get_2d_offset()
        Nn_Nn = numpy.zeros((3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Nn_Nn[i,j] = Nn_Nntmp[offset2[i,j]] 
        B_ia+= numpy.einsum('xpai,yxp,ypn->ian', ai_rho[1:], Nn_Nn, nrho1.conj(), optimize=True) # Nn_Nn
        
        B_ia+= numpy.einsum('zpai,zyxp,yxpn->ian', ai_rho[1:], Nn_Ns, nM1.conj(), optimize=True) # Nn_Ns
        B_ia+= numpy.einsum('xzpai,zyxp,ypn->ian', ai_s[:,1:], Nn_Ns, nrho1.conj(), optimize=True) # Ns_Nn
        
        Ns_Ns = numpy.zeros((3,3,3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Ns_Ns[i,j] = Ns_Nstmp[offset2[i,j]] 
        
        B_ia+= numpy.einsum('wzpai,zyxwp,yxpn->ian', ai_s[:,1:], Ns_Ns, nM1.conj(), optimize=True) # Ns_Ns
        
        # The orbital energy difference is calculated here
        A_ia+= numpy.einsum('ia,ian->ian', e_ia.reshape(nocc,nvir), x0.reshape(nocc,nvir,nstates), optimize=True)    
        
    else:
        raise NotImplementedError("Only LDA is implemented.")    
    
    # import pdb
    # pdb.set_trace()
    
    n2c = C_vir.shape[0]//2
    # The hartree potential term.
    dm1 = numpy.einsum('jbn,vj,ub->uvn', x0.reshape(nocc,nvir,nstates), C_occ.conj(), C_vir, optimize=True)
    eri = numpy.zeros(dm1.shape).astype(numpy.complex128)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] += mf.get_j(mf.mol, dm1[:,:,i], hermi=0)  # 
     
    erimo = numpy.einsum('uvn,ua,vi->ian',eri, C_vir.conj(), C_occ, optimize=True)
    A_ia += erimo
    
    # dm1 = numpy.einsum('jbn,vj,ub->uvn', x0.reshape(nocc,nvir,nstates), C_occ, C_vir.conj(), optimize=True)
    # eri = numpy.zeros(dm1.shape).astype(numpy.complex128)
    # for i in range(dm1.shape[-1]):
    #     eri[:,:,i] += mf.get_j(mf.mol, dm1[:,:,i], hermi=0)  # 
     
    # erimo = numpy.einsum('uvn,ua,vi->ian',eri, C_vir.conj(), C_occ, optimize=True)
    # B_ia += erimo
    
    # print(erimo[:,:,0])
    
    # Approach 2
    # n2c = C_vir.shape[0]//2
    # # C_vir[n2c:] *= (0.5/lib.param.LIGHT_SPEED)
    # # C_occ[n2c:] *= (0.5/lib.param.LIGHT_SPEED)
    
    # eri_LL = mf.mol.intor('int2e_spinor')
    # eri_LS = mf.mol.intor('int2e_spsp1_spinor')*(0.5/lib.param.LIGHT_SPEED)**2
    # eri_SS = mf.mol.intor('int2e_spsp1spsp2_spinor')*(0.5/lib.param.LIGHT_SPEED)**4
    # # # # # # transform the eri to mo space.
    # # n2c = C_occ.shape[0]//2
    
    # eri_LL_ao = numpy.einsum('uvwy,yb,wj,jbn->uvn', eri_LL,
    #                       C_vir[:n2c],C_occ[:n2c].conj(),x0.reshape(nocc,nvir,nstates)
    #                       ,optimize = True)
    # eri_LS_ao = numpy.einsum('uvwy,yb,wj,jbn->uvn', eri_LS,
    #                       C_vir[n2c:],C_occ[n2c:].conj(),x0.reshape(nocc,nvir,nstates)
    #                       ,optimize = True)
    # eri_SL_ao = numpy.einsum('uvwy,yb,wj,jbn->uvn', eri_LS.transpose(2,3,0,1),
    #                       C_vir[:n2c],C_occ[:n2c].conj(),x0.reshape(nocc,nvir,nstates)
    #                       ,optimize = True)
    # eri_SS_ao = numpy.einsum('uvwy,yb,wj,jbn->uvn', eri_SS,
    #                       C_vir[n2c:],C_occ[n2c:].conj(),x0.reshape(nocc,nvir,nstates)
    #                       ,optimize = True)
    
    # A_ia += numpy.einsum('uvn,ua,vi->ian', eri_LL_ao,
    #                       C_vir[:n2c].conj(),C_occ[:n2c],optimize = True)
    
    # A_ia += numpy.einsum('uvn,ua,vi->ian', eri_LS_ao,
    #                       C_vir[:n2c].conj(),C_occ[:n2c],optimize = True)   
    
    # A_ia += numpy.einsum('uvn,ua,vi->ian', eri_SL_ao,
    #                       C_vir[n2c:].conj(),C_occ[n2c:],optimize = True) 
    
    # A_ia += numpy.einsum('uvn,ua,vi->ian', eri_SS_ao,
    #                       C_vir[n2c:].conj(),C_occ[n2c:],optimize = True)   
    
    # The excat exchange is calculated
    omega, alpha, hyb = hyec
    if numpy.abs(hyb) >= 1e-10:
        dm2 = numpy.einsum('jbn,vj,ub->uvn', x0.reshape(nocc,nvir,nstates), C_occ.conj(), C_vir, optimize=True)
        eri = numpy.zeros(dm2.shape).astype(numpy.complex128)
        for i in range(dm2.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm2[:,:,i], hermi=0)
        eri *= hyb
        
        if abs(omega) > 1e-10:
            for i in range(dm2.shape[-1]):
                vklr = mf.get_k(mf.mol, dm2[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
                
        erimo = numpy.einsum('uvn,ua,vi->ain',eri,C_vir.conj(),C_occ, optimize=True)
        A_ai -= erimo

    A_ia = A_ia.reshape(-1,nstates)
    B_ia = B_ia.reshape(-1,nstates)
    TD_ia_ApB = A_ia + B_ia
    return TD_ia_ApB

def non_collinear_AmB_matx_r(e_ia, kernel, x0, xctype, weights, ais, uvs, mf,*args):
    fxc,hyec = kernel 
    mo_vir_L, mo_vir_S, mo_occ_L , mo_occ_S = ais
    nstates = x0.shape[-1]
    # import pdb
    # pdb.set_trace()
    x0 = x0.conj()
    C_vir, C_occ = uvs
    nocc = C_occ.shape[-1]
    nvir = C_vir.shape[-1]
    # import pdb
    # pdb.set_trace()
    if xctype == 'LDA':
        n_n,n_s,s_s = fxc 
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
        ai_s = numpy.array([ai_Mx,ai_My,ai_Mz])
        
        rho1 = numpy.einsum('pbj,jbn->pn', ai_rho, x0.reshape(nocc,nvir,nstates), optimize=True)
        M1 = numpy.einsum('xpbj,jbn->xpn', ai_s, x0.reshape(nocc,nvir,nstates), optimize=True)
        
        # A_ia = 0.0
        A_ia  = numpy.einsum('pai,p,pn->ian', ai_rho.conj(), n_n, rho1, optimize=True) # n_n
        A_ia += numpy.einsum('pai,xp,xpn->ian', ai_rho.conj(), n_s, M1, optimize=True) # n_s
        A_ia += numpy.einsum('xpai,xp,pn->ian', ai_s.conj(), n_s, rho1, optimize=True) # s_n
        A_ia += numpy.einsum('xpai,xyp,ypn->ian', ai_s.conj(), s_s, M1, optimize=True) # s_s
        
        # import pdb
        # pdb.set_trace()
        B_ia  = numpy.einsum('pai,p,pn->ian', ai_rho, n_n, rho1, optimize=True) # n_n
        B_ia += numpy.einsum('pai,xp,xpn->ian', ai_rho, n_s, M1, optimize=True) # n_s
        B_ia += numpy.einsum('xpai,xp,pn->ian', ai_s, n_s, rho1, optimize=True) # s_n
        B_ia += numpy.einsum('xpai,xyp,ypn->ian', ai_s, s_s, M1, optimize=True) # s_s
        # B_ia *= 0.0
        # The orbital energy difference is calculated here
        A_ia += numpy.einsum('ia,ian->ian', e_ia.reshape(nocc,nvir), x0.reshape(nocc,nvir,nstates), optimize=True)
        
    elif xctype == 'GGA':
        n_n, n_s, n_Nn, n_Ns, s_s, s_Nn, s_Ns, Nn_Nntmp, Nn_Ns, Ns_Nstmp = fxc
        
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
        
        ai_s = numpy.array([ai_Mx,ai_My,ai_Mz])
        ngrid = Nn_Nntmp.shape[-1]
        rho1 = numpy.einsum('pbj,jbn->pn', ai_rho[0], x0.reshape(nocc,nvir,nstates), optimize=True)
        M1 = numpy.einsum('xpbj,jbn->xpn', ai_s[:,0], x0.reshape(nocc,nvir,nstates), optimize=True)
        nrho1 = numpy.einsum('xpbj,jbn->xpn', ai_rho[1:], x0.reshape(nocc,nvir,nstates), optimize=True)
        # y means nabla_i, x means m_i
        nM1 = numpy.einsum('xypbj,jbn->yxpn', ai_s[:,1:], x0.reshape(nocc,nvir,nstates), optimize=True)
        
        # import pdb
        # pdb.set_trace()
        A_ia = numpy.einsum('pai,p,pn->ian', ai_rho[0].conj(), n_n, rho1, optimize=True).astype(numpy.complex128) # n_n
        
        A_ia+= numpy.einsum('pai,xp,xpn->ian', ai_rho[0].conj(), n_s, M1, optimize=True) # n_s
        A_ia+= numpy.einsum('xpai,xp,pn->ian', ai_s[:,0].conj(), n_s, rho1, optimize=True) # s_n
        
        A_ia+= numpy.einsum('pai,xp,xpn->ian', ai_rho[0].conj(), n_Nn, nrho1, optimize=True) # n_Nn
        A_ia+= numpy.einsum('xpai,xp,pn->ian', ai_rho[1:].conj(), n_Nn, rho1, optimize=True) # Nn_n
        
        A_ia+= numpy.einsum('pai,yxp,yxpn->ian', ai_rho[0].conj(), n_Ns, nM1, optimize=True) # n_Ns
        A_ia+= numpy.einsum('xypai,yxp,pn->ian', ai_s[:,1:].conj(), n_Ns, rho1, optimize=True) # Ns_n
        
        A_ia+= numpy.einsum('xpai,xyp,ypn->ian', ai_s[:,0].conj(), s_s, M1, optimize=True) # s_s
        
        A_ia+= numpy.einsum('xpai,yxp,ypn->ian', ai_s[:,0].conj(), s_Nn, nrho1, optimize=True) # s_Nn
        A_ia+= numpy.einsum('ypai,yxp,xpn->ian', ai_rho[1:].conj(), s_Nn, M1, optimize=True) # Nn_s
        
        A_ia+= numpy.einsum('xpai,zyxp,zypn->ian', ai_s[:,0].conj(), s_Ns, nM1, optimize=True) # s_Ns
        A_ia+= numpy.einsum('xzpai,zyxp,ypn->ian', ai_s[:,1:].conj(), s_Ns, M1, optimize=True) # Ns_s
        
        offset2 = numint_gksmc.get_2d_offset()
        Nn_Nn = numpy.zeros((3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Nn_Nn[i,j] = Nn_Nntmp[offset2[i,j]] 
        A_ia+= numpy.einsum('xpai,yxp,ypn->ian', ai_rho[1:].conj(), Nn_Nn, nrho1, optimize=True) # Nn_Nn
        
        A_ia+= numpy.einsum('zpai,zyxp,yxpn->ian', ai_rho[1:].conj(), Nn_Ns, nM1, optimize=True) # Nn_Ns
        A_ia+= numpy.einsum('xzpai,zyxp,ypn->ian', ai_s[:,1:].conj(), Nn_Ns, nrho1, optimize=True) # Ns_Nn
        
        Ns_Ns = numpy.zeros((3,3,3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Ns_Ns[i,j] = Ns_Nstmp[offset2[i,j]] 
        
        A_ia+= numpy.einsum('wzpai,zyxwp,yxpn->ian', ai_s[:,1:].conj(), Ns_Ns, nM1, optimize=True) # Ns_Ns
        
        B_ia = numpy.einsum('pai,p,pn->ian', ai_rho[0], n_n, rho1, optimize=True).astype(numpy.complex128) # n_n
        
        B_ia+= numpy.einsum('pai,xp,xpn->ian', ai_rho[0], n_s, M1, optimize=True) # n_s
        B_ia+= numpy.einsum('xpai,xp,pn->ian', ai_s[:,0], n_s, rho1, optimize=True) # s_n
        
        B_ia+= numpy.einsum('pai,xp,xpn->ian', ai_rho[0], n_Nn, nrho1, optimize=True) # n_Nn
        B_ia+= numpy.einsum('xpai,xp,pn->ian', ai_rho[1:], n_Nn, rho1, optimize=True) # Nn_n
        
        B_ia+= numpy.einsum('pai,yxp,yxpn->ian', ai_rho[0], n_Ns, nM1, optimize=True) # n_Ns
        B_ia+= numpy.einsum('xypai,yxp,pn->ian', ai_s[:,1:], n_Ns, rho1, optimize=True) # Ns_n
        
        B_ia+= numpy.einsum('xpai,xyp,ypn->ian', ai_s[:,0], s_s, M1, optimize=True) # s_s
        
        B_ia+= numpy.einsum('xpai,yxp,ypn->ian', ai_s[:,0], s_Nn, nrho1, optimize=True) # s_Nn
        B_ia+= numpy.einsum('ypai,yxp,xpn->ian', ai_rho[1:], s_Nn, M1, optimize=True) # Nn_s
        
        B_ia+= numpy.einsum('xpai,zyxp,zypn->ian', ai_s[:,0], s_Ns, nM1, optimize=True) # s_Ns
        B_ia+= numpy.einsum('xzpai,zyxp,ypn->ian', ai_s[:,1:], s_Ns, M1, optimize=True) # Ns_s
        
        offset2 = numint_gksmc.get_2d_offset()
        Nn_Nn = numpy.zeros((3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Nn_Nn[i,j] = Nn_Nntmp[offset2[i,j]] 
        B_ia+= numpy.einsum('xpai,yxp,ypn->ian', ai_rho[1:], Nn_Nn, nrho1, optimize=True) # Nn_Nn
        
        B_ia+= numpy.einsum('zpai,zyxp,yxpn->ian', ai_rho[1:], Nn_Ns, nM1, optimize=True) # Nn_Ns
        B_ia+= numpy.einsum('xzpai,zyxp,ypn->ian', ai_s[:,1:], Nn_Ns, nrho1, optimize=True) # Ns_Nn
        
        Ns_Ns = numpy.zeros((3,3,3,3,ngrid),dtype=numpy.complex128)
        for i in range(3):
            for j in range(3):
                Ns_Ns[i,j] = Ns_Nstmp[offset2[i,j]] 
        
        B_ia+= numpy.einsum('wzpai,zyxwp,yxpn->ian', ai_s[:,1:], Ns_Ns, nM1, optimize=True) # Ns_Ns
        
        # The orbital energy difference is calculated here
        A_ia+= numpy.einsum('ia,ian->ian', e_ia.reshape(nocc,nvir), x0.reshape(nocc,nvir,nstates), optimize=True)    
        
    else:
        raise NotImplementedError("Only LDA is implemented.")    
    
    # import pdb
    # pdb.set_trace()
    
    
    n2c = C_vir.shape[0]//2
    # The hartree potential term.
    dm1 = numpy.einsum('jbn,vj,ub->vun', x0.reshape(nocc,nvir,nstates), C_occ, C_vir.conj(), optimize=True)
    eri = numpy.zeros(dm1.shape).astype(numpy.complex128)
    for i in range(dm1.shape[-1]):
        eri[:,:,i] += mf.get_j(mf.mol, dm1[:,:,i], hermi=0)  # 
     
    erimo = numpy.einsum('vun,ua,vi->ian',eri, C_vir, C_occ.conj(), optimize=True)
    A_ia += erimo
   
    # dm1 = numpy.einsum('jbn,vj,ub->vun', x0.reshape(nocc,nvir,nstates), C_occ, C_vir.conj(), optimize=True)
    # eri = numpy.zeros(dm1.shape).astype(numpy.complex128)
    # for i in range(dm1.shape[-1]):
    #     eri[:,:,i] += mf.get_j(mf.mol, dm1[:,:,i], hermi=0)  # 
     
    # erimo = numpy.einsum('vun,ua,vi->ian',eri, C_vir.conj(), C_occ, optimize=True)
    # B_ia += erimo
   
    # Approach 2
    # n2c = C_vir.shape[0]//2
    # # C_vir[n2c:] *= (0.5/lib.param.LIGHT_SPEED)
    # # C_occ[n2c:] *= (0.5/lib.param.LIGHT_SPEED)
    
    # eri_LL = mf.mol.intor('int2e_spinor')
    # eri_LS = mf.mol.intor('int2e_spsp1_spinor')*(0.5/lib.param.LIGHT_SPEED)**2
    # eri_SS = mf.mol.intor('int2e_spsp1spsp2_spinor')*(0.5/lib.param.LIGHT_SPEED)**4
    # # # # # # transform the eri to mo space.
    # # n2c = C_occ.shape[0]//2
    
    # eri_LL_ao = numpy.einsum('uvwy,ub,vj,jbn->wyn', eri_LL,
    #                       C_vir[:n2c].conj(),C_occ[:n2c],x0.reshape(nocc,nvir,nstates)
    #                       ,optimize = True)
    # eri_LS_ao = numpy.einsum('uvwy,ub,vj,jbn->wyn', eri_LS,
    #                       C_vir[n2c:].conj(),C_occ[n2c:],x0.reshape(nocc,nvir,nstates)
    #                       ,optimize = True)
    # eri_SL_ao = numpy.einsum('uvwy,ub,vj,jbn->wyn', eri_LS.transpose(2,3,0,1),
    #                       C_vir[:n2c].conj(),C_occ[:n2c],x0.reshape(nocc,nvir,nstates)
    #                       ,optimize = True)
    # eri_SS_ao = numpy.einsum('uvwy,ub,vj,jbn->wyn', eri_SS,
    #                       C_vir[n2c:].conj(),C_occ[n2c:],x0.reshape(nocc,nvir,nstates)
    #                       ,optimize = True)
    
    # A_ia += numpy.einsum('uvn,va,ui->ian', eri_LL_ao,
    #                       C_vir[:n2c],C_occ[:n2c].conj(),optimize = True)
    
    # A_ia += numpy.einsum('uvn,va,ui->ian', eri_LS_ao,
    #                       C_vir[:n2c],C_occ[:n2c].conj(),optimize = True)   
    
    # A_ia += numpy.einsum('uvn,va,ui->ian', eri_SL_ao,
    #                       C_vir[n2c:],C_occ[n2c:].conj(),optimize = True) 
    
    # A_ia += numpy.einsum('uvn,va,ui->ian', eri_SS_ao,
    #                       C_vir[n2c:],C_occ[n2c:].conj(),optimize = True) 
   
    omega, alpha, hyb = hyec
    if numpy.abs(hyb) >= 1e-10:
        dm2 = numpy.einsum('jbn,vj,ub->uvn', x0.reshape(nocc,nvir,nstates), C_occ.conj(), C_vir, optimize=True)
        eri = numpy.zeros(dm2.shape).astype(numpy.complex128)
        for i in range(dm2.shape[-1]):
            eri[:,:,i] = mf.get_k(mf.mol, dm2[:,:,i], hermi=0)
        eri *= hyb
        
        if abs(omega) > 1e-10:
            for i in range(dm2.shape[-1]):
                vklr = mf.get_k(mf.mol, dm2[:,:,i], hermi=0, omega=omega)
                vklr*= (alpha - hyb)
                eri[:,:,i]+= vklr
                
        erimo = numpy.einsum('uvn,ua,vi->ain',eri,C_vir.conj(),C_occ, optimize=True)
        A_ai -= erimo

    A_ia = A_ia.reshape(-1,nstates)
    B_ia = B_ia.reshape(-1,nstates)
    TD_ia_AmB = A_ia - B_ia
    return TD_ia_AmB

def get_Diagelemt_of_O_nc_r(e_ia, kernel,xctype, weights, ais, uvs, mf,*args):
    # After tests, wrong convergence results are obtained. A new try about D matrix
    # is implenmented here without Coloumb and Exchange parts. 
    # (A-B)(A+B) X = w^2 X ---> (AA -BA + AB - BB), where A=A_kernel+orb, B=B_kernel
    
    fxc, hyec = kernel
    mo_vir_L, mo_vir_S, mo_occ_L , mo_occ_S = ais
    C_vir, C_occ = uvs
    nocc = C_occ.shape[-1]
    nvir = C_vir.shape[-1]
    e_ia = e_ia.reshape(nocc,nvir)
    
    # Initial D.
    D_ia = 0.0
    
    if xctype == 'LDA':
        n_n,n_s,s_s = fxc 
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
        ai_s = numpy.array([ai_Mx,ai_My,ai_Mz])
        
        # Calculate AA Part.
        D_ia += numpy.einsum('pai,p,pbj,pbj,p,pai->ia',ai_rho,n_n,ai_rho.conj(), ai_rho,n_n,ai_rho.conj(), optimize=True).astype(numpy.complex128) # n_n
        D_ia += numpy.einsum('pai,p,pbj,pbj,xp,xpai->ia',ai_rho,n_n,ai_rho.conj(), ai_rho,n_s,ai_s.conj(), optimize=True) # n_s
        D_ia += numpy.einsum('pai,p,pbj,xpbj,xp,pai->ia',ai_rho,n_n,ai_rho.conj(), ai_s,n_s,ai_rho.conj(), optimize=True) # s_n
        D_ia += numpy.einsum('pai,p,pbj,xpbj,xyp,ypai->ia',ai_rho,n_n,ai_rho.conj(), ai_s,s_s,ai_s.conj(), optimize=True) # s_s

        D_ia += numpy.einsum('pai,op,opbj,pbj,p,pai->ia',ai_rho,n_s,ai_s.conj(), ai_rho,n_n,ai_rho.conj(), optimize=True) # n_n
        D_ia += numpy.einsum('pai,op,opbj,pbj,xp,xpai->ia',ai_rho,n_s,ai_s.conj(), ai_rho,n_s,ai_s.conj(), optimize=True) # n_s
        D_ia += numpy.einsum('pai,op,opbj,xpbj,xp,pai->ia',ai_rho,n_s,ai_s.conj(), ai_s,n_s,ai_rho.conj(), optimize=True) # s_n
        D_ia += numpy.einsum('pai,op,opbj,xpbj,xyp,ypai->ia',ai_rho,n_s,ai_s.conj(), ai_s,s_s,ai_s.conj(), optimize=True) # s_s
        
        D_ia += numpy.einsum('opai,op,pbj,pbj,p,pai->ia',ai_s,n_s,ai_rho.conj(), ai_rho,n_n,ai_rho.conj(), optimize=True) # n_n
        D_ia += numpy.einsum('opai,op,pbj,pbj,xp,xpai->ia',ai_s,n_s,ai_rho.conj(), ai_rho,n_s,ai_s.conj(), optimize=True) # n_s
        D_ia += numpy.einsum('opai,op,pbj,xpbj,xp,pai->ia',ai_s,n_s,ai_rho.conj(), ai_s,n_s,ai_rho.conj(), optimize=True) # s_n
        D_ia += numpy.einsum('opai,op,pbj,xpbj,xyp,ypai->ia',ai_s,n_s,ai_rho.conj(), ai_s,s_s,ai_s.conj(), optimize=True) # s_s
        
        D_ia += numpy.einsum('opai,oqp,qpbj,pbj,p,pai->ia',ai_s,s_s,ai_s.conj(), ai_rho,n_n,ai_rho.conj(), optimize=True) # n_n
        D_ia += numpy.einsum('opai,oqp,qpbj,pbj,xp,xpai->ia',ai_s,s_s,ai_s.conj(), ai_rho,n_s,ai_s.conj(), optimize=True) # n_s
        D_ia += numpy.einsum('opai,oqp,qpbj,xpbj,xp,pai->ia',ai_s,s_s,ai_s.conj(), ai_s,n_s,ai_rho.conj(), optimize=True) # s_n
        D_ia += numpy.einsum('opai,oqp,qpbj,xpbj,xyp,ypai->ia',ai_s,s_s,ai_s.conj(), ai_s,s_s,ai_s.conj(), optimize=True) # s_s
        
        # add Aorb Part; *2 for orb A Part.
        D_ia += numpy.einsum('pai,p,pai,ia ->ia',ai_rho,n_n,ai_rho.conj(),e_ia, optimize=True)*2.0 # n_n
        D_ia += numpy.einsum('pai,xp,xpai,ia ->ia', ai_rho, n_s, ai_s.conj(),e_ia, optimize=True)*2.0 # n_s
        D_ia += numpy.einsum('xpai,xp,pai,ia ->ia', ai_s, n_s, ai_rho.conj(),e_ia, optimize=True)*2.0 # s_n
        D_ia += numpy.einsum('xpai,xyp,ypai,ia ->ia', ai_s, s_s, ai_s.conj(),e_ia, optimize=True)*2.0 # s_s

        # import pdb
        # pdb.set_trace()
        # add orborb Part.
        D_ia += e_ia*e_ia
    
        # Calculate BA Part.
        D_ia -= numpy.einsum('pai,p,pbj,pbj,p,pai->ia',    ai_rho,n_n,ai_rho, ai_rho,n_n,ai_rho.conj(), optimize=True).astype(numpy.complex128) # n_n
        # D_ia -= numpy.einsum('pai,p,pbj,pbj,xp,xpai->ia',   ai_rho,n_n,ai_rho, ai_rho,n_s,ai_s.conj(), optimize=True) # n_s
        # D_ia -= numpy.einsum('pai,p,pbj,xpbj,xp,pai->ia',   ai_rho,n_n,ai_rho, ai_s,n_s,ai_rho.conj(), optimize=True) # s_n
        # D_ia -= numpy.einsum('pai,p,pbj,xpbj,xyp,ypai->ia',ai_rho,n_n,ai_rho, ai_s,s_s,ai_s.conj(), optimize=True) # s_s

        # D_ia -= numpy.einsum('pai,op,opbj,pbj,p,pai->ia',    ai_rho,n_s,ai_s, ai_rho,n_n,ai_rho.conj(), optimize=True) # n_n
        # D_ia -= numpy.einsum('pai,op,opbj,pbj,xp,xpai->ia',  ai_rho,n_s,ai_s, ai_rho,n_s,ai_s.conj(), optimize=True) # n_s
        # D_ia -= numpy.einsum('pai,op,opbj,xpbj,xp,pai->ia',  ai_rho,n_s,ai_s, ai_s,n_s,ai_rho.conj(), optimize=True) # s_n
        # D_ia -= numpy.einsum('pai,op,opbj,xpbj,xyp,ypai->ia',ai_rho,n_s,ai_s, ai_s,s_s,ai_s.conj(), optimize=True) # s_s
        
        # D_ia -= numpy.einsum('opai,op,pbj,pbj,p,pai->ia',    ai_s,n_s,ai_rho, ai_rho,n_n,ai_rho.conj(), optimize=True) # n_n
        # D_ia -= numpy.einsum('opai,op,pbj,pbj,xp,xpai->ia',  ai_s,n_s,ai_rho, ai_rho,n_s,ai_s.conj(), optimize=True) # n_s
        # D_ia -= numpy.einsum('opai,op,pbj,xpbj,xp,pai->ia',  ai_s,n_s,ai_rho, ai_s,n_s,ai_rho.conj(), optimize=True) # s_n
        # D_ia -= numpy.einsum('opai,op,pbj,xpbj,xyp,ypai->ia',ai_s,n_s,ai_rho, ai_s,s_s,ai_s.conj(), optimize=True) # s_s
        
        # D_ia -= numpy.einsum('opai,oqp,qpbj,pbj,p,pai->ia',    ai_s,s_s,ai_s, ai_rho,n_n,ai_rho.conj(), optimize=True) # n_n
        # D_ia -= numpy.einsum('opai,oqp,qpbj,pbj,xp,xpai->ia',  ai_s,s_s,ai_s, ai_rho,n_s,ai_s.conj(), optimize=True) # n_s
        # D_ia -= numpy.einsum('opai,oqp,qpbj,xpbj,xp,pai->ia',  ai_s,s_s,ai_s, ai_s,n_s,ai_rho.conj(), optimize=True) # s_n
        # D_ia -= numpy.einsum('opai,oqp,qpbj,xpbj,xyp,ypai->ia',ai_s,s_s,ai_s, ai_s,s_s,ai_s.conj(), optimize=True) # s_s
    
        # Calculate AB Part.
        D_ia += numpy.einsum('pai,p,pbj,pbj,p,pai->ia',ai_rho,n_n,ai_rho.conj(), ai_rho,n_n,ai_rho, optimize=True).astype(numpy.complex128) # n_n
        # D_ia += numpy.einsum('pai,p,pbj,pbj,xp,xpai->ia',ai_rho,n_n,ai_rho.conj(),  ai_rho,n_s,ai_s, optimize=True) # n_s
        # D_ia += numpy.einsum('pai,p,pbj,xpbj,xp,pai->ia',ai_rho,n_n,ai_rho.conj(),  ai_s,n_s,ai_rho, optimize=True) # s_n
        # D_ia += numpy.einsum('pai,p,pbj,xpbj,xyp,ypai->ia',ai_rho,n_n,ai_rho.conj(), ai_s,s_s,ai_s, optimize=True) # s_s

        D_ia += numpy.einsum('pai,op,opbj,pbj,p,pai->ia',ai_rho,n_s,ai_s.conj(), ai_rho,n_n,ai_rho, optimize=True) # n_n
        # D_ia += numpy.einsum('pai,op,opbj,pbj,xp,xpai->ia',ai_rho,n_s,ai_s.conj(), ai_rho,n_s,ai_s, optimize=True) # n_s
        # D_ia += numpy.einsum('pai,op,opbj,xpbj,xp,pai->ia',ai_rho,n_s,ai_s.conj(), ai_s,n_s,ai_rho, optimize=True) # s_n
        # D_ia += numpy.einsum('pai,op,opbj,xpbj,xyp,ypai->ia',ai_rho,n_s,ai_s.conj(), ai_s,s_s,ai_s, optimize=True) # s_s
        
        D_ia += numpy.einsum('opai,op,pbj,pbj,p,pai->ia',ai_s,n_s,ai_rho.conj(), ai_rho,n_n,ai_rho, optimize=True) # n_n
        # D_ia += numpy.einsum('opai,op,pbj,pbj,xp,xpai->ia',ai_s,n_s,ai_rho.conj(), ai_rho,n_s,ai_s, optimize=True) # n_s
        # D_ia += numpy.einsum('opai,op,pbj,xpbj,xp,pai->ia',ai_s,n_s,ai_rho.conj(), ai_s,n_s,ai_rho, optimize=True) # s_n
        # D_ia += numpy.einsum('opai,op,pbj,xpbj,xyp,ypai->ia',ai_s,n_s,ai_rho.conj(), ai_s,s_s,ai_s, optimize=True) # s_s
        
        D_ia += numpy.einsum('opai,oqp,qpbj,pbj,p,pai->ia',ai_s,s_s,ai_s.conj(), ai_rho,n_n,ai_rho, optimize=True) # n_n
        # D_ia += numpy.einsum('opai,oqp,qpbj,pbj,xp,xpai->ia',ai_s,s_s,ai_s.conj(), ai_rho,n_s,ai_s, optimize=True) # n_s
        # D_ia += numpy.einsum('opai,oqp,qpbj,xpbj,xp,pai->ia',ai_s,s_s,ai_s.conj(), ai_s,n_s,ai_rho, optimize=True) # s_n
        # D_ia += numpy.einsum('opai,oqp,qpbj,xpbj,xyp,ypai->ia',ai_s,s_s,ai_s.conj(), ai_s,s_s,ai_s, optimize=True) # s_s
        
        # Note: Borb and orbB Parts cancels each other.
        
        # Calculate BB Part.
        D_ia -= numpy.einsum('pai,p,pbj,pbj,p,pai->ia',    ai_rho,n_n,ai_rho, ai_rho,n_n,ai_rho, optimize=True).astype(numpy.complex128) # n_n
        # D_ia -= numpy.einsum('pai,p,pbj,pbj,xp,xpai->ia',   ai_rho,n_n,ai_rho,   ai_rho,n_s,ai_s, optimize=True) # n_s
        # D_ia -= numpy.einsum('pai,p,pbj,xpbj,xp,pai->ia',   ai_rho,n_n,ai_rho,   ai_s,n_s,ai_rho, optimize=True) # s_n
        # D_ia -= numpy.einsum('pai,p,pbj,xpbj,xyp,ypai->ia',ai_rho,n_n,ai_rho,     ai_s,s_s,ai_s, optimize=True) # s_s

        # D_ia -= numpy.einsum('pai,op,opbj,pbj,p,pai->ia',    ai_rho,n_s,ai_s, ai_rho,n_n,ai_rho, optimize=True) # n_n
        # D_ia -= numpy.einsum('pai,op,opbj,pbj,xp,xpai->ia',  ai_rho,n_s,ai_s,   ai_rho,n_s,ai_s, optimize=True) # n_s
        # D_ia -= numpy.einsum('pai,op,opbj,xpbj,xp,pai->ia',  ai_rho,n_s,ai_s,   ai_s,n_s,ai_rho, optimize=True) # s_n
        # D_ia -= numpy.einsum('pai,op,opbj,xpbj,xyp,ypai->ia',ai_rho,n_s,ai_s,    ai_s,s_s,ai_s, optimize=True) # s_s
        
        # D_ia -= numpy.einsum('opai,op,pbj,pbj,p,pai->ia',    ai_s,n_s,ai_rho, ai_rho,n_n,ai_rho, optimize=True) # n_n
        # D_ia -= numpy.einsum('opai,op,pbj,pbj,xp,xpai->ia',  ai_s,n_s,ai_rho,   ai_rho,n_s,ai_s, optimize=True) # n_s
        # D_ia -= numpy.einsum('opai,op,pbj,xpbj,xp,pai->ia',  ai_s,n_s,ai_rho,   ai_s,n_s,ai_rho, optimize=True) # s_n
        # D_ia -= numpy.einsum('opai,op,pbj,xpbj,xyp,ypai->ia',ai_s,n_s,ai_rho,     ai_s,s_s,ai_s, optimize=True) # s_s
        
        # D_ia -= numpy.einsum('opai,oqp,qpbj,pbj,p,pai->ia',    ai_s,s_s,ai_s, ai_rho,n_n,ai_rho, optimize=True) # n_n
        # D_ia -= numpy.einsum('opai,oqp,qpbj,pbj,xp,xpai->ia',  ai_s,s_s,ai_s,   ai_rho,n_s,ai_s, optimize=True) # n_s
        # D_ia -= numpy.einsum('opai,oqp,qpbj,xpbj,xp,pai->ia',  ai_s,s_s,ai_s,   ai_s,n_s,ai_rho, optimize=True) # s_n
        # D_ia -= numpy.einsum('opai,oqp,qpbj,xpbj,xyp,ypai->ia',ai_s,s_s,ai_s,     ai_s,s_s,ai_s, optimize=True) # s_s

        # No orb Part.
        D_ia = D_ia.ravel()
    return D_ia


def get_TDz0(ApBz0,e_ia, kernel,xctype, weights, ais, uvs,mf):
    fxc,hyec = kernel 
    omega, alpha, hyb = hyec
    mo_vir_L, mo_vir_S, mo_occ_L , mo_occ_S = ais
    
    C_vir, C_occ = uvs
    nocc = C_occ.shape[-1]
    nvir = C_vir.shape[-1]
    nstates = ApBz0.shape[-1]
    # import pdb
    # pdb.set_trace()
    ApBz0 = ApBz0.reshape(nocc,nvir,-1)
    
    if xctype == 'LDA':
        n_n,n_s,s_s = fxc 
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
        ai_s = numpy.array([ai_Mx,ai_My,ai_Mz])
        
        # (A-B) (A+B)z0  --->  A(A+B)z0, orb(A+B)z0, -B(A+B)z0
        # Calculate A(A+B)z0.
        # import pdb
        # pdb.set_trace()
        TDz0_A  = numpy.einsum('pai,p,pbj,jbn->ian', ai_rho, n_n, ai_rho.conj(), ApBz0, optimize=True) # n_n
        TDz0_A += numpy.einsum('pai,xp,xpbj,jbn->ian', ai_rho, n_s, ai_s.conj(), ApBz0, optimize=True) # n_s
        TDz0_A += numpy.einsum('xpai,xp,pbj,jbn->ian', ai_s, n_s, ai_rho.conj(), ApBz0, optimize=True) # s_n
        TDz0_A += numpy.einsum('xpai,xyp,ypbj,jbn->ian', ai_s, s_s, ai_s.conj(), ApBz0, optimize=True) # s_s
        
        # TDz0_A=0.0
        # add the orb(A+B)z0 part.
        TDz0_A += numpy.einsum('ia,ian->ian', e_ia.reshape(nocc,nvir), ApBz0, optimize=True).reshape(nocc,nvir,-1)
        
        # add the Coloumb(A+B)z0 part.
        
        dm1 = numpy.einsum('vj,ub->vu',C_occ.conj(), C_vir, optimize=True)
        eri = mf.get_j(mf.mol, dm1, hermi=0)
        TDz0_A += numpy.einsum('vu,ub,vj,jbn->jbn', eri,C_vir.conj(), C_occ, ApBz0, optimize=True) 
        
        # n2c = C_vir.shape[0]//2
        # C_vir[n2c:] *= (0.5/lib.param.LIGHT_SPEED)
        # C_occ[n2c:] *= (0.5/lib.param.LIGHT_SPEED)
        
        # eri_LL = mf.mol.intor('int2e_spinor')
        # eri_LS = mf.mol.intor('int2e_spsp1_spinor')*(0.5/lib.param.LIGHT_SPEED)**2
        # eri_SS = mf.mol.intor('int2e_spsp1spsp2_spinor')*(0.5/lib.param.LIGHT_SPEED)**4
        # # # # # # transform the eri to mo space.
        # # n2c = C_occ.shape[0]//2
        
        # eri_LL_ao = numpy.einsum('uvwy,yb,wj,jbn->uvn', eri_LL,
        #                       C_vir[:n2c],C_occ[:n2c].conj(),ApBz0
        #                       ,optimize = True)
        # eri_LS_ao = numpy.einsum('uvwy,yb,wj,jbn->uvn', eri_LS,
        #                       C_vir[n2c:],C_occ[n2c:].conj(),ApBz0
        #                       ,optimize = True)
        # eri_SL_ao = numpy.einsum('uvwy,yb,wj,jbn->uvn', eri_LS.transpose(2,3,0,1),
        #                       C_vir[:n2c],C_occ[:n2c].conj(),ApBz0
        #                       ,optimize = True)
        # eri_SS_ao = numpy.einsum('uvwy,yb,wj,jbn->uvn', eri_SS,
        #                       C_vir[n2c:],C_occ[n2c:].conj(),ApBz0
        #                       ,optimize = True)
        
        # TDz0_A += numpy.einsum('uvn,ua,vi->ian', eri_LL_ao,
        #                       C_vir[:n2c].conj(),C_occ[:n2c],optimize = True)
        
        # TDz0_A += numpy.einsum('uvn,ua,vi->ian', eri_LS_ao,
        #                       C_vir[:n2c].conj(),C_occ[:n2c],optimize = True)   
        
        # TDz0_A += numpy.einsum('uvn,ua,vi->ian', eri_SL_ao,
        #                       C_vir[n2c:].conj(),C_occ[n2c:],optimize = True) 
        
        # TDz0_A += numpy.einsum('uvn,ua,vi->ian', eri_SS_ao,
        #                       C_vir[n2c:].conj(),C_occ[n2c:],optimize = True) 
        
        # if numpy.abs(hyb) >= 1e-10:
        #     dm2 = numpy.einsum('jbn,vj,ub->uvn', x0.reshape(nocc,nvir,nstates), C_occ.conj(), C_vir, optimize=True)
        #     eri = numpy.zeros(dm2.shape).astype(numpy.complex128)
        #     for i in range(dm2.shape[-1]):
        #         eri[:,:,i] = mf.get_k(mf.mol, dm2[:,:,i], hermi=0)
        #     eri *= hyb
            
        #     if abs(omega) > 1e-10:
        #         for i in range(dm2.shape[-1]):
        #             vklr = mf.get_k(mf.mol, dm2[:,:,i], hermi=0, omega=omega)
        #             vklr*= (alpha - hyb)
        #             eri[:,:,i]+= vklr
                    
        #     erimo = numpy.einsum('uvn,ua,vi->ain',eri,C_vir.conj(),C_occ, optimize=True)
        #     A_ai -= erimo
        
        # Calculate -B(A+B)z0.
        TDz0_B = numpy.einsum('pai,p,pbj,jbn->ian', ai_rho, n_n, ai_rho, ApBz0, optimize=True) # n_n
        TDz0_B += numpy.einsum('pai,xp,xpbj,jbn->ian', ai_rho, n_s, ai_s, ApBz0, optimize=True) # n_s
        TDz0_B += numpy.einsum('xpai,xp,pbj,jbn->ian', ai_s, n_s, ai_rho, ApBz0, optimize=True) # s_n
        TDz0_B += numpy.einsum('xpai,xyp,ypbj,jbn->ian', ai_s, s_s, ai_s, ApBz0, optimize=True) # s_s
        
        # add the Coloumb(A+B)z0 part.
        # dm2 = numpy.einsum('vj,ub->vu',C_occ, C_vir.conj(), optimize=True)
        # eri = mf.get_j(mf.mol, dm2, hermi=0)  # 
        # TDz0_B += numpy.einsum('vu,ub,vj,jbn->jbn', eri,C_vir.conj(), C_occ, ApBz0, optimize=True) 
       
        TDz0_A = TDz0_A.reshape(-1,nstates)
        TDz0_B = TDz0_B.reshape(-1,nstates)
       
        return TDz0_A-TDz0_B
    
        # Approach 2
        # n2c = C_vir.shape[0]//2
        # eri_LL = mf.mol.intor('int2e_spinor')*hyb
        # eri_LS = mf.mol.intor('int2e_spsp1_spinor')*(0.5/lib.param.LIGHT_SPEED)**2*hyb
        # eri_SS = mf.mol.intor('int2e_spsp1spsp2_spinor')*(0.5/lib.param.LIGHT_SPEED)**4*hyb
        # # transform the eri to mo space.
        # if abs(omega) >= 1e-10:
        #     with mf.mol.with_range_coulomb(omega=omega):
        #         eri_LL += eri_LL*alpha/hyb
        #         eri_LS += eri_LS*alpha/hyb
        #         eri_SS += eri_SS*alpha/hyb
        
        # eri_LL_ao = numpy.einsum('uvwy,vb,wj,jbn->uyn', eri_LL,
        #                       C_vir[:n2c],C_occ[:n2c].conj(),x0.reshape(nocc,nvir,nstates)
        #                       ,optimize = True)
        # eri_LS_ao = numpy.einsum('uvwy,vb,wj,jbn->uyn', eri_LS,
        #                       C_vir[n2c:],C_occ[n2c:].conj(),x0.reshape(nocc,nvir,nstates)
        #                       ,optimize = True)
        # eri_SL_ao = numpy.einsum('uvwy,vb,wj,jbn->uyn', eri_LS.transpose(2,3,0,1),
        #                       C_vir[:n2c],C_occ[:n2c].conj(),x0.reshape(nocc,nvir,nstates)
        #                       ,optimize = True)
        # eri_SS_ao = numpy.einsum('uvwy,vb,wj,jbn->uyn', eri_SS,
        #                       C_vir[n2c:],C_occ[n2c:].conj(),x0.reshape(nocc,nvir,nstates)
        #                       ,optimize = True)
        
        # TD_ia_top -= numpy.einsum('uyn,ua,yi->ian', eri_LL_ao,
        #                       C_vir[:n2c].conj(),C_occ[:n2c],optimize = True)
        
        # TD_ia_top -= numpy.einsum('uyn,ua,yi->ian', eri_LS_ao,
        #                       C_vir[:n2c].conj(),C_occ[:n2c],optimize = True)   
        
        # TD_ia_top -= numpy.einsum('uyn,ua,yi->ian', eri_SL_ao,
        #                       C_vir[n2c:].conj(),C_occ[n2c:],optimize = True) 
        
        # TD_ia_top -= numpy.einsum('uyn,ua,yi->ian', eri_SS_ao,
        #                       C_vir[n2c:].conj(),C_occ[n2c:],optimize = True)
        
        # eri_LL_ao = numpy.einsum('uvwy,wb,vj,jbn->uyn', eri_LL,
        #                       C_vir[:n2c].conj(),C_occ[:n2c],y0.reshape(nocc,nvir,nstates)
        #                       ,optimize = True)
        # eri_LS_ao = numpy.einsum('uvwy,wb,vj,jbn->uyn', eri_LS,
        #                       C_vir[n2c:].conj(),C_occ[n2c:],y0.reshape(nocc,nvir,nstates)
        #                       ,optimize = True)
        # eri_SL_ao = numpy.einsum('uvwy,wb,vj,jbn->uyn', eri_LS.transpose(2,3,0,1),
        #                       C_vir[:n2c].conj(),C_occ[:n2c],y0.reshape(nocc,nvir,nstates)
        #                       ,optimize = True)
        # eri_SS_ao = numpy.einsum('uvwy,wb,vj,jbn->uyn', eri_SS,
        #                       C_vir[n2c:].conj(),C_occ[n2c:],y0.reshape(nocc,nvir,nstates)
        #                       ,optimize = True)
        
        # TD_ia_top -= numpy.einsum('uyn,ua,yi->ian', eri_LL_ao,
        #                       C_vir[:n2c].conj(),C_occ[:n2c],optimize = True)
        
        # TD_ia_top -= numpy.einsum('uyn,ua,yi->ian', eri_LS_ao,
        #                       C_vir[:n2c].conj(),C_occ[:n2c],optimize = True)   
        
        # TD_ia_top -= numpy.einsum('uyn,ua,yi->ian', eri_SL_ao,
        #                       C_vir[n2c:].conj(),C_occ[n2c:],optimize = True) 
        
        # TD_ia_top -= numpy.einsum('uyn,ua,yi->ian', eri_SS_ao,
        #                       C_vir[n2c:].conj(),C_occ[n2c:],optimize = True)
        
        # eri_LL_ao = numpy.einsum('uvwy,wb,vj,jbn->uyn', eri_LL,
        #                       C_vir[:n2c].conj(),C_occ[:n2c],x0.conj().reshape(nocc,nvir,nstates)
        #                       ,optimize = True)
        # eri_LS_ao = numpy.einsum('uvwy,wb,vj,jbn->uyn', eri_LS,
        #                       C_vir[n2c:].conj(),C_occ[n2c:],x0.conj().reshape(nocc,nvir,nstates)
        #                       ,optimize = True)
        # eri_SL_ao = numpy.einsum('uvwy,wb,vj,jbn->uyn', eri_LS.transpose(2,3,0,1),
        #                       C_vir[:n2c].conj(),C_occ[:n2c],x0.conj().reshape(nocc,nvir,nstates)
        #                       ,optimize = True)
        # eri_SS_ao = numpy.einsum('uvwy,wb,vj,jbn->uyn', eri_SS,
        #                       C_vir[n2c:].conj(),C_occ[n2c:],x0.conj().reshape(nocc,nvir,nstates)
        #                       ,optimize = True)
        
        # TD_ia_bom += numpy.einsum('uyn,ua,yi->ian', eri_LL_ao,
        #                       C_vir[:n2c].conj(),C_occ[:n2c],optimize = True)
        
        # TD_ia_bom += numpy.einsum('uyn,ua,yi->ian', eri_LS_ao,
        #                       C_vir[:n2c].conj(),C_occ[:n2c],optimize = True)   
        
        # TD_ia_bom += numpy.einsum('uyn,ua,yi->ian', eri_SL_ao,
        #                       C_vir[n2c:].conj(),C_occ[n2c:],optimize = True) 
        
        # TD_ia_bom += numpy.einsum('uyn,ua,yi->ian', eri_SS_ao,
        #                       C_vir[n2c:].conj(),C_occ[n2c:],optimize = True)
        
        # eri_LL_ao = numpy.einsum('uvwy,vb,wj,jbn->uyn', eri_LL,
        #                       C_vir[:n2c],C_occ[:n2c].conj(),y0.conj().reshape(nocc,nvir,nstates)
        #                       ,optimize = True)
        # eri_LS_ao = numpy.einsum('uvwy,vb,wj,jbn->uyn', eri_LS,
        #                       C_vir[n2c:],C_occ[n2c:].conj(),y0.conj().reshape(nocc,nvir,nstates)
        #                       ,optimize = True)
        # eri_SL_ao = numpy.einsum('uvwy,vb,wj,jbn->uyn', eri_LS.transpose(2,3,0,1),
        #                       C_vir[:n2c],C_occ[:n2c].conj(),y0.conj().reshape(nocc,nvir,nstates)
        #                       ,optimize = True)
        # eri_SS_ao = numpy.einsum('uvwy,vb,wj,jbn->uyn', eri_SS,
        #                       C_vir[n2c:],C_occ[n2c:].conj(),y0.conj().reshape(nocc,nvir,nstates)
        #                       ,optimize = True)
        
        # TD_ia_bom += numpy.einsum('uyn,ua,yi->ian', eri_LL_ao,
        #                       C_vir[:n2c].conj(),C_occ[:n2c],optimize = True)
        
        # TD_ia_bom += numpy.einsum('uyn,ua,yi->ian', eri_LS_ao,
        #                       C_vir[:n2c].conj(),C_occ[:n2c],optimize = True)   
        
        # TD_ia_bom += numpy.einsum('uyn,ua,yi->ian', eri_SL_ao,
        #                       C_vir[n2c:].conj(),C_occ[n2c:],optimize = True) 
        
        # TD_ia_bom += numpy.einsum('uyn,ua,yi->ian', eri_SS_ao,
        #                       C_vir[n2c:].conj(),C_occ[n2c:],optimize = True)
        