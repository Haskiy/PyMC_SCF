#!/usr/bin/env python
'''
Author: Pu Zhichen
Date: 2021-07-02 09:45:09
LastEditTime: 2021-11-29 14:31:03
LastEditors: Pu Zhichen
Description: 
    Some subroutines for ploting the plot.
FilePath: \pyMC\tools\jupyter_plot_utils_r.py

 May the force be with you!
'''

import numpy as np

def read_and_concatenate(n):
    try:
        M = np.load(    r'M_relative_part1.npy')
        N = np.load(    r'NX_relative_part1.npy')
        Bxc1 = np.load(  r'Bxc_d0_part1.npy')
        Bxc2 = np.load(  r'Bxc_d1_part1.npy')
        Bxc3 = np.load(  r'Bxc_d2_part1.npy')
        for i in range(2,n+1):
            M1 = np.load(    r'M_relative_part'+str(i)+'.npy')
            N1 = np.load(    r'NX_relative_part'+str(i)+'.npy')
            Bxc1_1 = np.load( r'Bxc_d0_part'+str(i)+'.npy')
            Bxc2_1 = np.load( r'Bxc_d1_part'+str(i)+'.npy')
            Bxc3_1 = np.load( r'Bxc_d2_part'+str(i)+'.npy')
            M = np.concatenate((M,M1),axis=2)
            N = np.concatenate((N,N1),axis=2)
            Bxc1 = np.concatenate((Bxc1,Bxc1_1),axis=0)
            Bxc2 = np.concatenate((Bxc2,Bxc2_1),axis=0)
            Bxc3 = np.concatenate((Bxc3,Bxc3_1),axis=0)
        if M.shape.__len__() == 3:
            M2 = np.zeros((3,M.shape[-1]))
            M2[0] = M[0,0]
            M2[1] = M[1,0]
            M2[2] = M[2,0]
            M = M2
    except:
        Mx = np.load(    r'Mx_part1.npy')
        My = np.load(    r'My_part1.npy')
        Mz = np.load(    r'Mz_part1.npy')
        NXI = np.load(   r'NXI1.npy')
        NXII = np.load(  r'NXII1.npy')
        NXIII = np.load( r'NXIII1.npy')
        Bxc1 = np.load(  r'Bxc_d0_part1.npy')
        Bxc2 = np.load(  r'Bxc_d1_part1.npy')
        Bxc3 = np.load(  r'Bxc_d2_part1.npy')
        for i in range(2,n+1):
            Mx1 = np.load(    r'Mx_part'+str(i)+'.npy')
            My1 = np.load(    r'My_part'+str(i)+'.npy')
            Mz1 = np.load(    r'Mz_part'+str(i)+'.npy')
            NXI1 = np.load(   r'NXI'+str(i)+'.npy')
            NXII1 = np.load(  r'NXII'+str(i)+'.npy')
            NXIII1 = np.load( r'NXIII'+str(i)+'.npy')
            Bxc1_1 = np.load( r'Bxc_d0_part'+str(i)+'.npy')
            Bxc2_1 = np.load( r'Bxc_d1_part'+str(i)+'.npy')
            Bxc3_1 = np.load( r'Bxc_d2_part'+str(i)+'.npy')
            
            NXI = np.concatenate((NXI,NXI1),axis=1)
            NXII = np.concatenate((NXII,NXII1),axis=1)
            NXIII = np.concatenate((NXIII,NXIII1),axis=1)
            Mx = np.concatenate((Mx,Mx1),axis=1)
            My = np.concatenate((My,My1),axis=1)
            Mz = np.concatenate((Mz,Mz1),axis=1)
            Bxc1 = np.concatenate((Bxc1,Bxc1_1),axis=0)
            Bxc2 = np.concatenate((Bxc2,Bxc2_1),axis=0)
            Bxc3 = np.concatenate((Bxc3,Bxc3_1),axis=0)
        ngrid = Mx.shape[-1]
        M = np.zeros((3,ngrid))
        M[0] = Mx[0]
        M[1] = My[0]
        M[2] = Mz[0]
        N = np.zeros((3, 3, ngrid))
        N[0] = NXI
        N[1] = NXII
        N[2] = NXIII
    return M, N, Bxc1, Bxc2, Bxc3

def read_and_concatenate_MD(n):
    
    try:
        M = np.load(    r'M_relative_part1.npy')
        Bxc1 = np.load(  r'Bxc_tot_part1.npy')
        for i in range(2,n+1):
            M1 = np.load(    r'M_relative_part'+str(i)+'.npy')
            Bxc1_1 = np.load( r'Bxc_tot_part'+str(i)+'.npy')
            M = np.concatenate((M,M1),axis=2)
            Bxc1 = np.concatenate((Bxc1,Bxc1_1),axis=1)
        if M.shape.__len__() == 3:
            M2 = np.zeros((3,M.shape[-1]))
            M2[0] = M[0,0]
            M2[1] = M[1,0]
            M2[2] = M[2,0]
            M = M2
    except:
        Mx = np.load(    r'Mx_part1.npy')
        My = np.load(    r'My_part1.npy')
        Mz = np.load(    r'Mz_part1.npy')
        Bxc1 = np.load(  r'Bxc_tot_part1.npy')
        for i in range(2,n+1):
            Mx1 = np.load(    r'Mx_part'+str(i)+'.npy')
            My1 = np.load(    r'My_part'+str(i)+'.npy')
            Mz1 = np.load(    r'Mz_part'+str(i)+'.npy')
            Bxc1_1 = np.load( r'Bxc_tot_part'+str(i)+'.npy')
            Mx = np.concatenate((Mx,Mx1),axis=1)
            My = np.concatenate((My,My1),axis=1)
            Mz = np.concatenate((Mz,Mz1),axis=1)
            Bxc1 = np.concatenate((Bxc1,Bxc1_1),axis=1)
        ngrid = Mx.shape[-1]
        M = np.zeros((3,ngrid))
        M[0] = Mx[0]
        M[1] = My[0]
        M[2] = Mz[0]
               
    
    return M, Bxc1

def calculate_toque(NX, Bxc1, Bxc2, Bxc3, M):
    ngrid = NX.shape[-1]
    Bxc = np.zeros((3,ngrid))
    toque = np.zeros((3,ngrid))
    Bxc[0,:] = Bxc1*NX[0,0,:] + Bxc2*NX[1,0,:] + Bxc3*NX[2,0,:]
    Bxc[1,:] = Bxc1*NX[0,1,:] + Bxc2*NX[1,1,:] + Bxc3*NX[2,1,:]
    Bxc[2,:] = Bxc1*NX[0,2,:] + Bxc2*NX[1,2,:] + Bxc3*NX[2,2,:]
    for i in range(ngrid):
        toque[:,i] = np.cross(M[:,i],Bxc[:,i])
    Mnorm = np.zeros((ngrid))
    Mnorm = np.sqrt(M[0]**2 + M[1]**2 + M[2]**2)
    return Bxc, toque, Mnorm

def calculate_toque_MD(Bxc, M):
    ngrid = Bxc.shape[-1]
    toque = np.zeros((3,ngrid))
    
    for i in range(ngrid):
        toque[:,i] = np.cross(M[:,i],Bxc[:,i])
    Mnorm = np.zeros((ngrid))
    Mnorm = np.sqrt(M[0]**2 + M[1]**2 + M[2]**2)
    return toque, Mnorm

def get_m_times_r(coords, M, weights):
    
    m_times_r = np.cross(M.T, coords)
    m_times_r_tot = (m_times_r.T*weights).sum(axis=1)
    
    return m_times_r_tot

def distance(n,M0,M,nnorm):
    ngrid = M.shape[-1]
    d = np.zeros((ngrid))
    Mvec = np.zeros((3,ngrid))
    Mvec[0,:] = M[0,:] - M0[0]
    Mvec[1,:] = M[1,:] - M0[1]
    Mvec[2,:] = M[2,:] - M0[2]
    d[:] = Mvec[0,:]*n[0] + Mvec[1,:]*n[1] + Mvec[2,:]*n[2]
    return np.abs(d)/nnorm

def get_plane_slices(coords, M, Bxc, toque, weights, atoms, THRESHOLD = 0.1):
    n = np.cross(atoms[1]-atoms[0], atoms[2]-atoms[0])
    nnorm = np.sqrt(n[0]**2 + n[1]**2 + n[2]**2)
    d = distance(n,atoms[0],coords.T,nnorm)
    idx_plane = np.where(d<=THRESHOLD)[0]
    z_plane = coords[idx_plane,2]
    y_plane = coords[idx_plane,1]
    x_plane = coords[idx_plane,0]
    u_plane = M[0,idx_plane]
    v_plane = M[1,idx_plane]
    w_plane = M[2,idx_plane]
    Bxc_plane = Bxc[:,idx_plane]
    toque_plane = toque[:,idx_plane] 
    weights_plane = weights[idx_plane]
    return x_plane, y_plane, z_plane, u_plane, v_plane, \
            w_plane, Bxc_plane, toque_plane, weights_plane
            

def get_arbitary_plane_slices(coords, M, Bxc, toque, weights, atoms, slices = [-0.1, 0.1]):
    # n = np.cross(atoms[1]-atoms[0], atoms[2]-atoms[0])
    # nnorm = np.sqrt(n[0]**2 + n[1]**2 + n[2]**2)
    # d = distance(n,atoms[0],coords.T,nnorm)
    idx_plane_1 = np.where(coords[:,2]<slices[1])[0]
    idx_plane_2 = np.where(coords[:,2]>=slices[0])[0]
    idx_plane = np.array(list(set(idx_plane_1) & set(idx_plane_2)))
    # import pdb
    # pdb.set_trace()
    z_plane = coords[idx_plane,2]
    y_plane = coords[idx_plane,1]
    x_plane = coords[idx_plane,0]
    u_plane = M[0,idx_plane]
    v_plane = M[1,idx_plane]
    w_plane = M[2,idx_plane]
    Bxc_plane = Bxc[:,idx_plane]
    toque_plane = toque[:,idx_plane] 
    weights_plane = weights[idx_plane]
    return x_plane, y_plane, z_plane, u_plane, v_plane, \
            w_plane, Bxc_plane, toque_plane, weights_plane
            
def get_arbitary_plane_slices_2(coords, M, Bxc, toque, weights, atoms, slices = [-0.1, 0.1]):
    # n = np.cross(atoms[1]-atoms[0], atoms[2]-atoms[0])
    # nnorm = np.sqrt(n[0]**2 + n[1]**2 + n[2]**2)
    # d = distance(n,atoms[0],coords.T,nnorm)
    idx_plane_1 = np.where(coords[:,2]<slices[1])[0]
    idx_plane_2 = np.where(coords[:,2]>=slices[0])[0]
    idx_plane = np.array(list(set(idx_plane_1) & set(idx_plane_2)))
    

    z_plane = coords[idx_plane,2]
    y_plane = coords[idx_plane,1]
    x_plane = coords[idx_plane,0]
    u_plane = M[0,idx_plane]
    v_plane = M[1,idx_plane]
    w_plane = M[2,idx_plane]
    Bxc_plane = Bxc[:,idx_plane]
    toque_plane = toque[:,idx_plane] 
    weights_plane = weights[idx_plane]
    return x_plane, y_plane, z_plane, u_plane, v_plane, \
            w_plane, Bxc_plane, toque_plane, weights_plane
            
def get_arbitary_plane_slices_layer(coords, M, Bxc, toque, weights, atoms, Nslices = 2):
    
    coords_set = {(coords[i,0], coords[i,1]) 
               for i in range(coords.shape[0])}
    ngrid = len(coords_set)
    z_plane = np.zeros((ngrid))
    y_plane = np.zeros((ngrid))
    x_plane = np.zeros((ngrid))
    u_plane = np.zeros((ngrid))
    v_plane = np.zeros((ngrid))
    w_plane = np.zeros((ngrid))
    Bxc_plane = np.zeros((3, ngrid))
    toque_plane = np.zeros((3, ngrid))
    weights_plane = np.zeros((ngrid))
    
    for igrid in range(ngrid):
        print(igrid,ngrid)
        x_idx = np.where(coords[:,0] == np.array(list(coords_set)[igrid])[0])[0]
        y_idx = np.where(coords[:,1] == np.array(list(coords_set)[igrid])[1])[0]
        idx_plane = np.array(list(set(x_idx) & set(y_idx)))
        
    
    return x_plane, y_plane, z_plane, u_plane, v_plane, \
            w_plane, Bxc_plane, toque_plane, weights_plane

def toque_deviation(toque, weight):
    toque_norm_xy = np.linalg.norm(toque[:2],axis=0)
    toque_norm_z = np.abs(toque[2])
    toque_tot_xy = (toque_norm_xy*weight).sum()
    toque_tot_z = (toque_norm_z*weight).sum()
    print("toque_norm xy sum {0:12.8f}\ntoque_norm z sum {1:12.8f}\n\
        xy/z  {2:12.8f}".format(toque_tot_xy,toque_tot_z\
            ,toque_tot_xy/toque_tot_z))
    return toque_tot_xy,toque_tot_z,toque_tot_xy/toque_tot_z
            
def prt_cutoff_info_all(weights, idx, M, Bxc, toque):
    M_tot = (np.linalg.norm(M,axis = 0)*weights).sum()
    Bxc_tot = (np.linalg.norm(Bxc,axis = 0)*weights).sum()
    toque_tot = (np.linalg.norm(toque,axis = 0)*weights).sum()
    M_cutoff = (np.linalg.norm(M[:,idx],axis = 0)*weights[idx]).sum()
    Bxc_cutoff = (np.linalg.norm(Bxc[:,idx],axis = 0)*weights[idx]).sum()
    toque_cutoff = (np.linalg.norm(toque[:,idx],axis = 0)*weights[idx]).sum()
    
    print("M have part {0:12.8f}\n Bxc have part {1:12.8f}\n \
        Toque have part {2:12.8f}".format(M_cutoff/M_tot, Bxc_cutoff/Bxc_tot, toque_cutoff/toque_tot))

def prt_cutoff_info_plane(weights, idx, M, Bxc, toque):
    M = np.array(M)
    M_tot = (np.linalg.norm(M,axis = 0)*weights).sum()
    Bxc_tot = (np.linalg.norm(Bxc,axis = 0)*weights).sum()
    toque_tot = (np.linalg.norm(toque,axis = 0)*weights).sum()
    M_cutoff = (np.linalg.norm(M[:,idx],axis = 0)*weights[idx]).sum()
    Bxc_cutoff = (np.linalg.norm(Bxc[:,idx],axis = 0)*weights[idx]).sum()
    toque_cutoff = (np.linalg.norm(toque[:,idx],axis = 0)*weights[idx]).sum()
    
    print("M have part {0:12.8f}\n Bxc have part {1:12.8f}\n \
        Toque have part {2:12.8f}".format(M_cutoff/M_tot, Bxc_cutoff/Bxc_tot, toque_cutoff/toque_tot))  
    
def prt_info_tot(atoms, M, weights, coords):
    natom = atoms.shape[0]
    m_times_r_tot = get_m_times_r(coords, M, weights)
    M_tot = (M*weights).sum(axis = 1)
    grids_partition = partition_atom(atoms, coords)
    M_atom = np.zeros((natom,3))
    for iatm in range(natom):
        M_atom[iatm,0] = (M[0, grids_partition[iatm]]*weights[grids_partition[iatm]]).sum()
        M_atom[iatm,1] = (M[1, grids_partition[iatm]]*weights[grids_partition[iatm]]).sum()
        M_atom[iatm,2] = (M[2, grids_partition[iatm]]*weights[grids_partition[iatm]]).sum()
    
    print("M on each atom.")
    print(M_atom)
    print("m_times_r_tot", m_times_r_tot)
    print("M_tot", M_tot)
    return M_tot, m_times_r_tot, M_atom
    

def remove_near_atom_grids_all(coords, atoms, THRESHOLD = 1.0e-2):
    idx = []
    
    d1 = np.sqrt((coords[:,0] - atoms[0,0])**2+(coords[:,1] - atoms[0,1])**2\
            +(coords[:,2] - atoms[0,2])**2)
    d2 = np.sqrt((coords[:,0] - atoms[1,0])**2+(coords[:,1] - atoms[1,1])**2\
            +(coords[:,2] - atoms[1,2])**2)
    d3 = np.sqrt((coords[:,0] - atoms[2,0])**2+(coords[:,1] - atoms[2,1])**2\
            +(coords[:,2] - atoms[2,2])**2)
    for i in range(d1.shape[-1]):
        if (d1[i] <= THRESHOLD) or (d2[i] <= THRESHOLD) or (d3[i] <= THRESHOLD):
            continue
        idx.append(i)
    idx = np.array(idx)
    return idx
    
def remove_near_atom_grids_plane(coords, atoms, weights_plane, THRESHOLD = 1.0e-2
                                 , THRESHOLD_weight = 1.0e-12):
    x_plane, y_plane, z_plane = coords
    idx = []
    
    d1 = np.sqrt((x_plane - atoms[0,0])**2+(y_plane - atoms[0,1])**2\
        +(z_plane - atoms[0,2])**2)
    d2 = np.sqrt((x_plane - atoms[1,0])**2+(y_plane - atoms[1,1])**2\
            +(z_plane - atoms[1,2])**2)
    d3 = np.sqrt((x_plane - atoms[2,0])**2+(y_plane - atoms[2,1])**2\
            +(z_plane - atoms[2,2])**2)
    for i in range(d1.shape[-1]):
        if ((d1[i] <= THRESHOLD) or (d2[i] <= THRESHOLD) or (d3[i] <= THRESHOLD) 
            or np.abs(weights_plane[i])<=THRESHOLD_weight ):
            continue
        idx.append(i)
    idx = np.array(idx)
    
    return idx


def partition_atom(atoms, coords):
    
    natom = atoms.shape[0]
    ngrid = coords.shape[0]
    grids_partition = {i: [] for i in range(natom)}
    
    for igrid in range(ngrid):
        d = np.sqrt((coords[igrid,0]-atoms[:,0])**2
                   +(coords[igrid,1]-atoms[:,1])**2
                   +(coords[igrid,2]-atoms[:,2])**2)
        grids_partition[np.argmin(d)].append(igrid)
        
    return grids_partition 

def atom_info(atoms, coords, M, weights, threshold=1.0):
    
    natom = atoms.shape[0]
    ngrid = coords.shape[0]
    grids_partition = {i: [] for i in range(natom)}
    
    d1 = np.sqrt((coords[:,0] - atoms[0,0])**2+(coords[:,1] - atoms[0,1])**2\
            +(coords[:,2] - atoms[0,2])**2)
    d2 = np.sqrt((coords[:,0] - atoms[1,0])**2+(coords[:,1] - atoms[1,1])**2\
            +(coords[:,2] - atoms[1,2])**2)
    d3 = np.sqrt((coords[:,0] - atoms[2,0])**2+(coords[:,1] - atoms[2,1])**2\
            +(coords[:,2] - atoms[2,2])**2)
    
    for i in range(ngrid):
        if  d1[i] <= threshold:
            grids_partition[0].append(i)
        if  d2[i] <= threshold:
            grids_partition[1].append(i)
        if  d3[i] <= threshold:
            grids_partition[2].append(i)
    
    M_atom = np.zeros((natom,3))
    for iatm in range(natom):
        M_atom[iatm,0] = (M[0, grids_partition[iatm]]*weights[grids_partition[iatm]]).sum()
        M_atom[iatm,1] = (M[1, grids_partition[iatm]]*weights[grids_partition[iatm]]).sum()
        M_atom[iatm,2] = (M[2, grids_partition[iatm]]*weights[grids_partition[iatm]]).sum()
        
    print("M on each atom.")
    print(M_atom)
    return M_atom
        
        
def half_divide(rho, coords, weights, THRESHOLD = 0.01):
    
    ngrid = coords.shape[0]
    idxup = np.where(coords[:,1]>THRESHOLD)[0]
    idxdown = np.where(coords[:,1]<-1.0*THRESHOLD)[0]
    # import pdb
    # pdb.set_trace()
    idxin = np.array(list(set([i for i in range(ngrid)]) - set(idxup) - set(idxdown)))
    
    rhoup = (rho[idxup]*weights[idxup]).sum()
    rhodown = (rho[idxdown]*weights[idxdown]).sum()
    rhoin = (rho[idxin]*weights[idxin]).sum()*0.5
    rhoup += rhoin
    rhodown += rhoin
    
    return rhoup, rhodown
    