from numba import njit
from numba.types import float64,int64,int32,complex128
import numpy as np
import fast_scalar as fsc
import fast_spin as fsp

@njit(parallel = True)
def solve_on_grid_spin1D(dim_k,per,orb,norb,nsta,site_energies,hst,hind,hR,\
    mesh_arr,wfs,all_gaps,start_k):
    for i in range(mesh_arr[0]-1):
        # generate a kpoint
        kpt=[start_k[0]+float(i)/float(mesh_arr[0]-1)]
        # solve at that point
        (eval,evec)=fsp.solve_all(dim_k,per,orb,norb,nsta,site_energies,\
            hst,hind,hR,np.array([kpt]),eig_vectors=True)
        # store wavefunctions
        wfs[i]=evec[:,0,:,:]
        # store gaps
        if all_gaps.size != 0:
            all_gaps[i,:]=eval[1:,0]-eval[:-1,0]
@njit(parallel = False)
def solve_on_grid_spin2D(dim_k,per,orb,norb,nsta,site_energies,hst,hind,hR,\
    mesh_arr,wfs,all_gaps,start_k):
    for i in range(mesh_arr[0]-1):
        for j in range(mesh_arr[1]-1):
            kpt=[start_k[0]+float(i)/float(mesh_arr[0]-1),\
                    start_k[1]+float(j)/float(mesh_arr[1]-1)]
            eval, evec=fsp.solve_all(dim_k,per,orb,norb,nsta,site_energies,\
                hst,hind,hR,np.array([kpt]),eig_vectors=True)
            wfs[i,j]=evec[:,0,:,:]
            if all_gaps.size != 0:
                all_gaps[i,j,:]=eval[1:,0]-eval[:-1,0]
                pass
@njit(parallel = True)
def solve_on_grid_spin3D(dim_k,per,orb,norb,nsta,site_energies,hst,hind,hR,\
    mesh_arr,wfs,all_gaps,start_k):
    for i in range(mesh_arr[0]-1):
        for j in range(mesh_arr[1]-1):
            for k in range(mesh_arr[2]-1):
                kpt=[start_k[0]+float(i)/float(mesh_arr[0]-1),\
                        start_k[1]+float(j)/float(mesh_arr[1]-1),\
                        start_k[2]+float(k)/float(mesh_arr[2]-1)]
                (eval,evec)=fsp.solve_all(dim_k,per,orb,norb,nsta,site_energies,\
                    hst,hind,hR,np.array([kpt]),eig_vectors=True)
                wfs[i,j,k]=evec[:,0,:,:]
                if all_gaps.size != 0:
                    all_gaps[i,j,k,:]=eval[1:,0]-eval[:-1,0]
@njit(parallel = True)
def solve_on_grid_spin4D(dim_k,per,orb,norb,nsta,site_energies,hst,hind,hR,\
    mesh_arr,wfs,all_gaps,start_k):
    for i in range(mesh_arr[0]-1):
        for j in range(mesh_arr[1]-1):
            for k in range(mesh_arr[2]-1):
                for l in range(mesh_arr[3]-1):
                    kpt=[start_k[0]+float(i)/float(mesh_arr[0]-1),\
                                start_k[1]+float(j)/float(mesh_arr[1]-1),\
                                start_k[2]+float(k)/float(mesh_arr[2]-1),\
                                start_k[3]+float(l)/float(mesh_arr[3]-1)]
                    (eval,evec)=fsp.solve_all(dim_k,per,orb,norb,nsta,site_energies,\
                        hst,hind,hR,np.array([kpt]),eig_vectors=True)
                    wfs[i,j,k,l]=evec[:,0,:,:]
                    if all_gaps.size != 0:
                        all_gaps[i,j,k,l,:]=eval[1:]-eval[:-1]


@njit(parallel = True)
def solve_on_grid_scalar1D(dim_k,per,orb,norb,nsta,site_energies,hst,hind,hR,\
    mesh_arr,wfs,all_gaps,start_k):
    for i in range(mesh_arr[0]-1):
        # generate a kpoint
        kpt=[start_k[0]+float(i)/float(mesh_arr[0]-1)]
        # solve at that point
        (eval,evec)=fsc.solve_all(dim_k,per,orb,norb,nsta,site_energies,\
            hst,hind,hR,np.array([kpt]),eig_vectors=True)
        # store wavefunctions
        wfs[i]=evec[:,0,:]
        # store gaps
        if all_gaps.size != 0:
            all_gaps[i,:]=eval[1:,0]-eval[:-1,0]
@njit(parallel = True)
def solve_on_grid_scalar2D(dim_k,per,orb,norb,nsta,site_energies,hst,hind,hR,\
    mesh_arr,wfs,all_gaps,start_k):
    for i in range(mesh_arr[0]-1):
        for j in range(mesh_arr[1]-1):
            kpt=[start_k[0]+float(i)/float(mesh_arr[0]-1),\
                    start_k[1]+float(j)/float(mesh_arr[1]-1)]
            (eval,evec)=fsc.solve_all(dim_k,per,orb,norb,nsta,site_energies,\
                hst,hind,hR,np.array([kpt]),eig_vectors=True)
            wfs[i,j]=evec[:,0,:]
            if all_gaps.size != 0:
                all_gaps[i,j,:]=eval[1:,0]-eval[:-1,0]
@njit(parallel = True)
def solve_on_grid_scalar3D(dim_k,per,orb,norb,nsta,site_energies,hst,hind,hR,\
    mesh_arr,wfs,all_gaps,start_k):
    for i in range(mesh_arr[0]-1):
        for j in range(mesh_arr[1]-1):
            for k in range(mesh_arr[2]-1):
                kpt=[start_k[0]+float(i)/float(mesh_arr[0]-1),\
                        start_k[1]+float(j)/float(mesh_arr[1]-1),\
                        start_k[2]+float(k)/float(mesh_arr[2]-1)]
                (eval,evec)=fsc.solve_all(dim_k,per,orb,norb,nsta,site_energies,\
                    hst,hind,hR,np.array([kpt]),eig_vectors=True)
                wfs[i,j,k]=evec[:,0,:]
                if all_gaps.size != 0:
                    all_gaps[i,j,k,:]=eval[1:,0]-eval[:-1,0]
@njit(parallel = True)
def solve_on_grid_scalar4D(dim_k,per,orb,norb,nsta,site_energies,hst,hind,hR,\
    mesh_arr,wfs,all_gaps,start_k):
    for i in range(mesh_arr[0]-1):
        for j in range(mesh_arr[1]-1):
            for k in range(mesh_arr[2]-1):
                for l in range(mesh_arr[3]-1):
                    kpt=[start_k[0]+float(i)/float(mesh_arr[0]-1),\
                                start_k[1]+float(j)/float(mesh_arr[1]-1),\
                                start_k[2]+float(k)/float(mesh_arr[2]-1),\
                                start_k[3]+float(l)/float(mesh_arr[3]-1)]
                    (eval,evec)=fsc.solve_all(dim_k,per,orb,norb,nsta,site_energies,\
                        hst,hind,hR,np.array([kpt]),eig_vectors=True)
                    wfs[i,j,k,l]=evec[:,0,:]
                    if all_gaps.size != 0:
                        all_gaps[i,j,k,l,:,0]=eval[1:]-eval[:-1,0]

    