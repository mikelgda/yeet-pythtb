from numba import njit
from numba.np.ufunc import parallel
from numba.types import float64,int64,int32,complex128
import numpy as np

@njit
def _nicefy_eval(eval):
    "Sort eigenvaules and eigenvectors, if given, and convert to real numbers"
    # first take only real parts of the eigenvalues
    eval=eval.real
    # sort energies
    args=eval.argsort()
    eval=eval[args]
    return eval
@njit
def _nicefy_eig(eval,eig):
    "Sort eigenvaules and eigenvectors, if given, and convert to real numbers"
    # first take only real parts of the eigenvalues
    eval=eval.real
    # sort energies
    args=eval.argsort()
    eval=eval[args]
    eig=eig[args]
    return (eval,eig)


#spin implementation
# @njit('complex128[:,:,:,::1](int64,int32[::1],float64[:,::1],int64,complex128[:,:,::1],complex128[:,:,::1],int32[:,::1],int32[:,::1],float64[::1])')
@njit
def gen_ham(dim_k,per,orb,norb,site_energies,hst,hind,hR,k_input):
        """Generate Hamiltonian for a certain k-point,
        K-point is given in reduced coordinates!"""
        kpnt = k_input
        ham=np.zeros((norb,2,norb,2),dtype=complex128)
        # modify diagonal elements
        for i in range(norb):
            ham[i,:,i,:]=site_energies[i]
        # go over all hoppings
        for h in range(hst.shape[0]):
            # get all data for the hopping parameter
            amp=hst[h]
            i = hind[h,0]
            j = hind[h,1]
            # in 0-dim case there is no phase factor
            if dim_k>0:
                ind_R = hR[h]
                # vector from one site to another
                rv = -orb[i,:] + orb[j,:] + ind_R
                # Take only components of vector which are periodic
                rv = rv[per]
                # Calculate the hopping, see details in info/tb/tb.pdf
                phase=np.exp((2.0j) * np.pi * np.dot(kpnt,rv))
                amp = amp * phase
            # add this hopping into a matrix and also its conjugate

            ham[i,:,j,:] += amp
            ham[j,:,i,:] += amp.T.conjugate()
        return ham
#spin implementation
# @njit('Tuple((float64[::1],complex128[:,:,::1]))(complex128[:,:,:,::1],int64,int64,boolean)')
@njit
def sol_ham(ham,norb,nsta,eig_vectors=False):
        """Solves Hamiltonian and returns eigenvectors, eigenvalues"""
        # reshape matrix first
        ham_use=ham.reshape((2*norb,2*norb))
        # check that matrix is hermitian
        if np.real(np.max(ham_use-ham_use.T.conj()))>1.0E-9:
            raise Exception("\n\nHamiltonian matrix is not hermitian?!")
        #solve matrix
        if eig_vectors==False: # only find eigenvalues
            eval=np.linalg.eigvalsh(ham_use)
            # sort eigenvalues and convert to real numbers
            eval=_nicefy_eval(eval)
            return eval,np.zeros((1,1,1),dtype="complex128")
        else: # find eigenvalues and eigenvectors
            (eval,eig)=np.linalg.eigh(ham_use)
            # transpose matrix eig since otherwise it is confusing
            # now eig[i,:] is eigenvector for eval[i]-th eigenvalue
            eig=eig.T
            # sort evectors, eigenvalues and convert to real numbers
            (eval,eig)=_nicefy_eig(eval,eig)
            # reshape eigenvectors if doing a spinfull calculation
            eig=eig.reshape((nsta,norb,2))
            return eval,eig
#spin implementation
# @njit('Tuple((float64[:,::1],complex128[:,:,:,::1]))(int64,int32[::1],float64[:,::1],int64,int64,complex128[:,:,::1],complex128[:,:,::1],int32[:,::1],int32[:,::1],float64[:,::1],boolean)',parallel=True)
@njit(parallel=True)
def solve_all(dim_k,per,orb,norb,nsta,site_energies,hst,hind,hR,k_list,eig_vectors=False):
    nkp=len(k_list) # number of k points
    # first initialize matrices for all return data
    #    indices are [band,kpoint]
    ret_eval = np.zeros((nsta,nkp),dtype="float64")
    #    indices are [band,kpoint,orbital,spin]
    ret_evec = np.zeros((nsta,nkp,norb,2),dtype="complex128")
    # go over all kpoints
    for i in range(k_list.shape[0]):
        # generate Hamiltonian at that point
        ham = gen_ham(dim_k,per,orb,norb,site_energies,hst,hind,hR,k_list[i])
        # solve Hamiltonian
        if eig_vectors == False:
            eval, evec = sol_ham(ham,norb,nsta,eig_vectors=eig_vectors)
            ret_eval[:,i] = eval[:]
        else:
            eval, evec = sol_ham(ham,norb,nsta,eig_vectors=eig_vectors)
            ret_eval[:,i]=eval[:]
            ret_evec[:,i,:,:]=evec[:,:,:]
    # return stuff
    if eig_vectors==False:
        # indices of eval are [band,kpoint]
        return ret_eval, np.zeros((1,1,1,1),dtype="complex128")
    else:
        # indices of eval are [band,kpoint] for evec are [band,kpoint,orbital,(spin)]
        return (ret_eval,ret_evec)
