from numba import njit
from numba.types import float64,int64,int32,complex128
import numpy as np

#spin implementation
@njit('complex128[:,:,:,:](int64,int32[:],float64[:,:],int64,complex128[:,:,:],complex128[:,:,:],int32[:,:],int32[:,:],float64[:])')
def gen_ham(dim_k,per,orb,norb,site_energies,hst,hind,hR,k_input):
        """Generate Hamiltonian for a certain k-point,
        K-point is given in reduced coordinates!"""
        kpnt = k_input
        norb = orb.shape[0]
        ham=np.zeros((norb,2,norb,2),dtype=complex128)
        # modify diagonal elements
        for i in range(norb):
            ham[i,:,i,:]=site_energies[i]
        # go over all hoppings
        for i in range(hst.shape[0]):
            # get all data for the hopping parameter
            amp=hst[i]
            i = hind[i,0]
            j = hind[i,1]
            # in 0-dim case there is no phase factor
            if dim_k>0:
                ind_R = hR[i]
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
#scalar implementation
@njit('complex128[:,:](int64,int32[:],float64[:,:],int64,float64[:],complex128[:],int32[:,:],int32[:,:],float64[:])')
def gen_ham(dim_k,per,orb,norb,site_energies,hst,hind,hR,k_input):
        """Generate Hamiltonian for a certain k-point,
        K-point is given in reduced coordinates!"""
        kpnt = k_input
        norb = orb.shape[0]
        ham=np.zeros((norb,norb),dtype=complex128)
        # modify diagonal elements
        for i in range(norb):
            ham[i,i]=site_energies[i]
        # go over all hoppings
        for i in range(hst.shape[0]):
            # get all data for the hopping parameter
            amp=hst[i]
            i = hind[i,0]
            j = hind[i,1]
            # in 0-dim case there is no phase factor
            if dim_k>0:
                ind_R = hR[i]
                # vector from one site to another
                rv = -orb[i,:] + orb[j,:] + ind_R
                # Take only components of vector which are periodic
                rv = rv[per]
                # Calculate the hopping, see details in info/tb/tb.pdf
                phase=np.exp((2.0j) * np.pi * np.dot(kpnt,rv))
                amp = amp * phase
            # add this hopping into a matrix and also its conjugate

            ham[i,j] += amp
            ham[j,i] += np.conjugate(amp)
        return ham

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
@njit('Tuple((float64[:],complex128[:,:,:]))(complex128[:,:,:,::1],int64,int64,boolean)')
def sol_ham(ham,norb,nsta,eig_vectors=False):
        """Solves Hamiltonian and returns eigenvectors, eigenvalues"""
        # reshape matrix first
        ham_use = ham.reshape((2*norb,2*norb))
        # check that matrix is hermitian
        if np.real(np.max(ham_use-ham_use.T.conj()))>1.0E-9:
            raise Exception("\n\nHamiltonian matrix is not hermitian?!")
        #solve matrix
        if eig_vectors==False: # only find eigenvalues
            eval=np.linalg.eigvalsh(ham_use)
            # sort eigenvalues and convert to real numbers
            eval=_nicefy_eval(eval)
            return (eval,np.zeros((1,1,1),dtype="complex128"))
        else: # find eigenvalues and eigenvectors
            (eval,eig)=np.linalg.eigh(ham_use)
            # transpose matrix eig since otherwise it is confusing
            # now eig[i,:] is eigenvector for eval[i]-th eigenvalue
            eig=eig.T
            # sort evectors, eigenvalues and convert to real numbers
            (eval,eig)=_nicefy_eig(eval,eig)
            # reshape eigenvectors if doing a spinfull calculation
            eig=eig.reshape((nsta,norb,2))
            return (eval,eig)
#scalar implementation
@njit('Tuple((float64[:],complex128[:,:]))(complex128[:,:],int64,int64,boolean)')
def sol_ham(ham,norb,nsta,eig_vectors=False):
        """Solves Hamiltonian and returns eigenvectors, eigenvalues"""
        # reshape matrix first
        ham_use = ham
        # check that matrix is hermitian
        if np.real(np.max(ham_use-ham_use.T.conj()))>1.0E-9:
            raise Exception("\n\nHamiltonian matrix is not hermitian?!")
        #solve matrix
        if eig_vectors==False: # only find eigenvalues
            eval=np.linalg.eigvalsh(ham_use)
            # sort eigenvalues and convert to real numbers
            eval=_nicefy_eval(eval)
            return (eval,np.zeros((1,1),dtype="complex128"))
        else: # find eigenvalues and eigenvectors
            (eval,eig)=np.linalg.eigh(ham_use)
            # transpose matrix eig since otherwise it is confusing
            # now eig[i,:] is eigenvector for eval[i]-th eigenvalue
            eig=eig.T
            # sort evectors, eigenvalues and convert to real numbers
            (eval,eig)=_nicefy_eig(eval,eig)
            # reshape eigenvectors if doing a spinfull calculation
            return (eval,eig)