from PyQuante.DFunctionals import XC,need_gradients
import PyQuante.settings as settings
import numpy as np


def getXC(gr,nel,**kwargs):
    "Form the exchange-correlation matrix"

    functional = kwargs.get('functional',settings.DFTFunctional)
    do_grad_dens = need_gradients[functional]
    do_spin_polarized = kwargs.get('do_spin_polarized',
                                   settings.DFTSpinPolarized)
    
    gr.floor_density()  # Insure that the values of the density don't underflow
    gr.renormalize(nel) # Renormalize to the proper # electrons

    dens = gr.dens()
    weight = gr.weights()
    gamma = gr.get_gamma()
    npts = len(dens)

    if gr.version == 1:
        amdens = np.zeros((2,npts),'d')
        amgamma = np.zeros((3,npts),'d')
        amdens[0,:] = amdens[1,:] = 0.5*dens
        if gamma is not None:
            amgamma[0,:] = amgamma[1,:] = amgamma[2,:] = 0.25*gamma

    elif gr.version == 2:
        amdens = gr.density.T
        amgamma = gr.gamma.T

    fxc,dfxcdna,dfxcdnb,dfxcdgaa,dfxcdgab,dfxcdgbb = XC(amdens,amgamma,**kwargs)
    
    Exc = np.dot(weight,fxc)

    wva = weight*dfxcdna  # Combine w*v in a vector for multiplication by bfs

    # First do the part that doesn't depend upon gamma
    nbf = gr.get_nbf()
    Fxca = np.zeros((nbf,nbf),'d')
    for i in xrange(nbf):
        wva_i = wva*gr.bfgrid[:,i] 
        for j in xrange(nbf):
            Fxca[i,j] = np.dot(wva_i,gr.bfgrid[:,j])

    # Now do the gamma-dependent part.
    # Fxc_a += dot(2 dfxcdgaa*graddensa + dfxcdgab*graddensb,grad(chia*chib))
    # We can do the dot product as
    #  sum_grid sum_xyz A[grid,xyz]*B[grid,xyz]
    # or a 2d trace product
    # Here A contains the dfxcdgaa stuff
    #      B contains the grad(chia*chib)

    # Possible errors: gr.grad() here should be the grad of the b part?
    if do_grad_dens:
        # A,B are dimensioned (npts,3)
        A = transpose(0.5*transpose(gr.grad())*(weight*(2*dfxcdgaa+dfxcdgab)))
        for a in xrange(nbf):
            for b in xrange(a+1):
                B = gr.grad_bf_prod(a,b)
                Fxca[a,b] += sum(ravel(A*B))
                Fxca[b,a] = Fxca[a,b]
    if not do_spin_polarized: 
        return Exc,Fxca

    wvb = weight*dfxcdnb  # Combine w*v in a vector for multiplication by bfs

    # First do the part that doesn't depend upon gamma
    Fxcb = mp.zeros((nbf,nbf),'d')
    for i in xrange(nbf):
        wvb_i = wvb*gr.bfgrid[:,i] 
        for j in xrange(nbf):
            Fxcb[i,j] = np.dot(wvb_i,gr.bfgrid[:,j])

    # Now do the gamma-dependent part.
    # Fxc_b += dot(2 dfxcdgbb*graddensb + dfxcdgab*graddensa,grad(chia*chib))
    # We can do the dot product as
    #  sum_grid sum_xyz A[grid,xyz]*B[grid,xyz]
    # or a 2d trace product
    # Here A contains the dfxcdgaa stuff
    #      B contains the grad(chia*chib)

    # Possible errors: gr.grad() here should be the grad of the b part?
    if do_grad_dens:
        # A,B are dimensioned (npts,3)
        A = np.transpose(0.5*np.transpose(gr.grad())*(weight*(2*dfxcdgbb+dfxcdgab)))
        for a in xrange(nbf):
            for b in xrange(a+1):
                B = gr.grad_bf_prod(a,b)
                Fxcb[a,b] += sum(ravel(A*B))
                Fxcb[b,a] = Fxcb[a,b]

    return Exc,Fxca,Fxcb


if __name__ == '__main__':


    from PyQuante.MG2 import MG2 as MolecularGrid
    from libxc_itrf import *
    rho = np.array([0.15 ,0.2,0.3,0.4,0.5])

    xc_code = 'lda,vwn'
    exc,vxc,fxc,kxc = eval_xc(xc_code, rho, spin=0, relativity=0, deriv=2, verbose=None)


    dens = np.zeros((2,5))
    dens[0,:] = 0.5*rho
    dens[1,:] = 0.5*rho
    gamma = np.zeros((3,5))
    fxc,dfxcdna,dfxcdnb,dfxcdgaa,dfxcdgab,dfxcdgbb = XC(dens,gamma,functional='LDA')
    w = np.linspace(10,2,5)
    for i in range(len(rho)):
        print rho[i],fxc[i]/rho[i]
