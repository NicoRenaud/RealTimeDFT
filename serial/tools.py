import numpy as np 
import scipy.linalg as scla
import sys
import argparse
import os
import re
from tools import *

from PyQuante.LA2 import geigh,mkdens,trace2,mkdens_spinavg,simx,SymOrthCutoff
from PyQuante import SCF, Molecule
from PyQuante.NumWrap import arange
from PyQuante.cints import coulomb_repulsion
from PyQuante import configure_output
from PyQuante.Constants import hartree2ev,bohr2ang
from PyQuante.Ints import getbasis, getJ

#from Ints_MPI import getJ_mpi
from scipy.linalg import expm
from PyQuante.Convergence import DIIS
from PyQuante.DFunctionals import XC,need_gradients
import PyQuante.settings as settings

###########################################
## Get the value sof the XC functionals
###########################################
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



###########################################
## SELF CONSISTANT DFT
###########################################
def dft(mol,bfs,S,Hcore,Ints,gr,xc_func,mu=0,MaxIter=100,eps_SCF=1E-4,_diis_=True):

	##########################################################
	##					Get the system information
	##########################################################
	
	# size
	nbfs = len(bfs)

	# get the nuclear energy
	enuke = mol.get_enuke()
		
	# determine the number of electrons
	# and occupation numbers
	nelec = mol.get_nel()
	nclosed,nopen = mol.get_closedopen()
	nocc = nclosed

	# orthogonalization matrix
	X = SymOrthCutoff(S)


	if nopen != 0:
		print '\t\t ================================================================='
		print '\t\t Warning : using restricted HF with open shell is not recommended'
		print "\t\t Use only if you know what you're doing"
		print '\t\t ================================================================='

	# get a first DM
	#D = np.zeros((nbfs,nbfs))
	L,C = scla.eigh(Hcore,b=S)
	D = mkdens(C,0,nocc)
	
	# initialize the old energy
	eold = 0.

	# initialize the DIIS
	if _diis_:
		avg = DIIS(S)
	
	#print '\t SCF Calculations'
	for iiter in range(MaxIter):


		# set the denstity on the grid
		gr.setdens(D)
		
		# form the J matrix from the 
		# density matrix and  the 2electron integrals
		J = getJ(Ints,D)

		# get the XC Function
		Exc,XC = getXC(gr,nelec,functional=xc_func)

		# form the Fock matrix
		F = Hcore + 2*J + XC + mu

		# if DIIS
		if _diis_:
			F = avg.getF(F,D)

		# orthogonalize the Fock matrix
		Fp = np.dot(X.T,np.dot(F,X))
		
		# diagonalize the Fock matrix
		Lp,Cp = scla.eigh(Fp)
		
		# form the density matrix in the OB
		Dp = mkdens(Cp,0,nocc)

		# pass the eigenvector back to the AO
		C = np.dot(X,Cp)

		# form the density matrix in the AO
		D = mkdens(C,0,nocc)
		
		# compute the total energy
		Ej = 2*trace2(D,J)
		Eone = 2*trace2(D,Hcore)
		e = Eone + Ej + Exc + enuke
		
		print "\t\t Iteration: %d    Energy: %f    EnergyVar: %f"%(iiter,e.real,np.abs((e-eold).real))

		# stop if done
		if (np.abs(e-eold) < eps_SCF) :
			break
		else:
			eold = e

	if iiter < MaxIter:
		print("\t\t SCF for HF has converged in %d iterations, Final energy %1.3f Ha\n" % (iiter,e.real))
		
	else:
		print("\t\t SCF for HF has failed to converge after %d iterations")



	# compute the density matrix in the
	# eigenbasis of F
	P = np.dot(Cp.T,np.dot(Dp,Cp))
	#print D

	return Lp,C,Cp,F,Fp,D,Dp,P,X


#####################################################
#####################################################

#####################################################
##	Field
#####################################################
def compute_field(t,**kwargs):

	# get the argumetns
	field_form = kwargs.get('fform','sin')
	intensity = kwargs.get('fint')
	frequency = kwargs.get('ffreq')

	# continuous sinusoidal i.e. no enveloppe
	if field_form == 'sin':
		E = intensity*np.sin(frequency*t)

	# gaussian enveloppe
	elif field_form == 'gsin':
		t0 = kwargs.get('t0')
		s = kwargs.get('s')
		g = np.exp(-(t-t0)**2/s**2)
		E = g*intensity*np.sin(frequency*t)

	# triangular up and down
	elif field_form == 'linear':
		t0 = kwargs.get('t0')
		if t<t0:
			g = (1-(t0-t)/t0)
		elif t<2*t0:
			g = (1-(t-t0)/t0)
		else:
			g = 0
		E = g*intensity*np.sin(frequency*t)

	# triangular up-flat-dow
	elif field_form == 'linear_flat':
		tstep = 2*np.pi/frequency
		if t<tstep:
			g = t/tstep
		elif t<2*tstep:
			g=1.0
		elif t<3*tstep:
			g = (3.-t/tstep)
		else:
			g = 0
		E = g*intensity*np.sin(frequency*t)

	# if not recognized
	else:
		print 'Field form %s not recognized' %field_form
		sys.exit()
	return E

#####################################################
##	Propagate the field
#####################################################
def propagate_dm(D,F,dt,**kwargs):

	_prop_ = 'padme'
	method = kwargs.get('method')

	if method == 'relax':

		# direct diagonalization
		if _prop_ == 'direct':

			'''
			Warning : That doesn't work so well
			But I don't know why
			'''
			lambda_F,U_F = scla.eig(F)
			
			v = np.exp(1j*dt*lambda_F)
			df = np.diag(simx(D,U_F,'N'))
			prod = np.diag(v*df*np.conj(v))
			Dq = simx(prod,U_F,'T')

		# padde approximation			
		if _prop_ == 'padme':

			U = expm(1j*F*dt)
			D = np.dot(U,np.dot(D,np.conj(U.T)))
			#print D
		
	else:
		print 'Propagation method %s not regognized' %method
		sys.exit()

	return D


#####################################################
##	Compute non SC Fock Matrix
#####################################################
def compute_F(P,Hcore,X,gr,xc_func,Ints,nelec,mu=0):

	# get the matrix of 2J-K for a given fixed P
	# P is here given in the non- orthonormal basis
	J = getJ(Ints,P)

	# get the xc func
	Exc,XC = getXC(gr,nelec,functional=xc_func)

	# the G matrix
	G = 2*J+XC

	# pass the G matrix in the orthonormal basis
	Gp = simx(G,X,'T')
		
	# form the Fock matrix in the orthonomral basis
	F = Hcore + Gp + mu

	return F


#####################################################
##	Compute doverlap in atomic basis
#####################################################
def compute_idrj(bfs):
	nbf = len(bfs)
	mu_at = np.zeros((nbf,nbf))
	# form the matrix of the dm
	# between all the combinations
	# of possible ATOMIC orbitals 
	for i in range(nbf):
		bfi = bfs[i]
		for j in range(nbf):
			bfj = bfs[j]
			mu_at[i,j] = bfi.doverlap(bfj,0) + bfi.doverlap(bfj,1) + bfi.doverlap(bfj,2)

	return mu_at

#####################################################
##	Compute dipole moments in the atomic basis
#####################################################
def compute_dipole_atoms(bfs,fdir):
	nbf = len(bfs)
	mu_at = np.zeros((nbf,nbf))
	# form the matrix of the dm
	# between all the combinations
	# of possible ATOMIC orbitals 
	for i in range(nbf):
		bfi = bfs[i]
		for j in range(nbf):
			bfj = bfs[j]
			if fdir == 'x':
				mu_at[i,j] = bfi.multipole(bfj,1,0,0)
			elif fdir == 'y':
				mu_at[i,j] = bfi.multipole(bfj,0,1,0)
			elif fdir == 'z':
				mu_at[i,j] = bfi.multipole(bfj,0,0,1)
			elif fidr == 'sum':
				mu_at[i,j] = bfi.multipole(bfj,1,0,0)+bfi.multipole(bfj,0,1,0)+bfi.multipole(bfj,0,0,1)
			else:
				print '\t Error : direction %s for the field not recognized\n\t Options are x,y,z,sum\n' %fdir
				sys.exit()
	return mu_at

#####################################################
##	Compute molecular dipole moments
#####################################################
def compute_dipole_orbs(bfs,E,C,mu_at):	
	nbf = len(bfs)
	mu_orb = np.zeros((nbf,nbf),dtype='complex64')
	# form the matrix of the dm
	# between all the combinations
	# of possible ORBITALS orbitals 
	for a in range(nbf-1):
		for b in range(a+1,nbf):
			cmat = np.outer(C[:,a],C[:,b])
			mu_orb[a,b] = 1j/(E[a]-E[b])*np.sum(cmat*mu_at)
			mu_orb[b,a] = -mu_orb[a,b]

	# return
	return mu_orb

#####################################################
##	create a pdb file from the xyz
#####################################################
def create_pdb(pdb_file,xyz_file,units):

	# create the pdb file if it does not exists
	if not os.path.isfile(pdb_file):

		# if it was provided in bohr 
		# change to angstom
		if units == 'bohr':

			# read the file
			f = open(xyz_file,'r')
			data = f.readlines()
			f.close()

			#write a xyz file in angstrom
			name_mol = re.split(r'\.|/',xyz_file)[-2]
			fname = name_mol+'_angs.xyz'
			f = open(fname,'w')
			f.write('%s\n' %data[0])
			for i in range(2,len(data)):
				l = data[i].split()
				if len(l)>0:
					x,y,z = bohr2ang*float(l[1]),bohr2ang*float(l[2]),bohr2ang*float(l[3])
					f.write('%s %f %f %f\n' %(l[0],x,y,z))
			f.close()

			# convert to pdb
			os.system('obabel -ixyz %s -opdb -O %s' %(fname,pdb_file))



