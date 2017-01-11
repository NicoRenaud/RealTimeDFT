import numpy as np 
import scipy.linalg as scla
from scipy.linalg import expm
import sys,os,re
import argparse

from mpi4py import MPI
from PyQuante.LA2 import geigh,mkdens,trace2,mkdens_spinavg,simx,SymOrthCutoff
from PyQuante import SCF, Molecule
from PyQuante.NumWrap import arange
from PyQuante.cints import coulomb_repulsion
from PyQuante import configure_output
from PyQuante.Constants import hartree2ev,bohr2ang
from PyQuante.Convergence import DIIS
import PyQuante.settings as settings

#from PyQuante.DFunctionals import XC,need_gradients
#from PyQuante.Ints import getbasis, getJ

from DFunctionals_mpi import XC_mpi,need_gradients,gather_at_root
from Ints_MPI import getJ_mpi
#from tools import *

###########################################
## Get the value sof the XC functionals
###########################################
def getXC_mpi(gr,nel,rank,comm,_debug_=False,**kwargs):

	'''
	Form the exchange-correlation matrix
	'''

	######################################################################
	## Initialization of the calculations 
	## Master clean the molecular grid
	## then determine the work load for 1D communication
	## then we broadcast the send_cound and disp to all
	######################################################################

	# get the info about the functional we need
	functional = kwargs.get('functional',settings.DFTFunctional)
	do_grad_dens = need_gradients[functional]
	do_spin_polarized = kwargs.get('do_spin_polarized',settings.DFTSpinPolarized)
	nbprocs = comm.size

	# the master procs clean up the 
	# grid and extract the density and gamma
	# and determine the MPI load balance
	if rank == 0:

		# set up the density properly
		gr.floor_density()  # Insure that the values of the density don't underflow
		gr.renormalize(nel) # Renormalize to the proper # electrons

		# extract the density weight etc ..
		dens = gr.dens()
		weight = gr.weights()
		gamma = gr.get_gamma()
		npts = len(dens)

		# proper version of the density
		if gr.version == 1:
			amdens = np.zeros((2,npts),'d')
			amgamma = np.zeros((3,npts),'d')
			amdens[0,:] = amdens[1,:] = 0.5*dens
			if gamma is not None:
				amgamma[0,:] = amgamma[1,:] = amgamma[2,:] = 0.25*gamma

		elif gr.version == 2:
			amdens = gr.density.T
			amgamma = gr.gamma.T
			
		# determine the loab balance for the MPI
		# we determine here the number of column
		# but keep in mind that dens and gamma 
		# have also 2 lines
		# the XXX_1d are used to gather the fxc and derivatives
		# the XXX_2d are used to scatter the density and gamma
		npp = npts/nbprocs
		send_counts_1d = []
		disp_1d = []
		for ir in range(nbprocs):
			if ir < nbprocs-1:
				terms = range(ir*npp,(ir+1)*npp)
			else:
				terms = range(ir*npp,npts)

			send_counts_1d.append(len(terms))
			disp_1d.append(terms[0])

		# convert to np array and account
		# for the 2 lines
		send_counts_1d = np.array(send_counts_1d,dtype=np.int)
		send_counts_2d = 2*send_counts_1d
		disp_1d = np.array(disp_1d,dtype=np.int)
		disp_2d = 2*disp_1d
		
	# create the send_count and disp for the slaves
	if rank >0:
		send_counts_1d = np.empty(nbprocs,dtype=np.int)
		disp_1d = np.empty(nbprocs,dtype=np.int)
		send_counts_2d = np.empty(nbprocs,dtype=np.int)
		disp_2d = np.empty(nbprocs,dtype=np.int)

	# broadcast the send_count and disp to all the procs
	comm.Barrier()
	comm.Bcast([send_counts_1d,MPI.INT],root=0)
	comm.Bcast([disp_1d,MPI.INT],root=0)
	comm.Bcast([send_counts_2d,MPI.INT],root=0)
	comm.Bcast([disp_2d,MPI.INT],root=0)

	send_counts_1d = tuple(send_counts_1d)
	disp_1d = tuple(disp_1d)
	send_counts_2d = tuple(send_counts_2d)
	disp_2d = tuple(disp_2d)

	comm.Barrier()

	######################################################################
	## Compute the XC FUnctional and derivative will all procs
	## we first create local amdens and gamma for all procs
	## as well as NONE global amdens and gamma for slaves
	## the scatter the amdens and amgamma from master to slaves with scatterv
	## each procs computes its part of the functional and derivative
	## then we gather all the data to master with gatherv
	######################################################################

	# each procs create its own local density and gamma
	amdens_local = np.zeros(send_counts_2d[rank],dtype=np.double)
	amgamma_local = np.zeros(send_counts_2d[rank],dtype=np.double)

	# creates a flatten vector for the density and gamma
	# we use Fortran Colum Major style
	# create None global dens and gamma for the slaves
	if rank == 0:
		amdens_flat = amdens.flatten('F').astype('d')
		amgamma_flat = amgamma.flatten('F').astype('d')
	else:
		amdens_flat = amgamma_flat = None

	# we scatter the values of dens and gamm from master to
	# all slaves with the Scatterv command
	if rank == 0 and _debug_:
		print '\n\t\t Scatter the density/gradient to all procs'	
	comm.Barrier()
	comm.Scatterv([amdens_flat,send_counts_2d,disp_2d,MPI.DOUBLE],amdens_local,root=0)
	comm.Scatterv([amgamma_flat,send_counts_2d,disp_2d,MPI.DOUBLE],amgamma_local,root=0)
	#print ("process " + str(rank) + " has " +str(amdens_local))


	# reshape the local contained to 2lines x Ncolumns
	amdens_local = np.reshape(amdens_local,(2,send_counts_2d[rank]/2),order='F').astype('d')
	amgamma_local = np.reshape(amgamma_local,(2,send_counts_2d[rank]/2),order='F').astype('d')

	# all the procs compute their part of the XC functional
	if rank == 0 and _debug_ :
		print '\t\t Compute the XC functional and derivatives'	
	fxc,dfxcdna,dfxcdnb,dfxcdgaa,dfxcdgab,dfxcdgbb = XC_mpi(amdens_local,amgamma_local,**kwargs)
	
	# create containers for the total density and derivatives
	comm.Barrier()

	if rank == 0 :
		fxc_tot = np.zeros(npts,'d')
		dfxcdna_tot = np.zeros(npts,'d')
		dfxcdnb_tot  = np.zeros(npts,'d')
		dfxcdgaa_tot = np.zeros(npts,'d')
		dfxcdgab_tot = np.zeros(npts,'d')
		dfxcdgbb_tot = np.zeros(npts,'d')
	else:
		fxc_tot = None
		dfxcdna_tot = None
		dfxcdnb_tot  = None
		dfxcdgaa_tot = None
		dfxcdgab_tot = None
		dfxcdgbb_tot = None
	

	# the master procs gather all the data from the slaves
	# with the gatherv command
	
	if rank == 0 and _debug_ :
		print '\t\t Gather the density/gradient from all procs'
	comm.Barrier()

	comm.Gatherv(fxc,[fxc_tot,send_counts_1d,disp_1d,MPI.DOUBLE],root=0)
	comm.Gatherv(dfxcdna,[dfxcdna_tot,send_counts_1d,disp_1d,MPI.DOUBLE],root=0)
	comm.Gatherv(dfxcdnb,[dfxcdnb_tot,send_counts_1d,disp_1d,MPI.DOUBLE],root=0)
	comm.Gatherv(dfxcdgaa,[dfxcdgaa_tot,send_counts_1d,disp_1d,MPI.DOUBLE],root=0)
	comm.Gatherv(dfxcdgab,[dfxcdgab_tot,send_counts_1d,disp_1d,MPI.DOUBLE],root=0)
	comm.Gatherv(dfxcdgbb,[dfxcdgbb_tot,send_counts_1d,disp_1d,MPI.DOUBLE],root=0)

	
	# only the master procs does the rest so far
	if rank == 0:

		if _debug_:
			print '\t\t Compute the Matrix representation of the functional'
		# total energy. THis is different from XC lib that 
		# return something like np.dot(weight,fxc_tot)/dens
		#print fxc_tot
		Exc = np.dot(weight,fxc_tot)
		

		# ################################# #
		# we start with the spin up density
		# ################################# #

		# Combine w*v in a vector for multiplication by bfs
		wva = weight*dfxcdna_tot  

		# First do the part that doesn't depend upon gamma
		nbf = gr.get_nbf()
		Fxca = np.zeros((nbf,nbf),'d')
		
		# broadcast the wva vector to all procs
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
			A = transpose(0.5*transpose(gr.grad())*(weight*(2*dfxcdgaa_tot+dfxcdgab_tot)))
			for a in xrange(nbf):
				for b in xrange(a+1):
					B = gr.grad_bf_prod(a,b)
					Fxca[a,b] += sum(ravel(A*B))
					Fxca[b,a] = Fxca[a,b]

		# if we don not have spin polarization we only return 2 values
		# but we return 3  otherwize ... I can see that being problemactic
		# anywho we return here if no spin
		if not do_spin_polarized: 
			return Exc,Fxca
			
		# ################################# #
		# we deal with the spin down density
		# ################################# #

		# Combine w*v in a vector for multiplication by bfs
		wvb = weight*dfxcdnb_tot  

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
			A = np.transpose(0.5*np.transpose(gr.grad())*(weight*(2*dfxcdgbb_tot+dfxcdgab_tot)))
			for a in xrange(nbf):
				for b in xrange(a+1):
					B = gr.grad_bf_prod(a,b)
					Fxcb[a,b] += sum(ravel(A*B))
					Fxcb[b,a] = Fxcb[a,b]

		# The master returns everything
		return Exc,Fxca,Fxcb

	# all the slaves return Nones
	else:
		if not do_spin_polarized:
			return None,None
		else:
			return None,None, None
	

###########################################
## SELF CONSISTANT DFT
## In this MPI version the master proc
## use s the other procs to speed up the 
## calcultion of the XC
###########################################
def dft_mpi(mol,bfs,S,Hcore,Ints,gr,xc_func,rank,comm,mu=0,MaxIter=100,eps_SCF=1E-4,_diis_=True):

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

	# the rank 0 initialize the calculations
	if rank == 0:

		# orthogonalization matrix
		X = SymOrthCutoff(S)

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
	iiter = 1
	de = 1E3*np.ones(1)
	while (iiter < MaxIter) and (de > eps_SCF):


		# master proc initialize the grid
		if rank == 0:

			# set the denstity on the grid
			gr.setdens(D)
			
			# form the J matrix from the 
			# density matrix and  the 2electron integrals
			J = getJ_mpi(Ints,D)

		# get the XC Function all the procs contribute
		# only the master prcos hold the final Exc and XC matrix
		Exc,XC = getXC_mpi(gr,nelec,rank,comm,_debug_=True,functional=xc_func)
		
		# only the master procs continues here 
		# all the other diectly returns
		if rank == 0:

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

			# energy difference
			de[0] = np.abs(e-eold).real
			if de > eps_SCF:
				eold = e

			# print the SCF count
			print "\t\t Iteration: %d    Energy: %f    EnergyVar: %f"%(iiter,e.real,de)

		# broadcast the energy variation
		# so that all the process stops
		comm.Bcast(de,root=0)
		
		# increment the iterator
		iiter += 1
	
	if rank == 0:

		if iiter < MaxIter:
			print("\t\t SCF for HF has converged in %d iterations, Final energy %1.3f Ha\n" % (iiter,e.real))
			
		else:
			print("\t\t SCF for HF has failed to converge after %d iterations")


	# compute the density matrix in the
	# eigenbasis of F
	if rank == 0:
		P = np.dot(Cp.T,np.dot(Dp,Cp))
		return Lp,C,Cp,F,Fp,D,Dp,P,X
	else:
		return None,None,None,None,None,None,None,None,None
	


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
def compute_F_mpi(P,Hcore,X,gr,xc_func,Ints,nelec,rank,comm,mu=0):


	# get the matrix of J for a fixed density P
	# and set the density on the molecular grid
	# P is here given in the non-orthonormal, i.e. the AO basis
	if rank == 0:
		J = getJ_mpi(Ints,P)
		gr.setdens(P.real)

	# get the xc func
	Exc,XC = getXC_mpi(gr,nelec,rank,comm,functional=xc_func)

	# continue the rest of the calculation 
	# only with the master procs
	if rank == 0:

		# the G matrix
		G = 2*J+XC

		# pass the G matrix in the orthonormal basis
		Gp = simx(G,X,'T')
			
		# form the Fock matrix in the orthonomral basis
		F = Hcore + Gp + mu

	else:
		F = None

	# done
	return F


#####################################################
##	Compute non SC Fock Matrix MPI 
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



