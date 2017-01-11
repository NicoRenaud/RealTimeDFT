#!/usr/bin/env python

import numpy as np 
import scipy.linalg as scla
import scipy.constants
import sys
import argparse
import re

from Cube import *
from tools import *
from plot import *

from PyQuante.LA2 import geigh,mkdens,trace2,CholOrth,CanOrth,SymOrthCutoff,simx,sym
from PyQuante import SCF, Molecule
from PyQuante.NumWrap import arange
from PyQuante.cints import coulomb_repulsion
from PyQuante import configure_output
from PyQuante.Constants import hartree2ev
from PyQuante.Ints import getbasis
from PyQuante.Ints import getT,getS,getV,get1ints,getints,get2JmK, getJ

from PyQuante.MG2 import MG2 as MolecularGrid
from PyQuante.DFunctionals import XC,need_gradients

# print option for numpy
np.set_printoptions(precision=3)

# export flags
_plot_ = 1
_export_mo_ = 0
_export_mo_dynamics_ = 0
_export_blender_ = 0

# dyanmic flag
_predict_ = 1

# test/debug flags
_test_ = 1
_check_ortho_ = 0
_print_basis_ = 0
_verbose_ = 0

# constants
au2s = scipy.constants.value('atomic unit of time')

####################################################
##	MAIN FUNCTION
####################################################
def main(argv):

	parser = argparse.ArgumentParser(description='Real Time TDDFT simulations')

	# molecule information
	parser.add_argument('mol',help='xyz file of the molecule',type=str)
	parser.add_argument('-charge',default = 0, help='Charge of the system',type=float)
	parser.add_argument('-units',default = 'angs', help='Units in the xyz file',type=str)

	# DFT calculations
	parser.add_argument('-basis',default = 'sto-3g', help='basis set to be used in the calculation',type=str)
	parser.add_argument('-xcf',default = 'VWN', help='functional to use',type=str)

	# SCF onvergence
	parser.add_argument('-diis',default=True,help='Do we use DIIS for convergence or not',type=bool)
	parser.add_argument('-MaxIter',default = 100, help='Maximum number of SCF iterations',type=int)
	parser.add_argument('-eps_SCF',default = 1E-4, help='Criterion for SCF termination',type=float)

	# grid for integration
	parser.add_argument('-grid_nrad',default = 32, help='Number of radial shells per atom',type=int)
	parser.add_argument('-grid_fineness',default = 1, help='adial shell fineness. 0->coarse, 1->medium, 2->fine',type=int)

	# field information
	parser.add_argument('-ffreq',default = 0.1, help='Frequency of the field',type=float)
	parser.add_argument('-fint',default = 0.05, help='Intensity of the field',type=float)
	parser.add_argument('-ft0',default = 0.5, help='center of the field for gsin',type=float)
	parser.add_argument('-fsigma',default = 0.25, help='sigma of the field for gsin',type=float)
	parser.add_argument('-fform',default = 'sin', help='field form',type=str)
	parser.add_argument('-fdir',default = 'x', help='direction of the field',type=str)

	# Propagation
	parser.add_argument('-tmax',default = 200, help='maximum evolution time',type=float)
	parser.add_argument('-nT',default = 200, help='number of time step',type=int)
	parser.add_argument('-time_unit',default='au',help='unit of tmax',type=str)

	# export
	parser.add_argument('-nb_print_mo',default = 10, help='Number of orbitals to be written',type=int)
	parser.add_argument('-blender',default=False, help='Export the MO in blender',type=bool)

	'''
	Possible basis
	'3-21g' 'sto3g' 'sto-3g' 'sto-6g'
	'6-31g' '6-31g**' '6-31g(d,p)' '6-31g**++' '6-31g++**' '6-311g**' '6-311g++(2d,2p)'
    '6-311g++(3d,3p)' '6-311g++(3df,3pd)'
    'lacvp'
    
    'ccpvdz' 'cc-pvdz' 'ccpvtz' 'cc-pvtz' 'ccpvqz' 'cc-pvqz' 'ccpv5z' 'cc-pv5z' 'ccpv6z' 'cc-pv6z'

    'augccpvdz' 'aug-cc-pvdz' 'augccpvtz' 'aug-cc-pvtz' 'augccpvqz'
    'aug-cc-pvqz' 'augccpv5z' 'aug-cc-pv5z' 'augccpv6z' 'aug-cc-pv6z'    
    'dzvp':'dzvp',

	'''

	# done
	args=parser.parse_args()

	print '\n\n=================================================='
	print '== PyQuante - Real Time Time-dependent DFT      =='
	print '==================================================\n'

	#-------------------------------------------------------------------------------------------
	#
	#									PREPARE SIMULATIONS
	#
	#-------------------------------------------------------------------------------------------

	##########################################################
	##					Read Molecule
	##########################################################

	# read the xyz file of the molecule
	print '\t Read molecule position'

	f = open(args.mol,'r')
	data = f.readlines()
	f.close

	# get the molecule name
	name_mol = re.split(r'\.|/',args.mol)[-2]

	# create the molecule object
	xyz = []
	for i in range(2,len(data)):
		d = data[i].split()
		xyz.append((d[0],(float(d[1]),float(d[2]),float(d[3]))))

	natom = len(xyz)
	mol = Molecule(name=name_mol,units=args.units)
	mol.add_atuples(xyz)
	mol.set_charge(args.charge)
	nelec = mol.get_nel()
	
	if np.abs(args.charge) == 1:
		mol.set_multiplicity(2)
	if args.charge>1:
		print 'charge superior to one are not implemented'

	# get the basis function
	bfs = getbasis(mol,args.basis)
	nbfs = len(bfs)
	nclosed,nopen = mol.get_closedopen()
	nocc = nclosed
	print '\t\t Molecule %s' %args.mol
	print '\t\t Basis %s' %args.basis
	print '\t\t %d basis functions' %(nbfs)
	print '\t\t %s functional' %(args.xcf)
	
	if _print_basis_:
		for i in range(nbfs):
			print bfs[i]
	
	# compute all the integrals
	print '\n\t Compute the integrals and form the matrices'
	S,Hcore,Ints = getints(bfs,mol)
	
	print '\t Compute the transition dipole moments'
	mu_at = compute_dipole_atoms(bfs,args.fdir)

	print '\t Set up the molecular grid for density integration'
	# parameter of the grid
	grid_nrad = args.grid_nrad 
	grid_fineness = args.grid_fineness 

	# initialize the grid
	gr = MolecularGrid(mol,grid_nrad,grid_fineness) 
	gr.set_bf_amps(bfs)


	##########################################################
	##			Compute the HF GROUND STATE
	##########################################################

	print '\n\t Compute the ground state KS Ground State\n\t',
	print '-'*50
	L,C,Cp,F0,F0p,D,Dp,P,X = dft(mol,bfs,S,Hcore,Ints,gr,args.xcf,MaxIter=args.MaxIter,eps_SCF=args.eps_SCF,_diis_=args.diis)
	
	print '\t Energy of the KS orbitals\n\t',
	print '-'*50
	index_homo = nocc-1
	nb_print = int(min(nbfs,args.nb_print_mo)/2)
	for ibfs in range(index_homo-nb_print+1,index_homo+nb_print+1):
		print '\t\t orb %02d \t occ %1.1f \t\t Energy %1.3f Ha' %(ibfs,np.abs(2*P[ibfs,ibfs].real),L[ibfs].real)
	
	# store the field free eigenstates
	C0 = np.copy(C)
	C0p = np.copy(Cp) 

	# invert of the X matrix
	Xm1 = np.linalg.inv(X)

	##########################################################
	##		Transform the matrices in the OB
	##########################################################
	
	# pass the other matrices as well
	Hcore_p = simx(Hcore,X,'T')
	mu_at_p = simx(mu_at,X,'T')

	# copy the Fock matrix
	Fp = np.copy(F0p)
	
	# transtion matrix at t=0
	mup = mu_at_p

	# check if everythong is ok
	if _check_ortho_:
		w,u = np.linalg.eigh(F0p)
		if np.sum(np.abs(w-L)) > 1E-3:
			print '\t == Warning orthonormalisation issue'
			sys.exit()

	#-------------------------------------------------------------------------------------------
	#
	#						TEST IF EVERYTHING IS OF SO FAR
	#
	#-------------------------------------------------------------------------------------------

	# verify the basis transformation between 
	# Atomic Orbitals and Orthonormal orbitals
	#
	if _test_:

		print '\n\t Run Check on the matrices'

		#print'\n'
		#print'='*40
		print'\t\t Verify the basis transformation from diagonal to AO basis ',
			
		x = np.dot(np.diag(L),np.linalg.inv(C))
		x = np.dot(C,x)
		x = np.dot(S,x)
		
		if np.abs(np.sum(x-F0))<1E-3:
			print'\t Check'
		else:
			print '\t NOT OK'
		#print'='*40

		if _verbose_:
			print '\t\t reconstructed Fock matrix'
			print x
			print '\t\t original Fock matrix'
			print F0

		#print'\n\t'
		#print'='*40
		print'\t\t Verify the basis transformation from AO to diagonal basis ',


		y = np.dot(F0,C)
		y = np.dot(np.linalg.inv(S),y)
		y = np.dot(np.linalg.inv(C),y)

		if np.abs(np.sum(y-np.diag(L)))<1E-3:
			print'\t Check'
		else:
			print '\t NOT OK'

		#print'='*40

		if _verbose_:
			print '\t\t reconstructed eigenvalues'
			print y
			print '\t\t original eigenvalues'
			print L
	

	#
	# verify the basis transformation between 
	# Atomic Orbitals and Orthonormal orbitals
	#
	if _test_:
		#print'\n'
		#print'='*40
		print'\t\t Verify the basis transformation from AO basis to ORTHO basis ',
		#print'='*40

		if np.abs(np.sum(D-np.dot(X,np.dot(Dp,X.T))))<1E-3:
			print '\t Check'
		else:
			print '\t NOT OK'

		if _verbose_:
			print '\t\t reconstructed density in the AO'
			print np.dot(X,np.dot(Dp,X.T))
			print '\t\t Original density in the AO'
			print D
	

	#
	# verify the basis transformation between 
	# Atomic Orbitals and Orthonormal orbitals
	#
	if _test_:
		#print'\n'
		#print'='*40
		print'\t\t Verify the basis transformation from AO basis to ORTHO basis ',
		#print'='*40

		if np.abs(np.sum(Dp-np.dot(Xm1,np.dot(D,Xm1.T))))<1E-3:
			print '\t Check'
		else:
			print '\t NOT OK'
	
		if _verbose_:
			print '\t\t reconstructed density in the OB'
			print np.dot(Xm1,np.dot(D,Xm1.T))
			print '\t\t original density in the OB'
			print Dp
	
	# test if the Fock matrix and densitu matrix in OB
	# share the same eigenbasis
	# due to degeneracies in the density matrix only a few
	# eigenvectors might be the same
	if _verbose_:
		print'\t\t verify that the Fock matrix and the density matrix '
		print'\t\t in the ORTHOGONAL BASIS have the same eigenvector', 

		lf,cf = scla.eigh(F0p)
		r = 1E-6*np.random.rand(nbfs,nbfs)
		r += r.T
		ld,cd = scla.eigh(Dp+r)

		x1 = simx(Dp,cf,'N')
		x2 = simx(F0p,cd,'N')

		s1 = np.sum(np.abs(x1-np.diag(np.diag(x1))))
		s2 = np.sum(np.abs(x2-np.diag(np.diag(x2))))
		
		if s1 < 1E-6 and s2 < 1E-6:
			print'\t\t Check'
		else:
			print '\t\t NOT OK'
			if _verbose_:
				print'\t\tDensity matrix in eigenbasis of the Fock matrix'
				print np.array_str(x1,suppress_small=True)
				print'\t\t\Fock matrix in eigenbasis of the Density matrix'
				print np.array_str(x2,suppress_small=True)

			
		print'\n'
		print'\t\t',
		print'='*40
		print'\t\teigenvector/eigenvalues of the fock matrix'
		print lf
		print cf
		print''
		print'\t\teigenvector/eigenvalues of the density matrix'
		print ld
		print cd
		print'\t\t',
		print'='*40

	#
	# check the initial population of the molecular orbital
	#
	if _verbose_:

		print'\t\t',
		print'='*40
		print'\t\t Initial population of the molecular orbitals'
		print '\t\t ',
		for i in range(len(P)):
			print '%1.3f ' %(P[i,i].real),
		print ''
		print'\t\t',
		print'='*40
	#

	#-------------------------------------------------------------------------------------------
	#
	#								SIMUALTIONS
	#
	#-------------------------------------------------------------------------------------------

	##########################################################
	##					Define time and outputs
	##########################################################

	if args.time_unit == 'fs':
		tmax_convert =args.tmax*1E-15/au2s
	elif args.time_unit == 'ps':
		tmax_convert =args.tmax*1E-12/au2s
	elif args.time_unit == 'au':
		tmax_convert = args.tmax

	# a few shortcut
	ffreq= args.ffreq
	fint = args.fint
	ft0 = args.ft0*tmax_convert
	fsigma = args.fsigma*tmax_convert
	fform = args.fform

	# readjust the frequency in case it is not specified
	if ffreq<0:
		ffreq = L[nocc]-L[nocc-1]
		print '\n\t Field frequency adjusted to %f' %(ffreq)

	T = np.linspace(0,tmax_convert,args.nT)
	Tplot = np.linspace(0,args.tmax,args.nT)
	FIELD = np.zeros(args.nT)
	N = np.zeros((nbfs,args.nT))
	Q = np.zeros((natom,args.nT))
	mu0 = 0 
	for i in range(natom):
		Q[i,:] = mol[i].Z
		mu0 += mol[i].Z*mol.atoms[i].r[0]
	MU = mu0*np.ones(args.nT)	
		
	
	dt = T[1]-T[0]	
	

	##########################################################
	##					Loop over time
	##########################################################
	print '\n'
	print '\t Compute the TD-DFT dynamics'
	print '\t Simulation done at %d percent' %(0),
	for iT in range(0,args.nT):


		################################################################
		## 					TIMER
		################################################################
		if iT*10%int(args.nT) == 0:
			sys.stdout.write('\r\t Simulation done at %d percent' %(iT*100/args.nT))
			sys.stdout.flush()

		################################################################
		## Compute the observable
		################################################################

		# compute the Lowdin atomic charges
		for iorb in range(nbfs):
			atid = bfs[iorb].atid
			Q[atid,iT] -= 2*Dp[iorb,iorb].real
			
		# compute the population of the orbitals
		for iorb in range(nbfs):
			N[iorb,iT] = (np.dot(C0p[:,iorb],np.dot(Dp.real,C0p[:,iorb].T))/np.linalg.norm(C0p[:,iorb])**2).real
		
		# compute the instantaneous dipole
		MU[iT] -= np.trace(np.dot(mu_at,D)).real

		######################################
		##	Propagation
		######################################

		if _predict_ and iT < args.nT-1:

			# predict the density matrix
			dp = propagate_dm(Dp,Fp,dt,method='relax')

			#predicted dm in AO basis
			d = np.dot(X,np.dot(dp,X.T))

			# value of the field
			fp1 = compute_field(T[iT+1],ffreq= ffreq,fint=fint,t0 = ft0, s = fsigma, fform=fform)
			mup_p1 = mu_at_p*fp1

			# predicted fock matrix
			fp = compute_F(d,Hcore_p,X,gr,args.xcf,Ints,nelec,mup_p1)

			# form the intermediate fock matrix
			fm = 0.5*(Fp+fp)

			# propagte the density matrix with that
			Dp = propagate_dm(Dp,fm,dt,method='relax')

		else:

			# propagte the density matrix with that
			Dp = propagate_dm(Dp,Fp,dt,method='relax')
			
		######################################
		##	New Density Matrix in AO
		######################################

		# DM in AO basis
		D = np.dot(X,np.dot(Dp,X.T))

		######################################
		##	New Field
		######################################

		# value of the field
		FIELD[iT] = compute_field(T[iT],ffreq= ffreq,fint=fint,t0 = ft0, s = fsigma, fform=fform)
		mup = mu_at_p*FIELD[iT]

		######################################
		##	New Fock Matrix
		######################################

		# new Fock matrix
		Fp = compute_F(D,Hcore_p,X,gr,args.xcf,Ints,nelec,mup)

	sys.stdout.write('\r\t Simulation done at 100 percent')
	
	# save all the data
	print '\n\t Save data\n'
	np.savetxt('time.dat',Tplot)
	np.savetxt('orb_pops.dat',N)
	np.savetxt('charges.dat',Q)
	np.savetxt('dipole.dat',MU)
	np.savetxt('field.dat',FIELD)

	#-------------------------------------------------------------------------------------------
	#
	#						EXPORT THE RESULTS
	#
	#-------------------------------------------------------------------------------------------
	
	##########################################################
	##			PLOT THE DATA WITH MATPLOTLIB
	##########################################################
	if _plot_:
		plot(Tplot,FIELD,N,Q,MU,cutlow=0.05,cuthigh=0.9)


	##########################################################
	##			Export the MO in VMD Format
	##########################################################
	if _export_mo_:

		print '\t Export MO Gaussian Cube format'
		
		index_homo = nocc-1
		nb_print = int(min(nbfs,args.nb_print_mo)/2)
		fmo_names = []
		for ibfs in range(index_homo-nb_print+1,index_homo+nb_print+1):
			if ibfs <= index_homo:
				motyp = 'occ'
			else:
				motyp = 'virt'
			file_name =mol.name+'_mo'+'_'+motyp+'_%01d.cube' %(index)
			xyz_min,nb_pts,spacing = mesh_orb(file_name,mol,bfs,C0,ibfs)
			fmo_names.append(file_name)
		

		##########################################################
		##					Export the MO
		##				in bvox Blender format
		##########################################################

		if _export_blender_:
			print '\t Export MO volumetric data for Blender'
			bvox_files_mo = []
			for fname in fmo_names:
				fn = cube2blender(fname)
				bvox_files_mo.append(fn)

			##########################################################
			##					Create the 
			##				Blender script to visualize the orbitals
			##########################################################
			path_to_files = os.getcwd()
			pdb_file = name_mol+'.pdb'
			create_pdb(pdb_file,args.mol,args.units)

			# create the blender file
			blname = name_mol+'_mo_volumetric.py'
			create_blender_script_mo(blname,xyz_min,nb_pts,spacing,pdb_file,bvox_files_mo,path_to_files)

	##########################################################
	##			Export the MO DYNAMICS in VMD Format
	##########################################################
	if _export_mo_dynamics_ :

	    # step increment
		nstep_mo = 4

		# resolution i.e. point per angstrom
		ppa = 1

		# just a test
		norm = 1-np.min(N[0,:])

		# loop to create all the desired cube files
		fdyn_elec, fdyn_hole = [], []
		for iT in range(0,args.nT,nstep_mo):

			if iT*10%int(args.nT) == 0:
				sys.stdout.write('\r\t Export Cube File \t\t\t\t %d percent done' %(iT*100/args.nT))
				sys.stdout.flush()

			felec,fhole,xyz_min,nb_pts,spacing = mesh_exc_dens(mol,bfs,N,C0,iT,nocc,resolution=ppa)
			fdyn_elec.append(felec)
			fdyn_hole.append(fhole)

		sys.stdout.write('\r\t Export Cube File \t\t\t\t 100 percent done\n')

		# create the vmd script to animate the voxel
		create_vmd_anim(mol.name,args.nT,nstep_mo)

		##########################################################
		##					Export the DYN
		##				in bvox Blender format
		##########################################################
		if _export_blender_:
			print '\t Export volumetric data for Blender'

			# create the bvox files for electron
			bvox_files_dyn_elec = []
			for fname in fdyn_elec:
				fn = cube2blender(fname)
				bvox_files_dyn_elec.append(fn)

			# create the bvox files for holes
			bvox_files_dyn_hole = []
			for fname in fdyn_hole:
				fn = cube2blender(fname)
				bvox_files_dyn_hole.append(fn)
			
			# create the pdb file
			pdb_file = name_mol+'.pdb'
			create_pdb(pdb_file,args.mol,args.units)

			# path to files
			path_to_files = os.getcwd()

			# create the blender script
			blname = name_mol+'_traj_volumetric.py'
			create_blender_script_traj(blname,xyz_min,nb_pts,spacing,pdb_file,bvox_files_dyn_elec,bvox_files_dyn_hole,path_to_files)




	print '\n\n=================================================='
	print '==                       Calculation done       =='
	print '==================================================\n'


if __name__=='__main__':
	main(sys.argv[1:])
	