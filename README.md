# RealTimeDFT

RealTimeHatreeDFT computes the dynamics of a quantum system following the real-time time-dependent Khon-Sham equations. The code relies heavily on the library PyQuante (http://pyquante.sourceforge.net). A serial and a parallel versions are available. The parallel verion uses mpi4py to accelerate the calculations of the two-electrons integrals and the density integration on the grid. The dynamics is calculated following  predictor-corrector approach for stability. Several options are available to specify the basis-set and the exciting field. 


![Alt text](illust.png?raw=true "Dynamics of Pyrazole")
