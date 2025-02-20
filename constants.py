import scipy.constants
import numpy as np

h = scipy.constants.value("Planck constant")
hbar = h/(2*np.pi)
hartree_to_eV = 27.211386245981
J_to_hartree = 2.2937104486906E+17
au_to_kg = 1.6605402e-27
h_eV = scipy.constants.value("Planck constant in eV/Hz")
c = scipy.constants.c
eps_0 = scipy.constants.epsilon_0
debye_to_Cm = 3.33564e-30
au_to_debye = scipy.constants.e * scipy.constants.value("Bohr radius") / debye_to_Cm
