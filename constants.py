import scipy.constants
import numpy as np

"""This module defines physical constants used in the ISC, IC and PL rate calculations."""

h = scipy.constants.value("Planck constant")
hbar = h/(2*np.pi)
hartree_to_eV = 27.211386245981
J_to_hartree = 2.2937104486906E+17
amu_to_kg = 1.6605402e-27
h_eV = scipy.constants.value("Planck constant in eV/Hz")
c = scipy.constants.c
eps_0 = scipy.constants.epsilon_0

# convertion from Debye to SI units (Coulomb meter)
debye_to_Cm = 3.33564e-30

# defines conversion constant for dipole moments from atomic units (missing the electron charge unit) to Debye
au_to_debye = scipy.constants.e * scipy.constants.value("Bohr radius") / debye_to_Cm
