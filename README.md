# Defect-Rate-Estimator

This repo provides rudimentary scripts and functions to calculate overlap between phonon wavefunctions and use it to estimate transition rates in non-radiative and inter-system crossing transitions.

Methods are based on the use of the single-phonon approximation in transitions between to state structures.

The repo consists of loose python files, meant to be run in the same folder, with:
- constants.py
  - Definition of physical constants and conversion factors for the physical units of these methods.
- classes.py
  - Defines a State and Transition class containing the functionality to store relevant information and compute rates and other useful properties
- calc_rates.py
  - Provides functionalities for reading example data table files and calculate the corresponding rates. Meant to be the executable script of this workflow.
 
- ISCs.txt, PLs.txt, ICs.txt
  - Example files containing tabulated data for the transitions (one per row) between different states, providing necessary state descriptions in terms of the single-phonon parameters and coupling constants for the interaction Hamiltonian driving the transition in question.
