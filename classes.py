MIT License

Copyright (c) 2025 Oscar Bulancea Lindvall

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

import numpy as np
from constants import *
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import curve_fit
import scipy.integrate as integrate

class State:
    """Class to store intrinsic state properties for a state in the Q-E state space

    Attributes:
        Qs (list): List of Q values sampled, in amu^1/2 A
        energies (list): List of energies corresponding to Q values, in hartree
        min_E (float): Minimum energy of the state, in hartree
        eq_Q (float): Equilibrium Q coordinate of the state, in amu^1/2 A
        freq (float): Effective phonon frequency of the state, in hartree
        label (str): Optional label for the state

    """

    def __init__(self, Q_energy, effective_freq=None, label=None):
        self.Qs = [entry[0] for entry in Q_energy]
        self.energies = [entry[1] for entry in Q_energy]
        min_E_ind = np.argmin(self.energies)
        self.min_E = min(self.energies)
        self.eq_Q = self.Qs[min_E_ind]
        self.label = label
        if effective_freq is None:
            model = lambda x, w, E0: w**2*(x - self.eq_Q)**2/2 + E0
            params = curve_fit(model, self.Qs, self.energies, (1,self.min_E))
            w = np.sqrt(params[0]*hbar/amu_to_kg)*1e10 # in hartree
            self.freq = w
        else:
            self.freq = effective_freq

    def covering_q_range(self, norm=5):
        """Calculates a Q range covering the equilibrium point and decent spread correlated to the effective phonon frequency of the state"""
        alpha = self.freq/(hbar*(hbar*J_to_hartree)) * amu_to_kg * 1e-20 # in (au A^2)^-1
        return np.array([-1,1])*(norm/np.sqrt(alpha)) + self.eq_Q
    
    def plot_Q_curve(self, color="blue", offset=0):
        """Plots the Q-dependent energy curve of the state, with specified offset in the energy"""
        w = self.freq / (J_to_hartree * hbar)
        Q = self.eq_Q
        model = lambda x: J_to_hartree*w**2*((x - Q) * np.sqrt(amu_to_kg)*1e-10)**2/2 + self.min_E
        q_range = self.covering_q_range()
        x_vals = np.linspace(q_range[0], q_range[1], 1000)
        plt.plot(x_vals, model(x_vals)-offset, color=color)

    def plot_eigenmode(self, n, normalization=1, color="blue", offset=0):
        """Plots the nth eigenmode of the state, with specified offset in the energy"""
        q_range = self.covering_q_range()
        x_vals = np.linspace(q_range[0], q_range[1], 1000)
        mode = normalization*qosc_eigenfunc(x_vals, self.freq, n, self.eq_Q) + self.min_E + n*self.freq - offset + self.freq/2
        plt.plot(x_vals, mode, linewidth=0.8, color=color)
        plt.plot(x_vals, (self.min_E + (n+1/2)*self.freq - offset)*np.ones(x_vals.shape), "--", linewidth=0.5, color=color)

class Transition:
    """Class to represent a transition between two states, differing in structural coordinates and with a linearly Q-dependent coupling factor from the interaction Hamiltonian

    Attributes:
        state_i (State): Initial state of the transition
        state_f (State): Final state of the transition
        couplings (2-D numpy array): Coupling values at the equilibrium Q coordinates of the initial and final states, for each interaction operator considered. Shape: (number of operators, 2)
        trans_type (str): Type of the transition ("ISC", "IC", or "PL")
    """

    def __init__(self, init_state, final_state, coupling, trans_type):
        self.state_i = init_state
        self.state_f = final_state
        self.couplings = np.array(coupling)
        self.trans_type = trans_type

    def calc_huangrhys(self):
        """Calculates the Huang-Rhys factor for the Transition, using the effective phonon frequency and Q coord. positions of respective states

        Returns:
            float: Huang-Rhys factor for the transition
        """
        dQ2 = (self.state_i.eq_Q - self.state_f.eq_Q)**2 * (amu_to_kg)*1e-20
        w = np.mean([self.state_f.freq, self.state_i.freq])/(hbar*J_to_hartree)
        return dQ2*w/(2*hbar)


    def gen_linear_coupling(self, q_vals):
        """Create a linear interpolation between given coupling values for the provided Q range.

        Args:
            q_vals (1-D numpy array): Q values to evaluate couplings at

        Returns:
            2-D numpy array: Interpolated coupling values for each state transition at the given Q values. Q values along the columns, transitions along the rows.
        """
        coupling_vals = np.zeros((self.couplings.shape[0], len(q_vals)))
        for i in range(self.couplings.shape[0]):
            #print("debug")
            #print((self.state_i.eq_Q, self.state_f.eq_Q), (self.couplings[i,0], self.couplings[i,1]))
            fit = np.polyfit((self.state_i.eq_Q, self.state_f.eq_Q), (self.couplings[i,0], self.couplings[i,1]), deg=1)
            f = lambda x: fit[0]*x + fit[1]
            coupling_vals[i] = f(q_vals)
        return coupling_vals


    def calc_rate(self, nmax = 40, overlap_threshold=1e-6, neval=int(1e6), n_refr=None, spread=1e-3, plot=False):
        """Calculates the transition rates for the transition, by evaluating Fermi's golden rule for the corresponding
        type of transition this object represents. The Q-dependent overlap is evaluated numerically.

        Args:
            nmax (int, optional): Maximal phonon mode number to include in integration. Defaults to 40.
            overlap_threshold (float, optional): Threshold for overlap significance. Defaults to 1e-6.
            neval (int, optional): Number of evaluation points in integration. Defaults to int(1e6).
            n_refr (float, optional): Refractive index, required for PL transitions. Defaults to None.
            spread (float, optional): Standard deviation for delta function approximation. Defaults to 1e-3.
            plot (bool, optional): Whether to plot intermediate results. Defaults to False.

        Returns:
            4-tuple: List containing 1) Total transition rate, 2) Partial rates state, 3) Partial rates per mode and state, 4) Energy values used in evaluation
        """
        ### prepare entities
        eq_Q_i = self.state_i.eq_Q # equilibrium Q Coordinates for states i and f
        eq_Q_f = self.state_f.eq_Q
        w_i = self.state_i.freq # w_i, w_f in hartrees
        w_f = self.state_f.freq
        
        ### prepare Q range
        ranges = np.array([self.state_i.covering_q_range(), self.state_f.covering_q_range()])
        q_range = np.linspace(np.min(ranges,axis=(0,1)), np.max(ranges, axis=(0,1)), neval)
        ### generate coupling values
        coupling_vals = self.gen_linear_coupling(q_range)
        if plot:
            plt.plot(q_range, coupling_vals[0])
            plt.show()
        
        ### perform overlap sum
        dE = self.state_i.min_E - self.state_f.min_E
        overlaps = np.zeros(nmax)
        if self.trans_type == "PL":
            # interested in energy conservation with regard to emitted photon degree of freedom
            energies = np.linspace(dE-nmax*w_f-5*float(np.abs(spread)),dE+5*float(np.abs(spread) + w_f),neval)
        else:
            # interested in energy conservation with regard to zero photon emission (nonradiative)
            energies = np.linspace(-w_f, w_f,neval)

        # partial fermi contributions for each state, vibrational level and emitted phonon energy
        fermis = np.zeros((coupling_vals.shape[0], nmax, len(energies)))
        for i in range(coupling_vals.shape[0]): # iterate state transitions
            for n in range(nmax): # iterate vibrational levels
                overlap_val = overlap(0, n, [w_i, w_f], dq=np.abs(eq_Q_f - eq_Q_i), operand=coupling_vals[i], q_range=q_range)
                overlaps[n] = overlap_val
                if overlap_val == 0:
                    break
                delta_vals = delta(energies, shift=(dE - n*w_f), spread=spread)
                fermis[i,n,:] = overlap_val * delta_vals

        ### Calculate fermi rule
        if self.trans_type.upper() == "PL":
            if n_refr is None:
                raise Exception("Transition is type PL, but missing refractive index")
            #for i in range(fermis.shape[0]):
            #    for j in range(fermis.shape[1]):
            #        plt.plot(energies, fermis[i,j,:])
            #plt.show()
            #print(fermis)
            fermis *= debye_to_Cm**2 * J_to_hartree # dipole moments now in SI
            optical_prefactor = 1/J_to_hartree * (energies/(hbar*J_to_hartree))**3 * n_refr**2 / (3*np.pi * hbar * c**3 * eps_0) # distribution in (hartree * s)^-1
            optical_prefactor[energies < 0] = 0 # do not include phonon absorption regions
            fermis *= optical_prefactor
            total_fermi = np.sum(fermis, axis=1) # sum over vibrational levels
            total_rate = np.zeros(total_fermi.shape[0])
            ZPL_rate = np.zeros(total_fermi.shape[0])
            for i in range(total_rate.shape[0]):
                total_rate[i] = integrate.simpson(total_fermi[i,energies >= 0], energies[energies >= 0]) # integrate sum of phonon contributions
                ZPL_rate[i] = integrate.simpson(fermis[i,0,:], energies) # integrate spectra of ZPL only
            print(f"ZPL rate: {ZPL_rate}")
            print(f"total rate: {total_rate}")
            return total_rate, total_fermi, fermis, energies

        elif self.trans_type.upper() == "ISC":
            fermis *= 2*np.pi/(hbar*J_to_hartree)
            total_fermi = np.sum(fermis, axis=1)
            closest_to_zero = np.argmin(np.abs(energies))
            rate = total_fermi[:,closest_to_zero]
            return rate, total_fermi, fermis, energies

        elif self.trans_type.upper() == "IC":
            #print(f"Delta E: {dE}")
            #print("Overlaps")
            #print(overlaps)
            fermis *= 2*np.pi/(hbar*J_to_hartree) * dE**2
            total_fermi = np.sum(fermis, axis=1)
            closest_to_zero = np.argmin(np.abs(energies))
            rate = total_fermi[:,closest_to_zero]
            return rate, total_fermi, fermis, energies

        else:
            raise ValueError("Unknown transition type specified")

herm_cache = [np.array([1])]
def herm(x,n):
    """Evaluates the hermite polynomial of degree n at points x.

    Args:
        x (1D numpy array): Evaluation points
        n (int): Hermite polynomial degree

    Returns:
        1D numpy array: Evaluated Hermite polynomial values at points x
    """
    if len(herm_cache)-1 <= n:
        for i in range(len(herm_cache), n+1):
            cache = np.zeros(i+1)
            # construct hermite polynomial coefficients via recursion relations
            # a_(n,k) = 2a_(n-1,k-1) - (k+1)a_(n-1,k+1)
            cache[:i] = 2*herm_cache[i-1]
            cache[2:] -= np.arange(i-1,0,-1)*herm_cache[i-1][:-1]
            herm_cache.append(cache)
    return np.polyval(herm_cache[n], x)

def qosc_eigenfunc(q, w, n, q0=0):
    """Evaluates a quantum harmonic oscillator eigenstate at points q for phonon frequency w, for eigenmode n, optionally shifted by q0.

    Args:
        q (1-D numpy array): Evaluation points, in (amu)^1/2 A
        w (float): Phonon frequency, in Hartree
        n (int): Eigenmode number
        q0 (float, optional): Shift in the coordinate. Defaults to 0.

    Returns:
        1-D numpy array: Evaluated quantum harmonic oscillator eigenstate values at points q
    """
    # q in A(au)^1/2
    # w in Hartree
    #w_eV = w*hartree_to_eV
    alpha = w/(hbar*J_to_hartree)/hbar * amu_to_kg * 1e-20 # in (amu A^2)^-1
    herm_val = herm(np.sqrt(alpha)*(q-q0), n)
    factorial_list = np.arange(1,n+1)
    # bake prefactor into exponent for more convenient evaluation
    exponent = -alpha/2 * (q-q0)**2 - np.sum(np.log(factorial_list))/2 - np.log(2)*n/2
    exp_val = np.exp(exponent)
    #prefactor = 1/(np.sqrt(2**n)*np.sqrt(factorial(n)))
    return  (alpha/np.pi)**(0.25) * exp_val * herm_val

def overlap(n_i,n_f, w, dq=0, operand=None, q_range=None, neval=None):
    """Evaluates the overlap or matrix elements of two quantum harmonic oscillator states.

    Args:
        n_i (int): Eigenmode number of initial state
        n_f (int): Eigenmode number of final state
        w (float or list of two floats): Phonon frequency or frequencies
        dq (float, optional): Shift in the coordinate. Defaults to 0.
        operand (1-D numpy array, optional): Operand for matrix element calculation. Defaults to None.
        q_range (1-D numpy array, optional): Range of evaluation points. Defaults to None.
        neval (int, optional): Number of evaluation points. Defaults to None.

    Returns:
        1-D array: Wavefunction overlap, as a function of Q coord.
    """
    if isinstance(w, float):
        w_i,w_f = (w,w)
    elif isinstance(w, list):
        w_i,w_f = tuple(w)
    if neval is None:
        if q_range is None:
            neval = int(1e5)
        else:
            neval = len(q_range)
    if operand is None:
        operand = np.ones(neval)
    if q_range is None:
        w_eV = min(w)*hartree_to_eV
        alpha = 2*np.pi*w_eV/(h*h_eV) * amu_to_kg * 1e-20 # in (amu A^2)^-1
        max_range = 10/np.sqrt(alpha)
        q_range = np.linspace(-max_range, max_range, neval)
    
    eig_f = qosc_eigenfunc(q_range,w_i,n_i)
    eig_i = qosc_eigenfunc(q_range,w_f,n_f,dq)

    product = integrate.simpson(eig_i*operand*eig_f, q_range)
    return np.abs(product)**2

def delta(x, shift=0, spread=1e-3):
    """Generates a delta function, modelled as a gaussian distribution centered at "shift" and with sigma=spread

    Args:
        x (1-D numpy array): Evaluation points
        shift (float, optional): Center of the delta function. Defaults to 0.
        spread (float, optional): Standard deviation of the Gaussian approximation. Defaults to 1e-3.

    Returns:
        1-D numpy array: Evaluated delta function values at points x
    """
    return 1/np.sqrt(2*np.pi*np.abs(spread)**2)*np.exp(-(x-shift)**2/(2*np.abs(spread)**2))
