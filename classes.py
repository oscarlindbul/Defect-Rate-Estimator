import numpy as np
from constants import *
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import curve_fit
import scipy.integrate as integrate

class State:

    def __init__(self, Q_energy, effective_freq=None, label=None, soc=None, dipmoment=None, elph=None):
        self.Qs = [entry[0] for entry in Q_energy]
        self.energies = [entry[1] for entry in Q_energy]
        min_E_ind = np.argmin(self.energies)
        self.min_E = min(self.energies)
        self.eq_Q = self.Qs[min_E_ind]
        self.socs = soc
        self.label = label
        if effective_freq is None:
            model = lambda x, w, E0: w**2*(x - self.eq_Q)**2/2 + E0
            params = curve_fit(model, self.Qs, self.energies, (1,self.min_E))
            w = np.sqrt(params[0]*hbar/au_to_kg)*1e10 # in hartree
            self.freq = w
        else:
            self.freq = effective_freq
        self.dipole_moment = dipmoment
        self.elph = None

    def covering_q_range(self, norm=5):
        alpha = self.freq/(hbar*(hbar*J_to_hartree)) * au_to_kg * 1e-20 # in (au A^2)^-1
        #print(self.freq, self.energies, self.Qs, self.eq_Q, alpha)
        return np.array([-1,1])*(norm/np.sqrt(alpha)) + self.eq_Q
    
    def plot_Q_curve(self, color="blue", offset=0):
        w = self.freq / (J_to_hartree * hbar)
        Q = self.eq_Q
        model = lambda x: J_to_hartree*w**2*((x - Q) * np.sqrt(au_to_kg)*1e-10)**2/2 + self.min_E
        q_range = self.covering_q_range()
        x_vals = np.linspace(q_range[0], q_range[1], 1000)
        plt.plot(x_vals, model(x_vals)-offset, color=color)

    def plot_eigenmode(self, n, normalization=1, color="blue", offset=0):
        q_range = self.covering_q_range()
        x_vals = np.linspace(q_range[0], q_range[1], 1000)
        mode = normalization*qosc_eigenfunc(x_vals, self.freq, n, self.eq_Q) + self.min_E + n*self.freq - offset + self.freq/2
        plt.plot(x_vals, mode, linewidth=0.8, color=color)
        plt.plot(x_vals, (self.min_E + (n+1/2)*self.freq - offset)*np.ones(x_vals.shape), "--", linewidth=0.5, color=color)

class Transition:

    def __init__(self, init_state, final_state, coupling, trans_type):
        self.state_i = init_state
        self.state_f = final_state
        self.couplings = np.array(coupling)
        self.trans_type = trans_type

    def calc_huangrhys(self):
        dQ2 = (self.state_i.eq_Q - self.state_f.eq_Q)**2 * (au_to_kg)*1e-20
        w = np.mean([self.state_f.freq, self.state_i.freq])/(hbar*J_to_hartree)
        return dQ2*w/(2*hbar)


    def gen_linear_coupling(self, q_vals):
        coupling_vals = np.zeros((self.couplings.shape[0], len(q_vals)))
        for i in range(self.couplings.shape[0]):
            print("debug")
            print((self.state_i.eq_Q, self.state_f.eq_Q), (self.couplings[i,0], self.couplings[i,1]))
            fit = np.polyfit((self.state_i.eq_Q, self.state_f.eq_Q), (self.couplings[i,0], self.couplings[i,1]), deg=1)
            f = lambda x: fit[0]*x + fit[1]
            coupling_vals[i] = f(q_vals)
        return coupling_vals


    def calc_rate(self, nmax = 40, overlap_threshold=1e-6, neval=int(1e6), n_refr=None, spread=1e-3, plot=False):

        ### prepare entities
        eq_Q_i = self.state_i.eq_Q
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
        #print(w_i, w_f, self.couplings, ranges, np.min(q_range), np.max(q_range))
        ##plt.plot(q_range, coupling_vals[0])
        #plt.show()
        
        ### perform overlap sum
        dE = self.state_i.min_E - self.state_f.min_E
        overlaps = np.zeros(nmax)
        if self.trans_type == "PL":
            energies = np.linspace(dE-nmax*w_f-5*float(np.abs(spread)),dE+5*float(np.abs(spread) + w_f),neval)
        else:
            energies = np.linspace(-w_f, w_f,neval)
        fermis = np.zeros((coupling_vals.shape[0], nmax, len(energies)))
        for i in range(coupling_vals.shape[0]):
            for n in range(nmax):
                overlap_val = overlap(0, n, [w_i, w_f], dq=np.abs(eq_Q_f - eq_Q_i), operand=coupling_vals[i], q_range=q_range)
                overlaps[n] = overlap_val
                #print("n={}, {}".format(n,overlap_val))
                if overlap_val == 0:# or n > 0 and overlap_val < overlaps[0]*overlap_threshold:
                    break
                delta_vals = delta(energies, shift=(dE - n*w_f), spread=spread)
                ##plt.plot(energies, delta_vals)
                #plt.show()
                fermis[i,n,:] = overlap_val * delta_vals

        ### Calculate fermi rule
        if self.trans_type.upper() == "PL":
            if n_refr is None:
                print("Transition is type PL, but missing refractive index")
                return None
            for i in range(fermis.shape[0]):
                for j in range(fermis.shape[1]):
                    plt.plot(energies, fermis[i,j,:])
            plt.show()
            print(fermis)
            fermis *= debye_to_Cm**2 * J_to_hartree # dipole moments now in SI
            optical_prefactor = 1/J_to_hartree * (energies/(hbar*J_to_hartree))**3 * n_refr**2 / (3*np.pi * hbar * c**3 * eps_0) # distribution in (hartree * s)^-1
            optical_prefactor[energies < 0] = 0
            fermis *= optical_prefactor
            total_fermi = np.sum(fermis, axis=1)
            total_rate = np.zeros(total_fermi.shape[0])
            ZPL_rate = np.zeros(total_fermi.shape[0])
            for i in range(total_rate.shape[0]):
                total_rate[i] = integrate.simpson(total_fermi[i,energies >= 0], energies[energies >= 0])
                ZPL_rate[i] = integrate.simpson(fermis[i,0,:], energies)
            print(f"ZPL rate: {ZPL_rate}")
            #closest_to_ZPL = np.argmin(np.abs(energies - dE))
            #model_rate = optical_prefactor[closest_to_ZPL] * J_to_hartree * coupling_vals[0,0]
            #print(f"ZPL rate (max): {model_rate}")
            print(f"total rate: {total_rate}")
            return total_rate, total_fermi, fermis, energies

        elif self.trans_type.upper() == "ISC":
            fermis *= 2*np.pi/(hbar*J_to_hartree)
            total_fermi = np.sum(fermis, axis=1)
            closest_to_zero = np.argmin(np.abs(energies))
            rate = total_fermi[:,closest_to_zero]
            return rate, total_fermi, fermis, energies

        elif self.trans_type.upper() == "IC":
            print(f"Delta E: {dE}")
            print("Overlaps")
            print(overlaps)
            fermis *= 2*np.pi/(hbar*J_to_hartree) * dE**2
            total_fermi = np.sum(fermis, axis=1)
            closest_to_zero = np.argmin(np.abs(energies))
            rate = total_fermi[:,closest_to_zero]
            return rate, total_fermi, fermis, energies

        else:
            print("Unkown transition type specified")
            return None

        

    def PL_dist(self, n_refr):
        ### prepare entities
        eq_Q_i = self.state_i.eq_Q
        eq_Q_f = self.state_f.eq_Q
        w_i = self.state_i.freq * hbar * J_to_hartree
        w_f = self.state_f.freq * hbar * J_to_hartree
        
        ### prepare Q range
        ranges = np.array([self.state_i.covering_q_range(), self.state_f.covering_q_range()])
        q_range = np.linspace(np.min(ranges,axes=(0,1)), np.max(ranges, axes=(0,1)), neval)
        tdm_vals = np.zeros(2)
        tdm_fit = np.polyfit((eq_Q_i, eq_Q_f), (self.state_i.dipole_moment, self.state_f.dipole_moment), deg=1)
        f = lambda x: tdm_fit[0]*x + tdm_fit[1]
        tdm_vals = f(q_range)
        
        ### perform overlap sum
        dE = self.state_i.min_E - self.state_f.min_E
        overlaps = np.zeros((nmax, len(q_range)))
        energies = np.linspace(0, dE*1.1, 5000)
        fermis = np.zeros((nmax, len(energies)))
        for n in range(nmax):
            overlap_val = overlap(0,n, [w_i, w_f], dq=eq_Q_f - eq_Q_i, operand=tdm_vals, q_range=q_range)
            if overlap_val < 1e-10:
                break

            delta_vals = delta(energies, shift=(n*w_f - dE))
            fermis[n,:] = overlap_val * delta_vals

        ### Calculate fermi rule
        c = scipy.constants.c
        eps_0 = scipy.constants.epsilon_0
        optical_prefactor = energies**3 * n_refr**2 / (3*np.pi * hbar**4 * c**3 * eps_0)
        total_fermi = optical_prefactor * np.sum(fermis, axis=0)
        if plot_map:
            #plt.plot(energies, total_fermi)
            plt.show()
        total_rate = integrate.simpson(energies, total_fermi)
        return total_fermi, total_rate

    def ISC_rate(self, nmax = 40, neval=int(1e6), plot_map=False):
        ### prepare entities
        eq_Q_i = self.state_i.eq_Q
        eq_Q_f = self.state_f.eq_Q
        w_i = self.state_i.freq * hbar * J_to_hartree
        w_f = self.state_f.freq * hbar * J_to_hartree
        
        ### prepare Q range
        ranges = np.array([self.state_i.covering_q_range(), self.state_f.covering_q_range()])
        q_range = np.linspace(np.min(ranges,axes=(0,1)), np.max(ranges, axes=(0,1)), neval)
        soc_vals = np.zeros((len(self.state_i.socs), 2))
        for soc_i in range(len(self.state_i.socs)):
            soc_fit = np.polyfit((eq_Q_i, eq_Q_f), (self.state_i.socs[soc_i], self.state_f.socs[soc_i]), deg=1)
            f = lambda x: soc_fit[0]*x + soc_fit[1]
            soc_vals[soc_i] = np.abs(f(q_range))
        
        ### perform overlap sum
        dE = self.state_i.min_E - self.state_f.min_E
        overlaps = np.zeros((soc_vals.shape[0], nmax, len(q_range)))
        energies = np.linspace(-0.1,0.1,5000)
        fermis = np.zeros((soc_vals.shape[0], nmax, len(energies)))
        for i in range(len(soc_vals.shape[0])):
            for n in range(nmax):
                overlap_val = overlap(0,n, [w_i, w_f], dq=eq_Q_f - eq_Q_i, operand=soc_vals[i], q_range=q_range)
                if overlap_val < 1e-10:
                    break
                delta_vals = delta(energies, shift=(n*w_f - dE))
                fermis[i,n,:] = overlap_val * delta_vals

        ### Calculate fermi rule
        total_fermi = (2*np.pi)**2/h * np.sum(fermis, axis=1)
        #if plot_map:
        #    for i in range(soc_vals.shape[0]):
                #plt.plot(energies, total_fermi[i,:])
            #plt.show()
        closest_to_zero = np.argmin(np.abs(energies))
        rate = total_fermi[:,closest_to_zero]
        return rate
        

    def IC_rate(self):
        ### prepare entities
        eq_Q_i = self.state_i.eq_Q
        eq_Q_f = self.state_f.eq_Q
        w_i = self.state_i.freq * hbar * J_to_hartree
        w_f = self.state_f.freq * hbar * J_to_hartree
        
        ### prepare Q range
        ranges = np.array([self.state_i.covering_q_range(), self.state_f.covering_q_range()])
        q_range = np.linspace(np.min(ranges,axes=(0,1)), np.max(ranges, axes=(0,1)), neval)
        elph_vals = np.zeros(2)
        elph_fit = np.polyfit((eq_Q_i, eq_Q_f), (self.state_i.elph, self.state_f.elph), deg=1)
        f = lambda x: tdm_fit[0]*x + tdm_fit[1]
        elph_vals = f(q_range)
        
        ### perform overlap sum
        dE = self.state_i.min_E - self.state_f.min_E
        overlaps = np.zeros((nmax, len(q_range)))
        energies = np.linspace(-0.1, 0.1, 5000)
        fermis = np.zeros((nmax, len(energies)))
        for n in range(nmax):
            overlap_val = overlap(0,n, [w_i, w_f], dq=eq_Q_f - eq_Q_i, operand=elph_vals, q_range=q_range)
            if overlap_val < 1e-10:
                break

            delta_vals = delta(energies, shift=(n*w_f - dE))
            fermis[n,:] = overlap_val * delta_vals

        ### Calculate fermi rule
        total_fermi = 2*np.pi/hbar * dE**2 * np.sum(fermis, axis=0)
        if plot_map:
            #plt.plot(energies, total_fermi)
            plt.show()
        closest_to_zero = np.argmin(np.abs(energies))
        rate = total_fermi[:,closest_to_zero]
        return rate

herm_cache = [np.array([1])]
def herm(x,n):
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
    # q in A(au)^1/2
    # w in Hartree
    #w_eV = w*hartree_to_eV
    alpha = w/(hbar*J_to_hartree)/hbar * au_to_kg * 1e-20 # in (au A^2)^-1
    herm_val = herm(np.sqrt(alpha)*(q-q0), n)
    factorial_list = np.arange(1,n+1)
    # bake prefactor into exponent for more convenient evaluation
    exponent = -alpha/2 * (q-q0)**2 - np.sum(np.log(factorial_list))/2 - np.log(2)*n/2
    exp_val = np.exp(exponent)
    #prefactor = 1/(np.sqrt(2**n)*np.sqrt(factorial(n)))
    return  (alpha/np.pi)**(0.25) * exp_val * herm_val

def overlap(n_i,n_f, w, dq=0, operand=None, q_range=None, neval=None, analytical=False):
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
        alpha = 2*np.pi*w_eV/(h*h_eV) * au_to_kg * 1e-20 # in (au A^2)^-1
        max_range = 10/np.sqrt(alpha)
        q_range = np.linspace(-max_range, max_range, neval)
    if not analytical:
        eig_f = qosc_eigenfunc(q_range,w_i,n_i)
        eig_i = qosc_eigenfunc(q_range,w_f,n_f,dq)

        product = integrate.simpson(eig_i*operand*eig_f, q_range)
    else:
        pass
        product,_ = integrate.quad(lambda x: qosc_eigenfunc(x,w_i,n_i)*qosc_eigenfunc(x,w_f,n_f,dq), np.min(q_range), np.max(q_range))
    return np.abs(product)**2

def delta(x, shift=0, spread=1e-3):
    return 1/np.sqrt(2*np.pi*np.abs(spread)**2)*np.exp(-(x-shift)**2/(2*np.abs(spread)**2))
