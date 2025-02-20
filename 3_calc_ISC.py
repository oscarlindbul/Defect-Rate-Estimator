import numpy as np
from classes import *
from constants import *
import matplotlib.colors as mc
from colorsys import rgb_to_hls, hls_to_rgb
from mpl_toolkits.axisartist.axislines import AxesZero

def read_transitions(filename):
    print(f"Reading datafile {filename}")
    with open(filename, "r") as reader:
        lines = reader.readlines()
    for line in lines:
        if line.strip()[0] == "#":
            continue
        data = line.strip().split()
        trans_type = data[0]
        labels = data[1:3]
        Qs = list(map(float, data[3:5]))
        state1_E = list(map(float,data[5:7]))
        state2_E = list(map(float, data[7:9]))
        freqs = np.array(list(map(float, data[9:11]))) * hbar * J_to_hartree # in hartree
        couplings = np.array(list(map(float, data[11:])))
        print(f"Reading couplings for transition of type {trans_type}")
        if trans_type == "ISC":
            couplings *= 100*c*h*J_to_hartree # hartrees
            print(f"{couplings} Hartree")
        elif trans_type == "IC":
            couplings *= 1 # electron overlap has unit 1?
            print(f"{couplings} [1]")
            print(Qs, labels, trans_type)
            print(state1_E)
            print(state2_E)
            print(couplings)
        elif trans_type == "PL":
            couplings = np.sqrt(couplings)
            couplings *= au_to_debye # dipole moments in Debye
            print(f"{couplings} Debye")
        else:
            print("Unknown coupling type")
            exit()
        if len(couplings) % 2:
            print("Wrong number of coupling terms in file")
            return None
        n_couplings = len(couplings) // 2

        state_i = State(list(zip(Qs, state1_E)), freqs[0], label=labels[0])
        state_f = State(list(zip(Qs, state2_E)), freqs[1], label=labels[1])

        transition = Transition(state_i, state_f, couplings.reshape((n_couplings,2)), trans_type)
        yield transition

def make_level_plot(filename, included_modes=4):
    ground_color = rgb_to_hls(*mc.to_rgb("blue"))
    ex_color = rgb_to_hls(*mc.to_rgb("orange"))
    transitions = list(read_transitions(filename))
    for trans in transitions:

        fig = plt.figure(figsize=(8,8))
        for n in range(included_modes):
            g_color = tuple([ground_color[i] + (1-ground_color[i])*n/included_modes if i == 1 else ground_color[i] for i in range(len(ground_color))])
            e_color = tuple([ex_color[i] + (1-ex_color[i])*n/included_modes if i == 1 else ex_color[i] for i in range(len(ex_color))])
            trans.state_i.plot_eigenmode(n, 1e-3, color=hls_to_rgb(*e_color), offset=trans.state_f.min_E)
            trans.state_f.plot_eigenmode(n, 1e-3, color=hls_to_rgb(*g_color), offset=trans.state_f.min_E)
            q_i_range = trans.state_i.covering_q_range(norm=3.5)
            q_f_range = trans.state_f.covering_q_range(norm=3)

            E_i_mode = trans.state_i.min_E + trans.state_i.freq * (n+1/2) - trans.state_f.min_E
            E_f_mode = trans.state_f.min_E + trans.state_f.freq * (n+1/2) - trans.state_f.min_E
            plt.annotate(f"$n={n}$", (q_f_range[0], E_f_mode - 0.4*(trans.state_f.freq)))
            plt.annotate(f"$m={n}$", (q_i_range[0], E_i_mode + 0.15*(trans.state_f.freq)))
        trans.state_i.plot_Q_curve(color=hls_to_rgb(*ex_color), offset=trans.state_f.min_E)
        trans.state_f.plot_Q_curve(color=hls_to_rgb(*ground_color), offset=trans.state_f.min_E)
        plt.title(f"{trans.trans_type}: {trans.state_i.label} $\\rightarrow$ {trans.state_f.label}")
        plt.xlabel("Configuration coordinate $(\\text{amu})^{1/2}\\AA$")
        plt.ylabel("Energy (Hartree)")
        
        x_lims = np.array([state.covering_q_range(norm=3.25) for state in [trans.state_i,trans.state_f]])
        x_lims = [np.min(x_lims, axis=(0,1)), np.max(x_lims, axis=(0,1))]
        plt.xlim(x_lims)
        y_lims = np.array([-0.05, 1])*(E_i_mode+trans.state_i.freq)
        plt.ylim(y_lims)
        plt.savefig(f"levels_{trans.trans_type}_{trans.state_i.label}_{trans.state_f.label}.pdf")
        plt.show()
        fig.clf()

def make_coupling_plot(filename, ylabel, plot_labels, scaling=1):
    transitions = list(read_transitions(filename))
    for trans in transitions:
        fig = plt.figure()
        ax = fig.add_subplot(axes_class=AxesZero)
        for direction in ["xzero", "yzero"]:
            ax.axis[direction].set_axisline_style("-|>")
            ax.axis[direction].set_visible(True)
            ax.axis[direction].label.set_visible(False)

        for direction in ["left", "right", "bottom", "top"]:
            ax.axis[direction].set_visible(False)

        x_lims = np.array([state.covering_q_range() for state in [trans.state_i,trans.state_f]])
        x_lims = [np.min(x_lims, axis=(0,1)), np.max(x_lims, axis=(0,1))]
        x_vals = np.linspace(*x_lims, 500)
        couplings = trans.gen_linear_coupling(x_vals)*scaling
        for i in range(couplings.shape[0]):
            plt.plot(x_vals, couplings[i,:], label=plot_labels[i])
        plt.legend()
        plt.ylabel(ylabel)
        ax.set_axis_off()
        ax.text(np.max(x_vals),-(np.max(couplings) - np.min(couplings))*0.1, "$Q \\thinspace (\\text{amu})^{1/2}\\AA$")
        ax.text(-(np.max(x_vals)-np.min(x_vals))*0.05, np.max(couplings)*1.05, ylabel, horizontalalignment="right")
        plt.xlabel("$Q \\thinspace (\\text{amu})^{1/2}\\AA$")
        plt.savefig(f"couplings_{trans.trans_type}_{trans.state_i.label}_{trans.state_f.label}.pdf")
        plt.show()
        fig.clf()

def calculate_transition_rates(filename, data_filename, n_refr = None, maximize=False, neval=int(1e6), plot=False):
    data_file = open(data_filename, "w")
    for trans in read_transitions(filename):
    
        # find optimal delta spread
        def func_eval(spread):
            rate,_,_,_ = trans.calc_rate(spread=spread, nmax=10, n_refr=n_refr)
            if len(rate) > 1:
                rate = np.max(rate)
            print(rate)
            return -rate
        initial_guess = 1e-2
        if trans.trans_type == "PL":
            initial_guess = 1e-3
        if maximize:
            max_point = scipy.optimize.fmin(func_eval, initial_guess, disp=False)
            optimal_spread = max_point[0]
        else:
            optimal_spread = trans.state_f.freq*2/2.355
        if optimal_spread > 1 or optimal_spread < 1e-6:
            print(f"Optimal delta width {optimal_spread} Hartree too big")
            optimal_spread = 1e-3
        print(f"Using optimal spread {optimal_spread}")
        
        rate, spectrum, partial_spectras, energies = trans.calc_rate(spread=optimal_spread, nmax=40, n_refr = n_refr, neval=neval, plot=plot)
        
        closest_to_zero_mid = np.argmin(np.abs(energies))
        closest_to_zero_small = np.argmin(np.abs(energies + 0.1/hartree_to_eV))
        closest_to_zero_big = np.argmin(np.abs(energies - 0.1/hartree_to_eV))
        partial_rates = np.zeros(partial_spectras.shape[:2])
        if trans.trans_type == "PL":
            for i in range(partial_rates.shape[0]):
                for j in range(partial_rates.shape[1]):
                    partial_rates[i,j] = integrate.simpson(partial_spectras[i,j, energies >= 0], x = energies[energies >= 0])
        else:
            partial_rates_mid = partial_spectras[:,:,closest_to_zero_mid]
            partial_rates_small = partial_spectras[:,:,closest_to_zero_small]
            partial_rates_big = partial_spectras[:,:,closest_to_zero_big]
            rate = spectrum[:, closest_to_zero_mid]
            small_rate = spectrum[:, closest_to_zero_small]
            big_rate = spectrum[:, closest_to_zero_big]
            for i in range(partial_spectras.shape[0]):
                print(f"Transition {trans.state_i.label} to {trans.state_f.label} with couplings {trans.couplings[i,:]}", file=data_file)
                for j in range(partial_rates.shape[1]):
                    print(f"n={j}   {partial_rates_mid[i,j]} {partial_rates_small[i,j]} {partial_rates_big[i,j]}", file=data_file)
                print(f"Total rate {rate[i]}, {small_rate[i]} {big_rate[i]} Hz", file=data_file)
        print(rate)
            
        plt.plot(energies, spectrum[0])
        for i in range(partial_spectras.shape[1]):
            plt.plot(energies, partial_spectras[0,i,:])
        plt.show()
    data_file.close()


if __name__ == "__main__":
    #make_level_plot("ICs.txt", included_modes=10)
    #make_coupling_plot("ICs.txt", "El-phonon Coupling ($W$)", ["W"])
    calculate_transition_rates("ICs.txt", "IC_data.txt", n_refr = 2.42, neval=int(1e6), plot=True) 


        

            
