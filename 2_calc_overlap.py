import numpy as np
from math import factorial
import scipy, scipy.constants
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from classes import *


if __name__ == "__main__":
    # read this in later
    w_i = 0.1
    w_f = 0.1
    dq = 0.2
    energy_shift = 0.05
    n_max = 50

    freq_range = np.linspace(-0.1, n_max*max(w_i,w_f), int(1e6))
    deltas = np.zeros((n_max, len(freq_range)))
    overlaps = np.zeros(n_max)
    for i in range(n_max):
        overlaps[i] = overlap(0,i,[w_i,w_f], dq)
        deltas[i] = delta(freq_range, shift=(-energy_shift + i*w_f))
        if overlaps[i] < 1e-12:
            break

    for i in range(n_max):
        ol = overlap(i, i, [w_i,w_f])
        print(f"Overlap {i}: {ol}")


    fermi_rules = np.dot(overlaps, deltas)
    plt.plot(freq_range, fermi_rules)
    plt.vlines(0, -5, np.max(fermi_rules)*1.5, color="red")
    plt.ylim([0,np.max(fermi_rules)*1.1])
    plt.show()
    exit()
    print(f"Overlap {i}: {overlaps[i]}")


    w = 0.1 # 1 Hartree
    q_range = np.linspace(-1, 1, int(1e12))
    for i in range(3+1):
        for j in range(3+1):
            print("Overlap ({},{}): {}".format(i,j,overlap(i,j, w)))


