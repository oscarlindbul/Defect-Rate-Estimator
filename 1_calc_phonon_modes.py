import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("Zsolt_data/NV_analysis/MODEL3_s1_t1_irrep0grid_data1.txt", delimiter=",")
structures = 

x = data[:,0]
E1 = data[:,1]
E2 = data[:,2]
E3 = data[:,3]

coeffs_E1 = np.polyfit(x, E1, deg=2)
coeffs_E2 = np.polyfit(x, E2, deg=2)

equilib1 = -coeffs_E1[1]/(2*coeffs_E1[0])
equilib2 = -coeffs_E2[1]/(2*coeffs_E2[0])


print(f"Equilibrium points: {equilib1}, {equilib2}")

func1 = sum([p*x**(2-i) for i,p in enumerate(coeffs_E1)])
func2 = sum([p*x**(2-i) for i,p in enumerate(coeffs_E2)])



plt.plot(x, E1)
plt.plot(x, func1, "--")
plt.plot(x, E2)
plt.plot(x, func2, "--")
plt.plot(x, E3)
plt.show()

