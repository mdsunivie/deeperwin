import numpy as np
import matplotlib.pyplot as plt




polynomial_degree = 4
R_core = 0.5
j = np.arange(1, polynomial_degree + 1)
A = R_core ** 2 / (2 + j[np.newaxis, :] + j[:, np.newaxis] + 1)
b = 1 / (j + 1)
coeff = np.linalg.solve(A, b)
coeff = coeff[:, np.newaxis]


n_points = 500
r = np.linspace(0, R_core, n_points)
force_raw = 1/r**2

rn = (r/R_core)**j[:, np.newaxis]
force_poly = np.sum(coeff * rn, axis=0)

R_cut = 0.1
r_cut = r/np.tanh(r/R_cut)
force_tanh_cut = r / r_cut**3


plt.close("all")
plt.plot(r, force_raw, label="Raw")
plt.plot(r, force_poly, label="Poly")
plt.plot(r, force_tanh_cut, label="Tanh cut")
plt.grid()
plt.legend()
plt.ylim([0, 100])




