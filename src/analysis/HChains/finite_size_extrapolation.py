#%%
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit, polyval

n = np.array([12, 16, 20])
E_tot = np.array([-5.952, -8.0088, -10.031])
E = E_tot / n

plt.close("all")
plt.plot(1/n, E, marker='o')
coeffs = polyfit(1/n, E, 2)
x_plot = np.linspace(0, 1/12, 100)
plt.plot(x_plot, polyval(x_plot, coeffs))

E_40 = polyval(1/40, coeffs) * 40
print(f"E_40 = {E_40:.3f} Ha")
