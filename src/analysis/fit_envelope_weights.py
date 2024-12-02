import numpy as np
import matplotlib.pyplot as plt

def _get_effective_charge(Z: int, n: int):
    """
    Calculates the approximate effective charge for an electron, using Slater's Rule of shielding.

    Args:
        Z: Nuclear charge of an atom
        n: Principal quantum number of the elctron for which the effective charge is being calculated

    Returns:
        Z_eff (float): effective charge
    """
    shielding = 0
    for n_shell in range(1, n+1):
        n_electrons_in_shell = 2 * n_shell ** 2
        if n_shell == n:
            n_el_in_lower_shells = sum([2*k**2 for k in range(1,n_shell)])
            n_electrons_in_shell = min(Z - n_el_in_lower_shells, n_electrons_in_shell) - 1
        if n_shell == n:
            shielding += n_electrons_in_shell * 0.3
        elif n_shell == (n-1):
            shielding += n_electrons_in_shell * 0.85
        else:
            shielding += n_electrons_in_shell
    return Z - shielding

# P
Z = [15] * 9
n = [1, 2, 2, 2, 2, 3, 3, 3, 3]
l = [0, 0, 1, 1, 1, 0, 1, 1, 1]
alpha = [10, 2.2, 2.2, 2.2, 2.2, 0.9, 0.45, 0.45, 0.45]

# Cl
Z += [17] * 9
n += [1, 2, 2, 2, 2, 3, 3, 3, 3]
l += [0, 0, 1, 1, 1, 0, 1, 1, 1]
alpha += [10, 1.7, 1.7, 1.7, 1.7, 1.0, 0.7, 0.7, 0.7]

# O
Z += [8] * 5
n += [1, 2, 2, 2, 2]
l += [0, 0, 1, 1, 1]
alpha += [5, 1.3, 0.9, 0.9, 0.9]

Z = np.array(Z)
n = np.array(n)
l = np.array(l)

Z_eff_values = np.array([_get_effective_charge(Z_, n_) for Z_, n_ in zip(Z, n)])
alpha_analytical = Z_eff_values / n
alpha_analytical_scaled = alpha_analytical * 0.5

plt.close("all")
plt.figure(dpi=100)
plt.scatter(alpha, alpha_analytical)
plt.scatter(alpha, alpha_analytical_scaled)
plt.plot([0, 10], [0, 10], color='k', ls='--')
plt.grid(alpha=0.5)
plt.xlabel("Learned from constant init")
plt.ylabel("Analytical")



