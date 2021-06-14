import numpy as np
import scipy.optimize


def morsePotential(r, E, a, Rb, E0):
    """
    Returns the Morse potential :math:`E_0 + E (exp{-2x} - 2exp{-x})` where :math:`x = a (r-R_b)`
    """
    x = a * (r-Rb)
    return E*(np.exp(-2*x) - 2*np.exp(-x)) + E0


def fitMorsePotential(d, E, p0=None):
    """
    Fits a morse potential to given energy data.

    Args:
        d (list, np.array): array-like list of bondlengths
        E (list, np.array): array-like list of energies for each bond-length
        p0 (tuple): Initial guess for parameters of morse potential. If set to None, parameters will be guessed based on data.

    Returns:
        (tuple): Fit parameters for Morse potential. Can be evaluated using morsePotential(r, *params)
    """
    if p0 is None:
        p0 = (0.1, 1.0, np.mean(d), -np.min(E)+0.1)
    morse_params = scipy.optimize.curve_fit(morsePotential, d, E, p0=p0)[0]
    return tuple(morse_params)


def lennardJonesPotential(r, E, a, sigma, E0):
    """
    Returns the Lennard Jones potential for given bond-lengths and parameters.
    """
    return E0 + (E / (1-6/a)) * (6*np.exp(a*(1-r/sigma))/a - (sigma/r)**6)