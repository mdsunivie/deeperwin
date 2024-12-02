import matplotlib.pyplot as plt
import numpy as np
from utils import load_from_file, morse_potential, fit_morse_potential

distances = [2.6, 2.8, 3.0, 3.2, 3.4, 3.6]
# distances = np.arange(1.2, 1.81, 0.1)
FxLi_values = []
FxH_values = []
energies = []
sigma_energies = []
plt.close("all")
plot_evaluations = True

def calculate_mean_without_outliers(x, q=0):
    x_min = np.quantile(x, q)
    x_max = np.quantile(x, 1-q)
    return np.mean(x[(x_min <= x) & (x <= x_max)])

if plot_evaluations:
    plt.figure(figsize=(14,7))

for i,dist in enumerate(distances):
    # data = load_from_file(f"/home/mscherbela/runs/forces/LiH_highres/LiH_eval_highres_{dist:.1f}_True_False/results.bz2")
    # data = load_from_file(f"/home/mscherbela/runs/forces/H2_sweep/H2_baseline_{dist:.1f}/results.bz2")
    data = load_from_file(f"/home/mscherbela/runs/forces/LiH_noshift/LiH_noshift_{dist:.1f}/results.bz2")

    force_history = np.array(data['metrics']['forces'])

    FxLi = force_history[:,0,0]
    FxH = force_history[:,1,0]
    FxLi_values.append(calculate_mean_without_outliers(FxLi)*1000)
    FxH_values.append(calculate_mean_without_outliers(FxH)*1000)
    energies.append(float(data['metrics']['E_mean']))
    sigma_energies.append(float(data['metrics']['E_mean_sigma']))

    if plot_evaluations:
        plt.subplot(2,4,i+1)
        plt.plot(FxLi, alpha=0.4, color='C0')
        plt.plot(FxH, alpha=0.4, color='C1')
        plt.axhline(np.mean(FxLi), color='C0')
        plt.axhline(np.mean(FxH), color='C1')
        plt.title(str(dist))
        plt.grid()



FxLi_values = np.array(FxLi_values)
FxH_values = np.array(FxH_values)
F_attraction = (FxLi_values - FxH_values) * 0.5
F_average = (FxLi_values + FxH_values) / 2

morse_params = fit_morse_potential(distances, energies)
d_plot = np.linspace(min(distances), max(distances), 200)
E_morse = morse_potential(d_plot, *morse_params)
F_attraction_morse = 1000 * np.gradient(E_morse) / (d_plot[1] - d_plot[0])

plt.figure(figsize=(14,7))
plt.subplot(1,2,1)
plt.plot(distances, FxLi_values, label='Li', alpha=0.5)
plt.plot(distances, FxH_values, label='H', alpha=0.5)
plt.plot(distances, F_attraction, label='attraction', lw=2)
plt.plot(distances, F_average, label='Total force', alpha=0.5)
plt.plot(d_plot, F_attraction_morse, label='Attraction morse', color='gray', lw=2)
plt.axhline(0, color='k')
plt.grid()
plt.legend()

plt.subplot(1,2,2)
plt.errorbar(distances, energies, yerr=sigma_energies)
plt.plot(d_plot, E_morse, color='gray')
plt.grid()






