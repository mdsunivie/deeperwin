import numpy as np
import matplotlib.pyplot as plt

n_epochs = 2 ** np.arange(8, 15)
E_deeperwin = [-78.5484390258789,
               -78.5608978271484,
               -78.565803527832,
               -78.5704498291016,
               -78.5750045776367,
               -78.5772705078125,
               -78.5785675048828
               ]

E_ref = {'Hartree-Fock':-78.07071901,
         'CASSCF':-78.09774858,
         'CCSD(T)-F12': -78.47551815,
         'MRCI-F12':-78.53646845,
         'MRCI-D-F12':-78.57744597,
}
E0 = np.min(E_deeperwin) - 1e-3



plt.close("all")
plt.figure(figsize=(4,4))

plt.loglog(n_epochs, E_deeperwin - E0, label='DeepErwin', lw=3, color='k')
for i, (method, E) in enumerate(E_ref.items()):
    plt.axhline(E - E0, label=method, color=f'C{i}')
plt.legend(loc='upper right', framealpha=1.0)

ax = plt.gca()
xticks = 2 ** np.arange(8, 15, 2)
ax.set_xticks(xticks)
ax.set_xticklabels([f"{x:,d}" for x in xticks])
ax.set_xticks(2 ** np.arange(8, 15, 1), minor=True)
ax.grid(alpha=0.7, color='gray', which='major')
ax.grid(alpha=0.4, ls='--', color='gray', which='minor')
plt.xlabel("Optimization steps")
plt.ylabel("Energy error / Ha")
plt.title("Groundstate energy of Ethene\nDeepErwin v. established methods")
plt.tight_layout()
fname = "/home/mscherbela/ucloud/results/DeepErwin_vs_established_Ethene_kfac_indep.png"
plt.savefig(fname, dpi=800)
plt.savefig(fname.replace(".png", ".pdf"))


