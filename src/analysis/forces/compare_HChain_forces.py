import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deeperwin.checkpoints import load_from_file
import glob

def calculate_forces_from_finite_diff(fname, method):
    df = pd.read_csv(fname, delimiter=';')
    df['a'] = df.geom.apply(lambda g: float(g.split('_')[1]))
    df['x'] = df.geom.apply(lambda g: float(g.split('_')[2]))
    df['da'] = df.geom.apply(lambda g: float(g.split('_')[4]))
    df['dx'] = df.geom.apply(lambda g: float(g.split('_')[5][:-4]))
    eps = df.da.max()


    force_data = []
    for i, r in df.iterrows():
        if (r.dx != 0) or (r.da != 0):
            continue
        E = r[method]
        E_xp = df[(df.a == r.a) & (df.x == r.x) & (df.dx > 0)].iloc[0][method]
        E_xm = df[(df.a == r.a) & (df.x == r.x) & (df.dx < 0)].iloc[0][method]
        E_ap = df[(df.a == r.a) & (df.x == r.x) & (df.da > 0)].iloc[0][method]
        E_am = df[(df.a == r.a) & (df.x == r.x) & (df.da < 0)].iloc[0][method]
        Fx = np.nanmean([E - E_xp, E_xm - E]) / eps
        Fa = np.nanmean([E - E_ap, E_am - E]) / eps
        # Fx = np.mean([E - E_xp, E_xm - E]) / eps
        # Fa = np.mean([E - E_ap, E_am - E]) / eps
        force_data.append(dict(a=r.a, x=r.x, Fx_ref=Fx, Fa_ref=Fa, E_ref=E))
    return pd.DataFrame(force_data)

def get_forces_from_dpe(fname):
    data = load_from_file(fname)
    F_mean = data['metrics']['forces_mean']
    F_t = data['metrics']['forces']
    comment = data['config']['physical.comment']
    if (comment is None) or (comment == ''):
        original_directory = data['config']['restart.path']
        original_directory = original_directory.replace("/home/lv71376/scherbelam/", "/home/mscherbela/")
        original_data = load_from_file(original_directory + "/results.bz2")
        R = original_data['config']['physical.R']
    else:
        R = data['config']['physical.R']
    a = np.round(R[1][0] - R[0][0], 2)
    x = np.round(R[2][0] - R[1][0], 2)

    n_particles = len(F_mean)
    jacobi_mat = np.zeros([n_particles, 2])
    jacobi_mat[0::2, 0] = np.arange(n_particles // 2)
    jacobi_mat[1::2, 0] = np.arange(n_particles // 2) + 1
    jacobi_mat[0::2, 1] = np.arange(n_particles // 2)
    jacobi_mat[1::2, 1] = np.arange(n_particles // 2)

    F_t = F_t - np.mean(np.mean(F_t, axis=0), axis=0) # average over time and atoms
    Fa_std, Fx_std = np.std(F_t[...,0], axis=0) @ jacobi_mat

    Fa, Fx = (F_mean[:, 0] - np.mean(F_mean[:, 0])) @ jacobi_mat
    return dict(a=a, x=x, Fa=Fa, Fx=Fx, Fa_std=Fa_std, Fx_std=Fx_std)

if __name__ == '__main__':
    method = 'MRCI-D-F12'
    df_ref = calculate_forces_from_finite_diff("/home/mscherbela/runs/references/H10/H10_forces.csv", method)
    df_dpe = []
    for dir in glob.glob("/home/mscherbela/runs/PES/H10_forces/H10_forces_kfac_*"):
        df = pd.DataFrame([get_forces_from_dpe(f) for f in glob.glob(dir + "/*/results.bz2")])
        df['r_cut'] = float(dir.split('_')[-2])
        df['r_core'] = float(dir.split('_')[-1])
        df_dpe.append(df)
    df_dpe = pd.concat(df_dpe, axis=0, ignore_index=True)
    df = pd.merge(df_ref, df_dpe, on=['a', 'x'], how='inner')
    df['error'] = np.sqrt((df.Fx - df.Fx_ref)**2 + (df.Fa - df.Fa_ref)**2)
    df['F_std'] = np.sqrt(df.Fa_std**2 + df.Fx_std**2)

    pivot = df.groupby(['r_cut', 'r_core']).agg(dict(error='mean', F_std='mean')).reset_index()
    #%%
    plt.close("all")
    fig, axes = plt.subplots(1,2, figsize=(8, 5))
    for r_core in pivot.r_core.unique():
        df_filt = pivot[pivot.r_core == r_core]
        axes[0].plot(df_filt.r_cut, df_filt.error, label=f"r_core = {r_core:.2f}")
        axes[1].plot(df_filt.r_cut, df_filt.F_std, label=f"r_core = {r_core:.2f}")
    for ax in axes:
        ax.set_xlabel("r_cut")
        ax.legend()
        ax.grid(alpha=0.5, color='gray')
    axes[0].set_ylabel("MSE forces vs. MRCI")
    axes[1].set_ylabel("Std-dev of forces")
    axes[0].set_title("Bias", fontsize=14)
    axes[1].set_title("Variance", fontsize=14)


    plt.suptitle("Force evaluation for H6")
    plt.tight_layout()
    # plt.savefig("/home/mscherbela/ucloud/results/H6_forces_hyperparams.png", dpi=400)

    #%%
    plot_single_method = True
    if plot_single_method:
        # run_dir_adam = "/home/mscherbela/runs/PES/H6/H6_indep_adam_force_eval"
        # run_dir_kfac = "/home/mscherbela/runs/PES/H6/H6_indep_kfac_force_eval_0.02_0.5"
        run_dir_kfac = "/home/mscherbela/runs/PES/H10_forces/H10_forces_kfac_0.02_0.2"

        df_dpe_kfac = pd.DataFrame([get_forces_from_dpe(f) for f in glob.glob(run_dir_kfac + "/*/results.bz2")])
        df_dpe_kfac.rename(columns=dict(Fx="Fx_kfac", Fa="Fa_kfac"), inplace=True)
        # df = pd.merge(df_ref, df_dpe_adam, on=['a', 'x'], how='outer')
        df = pd.merge(df_ref, df_dpe_kfac, on=['a', 'x'], how='outer')

        df = df.sort_values(['a', 'x'])
        plt.close("all")
        figsize=(15,8)

        plt.figure(figsize=figsize)
        for i,a in enumerate(df_ref.a.unique()):
            color = f'C{i}'
            kwargs_ref=dict(label=f"a={a:.1f} (ref)", alpha=0.5, ls='--', color=color)
            kwargs_adam=dict(label=f"a={a:.1f} (adam 5k)", alpha=1.0, ls='-', color=color)
            kwargs_kfac=dict(label=f"a={a:.1f} (kfac 40k)", alpha=1.0, ls=':', color=color, lw=3)

            df_filt = df[df.a == a]
            plt.subplot(1,2,1)
            plt.plot(df_filt.x, df_filt.Fa_ref, **kwargs_ref)
            # plt.plot(df_filt.x, df_filt.Fa, **kwargs_adam)
            plt.plot(df_filt.x, df_filt.Fa_kfac, **kwargs_kfac)
            plt.xlabel("x")
            plt.ylabel("Fa")
            plt.legend()
            plt.grid(alpha=0.5)
            plt.subplot(1,2,2)
            plt.plot(df_filt.x, df_filt.Fx_ref, **kwargs_ref)
            # plt.plot(df_filt.x, df_filt.Fx, **kwargs_adam)
            plt.plot(df_filt.x, df_filt.Fx_kfac, **kwargs_kfac)
            plt.xlabel("x")
            plt.ylabel("Fx")
            plt.legend()
            plt.grid(alpha=0.5)

        plt.figure(figsize=figsize)
        for i,x in enumerate(df_ref.x.unique()):
            color = f'C{i}'
            kwargs_ref=dict(label=f"x={x:.1f} (ref)", alpha=0.5, ls='--', color=color)
            kwargs_adam=dict(label=f"x={x:.1f} (adam 5k)", alpha=1.0, ls='-', color=color)
            kwargs_kfac=dict(label=f"x={x:.1f} (kfac 30k)", alpha=1.0, ls='-', color=color, lw=2)

            df_filt = df[df.x == x]
            plt.subplot(1,2,1)
            plt.plot(df_filt.a, df_filt.Fa_ref, **kwargs_ref)
            # plt.plot(df_filt.a, df_filt.Fa, **kwargs_adam)
            plt.plot(df_filt.a, df_filt.Fa_kfac, **kwargs_kfac)
            plt.xlabel("a")
            plt.ylabel("Fa")
            plt.legend()
            plt.grid(alpha=0.5)
            plt.subplot(1,2,2)
            plt.plot(df_filt.a, df_filt.Fx_ref, **kwargs_ref)
            # plt.plot(df_filt.a, df_filt.Fx, **kwargs_adam)
            plt.plot(df_filt.a, df_filt.Fx_kfac, **kwargs_kfac)
            plt.xlabel("a")
            plt.ylabel("Fx")
            plt.legend()
            plt.grid(alpha=0.5)

        plt.suptitle("Force on H6: DeepErwin vs MRCI")
        plt.tight_layout()
        plt.savefig("/home/mscherbela/ucloud/results/H10_forces_vs_MRCI.png", dpi=400)







