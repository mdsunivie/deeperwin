import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deeperwin.utils.utils import get_extent_for_imshow
from deeperwin.checkpoints import load_from_file
import glob
import scipy.interpolate


def calculate_forces_from_finite_diff(fname, method):
    df = pd.read_csv(fname)
    df["a"] = df.geom.apply(lambda g: float(g.split("_")[1]))
    df["x"] = df.geom.apply(lambda g: float(g.split("_")[2]))
    df["da"] = df.geom.apply(lambda g: float(g.split("_")[4]))
    df["dx"] = df.geom.apply(lambda g: float(g.split("_")[5][:-4]))
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
        force_data.append(dict(a=r.a, x=r.x, Fx_ref=Fx, Fa_ref=Fa))
    return pd.DataFrame(force_data)


def get_energies(fname):
    data = load_from_file(fname)
    R = data["config"]["physical.R"]
    a = np.round(R[1][0] - R[0][0], 2)
    x = np.round(R[2][0] - R[1][0], 2)
    return dict(a=a, x=x, E=data["metrics"]["E_mean"], E_ref=data["config"]["physical.E_ref"])


def get_forces_from_dpe(fname):
    data = load_from_file(fname)
    F_mean = data["metrics"]["forces_mean"]
    F_t = data["metrics"]["forces"]
    comment = data["config"]["physical.comment"]
    if (comment is None) or (comment == ""):
        original_directory = data["config"]["restart.path"]
        original_directory = original_directory.replace("/home/lv71376/scherbelam/", "/home/mscherbela/")
        original_data = load_from_file(original_directory + "/results.bz2")
        R = original_data["config"]["physical.R"]
    else:
        R = data["config"]["physical.R"]
    a = np.round(R[1][0] - R[0][0], 2)
    x = np.round(R[2][0] - R[1][0], 2)

    n_particles = len(F_mean)
    jacobi_mat = np.zeros([n_particles, 2])
    jacobi_mat[0::2, 0] = np.arange(n_particles // 2)
    jacobi_mat[1::2, 0] = np.arange(n_particles // 2) + 1
    jacobi_mat[0::2, 1] = np.arange(n_particles // 2)
    jacobi_mat[1::2, 1] = np.arange(n_particles // 2)

    F_t = F_t - np.mean(np.mean(F_t, axis=0), axis=0)  # average over time and atoms
    Fa_std, Fx_std = np.std(F_t[..., 0], axis=0) @ jacobi_mat

    Fa, Fx = (F_mean[:, 0] - np.mean(F_mean[:, 0])) @ jacobi_mat
    return dict(a=a, x=x, Fa=Fa, Fx=Fx, Fa_std=Fa_std, Fx_std=Fx_std)


if __name__ == "__main__":
    dpe_forces_dir = "/PES/H10_forces/H10_forces_kfac_0.01_0.2"
    dpe_energies_dir = "/PES/H10_forces/h10_indep_kfac"
    df_dpe_energies = pd.DataFrame([get_energies(f) for f in glob.glob(dpe_energies_dir + "/*/results.bz2")])
    df_dpe_forces = pd.DataFrame([get_forces_from_dpe(f) for f in glob.glob(dpe_forces_dir + "/*/results.bz2")])
    df = calculate_forces_from_finite_diff("/home/mscherbela/runs/references/H10/H10_forces.csv", "MRCI-D-F12")
    df = pd.merge(df, df_dpe_energies, on=["a", "x"], how="outer")
    df = pd.merge(df, df_dpe_forces, on=["a", "x"], how="outer")
    # df = df[df.a > 1.2]
    df_with_ref = df[~(df.Fx_ref.isna() | df.Fx.isna())]

    keys = ["E", "Fx", "Fa", "E_ref", "Fx_ref", "Fa_ref"]
    pivot = df.pivot("a", "x", keys)
    E, Fx, Fa, E_ref, Fx_ref, Fa_ref = [np.array(pivot[k].values, float) for k in keys]
    a_values = np.array(df.a.unique())
    x_values = np.array(df.x.unique())
    aa, xx = np.meshgrid(a_values, x_values, indexing="ij")

    print("Refining PES for plotting by cubic interpolation...")
    n_refine = 200
    x_values_fine = np.linspace(x_values.min(), x_values.max() + 0.01, n_refine)
    a_values_fine = np.linspace(a_values.min(), a_values.max(), n_refine)
    E_interp_func = scipy.interpolate.interp2d(x_values, a_values, E, kind="cubic")
    E_interp = E_interp_func(x_values_fine, a_values_fine)

    # Fx_2D = df.pivot('a', 'x', 'Fx').values
    # Fx_ref_2D = df.pivot('a', 'x', 'Fx_ref').values
    # %%
    plt.close("all")

    # plt.subplot(1,2,2)
    # plt.contour(E_interp, levels=30, extent=get_extent_for_imshow(x_values_fine, a_values_fine))
    # plt.quiver(xx, aa, Fx, Fa)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4), gridspec_kw={"width_ratios": [1.2, 1, 1]}, dpi=100)
    axes[0] = plt.subplot(1, 3, 1, projection="3d")
    ax_3d, ax_PES, ax_scatter = axes

    ax_3d.plot_surface(aa, xx, E, cmap="viridis")
    ax_3d.view_init(20.0, 70.0)
    ax_3d.set_xticks([1.8, 1.6, 1.4, 1.2])
    ax_3d.set_yticks([1.6, 2.0, 2.4, 2.8])
    ax_3d.set_zticks(np.arange(np.round(E.min(), 1), np.round(E.max(), 1) + 0.1, 0.1))
    ax_3d.set_xlabel("a / bohr")
    ax_3d.set_ylabel("x / bohr")
    ax_3d.set_zlabel("E / Ha")

    img_ax = ax_3d.inset_axes([0.05, 0.8, 0.85, 0.2])
    img_ax.imshow(plt.imread("/home/mscherbela/ucloud/results/paper_figures/jax/H10_Chain.png"))
    img_ax.axis("off")

    # ax_PES.contour(E.T, levels=30, extent=get_extent_for_imshow(a_values, x_values), zorder=0)
    ax_PES.clear()
    ax_PES.contour(E_interp.T, levels=30, extent=get_extent_for_imshow(a_values_fine, x_values_fine), zorder=0)
    ax_PES.quiver(
        aa,
        xx,
        Fa_ref,
        Fx_ref,
        color="r",
        alpha=0.8,
        zorder=2,
        scale=5,
        width=0.02,
        headlength=1.5,
        headaxislength=1.5,
        headwidth=2,
    )
    ax_PES.quiver(aa, xx, Fa, Fx, color="k", alpha=0.8, zorder=2, scale=5, headlength=4, headaxislength=4)
    ax_PES.set_xlabel("a / bohr")
    ax_PES.set_ylabel("x / bohr")

    r2_a = np.corrcoef(df_with_ref.Fa, df_with_ref.Fa_ref)[0, 1] ** 2
    r2_x = np.corrcoef(df_with_ref.Fx, df_with_ref.Fx_ref)[0, 1] ** 2
    ax_scatter.scatter(
        df_with_ref.Fx_ref, df_with_ref.Fx, label=f"$F_x$: ($r^2$={r2_x:.3f})", color="C4", marker="D", alpha=0.7
    )
    ax_scatter.scatter(
        df_with_ref.Fa_ref, df_with_ref.Fa, label=f"$F_a$: ($r^2$={r2_a:.3f})", color="C3", marker="o", alpha=0.7
    )
    ax_scatter.legend(loc="upper left")
    ax_scatter.set_xlabel("Force by MRCI-D-F12 / atomic. units")
    ax_scatter.set_ylabel("Force by DeepErwin / atomic. units")
    ax_scatter.grid(alpha=0.5, color="gray")

    F_minmax = np.array([df_with_ref.Fa.min() - 0.1, df_with_ref.Fa.max() + 0.1])
    ax_scatter.plot(F_minmax, F_minmax, color="k")

    for i, ax in enumerate(axes[1:]):
        ax.text(0.0, 1.02, f"({chr(98 + i)})", dict(fontweight="bold", fontsize=12), transform=ax.transAxes)
    plt.tight_layout()
    # Cannot add text to 3d-axis; add it to 2nd axis again, but make sure not to touch tight_layout again because 2nd axis now extends far to the left
    axes[1].text(-1.52, 1.02, "(a)", dict(fontweight="bold", fontsize=12), transform=axes[1].transAxes)

    fname = "/home/mscherbela/ucloud/results/paper_figures/jax/H10_PES.pdf"
    plt.savefig(fname, dpi=400, bbox_inches="tight")
    plt.savefig(fname.replace(".pdf", ".png"), bbox_inches="tight")

    # for i, ax in enumerate(axes):
    #     renderer = fig.canvas.get_renderer()
    #     bbox = ax.get_tightbbox(renderer)
    #     bbox = bbox.transformed(fig.dpi_scale_trans.inverted()).padded(0.1)
    #     plt.savefig(fname.replace(".pdf", f"_{i}.pdf"), bbox_inches=bbox)
    #
    df = df.rename(
        columns={
            "x": "x in bohr",
            "a": "a in bohr",
            "E": "energy in Ha",
            "E_ref": "reference energy (MRCI) in Ha",
            "Fx": "DeepErwin: Force in x-direction in Ha/bohr",
            "Fa": "DeepErwin: Force in a-direction in Ha/bohr",
            "Fx_ref": "MRCI finite differences: Force in x-direction in Ha/bohr",
            "Fa_ref": "MRCI finite differences: Force in a-direction in Ha/bohr",
        }
    )
    df.insert(0, "geometry", np.arange(len(df)))
    df.to_csv("/home/mscherbela/ucloud/results/paper_figures/jax/figure_data/Fig5_H10_forces.csv", index=False)
