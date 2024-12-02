import json
import pickle
import numpy as np
import pandas as pd


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def build_subset(full_data, keys, molecules=None):
    subset = {}
    molecules = molecules or full_data.keys()
    for molecule in molecules:
        subset[molecule] = {}
        for key in keys:
            if key not in full_data[molecule]:
                print(f"No data for {molecule}.{key}")
                continue
            subset[molecule][key] = full_data[molecule][key]._asdict()
    return subset


def save_as_json(data, fname):
    with open(fname, "w") as f:
        json.dump(data, f, cls=NumpyEncoder, indent=2)


def save_as_csv(data, fname, include_subfigure=True):
    flat_data = []
    labels = dict(
        Indep="Independent opt. (blue)",
        Shared_75="75% of weights shared (orange)",
        Shared_95="95% of weights shared (red)",
        ReuseIndep="Independent opt. (blue)",
        ReuseShared95="Pre-trained by shared opt. (violet)",
        ReuseFromSmallerShared="Pre-trained by shared opt. of smaller molecule (red)",
        ReuseFromIndepSingleLR="Full-weight-reuse from indep. opt. (green)",
    )
    subfigure = {"H4p": "a", "H6": "b", "H10": "c", "Ethene": "d"}
    for molecule in data:
        for curve_type in data[molecule]:
            for n_epochs, errors in zip(data[molecule][curve_type]["n_epochs"], data[molecule][curve_type]["errors"]):
                if include_subfigure:
                    row = {"subfigure": subfigure[molecule]}
                else:
                    row = {}
                row.update(
                    {
                        "molecule": molecule,
                        "Curve (color in plot)": labels[curve_type],
                        "Nr of epochs per geometry": n_epochs,
                    }
                )
                for i, E in enumerate(errors):
                    row[f"energy error of geometry {i:03d} in mHa"] = E
                flat_data.append(row)
    df = pd.DataFrame(flat_data)
    df.to_csv(fname, index=False)


def load_data(cache_fname):
    with open(cache_fname, "rb") as f:
        full_plot_data = pickle.load(f)
    if "Methane" in full_plot_data:
        del full_plot_data["Methane"]
    del full_plot_data["H4p"]["ReuseFromSmallerShared"]
    return full_plot_data


output_dir = "/home/mscherbela/ucloud/results/paper_figures/jax/figure_data/"

full_data_kfac = load_data("/home/mscherbela/tmp/data_shared_vs_indep_kfac.pkl")
fig2_data = build_subset(full_data_kfac, ["Indep", "Shared_75", "Shared_95"])
save_as_csv(fig2_data, output_dir + "Fig2_weight_sharing.csv")
fig3_data = build_subset(full_data_kfac, ["ReuseIndep", "ReuseShared95", "ReuseFromSmallerShared"])
save_as_csv(fig3_data, output_dir + "Fig3_weight_reuse.csv")
si_fig3_data = build_subset(
    full_data_kfac, ["Indep", "ReuseFromIndepSingleLR", "ReuseShared95", "Shared_95"], ["Ethene"]
)
save_as_csv(
    si_fig3_data, output_dir + "Supplementary_Data_Suppl_Fig3_reuse_from_indep_kfac.csv", include_subfigure=False
)

full_data_adam = load_data("/home/mscherbela/tmp/data_shared_vs_indep_adam.pkl")
si_fig1_data = build_subset(full_data_adam, ["Indep", "Shared_75", "Shared_95"])
save_as_csv(si_fig1_data, output_dir + "Supplementary_Data_Suppl_Fig1_weight_sharing_adam.csv")
si_fig2_data = build_subset(full_data_adam, ["Indep", "ReuseShared95"])
save_as_csv(si_fig2_data, output_dir + "Supplementary_Data_Suppl_Fig2_weight_reuse_adam.csv")

# %%
