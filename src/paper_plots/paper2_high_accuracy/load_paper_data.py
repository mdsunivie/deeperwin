from paper_plots.paper2_high_accuracy.load_wandb_data import load_wandb_data, build_overview, load_raw_data
import warnings

warnings.filterwarnings("ignore", module="urllib3")  # suppress spurious SystemTimeWarnings
tmp_dir = "/home/mscherbela/tmp/"
EVAL_EPOCHS = [20, 50, 100]
SMOOTH_EPOCHS = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 80, 100]

# %% Large molecules
project = "largemol"


def category_func(run):
    if run.config.get("computation.disable_tensor_cores", False) == False:
        return "tf32"
    damping = float(run.config["optimization.optimizer.damping"])
    pretraining = int(run.config.get("pre_training.n_epochs", 0))
    intersteps = int(run.config["mcmc.n_inter_steps"])
    return f"{damping:.3f}damp_{pretraining:d}pre_{intersteps}inter"


df = load_wandb_data(project, fname=tmp_dir + project + ".csv", category_func=category_func)
df_overview = build_overview(df, EVAL_EPOCHS, SMOOTH_EPOCHS)
df_overview.to_clipboard(index=False, sep="\t", header=False)
print(df_overview.to_string(index=False))

# %% Large atoms
project = "largeatom"


def category_func(run):
    if run.config.get("computation.disable_tensor_cores", False) == False:
        return "tf32"
    damping = float(run.config["optimization.optimizer.damping"])
    pretraining = int(run.config.get("pre_training.n_epochs", 0))
    intersteps = int(run.config["mcmc.n_inter_steps"])
    return f"{damping:.3f}damp_{pretraining:d}pre_{intersteps}inter"


df = load_wandb_data(project, fname=tmp_dir + project + ".csv", category_func=category_func)
df_overview = build_overview(df, EVAL_EPOCHS, SMOOTH_EPOCHS)
df_overview.to_clipboard(index=False, sep="\t", header=False)
print(df_overview.to_string(index=False))


# %% Alpha initialization
project = "alpha_init"


def category_func(run):
    name = run.name
    damping = float(run.config["optimization.optimizer.damping"])
    pretraining = int(run.config.get("pre_training.n_epochs", 0))
    intersteps = int(run.config["mcmc.n_inter_steps"])
    for k in ["analytical", "constant", "reuse"]:
        if k in name:
            return f"{k}_{damping:.3f}damp_{pretraining:d}pre_{intersteps}inter"
    return "other"


df = load_wandb_data(project, fname=tmp_dir + project + ".csv", category_func=category_func)
df_overview = build_overview(df, EVAL_EPOCHS, SMOOTH_EPOCHS)
df_overview.to_clipboard(index=False, sep="\t", header=False)
print(df_overview.to_string(index=False))

# %%
runs = dict(
    # K_hf_init="schroedinger_univie/alpha_init/22w11h1y",
    #         K_constant_init="schroedinger_univie/largeatom/runs/zui2id4c",
    #         Fe_hf_init="schroedinger_univie/alpha_init/1rwuhq5y",
    #         Fe_constant_init="schroedinger_univie/largeatom/330utvbl",
    Ar_hf_init="schroedinger_univie/regression_nips/runs/1jl20zv1",
    Ar_constant_init="schroedinger_univie/regression_nips/runs/3kxs3v1a",
)
for run, id_string in runs.items():
    print(f"Loading {run}")
    df = load_raw_data(id_string)
    df.to_csv(tmp_dir + f"full_runs/{run}.csv", index=False, sep=";")

# %% Ablation v2 (TF32 disabled)
project = "ablation_v2"


def category_func(run):
    config = run.config
    use_baseline = config.get("model.orbitals.baseline_orbitals.use_bf_factor", False)
    full_det = config["model.orbitals.use_full_det"]
    dpe_hp = abs(config["optimization.schedule.decay_time"] - 6000) < 1e-3
    small_batch = config.get("optimization.batch_size", 2048) == 2048
    use_schnet = config.get("model.embedding.use_schnet_features", False)
    use_local = config["model.features.use_local_coordinates"]
    use_init = config["model.orbitals.envelope_orbitals.initialization"] != "constant"
    el_ion_stream = "no_el_ion" not in run.name

    if (
        (not use_baseline)
        and (not full_det)
        and (not dpe_hp)
        and (not small_batch)
        and (not use_schnet)
        and (not use_local)
    ):
        return "01_fermi_iso"
    elif (
        (not use_baseline) and full_det and (not dpe_hp) and (not small_batch) and (not use_schnet) and (not use_local)
    ):
        return "02_fermi_iso_fulldet"
    elif (not use_baseline) and full_det and dpe_hp and (not small_batch) and (not use_schnet) and (not use_local):
        return "03a_fermi_iso_fulldet_hp_4096batch"
    elif (not use_baseline) and full_det and dpe_hp and small_batch and (not use_schnet) and (not use_local):
        return "03_fermi_iso_fulldet_hp"
    elif (
        (not use_baseline)
        and full_det
        and dpe_hp
        and small_batch
        and use_schnet
        and (not el_ion_stream)
        and (not use_local)
    ):
        return "04a_fermi_iso_fulldet_hp_emb_no_el_ion"
    elif (
        (not use_baseline) and full_det and dpe_hp and small_batch and use_schnet and el_ion_stream and (not use_local)
    ):
        return "04_fermi_iso_fulldet_hp_emb"
    elif (not use_baseline) and full_det and dpe_hp and use_schnet and use_local and (not use_init):
        return "05_dpe11"
    elif (not use_baseline) and full_det and dpe_hp and use_schnet and use_local and use_init:
        return "06_dpe11_init"
    return "other"


df = load_wandb_data(project, fname=tmp_dir + project + ".csv", category_func=category_func)
df_overview = build_overview(df, EVAL_EPOCHS, SMOOTH_EPOCHS)
df_overview.to_clipboard(index=False, sep="\t", header=False)
print(df_overview.to_string(index=False))

# %% N2 sweep
project = "N2_sweep"


def category_func(run):
    is_dpe = run.config["model.embedding.use_schnet_features"]
    dist = float(run.config["physical.R"][1][0])
    initialization = run.config["model.orbitals.envelope_orbitals.initialization"] != "constant"
    if is_dpe:
        category = "dpe_init" if initialization else "dpe"
    else:
        category = "fermi"
    category = category + f"_{dist:.3f}"
    return category


df = load_wandb_data(project, fname=tmp_dir + project + ".csv", category_func=category_func)
df_overview = build_overview(df, EVAL_EPOCHS, SMOOTH_EPOCHS)
df_overview.to_clipboard(index=False, sep="\t", header=False)
print(df_overview.to_string(index=False))

# %% Large molecules
project = "bent_h10"
df = load_wandb_data(project, fname=tmp_dir + project + ".csv", category_func=None)
df_overview = build_overview(df, EVAL_EPOCHS, SMOOTH_EPOCHS)
df_overview.to_clipboard(index=False, sep="\t", header=False)
print(df_overview.to_string(index=False))

# %% Large molecules
project = "pretrainingV3"
# def filter_func(name):
#     for token in ['prod_cas_NH3', 'prod_hf_NH3']:
#         if token in name:
#             return True
#     return False


def category_func(run):
    method = "hf" if run.config["pre_training.use_only_leading_determinant"] else "cas"
    n_pre = int(run.config["pre_training.n_epochs"])
    return f"{method}_{n_pre}"


df = load_wandb_data(project, fname=tmp_dir + project + ".csv", run_name_filter_func=None, category_func=category_func)
df_overview = build_overview(df, EVAL_EPOCHS, SMOOTH_EPOCHS)
df_overview.to_clipboard(index=False, sep="\t", header=False)
print(df_overview.to_string(index=False))
