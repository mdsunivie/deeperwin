import pandas as pd
import matplotlib.pyplot as plt
import wandb
def load_wandb():
    project = "pretrainingV2"
    code_version = "commit 8fbb0309d7477ba4c394a6c4dcc159b82f623f70; Merge: 235c858 bfa2d80; Author: leongerard <leongerard@t-online.de>; Date:   Fri Apr 29 11:55:28 2022 +0200; ;     Merge branch 'michael' into leon_dev; "
    api = wandb.Api()
    runs = api.runs(f"schroedinger_univie/{project}")

    filter_names = tuple(['0opt_50keval']) #

    data = []
    for counter, run in enumerate(runs):

        if counter % 10==0:
            print(f"Rows {counter}")

        if run.state == 'running' or run.state == 'crashed':
            continue
        if all(f not in run.name for f in filter_names):
            continue
        if run.config['code_version'].replace(" ", "") != code_version.replace(" ", ""):
            continue


        method = "_".join(run.name.split("_")[2:4])
        print(method, run.config['code_version'].replace(" ", ""))
        d = dict(name=run.name, molecule=run.config['physical.name'], method=method,
                 error_final=run.summary['error_eval'], epochs=run.config['pre_training.n_epochs'])

        data.append(d)

    print("finished")
    name = "pretrainingNoOpt.xlsx"
    excel_file_path = "/Users/leongerard/Desktop/data_schroedinger/" + name

    df = pd.DataFrame(data)
    df.to_excel(excel_file_path)
load_wandb()

#%%

def correct_nb_epochs(x):
    if "fixed" in x:
        nb_epochs = x.split("_")[-3]
        return int(nb_epochs)
    nb_epochs = x.split("_")[-2]
    return int(nb_epochs)


name = "pretrainingNoOpt.xlsx"
excel_file_path = "/Users/leongerard/Desktop/data_schroedinger/" + name
df = pd.read_excel(excel_file_path)

df_hf = df[df['method'] == 'prod_hf']
df_cas = df[df['method'] == 'prod_cas']
data_hf = df_hf.groupby(['epochs']).agg(['mean', 'std']).reset_index()
data_cas = df_cas.groupby(['epochs']).agg(['mean', 'std']).reset_index()


cas_error = (-56.294383 + 56.56439971923828)*1000
hf_error = (-56.177129 + 56.56439971923828)*1000

fig, ax = plt.subplots(1, 1)
ax.bar([0,1], data_hf['error_final']['mean'], label='HF Pretraining', color='slategrey')
ax.bar([2, 3], data_cas['error_final']['mean'], label='CAS Pretraining', color='tomato')

ax.axhline(cas_error, color='black', ls='--', label='CAS Baseline')
ax.axhline(hf_error, color='grey', ls='--', label='HF Baseline')

ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels([5000, 200000, 5000, 200000])
ax.set_xlabel("Pretraining steps")
ax.set_ylabel("Error / mHa")
ax.legend()
ax.set_title("Pretraining HF/ CAS with zero optimization")