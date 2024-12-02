

import numpy as np
import pandas as pd
import wandb


projects = ["reuse"]
setting_names = ["physical.name", "physical.R", "optimization.intermediate_eval.n_opt_epochs"]#, "optimization.learning_rate", "optimization.schedule.decay_time", "optimization.n_epochs",
                 #"model.embedding.one_el_input", "model.distance_feature_powers", "model.envelope_type", "model.use_rbf_features", "model.use_differences_features"]
metrics = [] #["error_eval", "error_plus_2_stdev", "sigma_error_eval", "E_mean"]


df = pd.DataFrame()
nb_rows = 1000
physical_name = "HChain6"
filter_names = tuple(['bench_h6_28_2_v11_256', 'reuse_h6_28_2_no_pretrainig_all_geom_',
                      'reuse_h6_28_2_no_pretrainig_v11_256', 'reuse_h6_28_2_no_pretrainig_v11_256_no_orbs',
                      'reuse_h6_28_2_scaled_pretrainig_v11_256'])

for project in projects:
    api = wandb.Api()
    runs = api.runs(f"schroedinger_univie/{project}")

    counter = 0
    for counter, run in enumerate(runs):
        print(run.name)
        if run.state == 'running':
            continue
        if not run.name.startswith(filter_names):
            continue
        row = {}
        config = run.config
        row['name'] = run.name + "_" + str(config['physical.comment'])

        if config['physical.name'] == physical_name:
            eval_epochs = config['optimization.intermediate_eval.opt_epochs']
            error = [r['error_intermed_eval'] for r in run.scan_history(keys=['error_intermed_eval'])]
            for e, r in zip(eval_epochs, error):
                row[e] = r

            for m in metrics:
                    row[m] = run.summary[m]

            try:
                row['comment'] = config['physical.comment']
            except:
                row['comment'] = ""

            df = df.append(row, ignore_index=True)
            print(row)

            if counter > nb_rows:
                break

name = "reuse_2.xlsx"
excel_file_path = "/Users/leongerard/Desktop/data_schroedinger/excel_analysis/" + name
df.to_excel(excel_file_path)

#%%
import matplotlib.pyplot as plt

df = pd.read_excel(excel_file_path)
df_bench = df[df['name'].str.contains('bench')]
df_reuse = df[df['name'].str.contains('reuse')]

