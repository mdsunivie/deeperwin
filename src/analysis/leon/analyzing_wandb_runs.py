import numpy as np
import pandas as pd
import wandb


projects = ["mixture"]
setting_names = ["optimization.learning_rate", "optimization.schedule.decay_time", "optimization.n_epochs",
                 "model.embedding.one_el_input", "model.distance_feature_powers", "model.envelope_type", "model.use_rbf_features", "model.use_differences_features"]

metrics = ["error_eval", "error_plus_2_stdev", "sigma_error_eval", "E_mean"]


df = pd.DataFrame()
nb_rows = 13


for project in projects:
    api = wandb.Api()
    runs = api.runs(f"schroedinger_univie/{project}")

    counter = 0

    for counter, run in enumerate(runs):

        if run.state == 'running':
            continue
        row = {}
        row['name'] = run.name
        config = run.config

        if config['physical.name'] == "O":
            for s in setting_names:
                try:
                    if s == "optimization.schedule.decay_time":
                        row[s.split(".")[-1]] = float(config[s])
                    else:
                        row[s.split(".")[-1]] = config[s]
                except:
                    row[s.split(".")[-1]] = "Not existing for this type"

            for m in metrics:
                    row[m] = run.summary[m]


            df = df.append(row, ignore_index=True)

            print(row)

            if counter > nb_rows:
                break

name = "test.xlsx"
excel_file_path = "/Users/leongerard/Desktop/data/excel_analysis/" + name
df.to_excel(excel_file_path)
