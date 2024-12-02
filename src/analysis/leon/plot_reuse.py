

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches

name = "reuse_2.xlsx"
excel_file_path = "/Users/leongerard/Desktop/data_schroedinger/excel_analysis/" + name
filter_names = tuple(['bench_h6_28_2_v11_256', 'reuse_h6_28_2_no_pretrainig_all_geom_',
                      'reuse_h6_28_2_no_pretrainig_v11_256', 'reuse_h6_28_2_no_pretrainig_v11_256_no_orbs',
                      'reuse_h6_28_2_scaled_pretrainig_v11_256'])
df = pd.read_excel(excel_file_path)
#df_reuse_v4 = df[df['name'].str.contains('reuse_h6_28_2_no_pretrainig_all_geom_')]

df_bench = df[df['name'].str.contains('bench')]
bench = (df_bench, 'Bench')

df_reuse_v11 = df[df['name'].str.startswith('reuse_h6_28_2_no_pretrainig_v11_256')][~df['name'].str.startswith('reuse_h6_28_2_no_pretrainig_v11_256_no_orbs')]
r_v11 = (df_reuse_v11, 'Reuse V11')

df_reuse_v11_no_orbs = df[df['name'].str.contains('reuse_h6_28_2_no_pretrainig_v11_256_no_orbs')]
r_v11_no_orbs = (df_reuse_v11_no_orbs, 'Reuse V11 w/o orbs')


df_reuse_v11_scl_pre = df[df['name'].str.contains('reuse_h6_28_2_scaled_pretrainig_v11_256')]
r_v11_scl_pre = (df_reuse_v11_scl_pre, 'Reuse V11 w/o orbs + sc. pretraining')

data = [bench, r_v11, r_v11_no_orbs, r_v11_scl_pre]

bench_epochs = [64, 128, 256, 512, 1024, 2048, 4096]
reuse_epochs = [128, 256, 512, 1024, 2048, 4096]


# error_bench = [np.array(df_bench[e])[~np.isnan(np.array(df_bench[e]))] for e in bench_epochs]
# error_reuse_v4 = [np.array(df_reuse_v4[e]) for e in reuse_epochs]
# error_reuse_v11 = [np.array(df_reuse_v11[e])[~np.isnan(np.array(df_reuse_v11[e]))] for e in reuse_epochs]
labels = []


def add_label(violin, label):
    color = violin["bodies"][0].get_facecolor().flatten()
    labels.append((mpatches.Patch(color=color), label))

fig, ax = plt.subplots(1, 1)
for i, d in enumerate(data):
    print(len(d[0]))
    if i == 0:
        error = [np.array(d[0][e])[~np.isnan(np.array(d[0][e]))] for e in bench_epochs]
    else:
        error = [np.array(d[0][e])[~np.isnan(np.array(d[0][e]))] for e in reuse_epochs]

    # ax.semilogx(bench_epochs, error_bench, label="Benchmark V11-256", color="grey")
    # ax.semilogx(reuse_epochs, error_reuse, label="Reuse", color="darkblue")



    add_label(ax.violinplot(error,
                      showmeans=True,
                      showmedians=False), d[1])

# add_label(ax.violinplot(error_reuse_v11,
#                   showmeans=True,
#                   showmedians=False), "Reuse V11")
#
# add_label(ax.violinplot(error_bench,
#                   showmeans=True,
#                   showmedians=False), "Benchmark V11-256")
ax.legend(*zip(*labels))
ax.grid(alpha=0.5)

# ax.set_xticks(bench_epochs)
ax.set_xticklabels(bench_epochs)
ax.set_xlabel("Epochs")
ax.set_ylabel("Error / mHa")
ax.set_ylim([0, 5])
