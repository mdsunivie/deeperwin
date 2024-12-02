import numpy as np
import matplotlib.pyplot as plt


def plot(result_dataset, translation_title, translation_x=None):
    fig, ax = plt.subplots(1, len(result_dataset.keys()))

    for i, name in enumerate(result_dataset):
        setting = result_dataset[name]
        error = [setting[s][0] for s in setting]
        std_dev = [setting[s][1] for s in setting]

        x = np.arange(len(setting))
        if translation_x is None:
            label_names = [s for s in setting]
        else:
            label_names = [translation_x[name][s] for s in translation_x[name]]

        if len(result_dataset.keys()) == 1:
            ax.set_xticks(x)
            ax.set_xticklabels(label_names)
            ax.bar(x, error, yerr=std_dev, color=["darkblue", "slategray", "dodgerblue"])
            ax.set_title(f"{translation_title[name]}")
        else:
            ax[i].set_xticks(x)
            ax[i].set_xticklabels(label_names)
            ax[i].bar(x, error, yerr=std_dev, color=["darkblue", "slategray", "dodgerblue"])
            ax[i].set_title(f"{translation_title[name]}")

    if len(result_dataset.keys()) == 1:
        ax.set_ylabel("Error (mHa)")
    else:
        ax[0].set_ylabel("Error (mHa)")

'''

General setting for benchmark:
- Decay 10k
- Epochs 60k
- One el. True
- Distance feat. -1
- RBF feat. True
- Diff. feat. True

SchNet 2.0 + CAS Orb.  

'''
result_dataset = dict(
    one_el = dict(true=[2.1, 0.36], false=[1.0, 0.34]),
    distance_power = dict(pos_1=[2.3, 0.64], neg_1=[2.1, 0.36]),
    decay = dict(d_10000=[2.1, 0.36], d_5000=[2.5, 0.42], d_1000=[2.3, 0.37])
)
translation_title = dict(
    one_el = "Use one electron feat.",
    distance_power = "Distance power",
    decay = "LR Decay"
)

#plot(result_dataset, translation_title)


'''

General setting for benchmark:
- Decay 10k
- Epochs 60k
- One el. False
- Distance feat. -1
- RBF feat. True
- Diff. feat. True

SchNet 2.0 + CAS Orb.  

'''

result_dataset = dict(
    rbf=dict(true=[1.8, 0.39], false=[1.4, 0.37], true_lr=[2.3, 0.51])
)

translation_title = dict(rbf="Use RBF Features")
translation_x = dict(
    rbf=dict(true="True", false="False", true_lr="True + Small LR (5x10-3)")
)

#plot(result_dataset, translation_title, translation_x)


'''

General setting for benchmark:
- Decay 10k
- Epochs 60k
- One el. False
- Distance feat. 1
- RBF feat. True
- Diff. feat. True

SchNet 2.0 + CAS Orb.  

'''


result_dataset = dict(
    rbf=dict(true=[1.2, 0.27], false=[1.5, 0.22], true_one_el=[2.3, 0.64])
)

translation_title = dict(rbf="Use RBF Features with pos. distance feat.")
translation_x = dict(
    rbf=dict(true="True", false="False", true_one_el="True + One el. input")
)

plot(result_dataset, translation_title, translation_x)