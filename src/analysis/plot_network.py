import numpy as np
import matplotlib
import matplotlib.pyplot as plt

target = ["True", "False"]
el_decay = ["True", "False"]

error = np.array([[4.478, 3.483],
                    [3.647, 2.502]])


fig, ax = plt.subplots()
im = ax.imshow(error)

# We want to show all ticks...
ax.set_xticks(np.arange(len(el_decay)))
ax.set_yticks(np.arange(len(target)))
# ... and label them with the respective list entries
ax.set_xticklabels(el_decay)
ax.set_yticklabels(target)
ax.set_ylabel("Target electron")
ax.set_xlabel("Electron-Electron shift decay")

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(target)):
    for j in range(len(el_decay)):
        text = ax.text(j, i, error[i, j],
                       ha="center", va="center", color="w")

ax.set_title("Shift: Energy error (mHa) for Nitrogen")
fig.tight_layout()
plt.show()