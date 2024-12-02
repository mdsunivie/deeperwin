from deeperwin.utils.utils import get_number_of_params
from deeperwin.checkpoints import load_from_file

#%%


path = "/Users/leongerard/Desktop/data_schroedinger/largeatom_fe_rep1/chkpt079999/results.bz2"
data = load_from_file(path)
#weights = data['weights']['trainable']

#%%

print(get_number_of_params(weights))

print([(key, get_number_of_params(weights[key])) for key in weights])