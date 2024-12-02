from utils import load_from_file, get_distance_matrix
from model import build_backflow_shift, get_rbf_features
import numpy as np
import matplotlib.pyplot as plt

fname = '/users/mscherbela/runs/jaxtest/config/test_LiH_normal/results.bz2'
data = load_from_file(fname)
