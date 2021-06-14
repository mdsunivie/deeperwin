# For relative imports to work in Python 3.6
import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))


# Importing a few commonly used objects to the package level for convenient access via deeperwin.xyz
from .main import WaveFunction
from .utilities.erwinConfiguration import DefaultConfig
from .utilities.postprocessing import loadRuns
from .models.DeepErwinModel import DeepErwinModel
