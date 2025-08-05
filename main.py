import os
import keras
import bayesflow as bf
import numpy as np
from simulation import three_body_simulation
import warnings
from models import GRU, RegularizedCNNBiGRU
import json

warnings.filterwarnings(action="ignore")

if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "jax"
