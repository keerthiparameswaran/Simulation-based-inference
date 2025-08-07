import os
import keras
import bayesflow as bf
import numpy as np
from simulation import three_body_simulation
import warnings
from models import GRU
import json

warnings.filterwarnings(action="ignore")
config_file = "simulation\config.yml"
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "jax"

sim = three_body_simulation.ThreeBodySimulation(config_path=config_file)


def prior():
    x = sim.prior()
    return x


def tb_simulator(
    x1, y1, z1, x2, y2, z2, x3, y3, z3, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3
):
    return sim.compute_trajectories(
        x1, y1, z1, x2, y2, z2, x3, y3, z3, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3
    )


simulator = bf.make_simulator([prior, tb_simulator])
train_data = simulator.sample(1000)
train_data = {k: v.tolist() for k, v in train_data.items()}

print("Generated Train data.")
test_data = simulator.sample(100)
test_data = {k: v.tolist() for k, v in test_data.items()}

print("Generated Test data")
# validation_data = simulator.sample(500)
# print("Generated Validation data")

os.makedirs("data", exist_ok=True)
with open("data/train.json", "w") as file:
    json.dump(train_data, file)
print("Completed Writing data to train.json")
with open("data/test.json", "w") as file:
    json.dump(test_data, file)
print("Completed Writing data to test.json")
# with open("data/validation.json", "w") as file:
#     json.dump(validation_data, file)
# print("Completed Writing data to validation.json")
