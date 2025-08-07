import os
import json
import warnings
import numpy as np
import keras
import bayesflow as bf
from models import GRU, RegularizedCNNBiGRU
from simulation import three_body_simulation

# ----------------------------- #
# ENVIRONMENT SETUP
# ----------------------------- #
warnings.filterwarnings(action="ignore")

if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "jax"

# Define paths
config_file = os.path.join("simulation", "config.yml")
train_data_path = os.path.join("data", "train.json")
test_data_path = os.path.join("data", "test.json")
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

# ----------------------------- #
# SIMULATION SETUP
# ----------------------------- #
print("Initializing Three Body Simulation...")
sim = three_body_simulation.ThreeBodySimulation(config_path=config_file)


def prior():
    return sim.prior()


def tb_simulator(
    x1, y1, z1, x2, y2, z2, x3, y3, z3, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3
):
    return sim.compute_trajectories(
        x1, y1, z1, x2, y2, z2, x3, y3, z3, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3
    )


print("Wrapping simulator...")
simulator = bf.make_simulator([prior, tb_simulator])

# ----------------------------- #
# LOAD TRAINING AND TEST DATA
# ----------------------------- #
print("Loading Training Data...")
with open(train_data_path, "r") as file:
    training_data = {k: np.array(v) for k, v in json.load(file).items()}
print("Training Data Loaded.")

print("Loading Validation Data...")
with open(test_data_path, "r") as file:
    validation_data = {k: np.array(v) for k, v in json.load(file).items()}
print("Validation Data Loaded.")

# ----------------------------- #
# DATA ADAPTER SETUP
# ----------------------------- #
print("Setting up Adapter...")
adapter = (
    bf.adapters.Adapter()
    .as_time_series("trjs")
    .convert_dtype("float64", "float32")
    .concatenate(
        [
            "x1",
            "y1",
            "z1",
            "x2",
            "y2",
            "z2",
            "x3",
            "y3",
            "z3",
            "vx1",
            "vy1",
            "vz1",
            "vx2",
            "vy2",
            "vz2",
            "vx3",
            "vy3",
            "vz3",
        ],
        into="inference_variables",
    )
    .rename("trjs", "summary_variables")
)

# ----------------------------- #
# MODEL SETUP
# ----------------------------- #
print("Initializing Summary and Inference Networks...")
summary_net = RegularizedCNNBiGRU()
inference_net = bf.networks.CouplingFlow()

workflow = bf.BasicWorkflow(
    simulator = simulator,
    adapter=adapter,
    inference_network=inference_net,
    summary_network=summary_net,
)

# ----------------------------- #
# TRAINING
# ----------------------------- #
print("Starting Training...")
history = workflow.fit_offline(
    data=training_data,
    epochs=50,
    batch_size=64,
    validation_data=validation_data,
)
print("Training Completed.")

# Save loss plot
loss_plot_path = os.path.join(plots_dir, "loss_plot.png")
loss_plot = bf.diagnostics.plots.loss(history)
loss_plot.savefig(loss_plot_path)
print(f"Loss plot saved to {loss_plot_path}")

# ----------------------------- #
# POSTERIOR SAMPLING & DIAGNOSTICS
# ----------------------------- #
print("Simulating Posterior Draws...")
num_datasets = 300
num_samples = 1000

test_sims = workflow.simulate(num_datasets)
samples = workflow.sample(conditions=test_sims, num_samples=num_samples)

parameter_keys = [
    "x1",
    "y1",
    "z1",
    "x2",
    "y2",
    "z2",
    "x3",
    "y3",
    "z3",
    "vx1",
    "vy1",
    "vz1",
    "vx2",
    "vy2",
    "vz2",
    "vx3",
    "vy3",
    "vz3",
]

# Save diagnostics plots
print("Generating Diagnostic Plots...")

#Need to Fix this, generates some error. error was also noticed in the notebook.
# cal_hist_path = os.path.join(plots_dir, "calibration_histogram.png")
# calibration_histogram = bf.diagnostics.plots.calibration_histogram(
#     samples, test_sims, variable_keys=parameter_keys
# )
# calibration_histogram.savefig(cal_hist_path)
# print(f"Calibration histogram saved to {cal_hist_path}")

cal_ecdf_path = os.path.join(plots_dir, "calibration_ecdf.png")
calibration_ecdf = bf.diagnostics.plots.calibration_ecdf(
    samples, test_sims, difference=True, variable_keys=parameter_keys
)
calibration_ecdf.savefig(cal_ecdf_path)
print(f"Calibration ECDF saved to {cal_ecdf_path}")

recovery_path = os.path.join(plots_dir, "recovery.png")
recovery = bf.diagnostics.plots.recovery(
    samples, test_sims, variable_keys=parameter_keys
)
recovery.savefig(recovery_path)
print(f"Recovery plot saved to {recovery_path}")

# ----------------------------- #
# METRICS & DEFAULT DIAGNOSTICS
# ----------------------------- #
print("Computing and Saving Diagnostic Metrics...")
metrics = workflow.compute_default_diagnostics(test_data=300)
metrics_path = "metrics.csv"
metrics.to_csv(metrics_path)
print(f"Metrics saved to {metrics_path}")

print("Generating Default Diagnostic Summary Plot...")
default_diag_path = os.path.join(plots_dir, "default_diagnostics.png")
dd = workflow.plot_default_diagnostics(
    test_data=300,
    loss_kwargs={"figsize": (15, 3), "label_fontsize": 12},
    recovery_kwargs={"figsize": (15, 9), "label_fontsize": 12},
    calibration_ecdf_kwargs={
        "figsize": (15, 9),
        "legend_fontsize": 8,
        "difference": True,
        "label_fontsize": 12,
    },
    z_score_contraction_kwargs={"figsize": (15, 9), "label_fontsize": 12},
)
for key, value in dd.items():
    value.savefig(f"plots\\{key}.png")
print(f"Default diagnostics plot saved to {default_diag_path}")
