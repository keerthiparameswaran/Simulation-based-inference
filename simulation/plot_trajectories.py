import os
import plotly.graph_objects as go
from three_body_simulation import ThreeBodySimulation

def plot_and_save_trajectories(config_path: str = "./config.yml", output_path: str = "../plots/trajectories.html"):
    sim = ThreeBodySimulation(config_path)
    prior_data = sim.prior()
    
    result = sim.compute_trajectories(
        prior_data["x1"], prior_data["y1"], prior_data["z1"],
        prior_data["x2"], prior_data["y2"], prior_data["z2"],
        prior_data["x3"], prior_data["y3"], prior_data["z3"],
        prior_data["vx1"], prior_data["vy1"], prior_data["vz1"],
        prior_data["vx2"], prior_data["vy2"], prior_data["vz2"],
        prior_data["vx3"], prior_data["vy3"], prior_data["vz3"]
    )

    trjs = result["trjs"]

    # Extract positions over time
    x1, y1, z1 = trjs[:, 0], trjs[:, 1], trjs[:, 2]
    x2, y2, z2 = trjs[:, 6], trjs[:, 7], trjs[:, 8]
    x3, y3, z3 = trjs[:, 12], trjs[:, 13], trjs[:, 14]

    fig = go.Figure()

    # Trajectories
    fig.add_trace(go.Scatter3d(x=x1, y=y1, z=z1, mode='lines', name='Body 1'))
    fig.add_trace(go.Scatter3d(x=x2, y=y2, z=z2, mode='lines', name='Body 2'))
    fig.add_trace(go.Scatter3d(x=x3, y=y3, z=z3, mode='lines', name='Body 3'))

    # Start points (green)
    fig.add_trace(go.Scatter3d(
        x=[x1[0], x2[0], x3[0]],
        y=[y1[0], y2[0], y3[0]],
        z=[z1[0], z2[0], z3[0]],
        mode='markers',
        marker=dict(size=6, color='green'),
        name='Start'
    ))

    # End points (red)
    fig.add_trace(go.Scatter3d(
        x=[x1[-1], x2[-1], x3[-1]],
        y=[y1[-1], y2[-1], y3[-1]],
        z=[z1[-1], z2[-1], z3[-1]],
        mode='markers',
        marker=dict(size=6, color='red'),
        name='End'
    ))

    # Layout
    fig.update_layout(
        title='3D Trajectories of Three-Body Simulation',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig.write_html(output_path)
    print(f"Trajectory plot saved to {output_path}")


if __name__ == "__main__":
    plot_and_save_trajectories(output_path="../plots/trajectories.html")
