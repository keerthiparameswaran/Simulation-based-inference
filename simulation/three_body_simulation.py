import numpy as np
from scipy.integrate import solve_ivp
import yaml


class ThreeBodySimulation:
    """
    Class defining 3-Body simulation.

    """

    def __init__(self, config_path: str = "./config.yml", rng_seed: int = 5):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.G = self.config["simulation"]["G"]
        self.mass = self.config["simulation"]["mass"]
        self.num_bodies = self.config["simulation"]["num_bodies"]
        self.alpha = self.config["simulation"]["alpha"]

        self.t_start = self.config["simulation"]["t_start"]
        self.t_end = self.config["simulation"]["t_end"]
        self.time_steps = self.config["simulation"]["time_steps"]

        self.velocity_std = self.config["velocity_std"]
        self.noise_std = self.config["noise"]["std_dev"]
        self.noise_apply = self.config["noise"]["apply"]

        self.remove_initial_apply = self.config["remove_initial"]["apply"]
        self.remove_initial_time = self.config["remove_initial"]["time"]

        self.masses = np.array([self.mass] * self.num_bodies)
        self.RNG = np.random.default_rng(rng_seed)

    def compute_velocity_sigma(self, positions):
        """
        Method to compute the escape velocity for velocity prior.

        """
        r12 = np.linalg.norm(positions[0] - positions[1])
        r13 = np.linalg.norm(positions[0] - positions[2])
        r23 = np.linalg.norm(positions[1] - positions[2])

        r_avg = (r12 + r13 + r23) / 3.0
        total_mass = np.sum(self.masses)
        v_escape = np.sqrt(2 * self.G * total_mass / r_avg)

        sigma = self.alpha * v_escape
        return sigma

    def adjust_to_com_frame(self, positions, velocities):
        """
        Method to adjust for Center of Mass.
        """
        total_mass = np.sum(self.masses)

        com_position = np.sum(self.masses[:, None] * positions, axis=0) / total_mass
        com_velocity = np.sum(self.masses[:, None] * velocities, axis=0) / total_mass

        positions_com = positions - com_position
        velocities_com = velocities - com_velocity

        return positions_com, velocities_com

    def prior(self):
        positions = self.RNG.uniform(-4, 4, size=(self.num_bodies, 3))

        velocities = np.stack(
            [
                self.RNG.normal(0, self.velocity_std[0], size=3),
                self.RNG.normal(0, self.velocity_std[1], size=3),
                self.RNG.normal(0, self.velocity_std[2], size=3),
            ]
        )

        positions_com, velocities_com = self.adjust_to_com_frame(positions, velocities)
        x1, y1, z1 = positions_com[0]
        x2, y2, z2 = positions_com[1]
        x3, y3, z3 = positions_com[2]

        vx1, vy1, vz1 = velocities_com[0]
        vx2, vy2, vz2 = velocities_com[1]
        vx3, vy3, vz3 = velocities_com[2]

        return {
            "x1": x1,
            "y1": y1,
            "z1": z1,
            "x2": x2,
            "y2": y2,
            "z2": z2,
            "x3": x3,
            "y3": y3,
            "z3": z3,
            "vx1": vx1,
            "vy1": vy1,
            "vz1": vz1,
            "vx2": vx2,
            "vy2": vy2,
            "vz2": vz2,
            "vx3": vx3,
            "vy3": vy3,
            "vz3": vz3,
        }

    def derivatives(self, t, y):
        m1, m2, m3 = self.masses
        x1, y1, z1, vx1, vy1, vz1 = y[0:6]
        x2, y2, z2, vx2, vy2, vz2 = y[6:12]
        x3, y3, z3, vx3, vy3, vz3 = y[12:18]

        r1 = np.array([x1, y1, z1])
        r2 = np.array([x2, y2, z2])
        r3 = np.array([x3, y3, z3])

        r12 = np.linalg.norm(r2 - r1)
        r13 = np.linalg.norm(r3 - r1)
        r23 = np.linalg.norm(r3 - r2)

        a1 = self.G * m2 * (r2 - r1) / r12**3 + self.G * m3 * (r3 - r1) / r13**3
        a2 = self.G * m1 * (r1 - r2) / r12**3 + self.G * m3 * (r3 - r2) / r23**3
        a3 = self.G * m1 * (r1 - r3) / r13**3 + self.G * m2 * (r2 - r3) / r23**3

        return np.array(
            [
                vx1,
                vy1,
                vz1,
                a1[0],
                a1[1],
                a1[2],
                vx2,
                vy2,
                vz2,
                a2[0],
                a2[1],
                a2[2],
                vx3,
                vy3,
                vz3,
                a3[0],
                a3[1],
                a3[2],
            ]
        )

    def compute_trajectories(
        self,
        x1,
        y1,
        z1,
        x2,
        y2,
        z2,
        x3,
        y3,
        z3,
        vx1,
        vy1,
        vz1,
        vx2,
        vy2,
        vz2,
        vx3,
        vy3,
        vz3,
    ) -> dict:
        """
        Compute trajectories for given timesteps. Timesteps should be defined in config.yml.
        Args:
            positions : (x, y, z) of all 3 bodies.
            velocities : (x, y, z) for all 3 bodies.
        Returns:
            Dict with trajectories.

        """
        y0 = np.array(
            [
                x1,
                y1,
                z1,
                vx1,
                vy1,
                vz1,  # Body 1
                x2,
                y2,
                z2,
                vx2,
                vy2,
                vz2,  # Body 2
                x3,
                y3,
                z3,
                vx3,
                vy3,
                vz3,
            ]
        )

        t_eval = np.linspace(self.t_start, self.t_end, self.time_steps)
        sol = solve_ivp(
            self.derivatives,
            (self.t_start, self.t_end),
            y0,
            method="RK45",
            t_eval=t_eval,
            rtol=1e-9,
            atol=1e-9,
        )

        trjs = np.transpose(sol.y, (1, 0))

        if self.noise_apply:
            trjs = self.add_noise_to_trajectories(trjs)

        if self.remove_initial_apply:
            trjs, t_eval = self.remove_initial_points(trjs, t_eval)

        return dict(trjs=trjs)

    def add_noise_to_trajectories(self, trjs):
        noise = self.RNG.normal(0, self.noise_std, size=trjs.shape)
        return trjs + noise

    def remove_initial_points(self, trjs, t_eval):
        mask = t_eval >= self.remove_initial_time
        return trjs[mask], t_eval[mask]




#------------ OLD CODE ------
# import numpy as np
# from scipy.integrate import solve_ivp
# import yaml


# class ThreeBodySimulation:
#     """
#     Class defining 3-Body simulation.

#     """

#     def __init__(self, config_path: str = "./config.yml", rng_seed: int = 0):
#         with open(config_path, "r") as f:
#             self.config = yaml.safe_load(f)

#         self.G = self.config["simulation"]["G"]
#         self.mass = self.config["simulation"]["mass"]
#         self.num_bodies = self.config["simulation"]["num_bodies"]
#         self.alpha = self.config["simulation"]["alpha"]

#         self.t_start = self.config["simulation"]["t_start"]
#         self.t_end = self.config["simulation"]["t_end"]
#         self.time_steps = self.config["simulation"]["time_steps"]

#         self.velocity_std = self.config["velocity_std"]
#         self.noise_std = self.config["noise"]["std_dev"]
#         self.noise_apply = self.config["noise"]["apply"]

#         self.remove_initial_apply = self.config["remove_initial"]["apply"]
#         self.remove_initial_time = self.config["remove_initial"]["time"]

#         self.masses = np.array([self.mass] * self.num_bodies)
#         self.RNG = np.random.default_rng(rng_seed)

#     def compute_velocity_sigma(self, positions):
#         """
#         Method to compute the escape velocity for velocity prior.

#         """
#         r12 = np.linalg.norm(positions[0] - positions[1])
#         r13 = np.linalg.norm(positions[0] - positions[2])
#         r23 = np.linalg.norm(positions[1] - positions[2])

#         r_avg = (r12 + r13 + r23) / 3.0
#         total_mass = np.sum(self.masses)
#         v_escape = np.sqrt(2 * self.G * total_mass / r_avg)
#         sigma = self.alpha * v_escape
#         return sigma

#     def adjust_to_com_frame(self, positions, velocities):
#         """
#         Method to adjust for Center of Mass.
#         """
#         total_mass = np.sum(self.masses)
#         com_position = np.sum(self.masses[:, None] * positions, axis=0) / total_mass
#         com_velocity = np.sum(self.masses[:, None] * velocities, axis=0) / total_mass

#         return positions - com_position, velocities - com_velocity

#     def prior(self):
#         positions = self.RNG.uniform(-4, 4, size=(self.num_bodies, 3))
#         velocities = np.stack(
#             [self.RNG.normal(0, std, size=3) for std in self.velocity_std]
#         )
#         positions_com, velocities_com = self.adjust_to_com_frame(positions, velocities)
#         x1, y1, z1 = positions_com[0]
#         x2, y2, z2 = positions_com[1]
#         x3, y3, z3 = positions_com[2]

#         vx1, vy1, vz1 = velocities_com[0]
#         vx2, vy2, vz2 = velocities_com[1]
#         vx3, vy3, vz3 = velocities_com[2]

#         return {
#             "x1": x1,
#             "y1": y1,
#             "z1": z1,
#             "x2": x2,
#             "y2": y2,
#             "z2": z2,
#             "x3": x3,
#             "y3": y3,
#             "z3": z3,
#             "vx1": vx1,
#             "vy1": vy1,
#             "vz1": vz1,
#             "vx2": vx2,
#             "vy2": vy2,
#             "vz2": vz2,
#             "vx3": vx3,
#             "vy3": vy3,
#             "vz3": vz3,
#         }

#     def derivatives(self, t, y):
#         m1, m2, m3 = self.masses
#         x1, y1, z1, vx1, vy1, vz1 = y[0:6]
#         x2, y2, z2, vx2, vy2, vz2 = y[6:12]
#         x3, y3, z3, vx3, vy3, vz3 = y[12:18]

#         r1, r2, r3 = (
#             np.array([x1, y1, z1]),
#             np.array([x2, y2, z2]),
#             np.array([x3, y3, z3]),
#         )

#         r12 = np.linalg.norm(r2 - r1)
#         r13 = np.linalg.norm(r3 - r1)
#         r23 = np.linalg.norm(r3 - r2)

#         a1 = self.G * m2 * (r2 - r1) / r12**3 + self.G * m3 * (r3 - r1) / r13**3
#         a2 = self.G * m1 * (r1 - r2) / r12**3 + self.G * m3 * (r3 - r2) / r23**3
#         a3 = self.G * m1 * (r1 - r3) / r13**3 + self.G * m2 * (r2 - r3) / r23**3

#         return np.array([vx1, vy1, vz1, *a1, vx2, vy2, vz2, *a2, vx3, vy3, vz3, *a3])

#     def compute_trajectories(self, positions, velocities) -> dict:
#         """
#         Compute trajectories for given timesteps. Timesteps should be defined in config.yml.
#         Args:
#             positions : (x, y, z) of all 3 bodies.
#             velocities : (x, y, z) for all 3 bodies.
#         Returns:
#             Dict with trajectories.

#         """
#         y0 = np.hstack(
#             [
#                 positions[0],
#                 velocities[0],
#                 positions[1],
#                 velocities[1],
#                 positions[2],
#                 velocities[2],
#             ]
#         )

#         t_eval = np.linspace(self.t_start, self.t_end, self.time_steps)
#         sol = solve_ivp(
#             self.derivatives,
#             (self.t_start, self.t_end),
#             y0,
#             method="RK45",
#             t_eval=t_eval,
#             rtol=1e-9,
#             atol=1e-9,
#         )

#         trjs = np.transpose(sol.y, (1, 0))
#         if self.noise_apply:
#             trjs = self.add_noise_to_trajectories(trjs)

#         if self.remove_initial_apply:
#             trjs, t_eval = self.remove_initial_points(trjs, t_eval)

#         return dict(trjs=trjs)

#     def add_noise_to_trajectories(self, trjs):
#         noise = self.RNG.normal(0, self.noise_std, size=trjs.shape)
#         return trjs + noise

#     def remove_initial_points(self, trjs, t_eval):
#         mask = t_eval >= self.remove_initial_time
#         return trjs[mask], t_eval[mask]