from floris.tools import visualization
from floris.tools.floris_interface import FlorisInterface
from util.geometry import check_inside_sliced_unit_circle
from util.mechanics_equations import compute_Cp_Ct, compute_swept_area

import collections
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
import warnings

class Synthetic_Scenario():

    def __init__(self, config, actions, dist_threshold, range_, max_degree, wind_direction):

        center = [np.mean(config["farm"]["properties"]["layout_x"]), np.mean(config["farm"]["properties"]["layout_y"])]
        angle = np.deg2rad((wind_direction - 225.0) % 360)
        self.config = self._rotate_farm_around_point(config, center, angle)

        self.air_density = self.config["farm"]["properties"]["air_density"]
        rotor_diameter = self.config["turbine"][0]["properties"]["rotor_diameter"]
        self.swept_area = compute_swept_area(rotor_diameter)
        self.wind_speed = np.array(self.config["turbine"][0]["properties"]["power_thrust_table"]["wind_speed"]).copy()
        self.wind_direction = 225.0  # Always 225 deg after rotation
        self.power_curve = np.array(self.config["turbine"][0]["properties"]["power_curve"]).copy()
        self.thrust_curve = np.array(self.config["turbine"][0]["properties"]["thrust_curve"]).copy()

        self.actions = actions
        self.agents = self.config["farm"]["properties"]["turbine_names"]

        self.local_conditionals = None
        self._create_dependency_graph(dist_threshold, range_, max_degree)
        self._create_local_conditionals()

        self.n_groups_per_agent = collections.defaultdict(int)
        for agent in self.agents:
            for group in self.local_conditionals:
                if agent in group.columns:
                    self.n_groups_per_agent[agent] += 1

    def execute(self, joint_arm, noise=None):
        # Set joint arm
        self._set_joint_arm(joint_arm)

        config = self.config.copy()
        if noise is not None:
            wind_speed = sp.stats.norm(loc=config["farm"]["properties"]["wind_speed"], scale=noise).rvs(1)[0]
            config["farm"]["properties"]["wind_speed"] = [wind_speed]
        config['turbine'] = config['turbine'][0]

        # Compute wake field and turbine powers
        self.floris = FlorisInterface(input_dict=config)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.floris.calculate_wake()
        rewards = self.floris.get_turbine_power()
        plt.plot(np.arange(len(rewards)), rewards, 'o')

        return rewards


    def plot_dependency_graph(self):
        plt.clf()

        for n2 in self.nodes.keys():
            for n1 in self.parents[n2]:
                x1, y1 = self.nodes[n1]
                x2, y2 = self.nodes[n2]
                plt.arrow(x1, y1, 0.8*(x2-x1), 0.8*(y2-y1), head_width=100)

        for n, (x, y) in self.nodes.items():
            plt.plot(x, y, 'r.')

        plt.xlabel('x-position (in m)')
        plt.ylabel('y-position (in m)')
        plt.savefig('dependency_graph.pdf')

    def plot_wake_field(self):
        plt.clf()

        # Get horizontal plane at default height (hub-height)
        hor_plane = self.floris.get_hor_plane()

        # Plot
        fig, ax = plt.subplots()
        visualization.visualize_cut_plane(hor_plane, ax=ax)

        plt.xlabel('x-position (in m)')
        plt.ylabel('y-position (in m)')
        plt.savefig('wake_field.pdf')

    def _set_joint_arm(self, joint_arm):
        for ws, turbine_config in zip(joint_arm, self.config['turbine']):
            setpoint_index = np.where(self.wind_speed == ws)[0][0]
            setpoint = self.power_curve[setpoint_index]

            # Cut power curve
            power_curve = self.power_curve.copy()
            if setpoint < 8000:
                power_curve[setpoint_index:] = setpoint

            # Shift peak of thrust curve
            thrust_curve = self.thrust_curve.copy()
            peak_index = np.argmax(thrust_curve)
            if setpoint_index <= peak_index:
                shift_y = thrust_curve[peak_index] - thrust_curve[setpoint_index]
                after_peak = thrust_curve[peak_index:]
                before_setpoint = self.thrust_curve[:setpoint_index]
                thrust_curve = np.hstack((before_setpoint, after_peak - shift_y))
                padding = (len(self.wind_speed) - len(thrust_curve))*[thrust_curve[-1]]
                thrust_curve = np.hstack((thrust_curve, padding))

            # Compute coefficients
            Cp, Ct = compute_Cp_Ct(self.air_density, self.swept_area, self.wind_speed, power_curve, thrust_curve)
            turbine_config["properties"]["power_thrust_table"]["power"] = Cp
            turbine_config["properties"]["power_thrust_table"]["thrust"] = Ct

    def _create_dependency_graph(self, threshold, range_, max_degree):
        X = self.config["farm"]["properties"]["layout_x"]
        Y = self.config["farm"]["properties"]["layout_y"]
        agents = self.config["farm"]["properties"]["turbine_names"]

        # Create nodes and parents
        self.nodes = dict(zip(agents, zip(X, Y)))

        self.parents = collections.defaultdict(list)
        dists = collections.defaultdict(dict)
        for (a1, a2) in itertools.combinations(agents, 2):
            x1, y1 = self.nodes[a1]
            x2, y2 = self.nodes[a2]
            dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if dist <= threshold:
                start_angle = (self.wind_direction + 180 - range_) % 360
                end_angle = (self.wind_direction + 180 + range_) % 360
                if np.sqrt(x1 ** 2 + y1 ** 2) <= np.sqrt(x2 ** 2 + y2 ** 2):
                    if check_inside_sliced_unit_circle((x2 - x1) / threshold, (y2 - y1) / threshold, start_angle, end_angle):
                        self.parents[a2].append(a1)
                        dists[a2][a1] = dist
                else:
                    if check_inside_sliced_unit_circle((x1 - x2) / threshold, (y1 - y2) / threshold, start_angle, end_angle):
                        self.parents[a1].append(a2)
                        dists[a1][a2] = dist

        # Max degree restriction
        for a in self.parents.keys():
            neighbors = sorted(dists[a].items(), key=lambda x: x[1])[:max_degree]
            self.parents[a] = list(list(zip(*neighbors))[0])

        # Create children dict
        self.children = collections.defaultdict(list)
        for node, children in self.parents.items():
            for child in children:
                if node not in self.children[child]:
                    self.children[child].append(node)

    def _create_local_conditionals(self):
        # Create groups
        self.local_conditionals = []
        for n in self.agents:
            group_agents = [n] + self.parents[n]
            arms = itertools.product(self.actions, repeat=len(group_agents))
            group = pd.DataFrame(arms, columns=group_agents)
            self.local_conditionals.append(group)

    def _rotate_farm_around_point(self, config, center, angle):

        X = config["farm"]["properties"]["layout_x"]
        Y = config["farm"]["properties"]["layout_y"]

        s = np.sin(angle)
        c = np.cos(angle)

        for i in range(len(X)):
            x, y = X[i], Y[i]
            x -= center[0]
            y -= center[1]

            x = x * c - y * s
            y = x * s + y * c

            x += center[0];
            y += center[1];

            X[i] = x; Y[i] = y

        config["farm"]["properties"]["layout_x"] = X
        config["farm"]["properties"]["layout_y"] = Y

        return config
