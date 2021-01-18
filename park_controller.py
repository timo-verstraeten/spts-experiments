from mats.linear_programming import Directed_Coordination_Graph
from mats.posteriors import WindFarmGaussianPosterior
from mats.thompson_sampling import MultiAgentThompsonSampling
from scenario import Synthetic_Scenario

import json
import logging
import numpy as np
import pandas as pd


class SPTS:

    def __init__(self, wind_speed, wind_direction, turbulence_intensity, demand, penalized_machines, demand_allocation=None):
        actions = [13.5, 10.0, 6.5]

        self.demand = demand
        self.wind_speed_noise = 1e-5  # jitter for numerical stability

        max_power = 8*10**6 * 24

        with open('configs/farm_specs.json', 'rb') as f:
            config = json.load(f)

        # Set environment parameters
        config["farm"]["properties"]["wind_speed"] = [wind_speed]
        config["farm"]["properties"]["wind_direction"] = [225.0]
        config["farm"]["properties"]["turbulence_intensity"] = [turbulence_intensity]
        config["farm"]["properties"]["wind_x"] = [0]
        config["farm"]["properties"]["wind_y"] = [0]

        # Create environment
        self.env = Synthetic_Scenario(config, actions, 1000, 20, 50, wind_direction)

        # Create controller
        wind_speed_bins = config["turbine"][0]["properties"]["power_thrust_table"]["wind_speed"]
        power_curve = config["turbine"][0]["properties"]["power_curve"]
        power_curve = pd.Series(power_curve, index=wind_speed_bins)*1000
        power_curve[0.0] = 0.0

        self.demand = demand
        priors = [[WindFarmGaussianPosterior(1, np.min([arm.iloc[0], wind_speed]), power_curve) for _, arm in arms.iterrows()] for arms in self.env.local_conditionals]
        graph = Directed_Coordination_Graph(self.env.local_conditionals, actions, penalized_machines, self.demand, demand_allocation)
        self.mats = MultiAgentThompsonSampling(self.env.local_conditionals, priors, actions, graph, max_power)

    def train(self, n_iter):
        # Run MATS
        rewards = []
        chosen_arms = []
        best_reward = float('-inf')
        expected_rewards = []

        results = []
        for i in range(n_iter):
            logging.info(f"Iteration {i}")
            # Do step with MATS
            logging.info("-- pull...")
            joint_arm, expected_reward = self.mats.pull()
            logging.info("-- execute...")
            local_rewards = self.env.execute(joint_arm, self.wind_speed_noise)
            logging.info("-- update...")
            self.mats.update(joint_arm, local_rewards)

            # Logging
            reward = sum(local_rewards)
            if abs(best_reward - self.demand) > abs(reward - self.demand):
                best_reward = reward
            logging.info(f"{i}\t{reward}\t{joint_arm.values}")
            rewards.append(reward)
            chosen_arms.append(joint_arm)
            expected_rewards.append(expected_reward)

            for agent, arm, reward in zip(self.env.agents, joint_arm, local_rewards):
                results.append([i, agent, arm, reward])

        columns = ['iter', 'machine', 'action', 'reward']
        results = pd.DataFrame(results, index=np.arange(len(results)), columns=columns)

        return results

class Heuristic_Approach:

    def __init__(self, wind_speed, wind_direction, turbulence_intensity, demand):
        self.actions = [13.5, 10.0, 6.5]

        self.demand = demand

        with open('configs/farm_specs.json', 'rb') as f:
            config = json.load(f)

        # Set environment parameters
        config["farm"]["properties"]["wind_speed"] = [wind_speed]
        config["farm"]["properties"]["wind_direction"] = [wind_direction]
        config["farm"]["properties"]["turbulence_intensity"] = [turbulence_intensity]
        config["farm"]["properties"]["wind_x"] = [0]
        config["farm"]["properties"]["wind_y"] = [0]

        # Create environment
        self.env = Synthetic_Scenario(config, self.actions, 1000, 20, 50, wind_direction)

    def optimize(self):
        # PHASE 1: Order turbines from back to front
        X = np.array(self.env.config["farm"]["properties"]["layout_x"])
        Y = np.array(self.env.config["farm"]["properties"]["layout_y"])
        agents = self.env.config["farm"]["properties"]["turbine_names"]
        dist = np.sqrt(X**2 + Y**2)
        agents = sorted(zip(dist, agents))
        agents = list(reversed(list(zip(*agents))[1]))

        joint_arm = pd.Series([self.actions[-1]]*len(self.env.nodes), index=self.env.nodes)
        best_err, best_joint_arm = float("+inf"), None

        while True:
            rewards = self.env.execute(joint_arm)
            err = sum(rewards) - self.demand

            if np.abs(err) < best_err:
                best_err = np.abs(err)

            if err < 0:
                action = joint_arm[agents[0]]
                index = np.where(np.array(self.actions) == action)[0][0]
                index = index - 1
                if index < 0:
                    agents = agents[1:]
                    index = len(self.actions) - 1
                if len(agents) == 0:
                    break
                joint_arm[agents[0]] = self.actions[index]
            else:
                break

        rewards = self.env.execute(joint_arm)
        results = joint_arm.to_frame().rename(columns={0: 'action'})
        results['reward'] = rewards
        results = results.reset_index().rename(columns={'index': 'machine'})

        return results