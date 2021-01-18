import collections
import itertools
import logging
import numpy as np
import pandas as pd
import pulp

class Directed_Coordination_Graph:
    def __init__(self, groups, actions, penalized_machines, demand, demand_allocation):
        self.actions = actions
        self.nodes = []
        self.parents = collections.defaultdict(list)
        for group in groups:
            node = group.columns[0]
            self.nodes.append(node)
            self.parents[node] = group.columns[1:]
        self.demand = demand

        if demand_allocation is None:
            self.demand_allocation = pd.DataFrame()
            self.demand_allocation['machine'] = self.nodes
            self.demand_allocation['demand'] = self.demand
            self.demand_allocation['regime'] = 0
            self.demand_allocation['regime_demand'] = 1.0
        else:
            self.demand_allocation = demand_allocation
        self.n_regimes = self.demand_allocation['regime'].unique().size

        self.penalized_machines = penalized_machines
        print("Penalized machines:\t", self.penalized_machines)

    def resolve_LP(self, group_means, operation):
        rewards = {}
        for group in group_means:
            agent = group.columns[0]
            rewards[agent] = {}
            for _, entry in group.iterrows():
                local_action, reward = tuple(entry[:-1].tolist()), entry[-1]
                rewards[agent][local_action] = reward

        prob = pulp.LpProblem("Wind Farm Control - Setpoint Configuration", pulp.LpMinimize)

        reward_sum = 0
        reward_regime_sum = self.n_regimes*[0.]

        penalty_sum = 0
        chosen_action_sum = collections.defaultdict(lambda: collections.defaultdict(int))  # Restriction 2 - Part A

        resolved_nodes = set()
        all_variables = []
        while True:
            current = self._select_independent_node(resolved_nodes)
            if current is None:
                break

            parents = self.parents[current]
            total_sum = 0

            regime = self.demand_allocation['regime'].loc[self.demand_allocation['machine'] == current]
            regime = regime.iloc[0]

            if len(parents) == 0:
                for a in self.actions:
                    local_action = (a,)
                    x = pulp.LpVariable(f"x_{current}_" + '-'.join(list(map(str, local_action))), lowBound=0, upBound=1, cat=pulp.LpInteger)
                    all_variables.append(x)

                    reward = rewards[current][local_action]
                    reward_sum += reward * x
                    reward_regime_sum[regime] += reward * x
                    penalty_sum += self._penalty(current, reward) * x

                    total_sum += x
                    chosen_action_sum[current][a] += x  # Restriction 2 - Part A
            else:
                parent_action_sum = collections.defaultdict(
                    lambda: collections.defaultdict(int))  # Restriction 2 - Part B

                for a in self.actions:
                    for parent_actions in itertools.product(*([self.actions] * len(parents))):
                        local_action = (a,) + parent_actions
                        x = pulp.LpVariable(f"x_{current}_" + 'x'.join(list(map(str, local_action))), lowBound=0, upBound=1, cat=pulp.LpInteger)
                        all_variables.append(x)

                        reward = rewards[current][local_action]
                        reward_sum += reward * x
                        reward_regime_sum[regime] += reward * x
                        penalty_sum += self._penalty(current, reward) * x
                        total_sum += x
                        chosen_action_sum[current][a] += x  # Restriction 2 - Part A

                        for parent, a2 in zip(parents, parent_actions):
                            parent_action_sum[parent][a2] += x  # Restriction 2 - Part B

                # Restriction 2: Chosen action of parent should match the conditional action of the current
                for parent in parents:
                    for a2 in self.actions:
                        prob += chosen_action_sum[parent][a2] == parent_action_sum[parent][a2]

            # Restriction 1: Exactly one action per agent
            prob += total_sum == 1

            resolved_nodes.add(current)


        if operation == 'desired':
            # Objective function: Bring power production close to demand
            eps = 0.0
            for regime, df in self.demand_allocation.groupby('regime'):
                eps_regime = pulp.LpVariable(f"eps{regime}", lowBound=0, cat=pulp.LpContinuous)
                prob += reward_regime_sum[regime] <= self.demand*df['regime_demand'].iloc[0] + eps_regime
                prob += reward_regime_sum[regime] >= self.demand*df['regime_demand'].iloc[0] - eps_regime
                eps += eps_regime
            prob += eps + penalty_sum
        elif operation == 'max':
            prob += -reward_sum + penalty_sum
        else:
            ValueError(f"Operation '{operation}' does not exist.")

        logging.info("LP solving...")
        prob.solve()
        logging.info(f"LP status: {pulp.LpStatus[prob.status]}")

        # Construct best joint action
        max_joint_action = pd.Series(np.nan, index=self.nodes)
        r_test = 0
        for x in all_variables:
            if pulp.value(x) == 1:
                agent, action = x.name.split('_')[1:]
                action = action.split('x')[0]
                max_joint_action[agent] = float(action)

                joint_action = x.name.split('_')[-1].split('x')
                joint_action = tuple(float(a) for a in joint_action)
                r_test += rewards[agent][joint_action]
        return max_joint_action, pulp.value(reward_sum)

    def _select_independent_node(self, resolved_nodes):
        for node in self.nodes:
            if node not in resolved_nodes and len(set(self.parents[node]) - resolved_nodes) == 0:
                return node
        return None

    def _penalty(self, agent, power):
        if agent in self.penalized_machines and power >= 5000000:
            return 1.0e10
        else:
            return 0.0
