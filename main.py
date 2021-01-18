from park_controller import SPTS, Heuristic_Approach

import argparse
import json
import logging
import numpy as np
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SPTS experiments')
    parser.add_argument('--method', choices=['spts', 'heuristic'], type=str, help="The control method used for allocating set-points")
    parser.add_argument('--seed', type=int, help="Random seed")
    parser.add_argument('--wind_30_degrees', action='store_true', help="Incoming wind direction is 30 degrees (default: dominant wind direction, 0 degrees)")
    parser.add_argument('--demand', type=float, help="Total farm power demand")
    parser.add_argument('--n_penalized_machines', type=int, help="Number of high-risk turbines")
    parser.add_argument('--file_dir', type=str, help="Location where the results will be written")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger().setLevel(logging.INFO)

    # Read farm config
    with open('configs/farm_specs.json', 'rb') as f:
        config_farm = json.load(f)

    seed = args.seed
    wind_speed = 11.0
    if args.wind_30_degrees:
        wind_direction = 255.0  # 30 degrees offset from dominant wind direction
    else:
        wind_direction = 225.0  # South-east is the dominant wind direction
    demand = args.demand*1.0e6
    n_penalized_machines = args.n_penalized_machines

    np.random.seed(seed)

    if args.method == 'spts':
        n_iter = 200

        # Regimes based on clustering outcomes
        regimes = [['A0', 'B0', 'C0', 'D0']]
        regimes += [np.hstack([[f'A{i}', f'B{i}', f'C{i}', f'D{i}'] for i in range(1,2)])]
        regimes += [np.hstack([[f'A{i}', f'B{i}', f'C{i}', f'D{i}'] for i in range(2,4)])]
        regimes += [np.hstack([[f'A{i}', f'B{i}', f'C{i}', f'D{i}'] for i in range(4,6)])]

        # Demand allocation
        loads = pd.read_csv('loads.csv').rename(columns={'Unnamed: 0': 'machine', '0': 'load'})
        loads['load'] = 1 - loads['load']
        loads['load'] = loads['load'] / loads['load'].sum()
        dfs = []
        for _, entry in loads.iterrows():
            for i, regime in enumerate(regimes):
                if entry[0] in regime:
                    entry['regime'] = i
                    break
            dfs.append(entry)
        loads = pd.concat(dfs, axis=1, sort=False).T
        demand_allocation = loads.merge(loads.groupby('regime').sum()['load'].reset_index(), on="regime", how="left")
        demand_allocation.rename(columns={'load_y': 'regime_demand', 'load_x': 'demand'}, inplace=True)
        demand_allocation['demand'] = demand_allocation['demand'] / demand_allocation['regime_demand']

        # Randomly choose high-risk turbines
        penalized_machines = []
        remaining_regimes = regimes.copy()
        np.random.shuffle(remaining_regimes)
        for i in range(n_penalized_machines):
            regime = remaining_regimes[0]
            remaining_regimes = remaining_regimes[1:]
            penalized_machines.append(np.random.choice(regime))

        # Optimization
        controller = SPTS(wind_speed, wind_direction, 0.2, demand, penalized_machines, demand_allocation)
        controller.env.plot_dependency_graph()
        max_power = sum(controller.env.execute([13.5] * 24))
        controller.env.plot_wake_field()
        results = controller.train(n_iter)

        machines = pd.Series(0, index=controller.env.nodes)
        machines.loc[penalized_machines] = 1
        machines = machines.to_frame().reset_index().rename(columns={0: 'penalized', 'index': 'machine'})
        results = results.merge(machines, on='machine', how='left')

        # WRITE FILE
        filename = f'{args.file_dir}/{int(wind_direction)}_{"{:.1f}".format(wind_speed)}_{int(demand/1e6)}MW_{n_penalized_machines}_{seed}_spts.csv'
        results.to_csv(filename, index=False)

    elif args.method == 'heuristic':
        controller = Heuristic_Approach(wind_speed, wind_direction, 0.2, demand)
        results = controller.optimize()
        filename = f'{args.file_dir}/{int(wind_direction)}_{"{:.1f}".format(wind_speed)}_{int(demand/1e6)}MW_{n_penalized_machines}_{seed}_heuristic.csv'
        results.to_csv(filename, index=False)


