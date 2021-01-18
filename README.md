README
======

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

This repository contains the source code used for the experiments conducted in the paper titled "Scalable Optimization for Wind Farm Control using
Coordination Graphs" presented at AAMAS 2021.

```
@inproceedings{verstraeten2021spts,
  title={Scalable Optimization for Wind Farm Control using Coordination Graphs},
  author={Verstraeten, Timothy and Daems, Pieter-Jan and Bargiacchi, Eugenio and Roijers, Diederik M. and Libin, Pieter J.K. and Helsen Jan},
  booktitle={Proceedings of the 20th International Conference on Autonomous Agents and Multiagent Systems (AAMAS)},
  year={2021}
}
```

The paper proposes a new method for active power control in wind farms, called Set-Point Thompson Sampling (SPTS), which optimizes and allocates power set-points to each wind turbine to match a given power demand.
Moreover, the method considers a cost function that reflects the health status of turbines, and prevents damage-inducing set-points.

Requirements
------------

The project requires:

- Python 3.7
- FLORIS simulator (see requirements.txt)
- Python libraries (see requirements.txt)

Use the following command to install the required Python libraries:
```
pip install -r requirements.txt
```

Instructions
------------

Run the main.py file with parameters to perform an experiment.
For example,
```
python main.py --method=spts --seed=0 --demand=60 --n_penalized_machines=3 --file_dir=results
```

For more information about the arguments provided to the Main.py file, use the following command:
```
python main.py --help
```
