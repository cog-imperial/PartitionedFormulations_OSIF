# PartitionedFormulations_OSIF
This repository contains scripts for generating partition-based formulations for trained ReLU neural networks and several Optimal Sparse Input Features (OSIF) test instances implemented in Gurobi. More
details on the problems and methods can be found here: https://arxiv.org/abs/2202.05198.

Please cite this work as:
```
@article{kronqvist2022p,
  title    =   {{P-split formulations: A class of intermediate formulations between big-M and convex hull for disjunctive constraints}},
  author   =   {Kronqvist, Jan and Misener, Ruth and Tsay, Calvin},
  journal  =   {arXiv},
  volume   =   {2202.05198},
  year     =   {2022}
}
```

## Installing Gurobi
The solver software [Gurobi](https://www.gurobi.com) is required to run the examples. Gurobi is a commercial mathematical optimization solver and free of charge for academic research. It is available on Linux, Windows and Mac OS. 

Please follow the instructions to obtain a [free academic license](https://www.gurobi.com/academia/academic-program-and-licenses/). Once Gurobi is installed on your system, follow the steps to setup the Python interface [gurobipy](https://www.gurobi.com/documentation/9.0/quickstart_mac/the_grb_python_interface_f.html).

## Authors
* **[Jan Kronqvist](https://www.kth.se/profile/jankr)** ([jkronqvi](https://github.com/jkronqvi)) - KTH Royal Institute of Technology
* **[Ruth Misener](http://wp.doc.ic.ac.uk/rmisener/)** ([rmisener](https://github.com/rmisener)) - Imperial College London
* **[Calvin Tsay](https://www.imperial.ac.uk/people/c.tsay)** ([tsaycal](https://github.com/tsaycal)) - Imperial College London

## License
This repository is released under the Apache License 2.0. Please refer to the [LICENSE](https://github.com/cog-imperial/PartitionedFormulations_NN/blob/master/LICENSE) file for details.

## Acknowledgements
This work was supported by Engineering & Physical Sciences Research Council (EPSRC) Fellowships to CT and RM (grants EP/T001577/1 and EP/P016871/1), an Imperial College Research Fellowship to CT, a Royal Society Newton International Fellowship (NIF\R1\182194) to JK, a grant by the Swedish Cultural Foundation in Finland to JK, and a grant by the Swedish Research Council (2022-03502) to JK. The project was also in-part financially sponsored by  Digital Futures at KTH through JK and CT.
