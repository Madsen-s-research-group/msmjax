Example demonstrating use of msmJAX in simple MD simulations.
The system consists of particles with fixed charges that additionally interact over short range via a Lennard-Jones potential.
`ASE`'s MD engine is used for simplicity.

To run a simulation in the respective ensemble, navigate to one of the `NVE/`, `NVT/` or `NPT-temperature-ramp/` directories and execute `runscript.sh`, which will call the `run_md_ase.py` script with appropriate command-line arguments.

These directories contain all necessary input files for running the MD simulations, as well as example plots of various quantities over the trajectories, in the `plots/` subdirectories.
The trajectory data from which these plots were obtained are not included for file size reasons.

The plots were made with the [`plot_along_trajectories.ipynb`](plot_along_trajectories.ipynb) notebook, which you can use to plot your own trajectories as well.

The [`set_up_NPT-temperature-ramp_ase.ipynb`](set_up_NPT_ase.ipynb) and [`set_up_NVE_and_NVT_ase.ipynb`](set_up_NVE_and_NVT_ase.ipynb) notebooks are what was used to set up the inputs for the simulations, and are included for completeness.
You don't need to execute them yourselves for running the example.
If you wish to do so, you need [`Clinamen2`](https://pypi.org/project/Clinamen2/) and [`PACKMOL`](https://m3g.github.io/packmol/).
