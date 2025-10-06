Structures used in `examples/benchmarks/` as well as in some of the tests.
The structures are artificial (not meaning to represent any particular real system), but were constructed such that no two particles are too close together, by a combination of PACKMOL and post-processing that optimizes particle positions near the edges under periodic boundary conditions.
The code with which the structures were generated is contained in the [generate_structures.ipynb](generate_structures.ipynb) notebook.

For 100 and 10000 particles, which are the numbers of particles, at which accuracy benchmarks were run under `examples/benchmarks/cost_vs_accuracy/` there are 11 different structures in order to get better statistics on accuracy, while for all other particle numbers there is only one structure each.
