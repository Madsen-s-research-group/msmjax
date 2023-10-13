I came up with a structure for the basic user interface of the MSM. I believe it makes sense to more or less agree on this before starting in earnest on implementation details. If you find the time I would appreciate some feedback :)

I'm not extremely happy with the structure. I find it rather unwieldy, but failed to come up with a much more compact approach while being stateless.

A few comments:

- The intention is to use the neighbor lists built into jax-md for the directly evaluated contributions (lowest-level kernel, short-range). The workflow shown in the example code is (to the best of my knowledge) the way the neighbor lists are meant to be used.
- It might be overkill to use automatic differentiation on the whole potential energy function since the derivatives needed to calculate the forces are in fact fairly simple and used only in a few places:
  - Short-range contributions: The derivative of the lowest-level kernel function is needed. Both the energy and forces use the same neighbor list that can be updated once beforehand, hence the existence of the two functions `calculate_energy_contrib_shortrange` and `calculate_force_contrib_shortrange`.
  - For the grid contribution, the derivative of the B-spline is needed. Both the energy and forces use the same "grid potential" $e^{1+}$ that can be computed once beforehand, hence the existence of the two functions `calculate_energy_contrib_grids` and `calculate_force_contrib_grids`.
- I see a potential problem with the neighbor list allocation when doing distributed computations like in clinamen2. How to avoid reallocating the neighbor list from scratch for every new individual (the allocation isn't jittable and can take several seconds in my tests)?
