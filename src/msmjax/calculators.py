from dataclasses import dataclass
from functools import partial
from typing import Literal, Sequence

import jax
import jax.numpy as jnp
import numpy.typing as npt
from jax import Array
from jax.typing import ArrayLike

from grid_ops_experiment.grid_ops_experiment_multidim import J_zeroplus
from msmjax.core.longrange import (
    make_compute_u_oneplus,
    make_dyn_cell_longrange_fn,
    make_grid_pass_fn,
)
from msmjax.core.shortrange import (
    make_compute_u_zero,
    make_eval_pair_pot,
    make_eval_pair_pot_neighborlist,
)
from msmjax.kernels import SofteningFunctionOneOverR, split_one_over_r_kernel

# TODO: This should be defined elsewhere, since the longrange part will likely
#  also use it
# TODO: Name that is meaningful in all places where this is used?
CellMode = Literal["ortho", "general"]
ConvMeth = Literal["direct", "fft"]


@dataclass
class StaticCellMSMParams:
    # -------------------------------------------------------------------------
    # Basic MSM settings
    # -------------------------------------------------------------------------
    p: int  # TODO: Name: 'order'?
    mu: int  # TODO: One value per level?
    r_cut_0: float  # TODO: cutoff? r_cut? Mention 'level_zero'? Check if fits?
    h_1: Sequence[float]  # TODO: Allow only array? Name: grid_spacings?
    # TODO: The two 'max_level_*' are related to one another, to pbc and grids.
    #  Check consistency? Move to 'Long-range evaluation' or 'Info' sections?
    max_level_split: int
    max_level_eval: int
    # -------------------------------------------------------------------------
    # Geometry-related (arguably)
    # -------------------------------------------------------------------------
    cell: ArrayLike  # TODO: type?
    cell_mode: CellMode
    pbc: Sequence[bool]  # TODO: type?
    # -------------------------------------------------------------------------
    # Short-range evaluation
    # -------------------------------------------------------------------------
    # TODO: Make optional? Interaction with `use_neighbor_list`?
    #  (Where to) check if cutoff fits?
    supercell_diag: Sequence[int]
    use_neighborlist: bool  # TODO: Interaction with `supercell_diag`?
    ...
    # -------------------------------------------------------------------------
    # Long-range evaluation
    # -------------------------------------------------------------------------
    convolution_methods: Sequence[str]
    ...  # TODO: all grid stuff
    # -------------------------------------------------------------------------
    # Info
    # -------------------------------------------------------------------------
    n_dim: int
    # TODO: Store number of particles? In principle, they are not a fundamental
    #  model attribute. Users should be allowed to run the same model on
    #  different numbers of particles.
    n_particles: int
    omega: ArrayLike  # TODO: Non-negative-index part or full? Include at all?
    J: ArrayLike  # TODO: Non-negative-index part or full? Include at all? Lowercase name?
    kernel_stencils: Sequence[ArrayLike]  # TODO: Include??? Type?


def set_up_static_cell_msm(params: StaticCellMSMParams):
    kernel_fns = split_one_over_r_kernel(
        max_level=max_level_split,
        level_zero_cutoff=level_zero_cutoff,
        softening_function=SofteningFunctionOneOverR(p),
    )

    static_pair_eval_kwargs = {"pbc": pbc, "cell_mode": cell_mode}
    if use_neighborlist:
        pair_map_fn = partial(
            make_eval_pair_pot_neighborlist, **static_pair_eval_kwargs
        )
    else:
        static_pair_eval_kwargs["supercell_diag"] = supercell_diag
        pair_map_fn = partial(make_eval_pair_pot, **static_pair_eval_kwargs)

    # TODO: Close over cell (currently, this happens inside compute_energy
    #  -> do it here instead?)
    # TODO: Close over weights if use_neighborlist?
    compute_u_zero = make_compute_u_zero(
        kernel_fns=kernel_fns, pair_map_fn=pair_map_fn
    )

    # TODO: coefficients
    omega, _ = compute_coeffs_with_truncation(p, mu)
    J_zeroplus = compute_J_zeroplus(p)

    # TODO: grids

    # TODO: stencils

    compute_u_oneplus = make_compute_u_oneplus(...)  # TODO

    # TODO: Offer versions with and without neighbor list without having to
    #  define the energy function and all its derivatives twice, inside both
    #  both branches of an if-else that switches between neighbor list and
    #  no neighbor list?
    if use_neighborlist:

        def calc_energy(positions, charges, neighborlist):

            # TODO: Could cell and weights be fixed further up? This would
            #  allow writing a unified calc_energy function for both with
            #  and without neighbor list, if we accept that the
            #  no-neighbor-list version has an (unused) 'neighborlist'
            #  parameter as well.

            u_zero = compute_u_zero(
                positions,
                charges,
                cell=cell,
                neighbor_list=neighborlist,
                weights=1.0,
            )
            u_oneplus = compute_u_oneplus(positions, charges, kernel_stencils)
            return u_zero + u_oneplus

    def calc_forces(positions, charges, neighbor_list):
        return -jax.grad(calc_energy, argnums=0)(
            positions, charges, neighbor_list
        )

    def calc_energy_and_forces(positions, charges, neighbor_list):
        value, grad = jax.value_and_grad(calc_energy, argnums=0)(
            positions, charges, neighbor_list
        )
        return value, -grad

    # TODO: charge gradient?

    return calc_energy, calc_forces, calc_energy_and_forces


def set_up_dyn_cell_msm(params: DynCellMSMParams):
    # TODO: Functions for forces, energy and forces, charge gradient, ...
    #  + stress!!!
    pass  # TODO
