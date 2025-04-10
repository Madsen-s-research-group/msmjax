from dataclasses import dataclass
from functools import partial
from typing import Literal, Sequence

import numpy.typing as npt
from jax import Array
from jax.typing import ArrayLike

from msmjax.core.longrange import (
    make_compute_u_oneplus,
    make_dyn_cell_longrange_fn,
    make_grid_pass_fn,
)
from msmjax.core.shortrange import (
    make_compute_u_zero,
    make_eval_pair_pot,
    make_eval_pair_pot_with_neighbor_list,
)
from msmjax.kernels import SofteningFunctionOneOverR, split_one_over_r_kernel

# TODO: This should be defined elsewhere, since the longrange part will likely
#  also use it
# TODO: Name that is meaningful in all places where this is used?
CellMode = Literal["ortho", "general"]


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
    ...
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

    if use_neighborlist:
        pair_map_fn = partial(
            make_eval_pair_pot_with_neighbor_list,
            pbc=pbc,
            cell_mode=cell_mode,
        )
    else:
        pair_map_fn = partial(
            make_eval_pair_pot,
            pbc=pbc,
            cell_mode=cell_mode,
            supercell_diag=supercell_diag,
        )

    compute_u_zero = make_compute_u_zero(
        kernel_fns=kernel_fns, pair_map_fn=pair_map_fn
    )
    compute_u_oneplus = make_compute_u_oneplus(...)  # TODO

    # TODO: Offer versions with and without neighbor list without having to
    #  define the energy function and all its derivatives twice, inside
    #  inside both branches of an if-else that switches between
    #  neighbor list and no neighbor list?
    def compute_energy(positions, charges, neighbor_list):
        u_zero = compute_u_zero(
            positions,
            charges,
            cell=cell,
            neighbor_list=neighbor_list,
            weights=1.0,
        )
        u_oneplus = compute_u_oneplus(positions, charges, kernel_stencils)
        return u_zero + u_oneplus

    # TODO: Functions for forces, energy and forces, charge gradient, stress, ..?
    return compute_energy


def set_up_dyn_cell_msm(params: DynCellMSMParams):
    ...
