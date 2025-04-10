from dataclasses import dataclass
from typing import Literal, Sequence

import numpy.typing as npt
from jax import Array
from jax.typing import ArrayLike

from msmjax.core.longrange import (
    make_compute_u_oneplus,
    make_dyn_cell_longrange_fn,
    make_grid_pass_fn,
)

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
    use_neighbor_list: bool  # TODO: Interaction with `supercell_diag`?
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


def set_up_static_cell_msm(params: StaticCellMSMParams):
    ...


def set_up_dyn_cell_msm(params: DynCellMSMParams):
    ...
