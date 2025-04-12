import dataclasses
import json
from functools import partial
from pathlib import Path
from typing import Literal, Sequence

import jax
import jax.numpy as jnp
import numpy as onp
import numpy.typing as npt
from jax import Array
from jax.typing import ArrayLike

import msmjax
from msmjax.bspline_interpolation.coefficients import (
    compute_coeffs_with_truncation,
    compute_J_zeroplus,
)
from msmjax.bspline_interpolation.gridops import create_all_grid_to_grid_ops
from msmjax.convenience import set_up_kernels_grids_and_stencils
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


class CustomJSONEncoder(json.JSONEncoder):
    """Class that extends JSONEncoder to handle different data types."""

    # TODO: clinamen2 attribution

    def default(self, o):
        """Return a json-izable version of o or delegate on the base class."""
        if isinstance(o, onp.generic):
            # Deal with non-serializable types such as numpy.int64
            return o.item()
        elif isinstance(o, onp.ndarray):
            nruter = {
                "main_type": "NumPy/" + o.dtype.name,
                "data": o.tolist(),
            }
            return nruter
        return json.JSONEncoder.default(self, o)


class CustomJSONDecoder(json.JSONDecoder):
    """Class that extends the JSONDecoder to handle different data types."""

    # TODO: clinamen2 attribution

    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(
            self, object_hook=self.object_hook, *args, **kwargs
        )

    def object_hook(self, o):
        """Reencode numpy arrays from dictionary."""
        try:
            main_type, *extra = o["main_type"].split("/")
            if main_type == "NumPy":
                return onp.asarray(o["data"], dtype=extra[0])
        except (KeyError, ValueError):
            return o


@dataclasses.dataclass
class StaticCellMSMParams:
    # TODO: Include package version?
    # -------------------------------------------------------------------------
    # Basic MSM settings
    # -------------------------------------------------------------------------
    p: int  # TODO: Name: 'order'?
    mu: int  # TODO: One value per level?
    # TODO: Name: cutoff? r_cut? Mention 'level_zero'? Check if fits?
    #  Store the cutoffs at ALL levels? (Maybe under info?)
    r_cut_0: float
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
    # TODO: Is it possible/helpful to handle dynamic cells like this?
    #  The main intended benefit is to make it clear upon inspection of model
    #  params that a transform is used, so they don't assume an error because
    #  the grid spacings and extents don't match the input values.
    #  Note: cell_mode = "general" in combination with grids_defined_on_unitcube = False would be invalid
    grids_defined_on_unitcube: bool
    ...  # TODO: all other grid stuff
    # -------------------------------------------------------------------------
    # Info
    # -------------------------------------------------------------------------
    # n_dim: int
    # # TODO: Store number of particles? In principle, they are not a fundamental
    # #  model attribute. Users should be allowed to run the same model on
    # #  different numbers of particles.
    # n_particles: int
    # omega: ArrayLike  # TODO: Non-negative-index part or full? Include at all?
    # J: ArrayLike  # TODO: Non-negative-index part or full? Include at all? Lowercase name?
    # kernel_stencils: Sequence[ArrayLike]  # TODO: Include??? Type?
    version: str = msmjax.__version__

    def __post_init__(self):
        self.h_1 = onp.asarray(self.h_1)
        self.pbc = tuple(self.pbc)
        self.supercell_diag = (
            None if self.supercell_diag is None else tuple(self.supercell_diag)
        )
        self.convolution_methods = list(self.convolution_methods)
        # TODO: grid shapes to list (of tuples)
        # TODO: grid sizes to list (of int) (or don't include at all?)
        # TODO: spacings on all levels to list (of array? of tuple?)
        # TODO: cutoffs on all levels to list (of array? of tuple?)
        # TODO: J to onp.array
        # TODO: omega to onp.array
        # TODO: kernel_stencils to onp.array (or don't include at all?)

    def save_json(self, filename: str | Path, **kwargs) -> None:
        with open(filename, "w") as f:
            json.dump(
                dataclasses.asdict(self), f, cls=CustomJSONEncoder, **kwargs
            )

    @classmethod
    def load_json(cls, filename: str | Path):
        with open(filename, "r") as f:
            params_dict = json.load(f, cls=CustomJSONDecoder)
        return cls(**params_dict)


@dataclasses.dataclass
class DynCellMSMParams:
    # TODO
    p: int
    mu: int


def set_up_params_static_cell():
    pass  # TODO


def static_cell_msm(params: StaticCellMSMParams):
    kernel_fns = split_one_over_r_kernel(
        max_level=params.max_level_split,
        level_zero_cutoff=params.r_cut_0,
        softening_function=SofteningFunctionOneOverR(params.p),
    )

    if params.use_neighborlist:
        compute_u_zero = make_compute_u_zero(
            kernel_fns=kernel_fns,
            pair_map_fn=partial(
                make_eval_pair_pot_neighborlist,
                pbc=params.pbc,
                cell_mode=params.cell_mode,
            ),
        )
    else:
        compute_u_zero = make_compute_u_zero(
            kernel_fns=kernel_fns,
            # TODO: If I allow supercell_diag for make_eval_pair_pot_neighborlist
            #  as well, this if-else could be much cleaner.
            pair_map_fn=partial(
                make_eval_pair_pot,
                pbc=params.pbc,
                cell_mode=params.cell_mode,
                supercell_diag=params.supercell_diag,
            ),
        )

    # TODO: Should these be computed here, or be part of params?
    #  A thought: For debugging and post-calculation analysis, they should
    #             definitely be included in the params, whether the setup
    #             process does actually take them from there or not.
    omega, _ = compute_coeffs_with_truncation(params.p, params.mu)
    J_zeroplus = compute_J_zeroplus(params.p)

    # TODO: grids
    # TODO: stencils

    # TODO: Replace with more specific reworked functions for grid and stencil setup
    # TODO: Currently this only works for ortho cells!
    _, grids, kernel_stencils = set_up_kernels_grids_and_stencils(
        box_lengths=onp.diag(params.cell),
        pbc=params.pbc,
        level_one_gridspacing=params.h_1,
        level_zero_cutoff=params.r_cut_0,
        p=params.p,
        mu=params.mu,
        n_levels=params.max_level_split,
    )
    grids = grids[: params.max_level_eval + 1]
    kernel_stencils = kernel_stencils[: params.max_level_eval + 1]

    # TODO: compute_u_oneplus must include transformation to unit cube
    #  if cell_mode == "general"
    (
        restriction_fns,
        prolongation_fns,
        interaction_fns,
    ) = create_all_grid_to_grid_ops(grids, params.convolution_methods)
    grid_pass_fn = make_grid_pass_fn(
        restriction_fns, prolongation_fns, interaction_fns
    )
    compute_u_oneplus = make_compute_u_oneplus(
        singleparticle_basis_fn_lvl_one=grids[
            1
        ].evaluate_bspline_basis_one_particle,
        grid_pass_fn=grid_pass_fn,
        grid_shape_lvl_one=grids[1].shape,
    )

    def calc_energy(positions, charges, neighborlist=None):
        if params.use_neighborlist:
            u_zero = compute_u_zero(
                positions,
                charges,
                cell=params.cell,
                weights=1.0,
                neighborlist=neighborlist,
            )
        else:
            # TODO: Raise an error/warn if use_neighborlist is False
            #  but the neighborlist parameter is not None?
            u_zero = compute_u_zero(
                positions,
                charges,
                cell=params.cell,
            )
        u_oneplus = compute_u_oneplus(positions, charges, kernel_stencils)
        return u_zero + u_oneplus

    def calc_forces(positions, charges, neighborlist=None):
        return -jax.grad(calc_energy, argnums=0)(
            positions, charges, neighborlist
        )

    def calc_energy_and_forces(positions, charges, neighborlist=None):
        value, grad = jax.value_and_grad(calc_energy, argnums=0)(
            positions, charges, neighborlist
        )
        return value, -grad

    def calc_charge_gradient(positions, charges, neighborlist=None):
        return jax.grad(calc_energy, argnums=1)(
            positions, charges, neighborlist
        )

    # TODO: Option to return fns for short- and long-range part separately?

    return (
        calc_energy,
        calc_forces,
        calc_energy_and_forces,
        calc_charge_gradient,
    )


def dyn_cell_msm(params: DynCellMSMParams):
    # TODO: Functions for forces, energy and forces, charge gradient, ...
    #  + stress!!!
    pass  # TODO
