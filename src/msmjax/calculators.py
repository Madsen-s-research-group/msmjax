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
from msmjax.bspline_interpolation.gridops_clean import (
    create_all_grid_to_grid_ops,
    make_basis_evaluation_fn,
)
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
from msmjax.kernels import (
    SofteningFunctionOneOverR,
    make_construct_stencils,
    split_one_over_r_kernel,
)

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
    # -------------------------------------------------------------------------
    # Basic MSM settings
    # -------------------------------------------------------------------------
    p: int  # TODO: Name: 'order'?
    mu: int  # TODO: One value per level?
    max_level_split: int
    max_level_grids: int
    # TODO: Should grid_shapes, cutoff_radii, grid_spacings only include up to
    #  max_level_grids in the first place, or should the setup process take
    #  care of only using them up max_level_grids?
    grid_shapes: Sequence[tuple[int, ...]]
    cutoff_radii: Sequence[float]
    grid_spacings: Sequence[onp.ndarray]
    # -------------------------------------------------------------------------
    # Geometry-related (arguably)
    # -------------------------------------------------------------------------
    cell: onp.ndarray
    cell_mode: CellMode
    pbc: Sequence[bool]  # TODO: type? (for consistent serialization)
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
    convolution_methods: Sequence[ConvMeth]
    # TODO: Is it possible/helpful to handle dynamic cells like this?
    #  The main intended benefit is to make it clear upon inspection of model
    #  params that a transform is used, so they don't assume an error because
    #  the grid spacings and extents don't match the input values.
    #  Note: cell_mode = "general" in combination with grids_defined_on_unitcube = False would be invalid
    grids_defined_on_unitcube: bool
    stencil_extents_from_center: Sequence[tuple[int, ...]]
    # -------------------------------------------------------------------------
    # Info
    # -------------------------------------------------------------------------
    n_dim: int
    version: str = msmjax.__version__
    info: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        self.pbc = tuple(self.pbc)
        self.supercell_diag = (
            None if self.supercell_diag is None else tuple(self.supercell_diag)
        )
        self.convolution_methods = list(self.convolution_methods)
        # TODO: grid shapes to list (of tuples)
        # TODO: grid sizes to list (of int) (or don't include at all?)
        # TODO: stencil extents to list (of tuples)
        # TODO: spacings on all levels to list (of array? of tuple?)
        # TODO: cutoffs on all levels to list (of float?)
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
        level_zero_cutoff=params.cutoff_radii[0],
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

    # TODO: (Re-)compute omega here, or should it be part of params (maybe just in info)?
    #  - same for J
    omega, _ = compute_coeffs_with_truncation(params.p, params.mu)

    n_levels_intermed = params.max_level_split - 1
    include_toplevel = params.max_level_grids == params.max_level_split
    if n_levels_intermed > 0:
        k_lowest_intermed = kernel_fns[1]
        extents_intermed = params.stencil_extents_from_center[1]
    else:
        (k_lowest_intermed, extents_intermed) = (None, None)
    if include_toplevel:
        k_toplevel = kernel_fns[-1]
        grid_shape_toplevel = params.grid_shapes[-1]
    else:
        (k_toplevel, grid_shape_toplevel) = (None, None)
    construct_stencils = make_construct_stencils(
        omega=omega,
        n_levels_intermed=n_levels_intermed,
        include_toplevel=include_toplevel,
        k_lowest_intermed=k_lowest_intermed,
        extents_from_center_intermed=extents_intermed,
        k_toplevel=k_toplevel,
        grid_shape_toplevel=grid_shape_toplevel,
    )
    if params.cell_mode == "ortho":
        kernel_stencils = jax.jit(construct_stencils)(params.grid_spacings[1])
    elif params.cell_mode == "general":
        one_grid_cell = (
            params.cell
            * (params.grid_spacings[1] / onp.linalg.norm(params.cell, axis=1))[
                :, onp.newaxis
            ]
        )
        kernel_stencils = jax.jit(construct_stencils)(one_grid_cell)
    # TODO: (where to) check for invalid cell_mode?

    # TODO: compute_u_oneplus must include transformation to unit cube
    #  if cell_mode == "general"
    (
        restriction_fns,
        prolongation_fns,
        convolution_fns,
    ) = create_all_grid_to_grid_ops(
        grid_shapes=params.grid_shapes,
        p=params.p,
        pbc=params.pbc,
        convolution_methods=params.convolution_methods,
    )
    grid_pass_fn = make_grid_pass_fn(
        restriction_fns, prolongation_fns, convolution_fns
    )
    basis_evaluation_fn = make_basis_evaluation_fn(
        grid_shape=params.grid_shapes[1], p=params.p, pbc=params.pbc
    )
    compute_u_oneplus = make_compute_u_oneplus(
        singleparticle_basis_fn_lvl_one=partial(
            basis_evaluation_fn, spacings=params.grid_spacings[1]
        ),
        grid_pass_fn=grid_pass_fn,
        grid_shape_lvl_one=params.grid_shapes[1],
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
