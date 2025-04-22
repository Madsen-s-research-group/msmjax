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
    find_spacings_and_max_level_periodic,
    make_basis_evaluation_fn,
    set_up_grids_all_levels,
    suggest_max_grid_level_nonperiodic,
)
from msmjax.convenience import suggest_p
from msmjax.core.longrange import make_compute_u_oneplus, make_grid_pass_fn
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
ConvMeth = Literal["scipy-direct", "scipy-fft"]


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
class MSMParams:
    # -------------------------------------------------------------------------
    # Basic MSM settings
    # -------------------------------------------------------------------------
    p: int
    mu: int
    max_splitting_level: int
    max_grid_level: int
    cutoffs: Sequence[float]
    # -------------------------------------------------------------------------
    # Geometry-related (arguably)
    # -------------------------------------------------------------------------
    cell: onp.ndarray
    cell_mode: CellMode
    pbc: Sequence[bool]  # TODO: type? (for consistent serialization)
    dynamic_cell: bool
    # -------------------------------------------------------------------------
    # Short-range evaluation
    # -------------------------------------------------------------------------
    supercell_diag: Sequence[int]  # TODO: Check cutoff fits? (If yes, where?)
    use_neighborlist: bool  # TODO: Interaction with `supercell_diag`?
    # -------------------------------------------------------------------------
    # Long-range evaluation
    # -------------------------------------------------------------------------
    # TODO: cell_mode = "general" in combination with grids_defined_on_unitcube = False would be invalid
    grids_defined_on_unitcube: bool
    grid_shapes: Sequence[None | tuple[int, ...]]
    grid_spacings: Sequence[None | onp.ndarray]
    stencil_extents_from_center: Sequence[None | tuple[int, ...]]
    convolution_methods: Sequence[None | ConvMeth]
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


def _set_up_msm_params_base(
    # TODO: order of args (currently alphabetic) should match set_up_msm_params
    # TODO: make args keyword-only?
    cell,
    cell_mode,
    convolution_methods,
    level_one_spacings,
    level_zero_cutoff,
    max_splitting_level,
    mu,
    n_particles,
    p,
    passed_args,
    pbc,
    supercell_diag,
    use_neighborlist,
):
    cell = onp.asarray(cell)
    n_dim = cell.shape[0]
    side_lengths = onp.linalg.norm(cell, axis=1)
    level_one_spacings = onp.asarray(level_one_spacings)
    if onp.ndim(level_one_spacings) == 0:
        level_one_spacings = onp.full(n_dim, level_one_spacings)
    pbc = onp.asarray(pbc)

    # TODO: In periodic case, the spacings are adjusted further down, so this
    #  step calculates alpha and thus p and mu from the initial spacings
    #  pre-adjustment. Is this a problem? Which behavior is less surprising?
    alpha = int(onp.max(level_zero_cutoff / level_one_spacings))
    if p is None:
        p = suggest_p(alpha)  # TODO: does this have to be a separate function?
    # See section "1. Preprocessing" of the article
    # TODO: Allow different mus for each level? (The article suggests
    #  mu >= 3*p/2 for the highest grid level)
    if mu is None:
        mu = max(int(4 * alpha + p // 2), 3 * p // 2)

    if pbc.any():
        if max_splitting_level is not None:
            raise ValueError(
                "Leave max_splitting_level unfilled if at least one direction "
                "is periodic. It is determined automatically."
            )
        (
            adjusted_spacings,
            max_splitting_level,
        ) = find_spacings_and_max_level_periodic(
            side_lengths[pbc], level_one_spacings[pbc]
        )
        level_one_spacings[onp.where(pbc)[0]] = adjusted_spacings
        max_grid_level = max_splitting_level - 1
    else:
        if max_splitting_level is None:
            if n_particles is None:
                raise ValueError(
                    "n_particles is required for the automatic determination "
                    "of the number of grid levels in non-periodic systems. "
                    "Either specify max_splitting_level directly, "
                    "or n_particles."
                )
            # TODO: Warn/raise if max_splitting_level and n_particles are both given?
            # TODO: Print a message that max_level is being determined automatically?
            max_splitting_level = suggest_max_grid_level_nonperiodic(
                side_lengths=side_lengths,
                n_particles=n_particles,
                level_one_spacings=level_one_spacings,
                level_zero_cutoff=level_zero_cutoff,
                p=p,
            )
        max_grid_level = max_splitting_level

    cutoffs_all_levels = [
        2**lvl * level_zero_cutoff for lvl in range(max_splitting_level)
    ] + [onp.inf]

    gridshapes_all_levels, spacings_all_levels = set_up_grids_all_levels(
        side_lengths=side_lengths,
        level_one_spacings=level_one_spacings,
        pbc=pbc,
        max_grid_level=max_grid_level,
        p=p,
    )

    # TODO: These stencil sizes can be too small for non-ortho cells
    stencil_extents_from_center = [None]
    # TODO: Is +1 necessary?
    #  (I guess depends on how we have previously defined/rounded/int-cast alpha...)
    # TODO: Centralize this determination of the minimum intermediate stencil
    #  extents for a given cell in a function (that should handle both ortho and triclinic case)?
    extents_intermediate = (2 * alpha + 1,) * n_dim
    if pbc.any():
        stencil_extents_from_center += [extents_intermediate] * max_grid_level
    else:
        stencil_extents_from_center += [extents_intermediate] * (
            max_grid_level - 1
        )
        stencil_extents_from_center += [
            tuple(onp.array(gridshapes_all_levels[-1]) - 1)
        ]

    if isinstance(convolution_methods, str):
        convolution_methods = [None] + [convolution_methods] * max_grid_level

    params = MSMParams(
        p=p,
        mu=mu,
        max_splitting_level=max_splitting_level,
        max_grid_level=max_grid_level,
        cutoffs=cutoffs_all_levels,
        cell=cell,
        cell_mode=cell_mode,
        pbc=pbc,
        dynamic_cell=False,
        supercell_diag=supercell_diag,
        use_neighborlist=use_neighborlist,
        grids_defined_on_unitcube=None,  # TODO
        grid_shapes=gridshapes_all_levels,
        grid_spacings=spacings_all_levels,  # TODO
        stencil_extents_from_center=stencil_extents_from_center,
        convolution_methods=convolution_methods,
        n_dim=n_dim,
        info={"args_passed_during_setup": passed_args},
    )
    return params


def set_up_msm_params_static_cell(
    cell: ArrayLike,
    level_one_spacings: float | ArrayLike,
    level_zero_cutoff: float,
    pbc: Sequence[bool],
    cell_mode: CellMode,
    p: int = None,
    mu: int = None,
    n_particles: int = None,
    max_splitting_level: int = None,
    supercell_diag: Sequence[int] = None,
    use_neighborlist: bool = None,
    convolution_methods: ConvMeth | Sequence[ConvMeth] = "scipy-fft",
) -> MSMParams:
    """High-level convenience function for setting up MSM params."""
    # TODO: How necessary/useful is this? I want it to include only non-default
    #  arguments, but currently includes everything that is not None (convolution_methods as well)
    passed_args = {k: v for k, v in locals().items() if v is not None}

    params = _set_up_msm_params_base(
        cell,
        cell_mode,
        convolution_methods,
        level_one_spacings,
        level_zero_cutoff,
        max_splitting_level,
        mu,
        n_particles,
        p,
        passed_args,
        pbc,
        supercell_diag,
        use_neighborlist,
    )

    if cell_mode == "ortho":
        params.grids_defined_on_unitcube = False
    elif cell_mode == "general":
        params.grids_defined_on_unitcube = True
        side_lengths = onp.linalg.norm(cell, axis=1)
        params.grid_spacings = [
            (None if spacings is None else spacings / side_lengths)
            for spacings in params.grid_spacings
        ]
    else:
        # TODO: Where to check for this?
        raise ValueError("Illegal value for cell_mode")

    return params


def set_up_msm_params_dyn_cell(
    reference_cell: ArrayLike,
    reference_level_one_spacings: float | ArrayLike,
    level_zero_cutoff: float,
    pbc: Sequence[bool],
    cell_mode: CellMode,
    strain_limits: tuple[float, float] = None,
    stencil_extents_from_center=None,
    **kwargs,  # TODO: name (highlight that they will be passed to static-cell setup fn?)
):

    pass  # TODO


def create_msm(params: MSMParams):
    kernel_fns = split_one_over_r_kernel(
        max_level=params.max_splitting_level,
        level_zero_cutoff=params.cutoffs[0],
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

    n_levels_intermed = params.max_splitting_level - 1
    include_toplevel = params.max_grid_level == params.max_splitting_level
    if params.grids_defined_on_unitcube:
        scaled_spacings = params.grid_spacings[1]
    else:
        scaled_spacings = params.grid_spacings[1] / onp.linalg.norm(
            params.cell, axis=1
        )
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
        scaled_spacings=scaled_spacings,
        cell_mode=params.cell_mode,
        k_lowest_intermed=k_lowest_intermed,
        extents_from_center_intermed=extents_intermed,
        k_toplevel=k_toplevel,
        grid_shape_toplevel=grid_shape_toplevel,
    )

    # TODO: unnecessary duplication?
    if params.dynamic_cell:
        compute_u_oneplus = make_compute_u_oneplus(
            # TODO: Can this closure over spacings be made more compact?
            #  (Confusing to first define a basis eval function that takes
            #  spacings as arguments, and then define one that doesn't)
            singleparticle_basis_fn_lvl_one=partial(
                basis_evaluation_fn, spacings=params.grid_spacings[1]
            ),
            grid_pass_fn=grid_pass_fn,
            grid_shape_lvl_one=params.grid_shapes[1],
            transform_mode=params.cell_mode,
            kernel_stencil_construction_fn=construct_stencils,
        )
    else:
        compute_u_oneplus = make_compute_u_oneplus(
            # TODO: Can this closure over spacings be made more compact?
            #  (Confusing to first define a basis eval function that takes
            #  spacings as arguments, and then define one that doesn't)
            singleparticle_basis_fn_lvl_one=partial(
                basis_evaluation_fn, spacings=params.grid_spacings[1]
            ),
            grid_pass_fn=grid_pass_fn,
            grid_shape_lvl_one=params.grid_shapes[1],
            transform_mode=(
                params.cell_mode if params.grids_defined_on_unitcube else None
            ),
            kernel_stencils=jax.jit(construct_stencils)(params.cell),
        )

    def calc_energy(positions, charges, cell=None, neighborlist=None):
        # TODO: Raise an error if cell given if static cell, and if not given if dynamic cell?
        if params.use_neighborlist:
            # TODO: Better error message.
            if neighborlist is None:
                raise ValueError("neighborlist argument is required.")
            u_zero = compute_u_zero(
                positions,
                charges,
                cell=cell if params.dynamic_cell else params.cell,
                # TODO: Is weights=1.0 too restrictive?
                #  (setting to 1.0 enforces that neighbor list contains no duplicates)
                #  The problem would go away if I added support for neighbor list in
                #  matrix format and add a parameter 'neighborlist_format' such that
                #  users are forced to think about what they are using.
                weights=1.0,
                neighborlist=neighborlist,
            )
        else:
            # TODO: Raise an error/warn if use_neighborlist is False
            #  but the neighborlist parameter is not None?
            u_zero = compute_u_zero(
                positions,
                charges,
                cell=cell if params.dynamic_cell else params.cell,
            )
        u_oneplus = compute_u_oneplus(
            positions, charges, cell if params.dynamic_cell else params.cell
        )
        return u_zero + u_oneplus

    def calc_forces(positions, charges, cell=None, neighborlist=None):
        return -jax.grad(calc_energy, argnums=0)(
            positions, charges, cell, neighborlist
        )

    def calc_energy_and_forces(
        positions, charges, cell=None, neighborlist=None
    ):
        value, grad = jax.value_and_grad(calc_energy, argnums=0)(
            positions, charges, cell, neighborlist
        )
        return value, -grad

    def calc_charge_gradient(positions, charges, cell=None, neighborlist=None):
        return jax.grad(calc_energy, argnums=1)(
            positions, charges, cell, neighborlist
        )

    # TODO: Option to return fns for short- and long-range part separately?
    # TODO: calc_stress in dynamic-cell case

    return (
        calc_energy,
        calc_forces,
        calc_energy_and_forces,
        calc_charge_gradient,
    )
