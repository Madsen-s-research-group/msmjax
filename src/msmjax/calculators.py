import dataclasses
import json
from functools import partial
from pathlib import Path
from typing import Literal, Sequence

import jax
import numpy as onp
from babel.messages.frontend import parse_mapping
from jax.typing import ArrayLike

import msmjax
from msmjax.bspline.coefficients import compute_coeffs_with_truncation
from msmjax.bspline.gridops import (
    create_all_grid_to_grid_ops,
    find_spacings_and_max_level_periodic,
    make_basis_evaluation_fn,
    set_up_grids_all_levels,
    suggest_max_grid_level_nonperiodic,
)
from msmjax.core.longrange import make_compute_u_oneplus, make_grid_pass_fn
from msmjax.core.shortrange import (
    _gen_supercell,
    make_compute_u_zero,
    make_eval_pair_pot,
    make_eval_pair_pot_neighborlist,
)
from msmjax.kernels import (
    SoftenerOneOverR,
    determine_min_kernel_stencil_size,
    make_construct_stencils,
    split_one_over_r,
)
from msmjax.utils.general import CellMode, ConvMeth, get_max_cutoff_for_mic


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
    # TODO: make frozen?
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
    neighborlist_prefactor: float
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


def _suggest_p(alpha):
    """Find the interpolation order p that the article recommends.

    The article's criterion for choosing p for a given alpha is heuristic
    and is stated in section III.B.2.

    Args:
        alpha: The ratio between level-zero cutoff and level-one grid spacing.

    Returns:
        Suggested value of interpolation order.
    """
    list_of_ps = [4, 6, 8]
    idx_optimal_p = onp.argmin(
        onp.abs(onp.asarray(list_of_ps) - (1.25 * alpha + 0.25))
    )
    return list_of_ps[idx_optimal_p]


def find_stencil_extents_all_levels(
    cell,
    level_one_spacings,
    level_zero_cutoff,
    n_levels_intermed: int,
    include_toplevel: bool,
    grid_shape_toplevel: tuple[int, ...] = None,
):
    stencil_extents_from_center = [None]
    extents_intermediate = determine_min_kernel_stencil_size(
        cell, level_one_spacings, 2 * level_zero_cutoff
    )
    # TODO: Clipping of stencils to grid size along non-periodic directions in mixed-periodicity cases?
    #  -> Probably best to do this inside special_periodic_convolve_scipy since
    #     it is there that the shapes of the data arrays (=grid shapes) and the
    #     kernel stencils, and the pbc are the most conveniently available in one place.
    stencil_extents_from_center += [extents_intermediate] * n_levels_intermed
    if include_toplevel:
        stencil_extents_from_center += [
            tuple(onp.array(grid_shape_toplevel) - 1)
        ]
    return stencil_extents_from_center


def set_up_msm_params_base(
    cell: ArrayLike,
    level_one_spacings: float | ArrayLike,
    *,
    level_zero_cutoff: float,
    pbc: Sequence[bool],
    cell_mode: CellMode,
    p: int = None,
    mu: int = None,
    n_particles: int = None,
    max_splitting_level: int = None,
    supercell_diag: Sequence[int] = None,
    use_neighborlist: bool = None,  # TODO: neighborlist_format? prefactor?
    convolution_methods: ConvMeth | Sequence[ConvMeth] = "scipy-fft",
    extents_intermediate: tuple[int, ...] = None,  # TODO
):
    cell = onp.asarray(cell)
    n_dim = cell.shape[0]
    side_lengths = onp.linalg.norm(cell, axis=1)
    level_one_spacings = onp.asarray(level_one_spacings)
    if onp.ndim(level_one_spacings) == 0:
        level_one_spacings = onp.full(n_dim, level_one_spacings)
    pbc = onp.asarray(pbc, dtype=bool)

    # TODO: In periodic case, the spacings are adjusted further down, so this
    #  step calculates alpha and thus p and mu from the initial spacings
    #  pre-adjustment. Is this a problem? Which behavior is less surprising?
    alpha = int(onp.max(level_zero_cutoff / level_one_spacings))
    if p is None:
        p = _suggest_p(
            alpha
        )  # TODO: does this have to be a separate function?
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

    # TODO: Should stencil_extents_from_center be an (optional) argument and
    #  should there be a standalone function for finding it?
    #  That way, in set_up_msm_params_dyn_cell, we wouldn't need to set this
    #  attribute on the params after creating them with set_up_msm_params_base.
    # TODO: Clipping of stencils to grid size along non-periodic directions in mixed-periodicity cases?
    #  -> Probably best to do this inside special_periodic_convolve_scipy since
    #     it is there that the shapes of the data arrays (=grid shapes) and the
    #     kernel stencils, and the pbc are the most conveniently available in one place.
    stencil_extents_from_center = [None]
    if extents_intermediate is None:
        extents_intermediate = determine_min_kernel_stencil_size(
            cell, level_one_spacings, 2 * level_zero_cutoff
        )
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
        dynamic_cell=None,
        supercell_diag=supercell_diag,
        use_neighborlist=use_neighborlist,
        grids_defined_on_unitcube=None,  # TODO
        grid_shapes=gridshapes_all_levels,
        grid_spacings=spacings_all_levels,  # TODO
        stencil_extents_from_center=stencil_extents_from_center,
        convolution_methods=convolution_methods,
        n_dim=n_dim,
    )
    return params


def set_up_msm_params_static_cell(
    cell: ArrayLike,
    level_one_spacings: float | ArrayLike,
    **base_kwargs,  # TODO: name
) -> MSMParams:
    """High-level convenience function for setting up MSM params."""
    # TODO: How necessary/useful is this? I want it to include only non-default
    #  arguments, but currently includes everything that is not None (convolution_methods as well)
    passed_args = {k: v for k, v in locals().items() if v is not None}

    # TODO: check if cutoff fits?

    params = set_up_msm_params_base(
        cell,
        level_one_spacings,
        **base_kwargs,
    )

    params.dynamic_cell = False
    # TODO: ok to take cell_mode from kwargs? (more generally, is it ok that
    #  some kwargs are required?)
    if base_kwargs["cell_mode"] == "ortho":
        params.grids_defined_on_unitcube = False
    elif base_kwargs["cell_mode"] == "triclinic":
        params.grids_defined_on_unitcube = True
        side_lengths = onp.linalg.norm(cell, axis=1)
        params.grid_spacings = [
            (None if spacings is None else spacings / side_lengths)
            for spacings in params.grid_spacings
        ]
    else:
        # TODO: Where is the right place to check for this?
        raise ValueError("Illegal value for cell_mode")

    # TODO: add passed_args to params

    return params


def set_up_msm_params_dyn_cell(
    reference_cell: ArrayLike,
    reference_level_one_spacings: float | ArrayLike,
    stretch_ratio_limits: tuple[float, float] = None,
    stencil_extents_from_center=None,
    **base_kwargs,
) -> MSMParams:
    # TODO: How necessary/useful is this? I want it to include only non-default
    #  arguments, but currently includes everything that is not None (convolution_methods as well)
    passed_args = {k: v for k, v in locals().items() if v is not None}

    # TODO: what happens if both are None?
    if (
        stretch_ratio_limits is not None
        and stencil_extents_from_center is not None
    ):
        raise ValueError(
            "Do not specify both strain_limits and "
            "stencil_extents_from_center at the same time."
        )

    if stretch_ratio_limits is None:
        params = set_up_msm_params_base(
            cell=reference_cell,
            level_one_spacings=reference_level_one_spacings,
            **base_kwargs,
        )
        side_lengths = onp.linalg.norm(reference_cell, axis=1)
    else:
        # TODO: explain this in docstring, then remove comment
        #   strain_limits[0] => cell at max compression => determines stencil sizes
        #   strain_limits[1] => cell at max extension => determines grid spacing
        # TODO: Does the way stretch_ratio_limits is used below really make sense?
        # TODO: Should it be: level_one_spacings=reference_level_one_spacings * min(1 / stretch_ratio_limits[1], stretch_ratio_limits[0])?
        #  Or: level_one_spacings = reference_level_one_spacings / stretch_ratio_limits[1],
        #  followed by calling set_up_msm_params_base with stretch_ratio_limits[0] * level_one_spacings
        params = set_up_msm_params_base(
            cell=reference_cell * stretch_ratio_limits[0],
            level_one_spacings=reference_level_one_spacings
            / stretch_ratio_limits[1],
            **base_kwargs,
        )
        side_lengths = onp.linalg.norm(
            reference_cell * stretch_ratio_limits[0], axis=1
        )

    params.dynamic_cell = True
    params.grids_defined_on_unitcube = True
    params.cell = None
    params.grid_spacings = [
        (None if spacings is None else spacings / side_lengths)
        for spacings in params.grid_spacings
    ]
    # TODO: could this be moved up, inside the "if stretch_ratio_limits is None" check?
    if stencil_extents_from_center is not None:
        params.stencil_extents_from_center = stencil_extents_from_center

    # TODO: add passed_args to params

    return params


def set_up_msm_params(
    cell: ArrayLike,
    level_one_spacings: float | ArrayLike,
    level_zero_cutoff: float,
    pbc: Sequence[bool],
    cell_mode: CellMode,
    dynamic_cell: bool,
    p: int = None,
    mu: int = None,
    n_particles: int = None,
    max_splitting_level: int = None,
    supercell_diag: Sequence[int] = None,
    use_neighborlist: bool = False,  # TODO: neighborlist_format? prefactor?
    neighborlist_prefactor: float = None,
    intermediate_kernel_stencil_extents: tuple[int, ...] = None,
    convolution_methods: ConvMeth | Sequence[ConvMeth] = "scipy-fft",
):
    # TODO: error if neighborlist_prefactor not given but use_neighborlist = True

    cell = onp.asarray(cell)
    n_dim = cell.shape[0]
    side_lengths = onp.linalg.norm(cell, axis=1)
    level_one_spacings = onp.asarray(level_one_spacings)
    if onp.ndim(level_one_spacings) == 0:
        level_one_spacings = onp.full(n_dim, level_one_spacings)
    pbc = onp.asarray(pbc, dtype=bool)

    # TODO: In periodic case, the spacings are adjusted further down, so this
    #  step calculates alpha and thus p and mu from the initial spacings
    #  pre-adjustment. Is this a problem? Which behavior is less surprising?
    alpha = int(onp.max(level_zero_cutoff / level_one_spacings))
    if p is None:
        p = _suggest_p(alpha)
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

    # TODO: Clipping of stencils to grid size along non-periodic directions in mixed-periodicity cases?
    #  -> Probably best to do this inside special_periodic_convolve_scipy since
    #     it is there that the shapes of the data arrays (=grid shapes) and the
    #     kernel stencils, and the pbc are the most conveniently available in one place.
    stencil_extents_from_center = [None]
    if intermediate_kernel_stencil_extents is None:
        intermediate_kernel_stencil_extents = (
            determine_min_kernel_stencil_size(
                cell, level_one_spacings, 2 * level_zero_cutoff
            )
        )
    if pbc.any():
        stencil_extents_from_center += [
            intermediate_kernel_stencil_extents
        ] * max_grid_level
    else:
        stencil_extents_from_center += [
            intermediate_kernel_stencil_extents
        ] * (max_grid_level - 1)
        stencil_extents_from_center += [
            tuple(onp.array(gridshapes_all_levels[-1]) - 1)
        ]

    if dynamic_cell or cell_mode == "triclinic":
        grids_defined_on_unitcube = True
        side_lengths = onp.linalg.norm(cell, axis=1)
        spacings_all_levels = [
            (None if spacings is None else spacings / side_lengths)
            for spacings in spacings_all_levels
        ]
    elif cell_mode == "ortho":
        grids_defined_on_unitcube = False

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
        dynamic_cell=dynamic_cell,
        supercell_diag=supercell_diag,
        use_neighborlist=use_neighborlist,
        neighborlist_prefactor=neighborlist_prefactor,
        grids_defined_on_unitcube=grids_defined_on_unitcube,
        grid_shapes=gridshapes_all_levels,
        grid_spacings=spacings_all_levels,
        stencil_extents_from_center=stencil_extents_from_center,
        convolution_methods=convolution_methods,
        n_dim=n_dim,
    )
    return params


def create_msm(params: MSMParams):
    if not params.dynamic_cell:
        _ = check_cutoffs_and_spacings(params.cell, params)

    kernel_fns = split_one_over_r(
        max_level=params.max_splitting_level,
        level_zero_cutoff=params.cutoffs[0],
        softening_function=SoftenerOneOverR(params.p),
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
        # TODO: Raise an error if
        #  - cell arg is given, but static and ortho cell (cell_mode="ortho" and dynamic_cell=False),
        #  - cell arg is not given, but dynamic cell?
        if params.use_neighborlist:
            # TODO: Better error message.
            if neighborlist is None:
                raise ValueError("neighborlist argument is required.")
            u_zero = compute_u_zero(
                positions,
                charges,
                cell=cell if params.dynamic_cell else params.cell,
                weights=params.neighborlist_prefactor,
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


def check_cutoffs_and_spacings(cell: ArrayLike, params: MSMParams):
    if onp.any(params.pbc):
        placeholder_positions = onp.zeros((10, cell.shape[0]))
        placeholder_charges = onp.zeros((10,))
        _, _, supercell = _gen_supercell(
            placeholder_positions,
            placeholder_charges,
            cell,
            params.supercell_diag,
        )
        # A simple way of taking only directions with periodicity into account
        # in the cutoff determination is to make the cell vectors very large
        # (effectively infinite) along nonperiodic directions before
        # calculating the maximum allowed cutoff as if fully periodic.
        # We repeat this procedure for two different large elongation factors
        # and check that the resulting cutoffs are the same, to ensure that a
        # sufficient elongation of the cell was used.
        trial_cells = [
            supercell * onp.where(params.pbc, 1.0, elongation)[:, onp.newaxis]
            for elongation in [1.0e3, 1.0e4]
        ]
        trial_cutoffs = [get_max_cutoff_for_mic(c) for c in trial_cells]
        if not onp.allclose(*trial_cutoffs):
            raise ValueError(
                "Something went wrong while checking if the cutoff fits."
            )
        max_allowed_cutoff = trial_cutoffs[0]
        if not params.cutoffs[0] <= max_allowed_cutoff:
            raise ValueError(
                f"Level-zero cutoff radius too large for the given cell:\n"
                f"It is {params.cutoffs[0]}, but the cell can only "
                f"accommodate {max_allowed_cutoff}.\n"
                f"Consider reducing the cutoff or making a larger supercell "
                f"(either via supercell_diag or manually)."
            )

    # TODO: check at all levels, not just level 1?
    level_one_spacings = params.grid_spacings[1]
    if params.grids_defined_on_unitcube:
        level_one_spacings *= onp.linalg.norm(cell, axis=1)

    min_required_stencil_size = determine_min_kernel_stencil_size(
        cell=cell, spacings=level_one_spacings, cutoff=params.cutoffs[1]
    )
    given_stencil_size = params.stencil_extents_from_center[1]
    if not (
        onp.array(given_stencil_size) >= onp.array(min_required_stencil_size)
    ).all():
        raise ValueError(
            f"The kernel stencil is too small to cover the cutoff for the "
            f"given cell:\n"
            f"It is {given_stencil_size}, but needs to be (along each "
            f"component separately) at least {min_required_stencil_size}.\n"
            f"Try increasing the value of intermediate_kernel_stencil_extents "
            f"during setup, and make sure the cell is not unreasonably "
            f"compressed or deformed."
        )

    return level_one_spacings
