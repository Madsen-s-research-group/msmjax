import functools
import math

import jax
import jax.numpy as jnp
import numpy as onp
from neuralil.bessel_descriptors import gen_supercell

from msmjax import wrappers_old_code
from msmjax.bspline_interpolation.coefficients import compute_J_zeroplus
from msmjax.gridops_multidim import set_up_grids_all_levels
from msmjax.kernels import SofteningFunctionOneOverR, split_one_over_r_kernel


def suggest_p(alpha):
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


def suggest_max_gridlevel_nonPBC(
    max_pos, min_pos, nb_particles, level_one_gridspacing, level_zero_cutoff, p
):
    # TODO: Adapt to the interface of the new MSM code (`box_lengths` instead
    #  of `min_pos`, `max_pos`)
    # TODO: allow different grid spacings along each direction
    """Suggest a value for the highest grid level L, in non-periodic BC case.

    The suggested value of L is picked based on three considerations:

    1. The highest-level grid should not be coarser than the size of the
        simulation box itself.
        This criterion also catches the point where, due to the extension of
        the grid beyond the simulation box boundaries necessary in the case of
        non-periodic BCs, the number of grid points stops decreasing with
        increasing grid level.
    2. The cutoff radius a_{L-1} at the highest-but-one grid level L - 1 should
        not be so large that it covers the whole grid, or a too large portion
        of it, because that would be inefficient.
    3. The number of grid points at the highest level should be small enough
        that the evaluation of the interaction kernel at that level, which
        necessarily contains the infinite-range tail, can nonetheless be done
        cheaply.

    See Notes section for which conditions exactly are used in these criteria.

    The final suggested L is picked as the minimum out of the values from these
    different criteria (meaning that 1. and 2. take precedence over 3.).

    Notes:
        The condition used in 1. is that the highest-level grid spacing h_L
        does not exceed V^{1/ndim}, where V is the simulation box volume,
        and ndim the spatial dimension of the system.

        The condition used in 2. is that the number of grid points at
        the highest level should not go below (2 * alpha)^ndim,
        where alpha = level_zero_cutoff / level_one_gridspacing, and ndim the
        spatial dimension of the system.

        The condition used in 3. is that the number of grid of points
        at the highest level should be <= sqrt(nb_particles).

        Depending on simulation box size and value of p,
        conditions 2. and 3. may be unfulfillable.
        But condition 1. always gives a value.

    Args:
        min_pos (numpy.ndarray): Minimum possible values for particle position.
            Each entry corresponds to one cartesian coordinate.
        max_pos (numpy.ndarray): Minimum possible values for particle position.
            Each entry corresponds to one cartesian coordinate.
        nb_particles (int): Number of particles in simulation cell.
        level_one_gridspacing: The spacing of the finest grid.
        level_zero_cutoff: The cutoff radius of the level-zero kernel function
            (= of the direct term in the kernel splitting).
        p (int): The degree of the spline in the convention of the article.

    Returns:
        The suggested value for L.

    Raises:
        ValueError: If invalid simulation box boundaries.
    """
    if False in onp.greater(max_pos, min_pos):
        raise ValueError("max_pos must be greater than min_pos (element-wise)")
    if len(min_pos) != len(max_pos):
        raise ValueError(
            "Inconsistent spatial dimension in the simulation box boundaries."
        )

    alpha = level_zero_cutoff / level_one_gridspacing
    ndim = len(min_pos)
    boxvolume = math.prod(
        [
            maxpos_thisdirection - minpos_thisdirection
            for maxpos_thisdirection, minpos_thisdirection in zip(
                max_pos, min_pos
            )
        ]
    )

    # 1. Highest-level grid not coarser than simulation box size
    L_based_on_spacing = (
        int(onp.log2(boxvolume ** (1.0 / ndim) / level_one_gridspacing)) + 1
    )

    range_of_Ls = onp.arange(1, L_based_on_spacing + 1)

    gridspacings = 2 ** (range_of_Ls - 1) * level_one_gridspacing
    gridshapes = []
    for spacing in gridspacings:
        shape = tuple(
            int(sidelength_box / spacing) + 1 + p
            for sidelength_box in (max_pos - min_pos)
        )
        gridshapes.append(shape)
    nb_gridpoints = onp.array([math.prod(shape) for shape in gridshapes])

    # 2. Cutoff at highest-but-one level not too large relative to
    #    simulation box size
    try:
        L_based_on_cutoff = range_of_Ls[nb_gridpoints <= (2 * alpha) ** ndim][
            0
        ]
    except IndexError:
        L_based_on_cutoff = onp.nan

    # 3. Number of grid points at highest level small enough
    #    to evaluate interaction cheaply
    try:
        L_based_on_nb_of_gridpoints = range_of_Ls[
            nb_gridpoints <= math.sqrt(nb_particles)
        ][0]
    except IndexError:
        L_based_on_nb_of_gridpoints = onp.nan

    Ls_from_all_criteria = [
        L_based_on_spacing,
        L_based_on_cutoff,
        L_based_on_nb_of_gridpoints,
    ]
    suggested_L = min(Ls_from_all_criteria)

    # TODO: Remove or rework these messages. Perhaps don't print them by
    #  default, but add an extra return value that indicates the
    #  criterion/criteria used for selection?
    print(
        "Suggested value for max grid level L was found according to "
        "criterion/criteria:"
    )
    print(
        "(if multiple criteria listed, "
        "that means they agree on the suggested L)"
    )
    messages = [
        "*) Highest-level grid not coarser than "
        'simulation box size ("criterion 1.")',
        "*) Cutoff at highest-but-one level not too large relative to "
        'simulation box size ("criterion 2.")',
        "*) Number of grid points at highest level small enough "
        'to evaluate interaction cheaply ("criterion 3.")',
    ]
    for idx_of_criterion in [
        idx for idx, L in enumerate(Ls_from_all_criteria) if L == suggested_L
    ]:
        print(messages[idx_of_criterion])
    print()

    return suggested_L


def find_spacings_and_n_levels_periodic(box_lengths, level_one_spacings):
    box_lengths = onp.asarray(box_lengths)
    level_one_spacings = onp.asarray(level_one_spacings)

    powers = onp.round(onp.log2(box_lengths / level_one_spacings)).astype(int)
    # Make sure there is at least one level:
    powers = onp.where(powers >= 1, powers, 1)
    closest_valid_spacings = box_lengths / 2**powers
    # Define n_levels as including the highest level (which will be omitted
    #  in the calculation of the grid part, but still matters to get the
    #  correct splitting of the kernel, and for the self-interaction term of
    #  the short-range part)
    n_levels = max(powers) + 1

    return closest_valid_spacings, n_levels


def suggest_msm_params(
    box_lengths,
    pbcs,
    n_particles,
    level_one_gridspacing,
    level_zero_cutoff=None,
    alpha=None,
    p=None,
    mu=None,
    n_levels=None,
    **kwargs,
):
    # TODO: Let `level_one_gridspacing` default to None as well? (in which case
    #  it would be computed as the average particle spacing from `box_lengths`
    #  and `n_particles`
    # TODO: `n_particles` is in fact only needed in non-periodic case (as long
    #  as we do not need to infer `level_one_gridspacing` from the particle
    #  density)
    box_lengths = onp.asarray(box_lengths)
    pbcs = onp.asarray(pbcs)
    if not (onp.all(pbcs) or onp.all(~pbcs)):
        raise ValueError("Mixed boundary conditions currently not supported.")
    periodic = pbcs[0]

    if periodic:
        # FIXME: Return one spacing per direction, not just one value!
        actual_spacings, n_levels = find_spacings_and_n_levels_periodic(
            box_lengths=box_lengths,
            level_one_spacings=[level_one_gridspacing] * len(box_lengths),
        )
        level_one_gridspacing = actual_spacings[0]

    if alpha is None and level_zero_cutoff is not None:
        alpha = level_zero_cutoff / level_one_gridspacing
    elif level_zero_cutoff is None and alpha is not None:
        # TODO: The cutoff being computed from the grid spacing AFTER the grid
        #  spacing has been adapted for PBCs may lead to inconsistent or
        #  surprising results. Is this what we want?
        level_zero_cutoff = alpha * level_one_gridspacing
    else:
        raise ValueError(
            "Either `level_zero_cutoff` or `alpha` is required, "
            "and not both at the same time."
        )

    if p is None:
        p = suggest_p(alpha)
    # See section "1. Preprocessing" of the article
    # TODO: Allow different mus for each level? (The article suggests
    #  mu >= 3*p/2 for the highest grid level)
    if mu is None:
        mu = max(int(4 * alpha + p // 2), 3 * p // 2)
    if not periodic and n_levels is None:
        n_levels = suggest_max_gridlevel_nonPBC(
            min_pos=onp.zeros_like(box_lengths),
            max_pos=box_lengths,
            nb_particles=n_particles,
            level_one_gridspacing=level_one_gridspacing,
            level_zero_cutoff=level_zero_cutoff,
            p=p,
        )

    # TODO: check that n_levels is at least one (or is this function not the
    #  right place for that?)

    return {
        "level_one_gridspacing": float(level_one_gridspacing),
        "level_zero_cutoff": float(level_zero_cutoff),
        "p": int(p),
        "mu": int(mu),
        "n_levels": int(n_levels),
        **kwargs,
    }


def set_up_grids_and_kernels(
    box_lengths,
    pbcs,
    level_one_gridspacing,
    level_zero_cutoff,
    p,
    mu,
    n_levels,
):
    n_dim = len(pbcs)

    kernels = split_one_over_r_kernel(
        max_level=n_levels,
        level_zero_cutoff=level_zero_cutoff,
        softening_function=SofteningFunctionOneOverR(p),
    )
    grids = set_up_grids_all_levels(
        box_lengths=box_lengths,
        level_one_spacings=[level_one_gridspacing] * n_dim,
        pbcs=pbcs,
        n_levels=n_levels,
        p=p,
        J_zeroplus=compute_J_zeroplus(p),
    )
    kernel_stencils = wrappers_old_code._construct_kernel_stencils(
        kernels=kernels,
        box_lengths=box_lengths,
        level_one_gridspacing=level_one_gridspacing,
        level_zero_cutoff=level_zero_cutoff,
        n_levels=n_levels,
        p=p,
        mu=mu,
    )

    return kernels, grids, kernel_stencils
