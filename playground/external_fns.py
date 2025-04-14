def make_clean_prolongate_1d(
    axis_source_coarse: BSplineInterpolationAxis,
    axis_target_fine: BSplineInterpolationAxis,
):
    # TODO: Change function parameters to (n_points_in, n_points_out, p, periodic)
    #  => directly compute J from p during setup, this should be fine
    p = axis_source_coarse.p
    J_zeroplus = axis_source_coarse.J_zeroplus
    J = jnp.concatenate((J_zeroplus[::-1][:-1], J_zeroplus))

    n_points_in = axis_source_coarse.n_total
    n_points_out = axis_target_fine.n_total
    # TODO: check n_points_in >= n_points_out?
    periodic = axis_source_coarse.periodic

    # TODO: External factory function that creates both `zero_align_idx` and
    #  `to_positional_idx` from `periodic` and `p`

    def zero_align_idx(positional_idx):
        if periodic:
            return positional_idx
        else:
            return positional_idx - p // 2

    def to_positional_idx(zero_aligned_idx):
        if periodic:
            return zero_aligned_idx
        else:
            return zero_aligned_idx + p // 2


axis_source_coarse = grids[2].axes[0]
axis_target_fine = grids[1].axes[0]

restrict_1d = jax.jit(
    create_restriction_operator_1d(axis_source_coarse, axis_target_fine)
)
clean_restrict_1d = jax.jit(
    make_clean_restrict_1d(axis_source_coarse, axis_target_fine)
)

n_points_coarse = axis_source_coarse.n_total
n_points_fine = axis_target_fine.n_total

rng = onp.random.default_rng(2479)
gridpotential_coarse = rng.uniform(-1.0, 1.0, size=n_points_coarse)
gridpotential_coarse -= gridpotential_coarse.mean()
gridpotential_coarse = jax.device_put(gridpotential_coarse)
