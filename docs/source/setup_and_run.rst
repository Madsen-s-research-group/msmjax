Setting up and running a calculation
================================================

::

    import numpy as onp
    from msmjax.calculators import set_up_msm_params

    msm_params = set_up_msm_params(
        cell=onp.diag([10., 10., 10.]),
        level_one_spacings=onp.array([1., 1., 1.]),
        level_zero_cutoff=3.0,
        p=4,
        pbc=(False, False, False),
        cell_mode="ortho",
        dynamic_cell=False,
    )

::

    from msmjax.calculators import create_msm

    evaluation_fns = create_msm(msm_params)
    # jit for performance
    eval_energy_and_forces = jax.jit(
        evaluation_fns["energy_and_forces"]
    )
    energy, forces = eval_energy_and_forces(
        positions, charges
    )

::

    from msmjax.calculators import MSMParams

    msm_params.save_json("params.json")
    msm_params_restored = MSMParams.load_json("params.json")
