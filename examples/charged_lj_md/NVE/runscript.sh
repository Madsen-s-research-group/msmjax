python run_md_ase.py \
    --inputstruct initial_structure.xyz \
    --msm_params msm_params.json \
    --outdir out/ \
    --timestep_fs 1.0 \
    --loginterval_fs 500.0 \
    --simtime_ps 1000.0 \
    --ensemble NVE
