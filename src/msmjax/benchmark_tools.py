import platform
import shlex
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
import numpy as onp

path_input_structures = (
    Path(__file__).resolve().parents[2] / "data" / "benchmark" / "structures"
)


def get_git_commit_id(repository_path):
    proc = subprocess.run(
        shlex.split(f"git -C {Path(repository_path).resolve()} show -s"),
        capture_output=True,
    )
    commit_id = proc.stdout.decode("utf8").split()[1]
    return commit_id


def get_repository_info(path):
    try:
        commit_id = get_git_commit_id(path)
        return {"path": path, "commit_id": commit_id}
    except Exception as e:
        return {"path": path, "commit_id": None, "exception": repr(e)}


def get_metadata(additional_repository_paths: dict = None):
    metadata = {
        "timestamp": str(datetime.now()),
        "system_info": {
            "hostname": socket.gethostname(),
            "uname": platform.uname()._asdict(),
        },
        "python_info": {"version": sys.version, "path": sys.path},
        "jax_info": {
            "jax.__version__": jax.__version__,
            "jaxlib.__version__": jaxlib.__version__,
            "jax_device": jax.devices()[0].device_kind,
        },
    }

    path_msmjax = str(Path(__file__).resolve().parents[2])
    repository_info_msmjax = get_repository_info(path_msmjax)
    metadata["repository_info_msmjax"] = repository_info_msmjax

    if additional_repository_paths is not None:
        additional_repository_info = {}
        for label, path in additional_repository_paths.items():
            additional_repository_info[label] = get_repository_info(path)
        metadata["additional_repository_info"] = additional_repository_info

    return metadata


def time_set_of_structures(structures, pbc, setup_fn, **kwargs):
    timed_calc, info = setup_fn(
        structures=structures,
        pbc=pbc,
        **kwargs,
    )
    times_all = []
    for idx_structure in range(len(structures["positions"])):
        pos = jnp.array(structures["positions"][idx_structure])
        chg = jnp.array(structures["charges"][idx_structure])
        jax.device_put(pos)
        jax.device_put(chg)
        times_all.append(timed_calc(pos, chg))

    output = {
        "times": times_all.tolist(),
        "info": info,
    }

    return output
