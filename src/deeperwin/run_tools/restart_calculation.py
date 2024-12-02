import os
import re
from pathlib import Path
from deeperwin.checkpoints import load_run
from deeperwin.configuration import to_prettified_yaml
from ruamel.yaml import YAML
from deeperwin.cli import main


def _get_last_checkpoint(directory):
    directory = Path(directory)
    latest_chkpt = None
    latest_epoch = 0
    for fname in os.listdir(directory):
        full_fname = directory.joinpath(fname)
        if not full_fname.is_file():
            continue
        match = re.match(R"chkpt(\d+).zip", fname)
        if match and (n := int(match.group(1))) >= latest_epoch:
            latest_epoch = n
            latest_chkpt = str(full_fname.absolute())
    return latest_chkpt, latest_epoch


def _create_restart_config(chkpt_fname: str, n_epochs_done: int):
    chkpt_data = load_run(chkpt_fname, parse_config=True, load_pkl=False)
    restart_config = chkpt_data.config

    n_epochs_remaining = (
        restart_config.optimization.n_epochs_prev + restart_config.optimization.n_epochs - n_epochs_done
    )
    restart_expname = restart_config.experiment_name + f"_from{n_epochs_done}"
    restart_config = dict(
        experiment_name=restart_expname,
        reuse=dict(mode="restart", path=str(chkpt_fname)),
        optimization=dict(n_epochs=n_epochs_remaining),
        dispatch=dict(restart_config.dispatch),
        computation=dict(
            n_nodes=restart_config.computation.n_nodes, n_local_devices=restart_config.computation.n_local_devices
        ),
    )
    tmp_config_fname = restart_expname + ".yml"
    with open(tmp_config_fname, "w") as f:
        YAML().dump(to_prettified_yaml(restart_config), f)
    return tmp_config_fname


def restart_calculation(args):
    chkpt_fname, n_epochs_done = _get_last_checkpoint(args.directory)
    new_config_fname = _create_restart_config(chkpt_fname, n_epochs_done)
    cmd = f"setup -i {new_config_fname}"
    if args.force:
        cmd += " --force"
    if args.dry_run:
        cmd += " --dry-run"
    main(cmd)
    os.remove(new_config_fname)
