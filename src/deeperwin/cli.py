#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
import argparse
import sys

def main(args=None):
    args = _parse_input_args(args)
    if args.command == "setup":
        from deeperwin.run_tools.setup_calculations import setup_calculations
        setup_calculations(args)
    if args.command == "restart":
        from deeperwin.run_tools.restart_calculation import restart_calculation
        restart_calculation(args)
    elif args.command == "run":
        from deeperwin.process_molecule import process_single_molecule
        process_single_molecule(args.config_file)
    elif args.command == "convert-checkpoint":
        from deeperwin.run_tools.convert_checkpoint import convert_checkpoint
        convert_checkpoint(args.input_file, args.output_file)
    elif args.command == "run-multiple-shared":
        from deeperwin.process_molecule import process_multiple_molecules_shared
        process_multiple_molecules_shared(args.config_file)
    elif args.command == "wandb-agent":
        from deeperwin.run_tools.sweep_agent import run_sweep_agent
        run_sweep_agent()
    elif args.command == "setup-phisnet":
        from deeperwin.run_tools.setup_phisnet_calculations import setup_calculations
        setup_calculations(args)
    elif args.command == "train-phisnet":
        from deeperwin.run_tools.train_phisnet import train_phisnet
        train_phisnet(args.config_file)
    elif args.command == "select-gpus":
        from deeperwin.run_tools.available_gpus import assign_free_GPU_ids
        print(assign_free_GPU_ids(n_gpus=args.n_gpus, sleep_seconds=args.sleep))

def _parse_input_args(args=None):
    # Main CLI-parser
    parser = argparse.ArgumentParser(description="DeepErwin QMC code")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    # Sub-parser for setting-up calculations
    parser_setup = subparsers.add_parser("setup",
                                         help="Setup and dispatch one or multiple DeepErwin calculations, e.g. for a parameter sweep")
    parser_setup.add_argument("--input", "-i", default="config.yml", help="Path to input config file")
    parser_setup.add_argument('--parameter', '-p', nargs='+', action='append', default=[])
    parser_setup.add_argument('--force', '-f', action="store_true", help="Overwrite directories if they already exist")
    parser_setup.add_argument('--wandb-sweep', nargs=3, default=[],
                              help="Start a hyperparmeter sweep using wandb, with a given sweep-identifier, number of agents and number of runs per agent, e.g. --wandb-sweep schroedinger_univie/sweep_debug/cpxe3iq9 3 10")
    parser_setup.add_argument('--dry-run', action='store_true',
                              help="Only set-up the directories and config-files, but do not dispatch the actual calculation.")
    parser_setup.add_argument('--start-time-offset', default=0, type=int,
                              help="Add a delay (given in seconds) to each consecutive job at runtime, to avoid GPU-collisions")
    parser_setup.add_argument('--start-time-offset-first', default=0, type=int,
                              help="Add constant delay (given in seconds) to all jobs at runtime, to avoid GPU-collisions")

    # Sub-parser for single run
    parser_run = subparsers.add_parser("run", help="Run a single DeepErwin calculation with a given config.yml file")
    parser_run.add_argument("config_file", default="config.yml", help="Path to input config file")

    # Sub-parser for a run with multiple geometries using shared weights
    parser_run_multiple_shared = subparsers.add_parser("run-multiple-shared",
                                       help="Run a DeepErwin calculation for multiple molecules using weight-sharing")
    parser_run_multiple_shared.add_argument("config_file", default="config.yml", help="Path to input config file")

    # Sub-parser for training a PhiSNet-like model to generate orbital embeddings
    parser_phisnet = subparsers.add_parser("train-phisnet", help="Train a PhiSNet-like model to generate orbital embeddings")
    parser_phisnet.add_argument("config_file", default="config.yml", help="Path to input config file")

    # Sub-parser for setting-up PhisNet training runs
    parser_setup = subparsers.add_parser("setup-phisnet",
                                         help="Setup and dispatch one or multiple PhisNet training calculations, e.g. for a parameter sweep")
    parser_setup.add_argument("--input", "-i", default="config.yml", help="Path to input config file")
    parser_setup.add_argument('--parameter', '-p', nargs='+', action='append', default=[])
    parser_setup.add_argument('--force', '-f', action="store_true", help="Overwrite directories if they already exist")
    parser_setup.add_argument('--dry-run', action='store_true',
                              help="Only set-up the directories and config-files, but do not dispatch the actual calculation.")

    # Sub-parser for Weights&Biases sweep-agent
    subparsers.add_parser("wandb-agent", help="Agent to be called by wandb sweep-agent for automatic hyperparameter search")

    # Sub-parser for automatic restart of failed calculations
    parser_restart = subparsers.add_parser("restart",
                                           help="Restart a failed calculation (e.g. due to timeout) from latest checkpoint, re-using config from failed run")
    parser_restart.add_argument('--dry-run', action='store_true',
                                help="Only set-up the directories and config-files, but do not dispatch the actual calculation.")
    parser_restart.add_argument('--force', '-f', action="store_true",
                                help="Overwrite directories if they already exist")
    parser_restart.add_argument('directory', help="Path to directory containing failed run and last checkpoint")

    # Sub-parser for helper-tool that detects available GPUs (to be used to set CUDA_VISIBLE_DEVICES)
    parser_available_gpus = subparsers.add_parser("select-gpus",
                                         help="Setup and dispatch one or multiple DeepErwin calculations, e.g. for a parameter sweep")
    parser_available_gpus.add_argument("--n-gpus", default=1, type=int, help="Number of GPUs required")
    parser_available_gpus.add_argument("--sleep", default=0, type=int, help="Sleep for given number of seconds to avoid collisions before querying available gpus")

    # Sub-parser for herlper-tool which converts old checkpoints with incompatible naming schemes to newer versions
    parser_convert_chkpt = subparsers.add_parser("convert-checkpoint",
                                                 help="Convert a checkpoint from an old to a newer format to allow reuse with newer code versions")
    parser_convert_chkpt.add_argument("input_file", help="Filename of old checkpoint to convert")
    parser_convert_chkpt.add_argument("output_file", help="Target filename for converted checkpiont")

   
    # Allowed args: None (uses sys.argv[1:]), list or string
    if args is not None:
        if isinstance(args, str):
            args = args.split()
        args = [a for a in args if len(a) > 0]
    return parser.parse_args(args)

if __name__ == '__main__':
    main()
