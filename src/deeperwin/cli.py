#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
import argparse

def main():
    # Main CLI-parser
    parser = argparse.ArgumentParser(description="DeepErwin QMC code")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    # Sub-parser for single run
    parser_run = subparsers.add_parser("run", help="Run a single DeepErwin calculation with a given config.yml file")
    parser_run.add_argument("config_file", default="config.yml", help="Path to input config file")

    # Sub-parser for weight-sharing run
    parser_run_shared = subparsers.add_parser("run-shared",
                                       help="Run a DeepErwin calculation for multiple molecules using weight-sharing")
    parser_run_shared.add_argument("config_file", default="config.yml", help="Path to input config file")

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

    # Sub-parser for helper-tool that detects available GPUs (to be used to set CUDA_VISIBLE_DEVICES)
    parser_available_gpus = subparsers.add_parser("select-gpus",
                                         help="Setup and dispatch one or multiple DeepErwin calculations, e.g. for a parameter sweep")
    parser_available_gpus.add_argument("--n-gpus", default=1, type=int, help="Number of GPUs required")
    parser_available_gpus.add_argument("--sleep", default=0, type=int, help="Sleep for given number of seconds to avoid collisions before querying available gpus")

    # Sub-parser for Weights&Biases sweep-agent
    subparsers.add_parser("wandb-agent", help="Agent to be called by wandb sweep-agent for automatic hyperparameter search")


    # args = parser.parse_args("setup -i /home/mscherbela/develop/deeperwin_jaxtest/sample_configs/config_debug.yml -p physical.name LiH Ne".split())
    args = parser.parse_args()
    if args.command == "setup":
        from deeperwin.run_tools.setup_calculations import setup_calculations
        setup_calculations(args)
    elif args.command == "run":
        from deeperwin.process_molecule import process_molecule
        process_molecule(args.config_file)
    elif args.command == "run-shared":
        from deeperwin.process_molecules_shared import process_molecule_shared
        process_molecule_shared(args.config_file)
    elif args.command == "select-gpus":
        from deeperwin.run_tools.available_gpus import assign_free_GPU_ids
        print(assign_free_GPU_ids(n_gpus=args.n_gpus, sleep_seconds=args.sleep))
    elif args.command == "wandb-agent":
        from deeperwin.run_tools.sweep_agent import run_sweep_agent
        run_sweep_agent()

if __name__ == '__main__':
    main()
