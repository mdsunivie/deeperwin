import argparse
import glob
import pathlib
import subprocess

import yaml


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_files",
        nargs="*",
        default=None,
        help="Config files to run in consecutive order",
    )
    parser.add_argument(
        "-t",
        "--dep_type",
        default="afterany",
        choices=[
            "after",
            "afterany",
            "afterburstbuffer",
            "afterok",
            "aftercorr",
            "afternotok",
        ],
        help="Dependency type, see man sbatch.",
    )
    return parser


def main(args):
    prev_jobs = []
    for config_file in args.config_files:
        print(config_file)
        subprocess.run(["deeperwin", "setup", "-i", config_file, "--dry-run"])

        with open(config_file) as f:
            config = yaml.safe_load(f)

        exp_name = config["experiment_name"]
        job_files = sorted(glob.glob(f"{exp_name}/*.sh") + glob.glob(f"{exp_name}/*/*.sh"))
        _deps = ":".join(prev_jobs)
        prev_jobs = []

        for path in map(pathlib.Path, job_files):
            print("Running job", path)
            run_dir = path.parent
            dependencies = [f"--dependency={args.dep_type}:{_deps}"] if _deps else []
            cmd = ["sbatch"] + dependencies + ["job.sh"]
            out = subprocess.run(cmd, cwd=run_dir, stdout=subprocess.PIPE)
            stdout = out.stdout.decode("UTF-8")
            print(" ".join(cmd))
            print(stdout)
            jobid = stdout.split()[-1]
            prev_jobs.append(jobid)


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
