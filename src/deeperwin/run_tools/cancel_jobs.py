import subprocess
from collections import namedtuple
import re
import argparse
import getpass

Job = namedtuple("Job", "id state name")

def parse_squeue(sq_output):
    jobs = []
    for line in sq_output.split("\n"):
        if len(line) < 10:
            continue
        jobs.append(Job(
            id=int(line[:10].strip()),
            state=line[10:15].strip(),
            name=line[15:].strip()
        ))
    return jobs

def get_matching_jobs(jobs, pattern, require_full_match):
    jobs_to_cancel = []
    for j in jobs:
        if require_full_match:
            if re.match(pattern, j.name):
                jobs_to_cancel.append(j)
        else:
            if re.findall(pattern, j.name):
                jobs_to_cancel.append(j)
    return jobs_to_cancel

def confirm_job_cancel(jobs):
    print("Matches following jobs")
    for j in jobs:
        print(f"{j.id:10d} {j.state:<5} {j.name}")

    response = ""
    while response.lower() not in ["y", "yes", "n", "no"]:
        response = input(f"Do you want to cancel these {len(jobs)} jobs? [y/n]")
    return response in ["y", "yes"]

def cancel_jobs(jobs):
    subprocess.call(["scancel"] + [str(j.id) for j in jobs])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_match", action="store_true",
                        help="Uses re.match instead of re.find. The pattern thus needs to match the beginning of the job name, instead of an arbitrary substring",
                        default=False
                        )
    parser.add_argument("pattern", help="Regular expression of job names to match")

    args = parser.parse_args()
    username = getpass.getuser()
    sq_output = subprocess.check_output(["squeue", "-u", username, "-o", "%10A%5t%75j", "--noheader"], encoding="utf-8")
    jobs = parse_squeue(sq_output)
    jobs = get_matching_jobs(jobs, args.pattern, args.full_match)
    if jobs:
        cancel = confirm_job_cancel(jobs)
        if cancel:
            cancel_jobs(jobs)
    else:
        print("No matching jobs for this pattern")
