#!/usr/bin/env python
import argparse
import subprocess
import getpass
import re
import sys

DELIMITER = ';DELIM;'

def parseJob(line):
    """
    Parse the output of 'squeue --user username -o "%A{DELIMITER}%j{DELIMITER}%t"'

    Args:
        line (str): One output line of squeue

    Returns:
        (dict): Dict containing the keys id, name, and state
    """
    tokens = line.split(DELIMITER)
    assert len(tokens) == 3, str(tokens)
    data = dict(id=int(tokens[0]),
                name=tokens[1],
                state=tokens[2])
    return data

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--range', '-r', nargs='*', help="Select jobs with ids between first and second argument (including both ends)")
    parser.add_argument('--name', '-n', default=None, help="Select jobs with job-names matching the argument (interpreted as regex)")
    args = parser.parse_args()

    if args.range is None or len(args.range) == 0:
        filter_range = lambda j: True
    elif len(args.range) == 2:
        id_min = int(args.range[0])
        id_max = int(args.range[1])
        filter_range = lambda j: id_min <= j['id'] <= id_max
    else:
        raise ValueError("Range parameter must have exactly 2 arguments, specifying lower and upper job-id limit")

    if args.name is None:
        filter_name = lambda j: True
    else:
        filter_name = lambda j: re.match(args.name, j['name']) is not None

    username = getpass.getuser()
    sinfo_text = subprocess.check_output(['squeue', '--user', username, "-o", f"%A{DELIMITER}%j{DELIMITER}%t"]).decode('utf-8')

    jobs_to_cancel = []
    for i,l in enumerate(sinfo_text.split('\n')):
        if i == 0 or len(l) == 0:
            continue # skip header
        job = parseJob(l)
        if not filter_name(job):
            continue
        if not filter_range(job):
            continue
        jobs_to_cancel.append(job)

    if len(jobs_to_cancel) == 0:
        print("Matching no jobs. Exiting")
        sys.exit(1)
    else:
        print("Matching these jobs:")
        for j in jobs_to_cancel:
            print(f"{j['id']:<10}{j['name']:<60}{j['state']:>5}")

    while True:
        print(f"Do you want cancel {len(jobs_to_cancel)} jobs? [y/n]")
        selection = input().lower()
        if selection in ['y', 'n']:
            break
    if selection == 'n':
        print("Aborting. No jobs have been cancelled")
    else:
        print("Cancelling jobs")
        subprocess.run(['scancel'] + [str(j['id']) for j in jobs_to_cancel])

