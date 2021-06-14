#!/usr/bin/env python3
import subprocess
import os
import pwd
import argparse

def getUsername():
    """
    Get username on linux systems
    Returns:
        (str): Username

    """
    return pwd.getpwuid(os.getuid()).pw_name

class Job:
    """
    Small utility class representing a SLURM job.
    """
    def __init__(self, id, workdir, status):
        self.id = id
        self.workdir = workdir
        self.status = status

    def cancel(self):
        """
        Cancel the job using scancel.
        Returns:
            None
        """
        subprocess.run(["scancel", self.id])

    def changeQueueInJobScript(self, new_partition, new_qos=None, jobfname = 'job.sh'):
        """
        Change the queue/partition in the SLURM job-file (typically job.sh).

        This function modifies the jobfile of a potentially running job!
        Args:
            new_partition (str): New SLURM partition that will be used
            new_qos (str): New SLURM qos that will be used
            jobfname (str): Name of the jobfile

        Returns:
            None
        """
        fname = os.path.join(self.workdir, jobfname)
        if new_qos is None:
            new_qos = new_partition

        new_script_lines = []
        with open(fname) as f:
            for l in f:
                if '#SBATCH --partition' in l:
                    new_script_lines.append('#SBATCH --partition ' + new_partition)
                elif '#SBATCH --qos' in l:
                    new_script_lines.append('#SBATCH --qos ' + new_qos)
                else:
                    new_script_lines.append(l.strip())
        with open(fname, 'w') as f:
            f.write('\n'.join(new_script_lines))

    def submit(self):
        """
        Submit a job using sbatch.
        """
        subprocess.run(['sbatch', 'job.sh'], cwd=self.workdir)


def getJobs():
    """
    Get a list of all SLURM jobs of the current user by calling squeue
    Returns:
        (list): List of Job objects

    """
    p = subprocess.run(["squeue", "--user", getUsername(), "-o", "%i %Z %t"], capture_output=True, text=True)
    jobs = []
    for i, line in enumerate(p.stdout.strip().split('\n')):
        if i == 0:
            continue
        tokens = line.split()
        if len(tokens) != 3:
            continue
        jobs.append(Job(id=tokens[0],
                        workdir=tokens[1],
                        status=tokens[2]))
    return jobs


if __name__ == '__main__':
    partition_mapping = dict(gpu='gpu_gtx1080single', rtx='gpu_rtx2080ti', v100='gpu_v100', k20='gpu_k20m',
                             gtxmulti='gpu_gtx1080multi', gpu_gtx1080multi='gpu_gtx1080multi',
                             gpu_gtx1080single='gpu_gtx1080single', gpu_rtx2080ti='gpu_rtx2080ti', gpu_v100='gpu_v100',
                             gpu_k20m='gpu_k20m', gpu_gtx1080amd='gpu_gtx1080amd', gtxamd='gpu_gtx1080amd', cpu64='mem_0064', cpu128='mem_0128', cpu256='mem_0256',
                             cpu64p='vsc3plus_0064', cpu256p='vsc3plus_0256', mem_0064='mem_0064', mem_0128='mem_0128',
                             mem_0256='mem_0256', vsc3plus_0064='vsc3plus_0064', vsc3plus_0256='vsc3plus_0256',
                             mem_0096="mem_0096", mem_0384="mem_0384", mem_0768="mem_0768")

    parser = argparse.ArgumentParser()
    parser.add_argument("partition", choices=list(partition_mapping.keys()))
    parser.add_argument("--id", nargs='+', default=[])
    args = parser.parse_args()
    partition = partition_mapping[args.partition]

    jobs = getJobs()
    if len(args.id) > 0:
        valid_ids = [str(i).strip() for i in args.id]
        jobs = [j for j in jobs if str(j.id) in valid_ids]

    for j in jobs:
        if j.status != "R":
            j.cancel()
            j.changeQueueInJobScript(partition)
            j.submit()
            print(f"Switched queue for job {j.id}")
