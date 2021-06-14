#!/usr/bin/env python3
from datetime import datetime
import json
import re
import numpy as np
import argparse
import os
import deeperwin.references.references

TARGET_ENERGIES = deeperwin.references.references.get_reference_energies()

class TerminalColors:
    RED = '\033[38;5;1m'
    GREEN = '\033[38;5;2m'
    YELLOW = '\033[38;5;3m'
    BLUE = '\033[38;5;4m'
    BLUE2 = '\033[38;5;12m'
    VIOLET = '\033[38;5;5m'
    CYAN = '\033[38;5;6m'
    GREY = '\033[38;5;7m'
    RED2 = '\033[38;5;9m'
    GREEN2 = '\033[38;5;10m'
    ENDCOLOR = '\033[0m'
    BLACK = '\033[38;5;0m'
    WHITE = '\033[38;5;15m'


class ErwinLogFile:
    """
    Class that represents a parsed logfile of a DeepErwin calculation.

    Useful to get status of a calculation, estimated time to finish and a quick summary of the result (e.g. energy + std-dev).
    """
    STATES = ['train_burnin', 'train', 'eval_burnin', 'eval', 'finished', 'exception', 'unknown']
    TIME_FORMAT = '%Y-%m-%d %H:%M:%S,%f'
    color_dict = dict(finished=TerminalColors.GREEN2,
                      eval=TerminalColors.GREEN,
                      eval_burnin=TerminalColors.BLUE2,
                      train=TerminalColors.WHITE,
                      train_burnin=TerminalColors.YELLOW,
                      unknown=TerminalColors.RED,
                      timeout=TerminalColors.CYAN,
                      exception=TerminalColors.RED)

    PATTERN_TRAIN = re.compile(r'\n(.{23}) - erwin.* - DEBUG - Epoch (\d+), batch (\d+): loss=([-+]?[0-9]*\.?[0-9]*)')
    PATTERN_EVAL = re.compile(r'\n(.{23}) - erwin.* - INFO - evaluation iteration (\d+); mean local energy: ([-+]?[0-9]*\.?[0-9]*); total energy: ([-+]?[0-9]*\.?[0-9]*)')

    def __init__(self, fname):
        self.time_out = 10 * 60
        self.fname = ""
        self.content = ""
        self.lines = []
        self.config_dict = {}
        self.n_epochs_train = None
        self.n_epochs_eval = None
        self.name = ""
        self.energies = []
        self.epochs = []
        self.batches = []
        self.eval_energies = []
        self.eval_epochs = []
        self.status = "unknown"
        self.E_train = np.nan
        self.std_E_train = np.nan
        self.E_eval = np.nan
        self.std_E_eval = np.nan
        self.last_modified = os.path.getmtime(fname)

        self.load(fname)
        self.parse()

    def load(self, fname):
        """
        Load the file into memory
        """
        self.fname = fname
        with open(fname) as f:
            self.content = f.read()
            self.lines = [l for l in self.content.split('\n') if len(l) > 0]

    def parse(self):
        """
        Parse a log-file by extracting all relevant information and storing it in the attributes
        Returns:
            None
        """
        if 'CRITICAL - Uncaught exception' in self.content:
            # try to find a restart by finding the last working start of a calculation
            for i in range(len(self.lines)-1, -1, -1):
                if ' - INFO - Starting calculation' in self.lines[i]:
                    break
            self.lines = self.lines[i:]
            self.content = "\n".join(self.lines)

        if len(self.lines) > 2:
            self.is_parallel_training_master = self._is_parallel_training_masterfile()
            if not self.is_parallel_training_master:
                self.status = self.get_status()
                self.extract_config()
                self.extract_train_energy()
                self.extract_eval_energy()


    def get_calc_dir_shortened(self):
        """
        Return shortened string of the calculation directory that only contains the last 2 levels
        Returns:
            (str): Shortened calc-dir name
        """
        directory = os.path.dirname(self.fname)
        directory = '/'.join(directory.split('/')[-2:])
        return directory

    @staticmethod
    def _get_content(line):
        return ' - '.join(line.split(' - ')[3:])

    def string_to_float(self, string_list):
        return np.array([np.nan if x == '' else float(x) for x in string_list])

    @classmethod
    def _get_timestamp(cls, line):
        time_string = line.split(' - ')[0]
        return datetime.strptime(time_string, cls.TIME_FORMAT)

    def _is_parallel_training_masterfile(self):
        if 'Starting parallel wavefunction training' in self.content:
            return True
        return False

    def extract_config(self):
        """
        Extracts the configuration from the log-file and stores key parameters, e.g. self.n_epochs_train, self.n_epochs_eval, self.name
        """
        for line in self.lines:
            if 'physical.n_electrons' in line:
                break
        self.config_dict = json.loads(self._get_content(line))
        self.n_epochs_train = self.config_dict['optimization.n_epochs']
        try:
            self.n_epochs_eval = self.config_dict['evaluation.n_epochs_max']
        except KeyError:
            self.n_epochs_eval = self.config_dict['evaluation.n_epochs']
        self.name = self.config_dict['physical.name']

    def is_alive(self):
        """
        Check if a calculation is still runnning, by checking how long ago the logfile was written to.

        Returns:
            (bool): True, if estimated to be still running
        """
        td = datetime.now().timestamp() - self.last_modified
        return td <= self.time_out

    def get_status(self):
        """
        Get status of calculation by searching for key messages in the log-file.
        Returns:
            (str): Identifier of current status
        """
        if 'CRITICAL - Uncaught exception' in self.content: #and 'Error' in self.lines[-1]:
            return "exception"
        if 'INFO - Total energy:' in self.content:
            return 'finished'
        if 'INFO - evaluation iteration' in self.content:
            return 'eval'
        if 'DEBUG - Evaluate: Burn-in' in self.content:
            return 'eval_burnin'
        if ', batch ' in self.content: 
            return 'train'
        if 'Starting burn-in train...' in self.content:
            return 'train_burnin'
        return "unknown"

    def get_energy_error_string(self, E, std_E):
        """
        Return a formated string of the form value +- uncertainty
        """
        error = (E - TARGET_ENERGIES[self.name]) * 1000
        std_E = std_E * 1000
        return f"Error = {error: 6.1f} +- {std_E: 4.1f}"

    def matches_status_filter(self, value):
        """
        Returns True if the calculation matches a filter on the calculation status.
        Args:
            value (str): Status to filter for. Any of: 'running', 'timeout', 'train_burnin', 'train', 'eval_burnin', 'eval', 'finished', 'exception', 'unknown'

        Returns:
            (bool): True if it matches the filter
        """
        if value is None:
            return True
        if value == 'running':
            return (self.status != 'finished') and self.is_alive()
        elif value == 'timeout':
            return (self.status != 'finished') and (not self.self.is_alive())
        else:
            return (value == self.status)

    def matches_directory_filter(self, value):
        """
        Returns True if the calculation matches a filter on the directory name.
        Args:
            value (str): Regex to match the directory name against

        Returns:
            (bool): True if it matches the filter
        """
        if value is None:
            return True
        dir = self.get_calc_dir_shortened()
        return re.search(value, dir) is not None

    def _get_status_message(self):
        if self.status == "exception":
            msg = f"Uncaught exception: {self.lines[-1]}"
        elif self.status in ['finished', 'eval']:
            if self.status == 'finished':
                msg = f"Calculation finished. {self.get_energy_error_string(self.E_eval, self.std_E_eval)}"
            else:
                msg = f"Evaluation {self.eval_epochs[-1]:4d}/{self.n_epochs_eval:4d}: {self.get_energy_error_string(self.E_eval, self.std_E_eval)}"
        elif self.status in ['eval_burnin', 'train']:
            if self.status == 'eval_burnin':
                msg = f"Optimization finished, Evaluation burnin ongoing. Train {self.get_energy_error_string(self.E_train, self.std_E_train)}"
            else:
                msg = f"Optimization {self.epochs[-1]:4d}/{self.n_epochs_train:4d}: Train {self.get_energy_error_string(self.E_train, self.std_E_train)}"
        elif self.status == 'train_burnin':
            msg = "Optimization burn-in"
        else:
            msg = "Unknown status"

        n_nan = self._get_n_nan_batches()
        if n_nan > 50:
            msg += f" (Warning: {n_nan} batches with nan gradient)"
        return msg

    def _get_eta_string(self):
        if self.status in ['unknown', 'finished', 'train_burnin', 'exception']:
            return ""
        else:
            t = self.estimate_time_to_finish()
            if t is None:
                return ""
            else:
                try:
                    t = int(t)
                    hours = t // 3600
                    minutes = (t-hours*3600) // 60
                    return f"ETA: {hours:2d}h{minutes:02d}min"
                except ValueError:
                    return ""

    def _get_n_nan_batches(self):
        if len(self.lines) > 0:
            return len([l for l in self.lines if l.endswith('Gradient contained nan. Skipping this batch')])
        else:
            return 0

    def get_summary(self):
        """
        Returns a color-coded summary status message
        Returns:
            (str): Status message
        """
        directory = self.get_calc_dir_shortened()
        msg = self._get_status_message() + " " + self._get_eta_string()
        if self.is_alive() or self.status in ['finished', 'exception']:
            color = self.color_dict[self.status]
        else:
            color = self.color_dict['timeout']
            msg = "Timed out: " + msg

        summary = f"{color}{directory:<55}: {msg}{TerminalColors.ENDCOLOR}"
        return summary

    def extract_eval_energy(self):
        """
        Calculates last estimate of training energy as an avarage of the last n_epochs samples.

        Results are stored in self.eval_epochs, self.eval_energies, self.E_eval, self.std_E_eval
        Returns:
            None
        """
        res = self.PATTERN_EVAL.findall(self.content)
        if len(res) > 0:
            self.t_eval, self.eval_epochs, self.eval_energies, self.total_eval_energies = zip(*res)
            self.eval_epochs = np.array([int(e) for e in self.eval_epochs])
            self.eval_energies = self.string_to_float(self.eval_energies)
            self.total_eval_energies = self.string_to_float(self.total_eval_energies)
            self.E_eval = self.total_eval_energies[-1]
            self.std_E_eval = np.std(self.eval_energies) / np.sqrt(np.max(self.eval_epochs))

    def extract_train_energy(self, n_epochs=10):
        """
        Calculates last estimate of training energy as an avarage of the last n_epochs samples.

        Results are stored in self.epochs, self.batches, self.energies, self.E_train, self.std_E_train
        Args:
            n_epochs (int): Nr of samples to include in average

        Returns:
            None
        """
        res = self.PATTERN_TRAIN.findall(self.content)
        if len(res) > 0:
            self.t_train, self.epochs, self.batches, self.energies = zip(*res)
            self.epochs = np.array([int(e) for e in self.epochs])
            self.batches = np.array([int(b) for b in self.batches])
            self.energies = self.string_to_float(self.energies)
            include_in_avg = self.epochs >= (self.epochs[-1] - n_epochs)
            E_incl = self.energies[include_in_avg]
            self.E_train = np.mean(E_incl)
            self.std_E_train = np.std(E_incl) / np.sqrt(len(E_incl))

    def estimate_time_to_finish(self):
        """
        Estimate total time required to finish this calculation in seconds
        Returns:
            (float): Estimated time remaining in seconds
        """
        if self.status in ['unknown', 'train_burnin']:
            return None
        if self.status == 'finished':
            return 0
        if self.status in ['train', 'eval_burnin']:
            if np.max(self.epochs) < 5:
                return None
            ind_start = np.where((self.epochs >= 3) & (self.batches == 0))[0][0]
            ind_end = np.where(self.batches == 0)[0][-1]
            delta_t = datetime.strptime(self.t_train[ind_end], self.TIME_FORMAT) - datetime.strptime(self.t_train[ind_start], self.TIME_FORMAT)
            time_per_epoch = delta_t.total_seconds() / (self.epochs[ind_end] - self.epochs[ind_start])
            t_optim =  time_per_epoch * (self.n_epochs_train - np.max(self.epochs))
            t_eval = time_per_epoch * 0.5 * self.n_epochs_eval # assumes that evaluation is 2x faster than optimization
            return t_optim + t_eval
        elif self.status == 'eval':
            if len(self.eval_epochs) < 3:
                return None
            delta_t = datetime.strptime(self.t_eval[-1], self.TIME_FORMAT) - datetime.strptime(self.t_eval[1], self.TIME_FORMAT)
            time_per_epoch = delta_t.total_seconds() / (self.eval_epochs[-1] - self.eval_epochs[1])
            t_eval = time_per_epoch * (self.n_epochs_eval - np.max(self.eval_epochs))
            return t_eval

    def get_state_id(self):
        """
        Map current state of calculation to the corresponding state id.
        """
        return self.STATES.index(self.status)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', default='.', nargs='?')
    parser.add_argument('--status', '-s', nargs='?', default=None, choices=['running', 'timeout'] + ErwinLogFile.STATES,
                        help="Only show runs with matching status")
    parser.add_argument('--name', '-n', nargs='?', default=None, help="Regex filter for directory name")
    parser.add_argument('--time', '-t', nargs='?', default=np.inf, help="Filter for modification date in hours (i.e. only files changed withing the last XX hours will be displayed)")
    parser.add_argument('--sort', nargs='*', default=["state", "path", "energy"], choices=['state', 'name', 'path', 'energy'], help='Specify arguments by which to sort the output. Multiple arguments can be specified.')
    args = parser.parse_args()

    logfiles = []
    for root, dirs, files in os.walk(args.directory):
        if 'erwin.log' in files:
            fname = os.path.join(root, 'erwin.log')
            if (datetime.utcnow().timestamp() - os.path.getmtime(fname)) < (float(args.time) * 3600):
                try:
                    lf = ErwinLogFile(fname)
                    if not lf.matches_status_filter(args.status):
                        continue
                    if not lf.matches_directory_filter(args.name):
                        continue
                    if lf.is_parallel_training_master:
                        continue
                    logfiles.append(lf)
                except Exception as e:
                    print(f"ERROR: Could not parse {fname}; {e}")

    # Sort the logfiles by all the specified arguments
    data = []
    for i,lf in enumerate(logfiles):
        entry = []
        for sort_arg in args.sort:
            if sort_arg == 'name':
                entry.append(lf.name)
            elif sort_arg == 'path':
                entry.append(lf.fname)
            elif sort_arg == 'state':
                entry.append(lf.get_state_id())
            elif sort_arg == 'energy':
                entry.append(lf.E_eval)
            else:
                raise ValueError(f"Unknown sort argument: {sort_arg}")
        entry.append(i)
        entry.append(lf)
        data.append(entry)

    for entry in sorted(data):
        try:
            print(entry[-1].get_summary())
        except Exception as e:
            print(f"ERROR: Exception while summarizing: {entry[-1].fname}")
            raise e



