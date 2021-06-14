import time
from datetime import datetime


def getTimestampString():
    """
    Return pretty-printed current timestamp.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S.%f")[:-3]


class TimingStamp:
    """
    Small utility class that enables simple timing/profiling of code.

    Examples:
        >>> ts = TimingStamp("My Heavy computation")
        >>> someHeavyComputation()
        >>> ts.stop()
        >>> print(ts)
    """
    def __init__(self, name):
        """
        Init the timestamp and start the timer.
        """
        self.name = name
        self.t_start = time.time()

    def stop(self):
        """
        Stop the timing.
        """
        self.t_end = time.time()

    @property
    def t(self):
        """
        Elapsed time.
        """
        return self.t_end - self.t_start

    def __str__(self):
        """
        Pretty print name and duration of timed step.
        """
        s = f"{self.name:<30}: {self.t:.3f} sec"
        return s