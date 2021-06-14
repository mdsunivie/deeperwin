import logging
import sys

def _handle_exception(exc_type, exc_value, exc_traceback):
    logger = getLogger()
    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    # re-raise the exception so the program still terminates
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


def getLogger(name='erwin', fname = 'erwin.log', file_level = logging.DEBUG, console_level=logging.DEBUG, force_new=False, catch_exceptions=True):
    """
    Get a python logger configured to write to console and file.

    Args:
        name (str): Name of the logger
        fname (str): Filename where to dump the logfiles. Set to None to disable file logging
        file_level: Log-level for the file handler
        console_level: Log-level for the console handler
        force_new (bool): Force creation of a new logger. If set to False the same object will be returned when calling this function multiple times
        catch_exceptions (bool): If True, exceptions will be logged in the logfile and then re-raised

    Returns:
        (logging.Logger): Logger
    """
    logger = logging.getLogger(name)
    if force_new or (len(logger.handlers) == 0):
        logger.setLevel(logging.DEBUG)
        logger.handlers = []

        if fname is not None:
            fh = logging.FileHandler(fname)
            fh.setLevel(file_level)
            fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(console_level)
        ch.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(ch)

        if catch_exceptions:
            sys.excepthook = _handle_exception
    return logger