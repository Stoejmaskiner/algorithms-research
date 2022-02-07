import numpy as np
from scipy.interpolate import interp1d
import inspect

ANSI_RED = '\u001b[31m'
ANSI_YELLOW = '\u001b[33m'
ANSI_RESET = '\u001b[0m'
ANSI_BRIGHT_BLACK = '\u001b[30;1m'

def ez_resampler(data, fromSr, toSr) -> np.ndarray:
    xold = np.linspace(0, 10, len(data))
    yold = data
    xnew = np.linspace(0, 10, int(np.ceil(len(data)/fromSr * toSr)))
    f = interp1d(xold, yold, kind='cubic')
    ynew = f(xnew)
    return ynew

def dbfs_to_amplitude(dbfs: float) -> float:
    return 10**(dbfs / 20)

def amplitude_to_dbfs(amp: float) -> float:
    return 20 * np.log10(amp)

def dbg_error_formatter(msg: str) -> str:
    return f'{ANSI_RED}ERROR {msg}{ANSI_RESET}'

def dbg_unimportant_formatter(msg: str) -> str:
    return f'{ANSI_BRIGHT_BLACK}{msg}{ANSI_RESET}'
