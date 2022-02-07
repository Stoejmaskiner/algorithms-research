from scipy.fft import fft, ifft
from alive_progress import alive_bar
from alive_progress import config_handler
import numpy as np

config_handler.set_global(
    title_length = 30
)


def row_wise_fft(matrix: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """computes the 1-d fft of every row in the matrix, note that the length
    of each row is padded up to the closest power of 2
    
    ## Parameters
    - `matrix`: matrix of audio clips, rows are clips
    
    ## Returns
    A tuple of:
    - number of bins (always a power of 2)
    - matrix of amplitudes
    - matrix of phases
    """

    num = 2**int(np.ceil(np.log2(len(matrix[0]))))
    
    ret_rho = []
    ret_phi = []
    with alive_bar(len(matrix), title='row-wise fft') as bar:
        for row in matrix:
            bins = fft(row, num)
            bins_rho = np.abs(bins)
            bins_phi = np.angle(bins)
            ret_rho.append(bins_rho)
            ret_phi.append(bins_phi)
            bar()

    return (num, np.array(ret_rho), np.array(ret_phi))


def row_wise_ifft(num: int, rho_matrix: np.ndarray, phi_matrix: np.ndarray) -> np.ndarray:
    """computes the 1-d inverse fft of every row in the matrix, the length
    should be a power of 2
    
    ## Parameters:
    - `num`: size of each row, should be a power of 2
    - `rho_matrix`: matrix of amplitudes
    - `phi_matrix`: matrix of phases
    """
    out_row = []
    with alive_bar(len(rho_matrix), title='row-wise ifft') as bar:
        for rho_row, phi_row in zip(rho_matrix, phi_matrix):
            out_row.append(ifft(rho_row * np.exp(1j*phi_row)).real)
            bar()
    return np.array(out_row)
