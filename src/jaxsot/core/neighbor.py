import healpy as hp
import numpy as np


def calc_neighbor_weightmatrix(nside):
    """Cauculation of matrix to express neighboring pixels, borrowed from
    Spare, https://github.com/2ndmk2/Spare/ by @2ndmk2.

    Args:
        nside: the number of HealPix one side

    Returns:
        weight matrix for TSV
        neibour matrix
    """
    nside_now = nside
    Npix = 12 * nside ** 2
    Neighbor_matrix = np.zeros((Npix, Npix))
    Weight_tsv_matrix = np.zeros((Npix, Npix))
    for i in range(Npix):
        neighbor = hp.get_all_neighbours(nside_now, i)
        for j in range(8):
            neighbor_ind = neighbor[j]
            if neighbor_ind == -1:
                continue
            Neighbor_matrix[i][neighbor_ind] = 1
            Weight_tsv_matrix[i][i] += 0.5
            Weight_tsv_matrix[i][neighbor_ind] -= 0.5
            Weight_tsv_matrix[neighbor_ind][i] -= 0.5
            Weight_tsv_matrix[neighbor_ind][neighbor_ind] += 0.5

    return Weight_tsv_matrix, Neighbor_matrix


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    nside = 8

    wtsv, nmat = calc_neighbor_weightmatrix(nside)
#    hp.visufunc.mollview(nmat[120,:])
    hp.visufunc.mollview(wtsv[120, :])
    plt.show()
