"""test for lc."""

import pytest
import numpy as np
import healpy as hp
from jaxsot.core.neighbor import calc_neighbor_weightmatrix


def test_neighbor():
    nside = 8
    wtsv, nmat = calc_neighbor_weightmatrix(nside)
    n = np.arange(0, len(nmat))
    ref_neighbor = np.array([90, 91, 119, 121, 152, 153, 184])
    diff_neighbor = np.sum((n[nmat[120, :] == 1]-ref_neighbor)**2)
    assert diff_neighbor == 0


def test_wtsv():
    nside = 8
    centpix_val = 7.0  # why?
    wtsv, nmat = calc_neighbor_weightmatrix(nside)
    n = np.arange(0, len(nmat))
    ref_neighbor = np.array([90, 91, 119, 121, 152, 153, 184])
    diff = np.sum((wtsv[120, ref_neighbor]+1.0)**2) + \
        np.abs(wtsv[120, 120]-centpix_val)
    assert diff == 0


if __name__ == '__main__':
    test_neighbor()
    test_wtsv()
