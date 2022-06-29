import numpy as np
from jaxsot.core.block_coordinate_descent import opt_ref_vr, opt_map_l1tsv
import tqdm
import pytest

def test_bcd_l1tsv_vr():

    np.random.seed(1)
    Ni=3
    Nj=12
    Nk=2
    Nl=4
    W = np.random.rand(Ni,Nj)
    Ain = np.random.rand(Nj,Nk)
    Xin = np.random.rand(Nk,Nl)
    Y = np.dot(W, np.dot(Ain,Xin))

    A = np.random.rand(Nj,Nk)
    X = np.random.rand(Nk,Nl)
    laml1=10**(-3)
    lamtsv=10**(-2)
    lamX=10**(2)
    maxiter=5

    for i in range(maxiter):
        for k in range(0,Nk):
            X[k,:]=opt_ref_vr(k,Y,W,A,X,lamX)
            A[:, k] = opt_map_l1tsv(k, Y, W, A, X, laml1, lamtsv)

    Y = np.dot(W, np.dot(A,X))
    assert np.sum(Y) == pytest.approx(24.456713724102265)

if __name__=="__main__":
    test_bcd_l1tsv_vr()
