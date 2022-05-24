import numpy as np
from jaxsot.core.block_coordinate_descent import opt_map_l2, opt_ref_vr
import tqdm
import pytest

def test_bcd():

    np.random.seed(1)
    Ni=3
    Nj=8
    Nk=2
    Nl=4
    W = np.random.rand(Ni,Nj)
    Ain = np.random.rand(Nj,Nk)
    Xin = np.random.rand(Nk,Nl)
    Y = np.dot(W, np.dot(Ain,Xin))

    A = np.random.rand(Nj,Nk)
    X = np.random.rand(Nk,Nl)
    lamA=10**(-1)
    lamX=10**(2)
    maxiter=10

    for i in tqdm.tqdm(range(0,maxiter)):
        for k in range(0,Nk):
            X[k,:]=opt_ref_vr(k,Y,W,A,X,lamX)
            A[:,k]=opt_map_l2(k,Y,W,A,X,lamA)

    Y = np.dot(W, np.dot(A,X))
    assert np.sum(Y) == pytest.approx(20.90292796223254)

if __name__=="__main__":
    test_bcd()
