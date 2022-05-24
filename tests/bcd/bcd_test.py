import numpy as np
from jaxsot.core.block_coordinate_descent import opt_map_l2, opt_ref_vr
import tqdm

def test_bcd():

    np.random.seed(1)
    Ni=10
    Nj=20
    Nk=3
    Nl=5
    W = np.random.rand(Ni,Nj)
    Ain = np.random.rand(Nj,Nk)
    Xin = np.random.rand(Nk,Nl)
    Y = np.dot(W, np.dot(Ain,Xin))

    A = np.random.rand(Nj,Nk)
    X = np.random.rand(Nk,Nl)
    lamA=10**(-1)
    lamX=10**(2)
    maxiter=100

    for i in tqdm.tqdm(range(0,maxiter)):
        for k in range(0,Nk):
            X[k,:]=opt_ref_vr(k,Y,W,A,X,lamX)
            A[:,k]=opt_map_l2(k,Y,W,A,X,lamA)

    Y = np.dot(W, np.dot(A,X))
    refs=492.2236970510863
    assert np.abs(np.sum(Y)-refs)<1.e-16

if __name__=="__main__":
    test_bcd()
