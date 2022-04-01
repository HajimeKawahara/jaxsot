from sklearn.datasets import make_spd_matrix
import jaxopt
import numpy as np
import jax.numpy as jnp

def f(x,A,b):
    return 0.5*jnp.dot(x,jnp.dot(A,x)) - jnp.dot(b,x)

def test_spd():
    np.random.seed(1)
    N=10
    A = make_spd_matrix(N)
    b = np.random.rand(N)

    gd = jaxopt.GradientDescent(fun=f, maxiter=500)
    res = gd.run(init_params=np.random.rand(N),A=A,b=b)
    print(res.params)
    ref=12.169304
    assert np.abs(np.sum(res.params)-ref) == 0.0
    
if __name__=="__main__":
    test_spd()
