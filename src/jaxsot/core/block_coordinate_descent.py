import numpy as np
import jax.numpy as jnp
import jaxopt
import healpy as hp
from jaxsot.core.neighbor import calc_neighbor_weightmatrix

def QP_objective(params,W,b,shapeA0):
    """objective function, Q = 1/2 params^T W params + b^T params, for Quadratic Programming (QP)

    Args:
        params: parameter vector 
        W: symmetric matrix
        b: vector
        shapeA0: value for normalization

    Returns:
        normalized value of objective function
    
    """
    obj = 0.5*jnp.dot(params, jnp.dot(W,params)) + jnp.dot(b,params)
    return obj/shapeA0

def LS_TSV_objective(params, W, p, normxk, wtsv, lamtsv):
    """objective function, Q = 1/2 || p - W params ||^2 + lamtsv params^T wtsv params,
       for Least Squares method (LS) with TSV regularization

    Args:
        params: parameter vector 
        W: symmetric matrix
        p: vector
        normxk: value for normalization
        wtsv: negihbor matrix for TSV
        lamtsv: regularization parameter for TSV
        
    Returns:
        normalized value of objective function
    
    """
    residuals = p - jnp.dot(W,params)
    regtsv = lamtsv * jnp.dot(params, jnp.dot(wtsv, params))

    return 0.5 * jnp.sum(residuals ** 2) + regtsv / normxk


def opt_ref_vr(k,Y,W,A,X,lamX):
    """optimization for reflectivity of k-th Component (volume regularization)

    Args:
        k: the number of target Component
        Y: multiband light curve
        W: weight
        A: multiband map
        X: multiband reflectivity
        lamX: regularization parameter for X

    Returns: 
        updated reflectivity of k-th Component
 
    """
    Nl = np.shape(Y)[1]

    AX=np.dot(np.delete(A,obj=k,axis=1),np.delete(X,obj=k,axis=0))
    Delta=Y-np.dot(W,AX)
    ak=A[:,k]
    Wa=np.dot(W,ak)
    W_x=np.dot(Wa,Wa)*np.eye(Nl)
    bx=-np.dot(np.dot(Delta.T,W),ak)
    Xminus = np.delete(X,obj=k,axis=0)
    XXTinverse=np.linalg.inv(np.dot(Xminus,Xminus.T))
    K=np.eye(Nl) - np.dot(np.dot(Xminus.T,XXTinverse),Xminus)
    K=K*np.linalg.det(np.dot(Xminus,Xminus.T))*lamX   

    pg = jaxopt.ProjectedGradient(fun=QP_objective, projection=jaxopt.projection.projection_non_negative)
    state=pg.init_state(init_params=jnp.array(X[k,:]))
    params,state=pg.update(params=jnp.array(X[k,:]),state=state,W=W_x+K,b=bx,shapeA0=A.shape[0])

    return params

def opt_map_l2(k,Y,W,A,X,lamA):
    """optimization for map of k-th Component (L2 regularization)

    Args:
        k: the number of target Component
        Y: multiband light curve
        W: weight
        A: multiband map
        X: multiband reflectivity
        lamA: regularization parameter for A

    Returns: 
        updated map of k-th Component
 
    """
    Nj=np.shape(A)[0]
    AX=np.dot(np.delete(A,obj=k,axis=1),np.delete(X,obj=k,axis=0))
    Delta=Y-np.dot(W,AX)

    xk=X[k,:]
    W_a=(np.dot(xk,xk))*(np.dot(W.T,W))
    b=-np.dot(np.dot(W.T,Delta),xk)
    T_a=lamA*np.eye(Nj)

    pg = jaxopt.ProjectedGradient(fun=QP_objective, projection=jaxopt.projection.projection_non_negative)
    state=pg.init_state(init_params=jnp.array(A[:,k]))
    params,state=pg.update(params=jnp.array(A[:,k]),state=state,W=W_a+T_a,b=b,shapeA0=A.shape[0])

    return params

def opt_map_l1tsv(k, Y, W, A, X, laml1, lamtsv):
    """optimization for map of k-th Component (L1+TSV regularization)

    Args:
        k: the number of target Component
        Y: multiband light curve
        W: weight
        A: multiband map
        X: multiband reflectivity
        laml1: regularization parameter for L1
        lamtsv: regularization parameter for TSV

    Returns:
        updated map of k-th Component

    """
    AX = np.dot(np.delete(A, obj=k, axis=1), np.delete(X, obj=k, axis=0))
    Delta = Y - np.dot(W, AX)

    xk = X[k, :]
    normxk = np.sum(xk ** 2)
    p = np.dot(Delta, xk) / normxk
    nside = hp.npix2nside(np.shape(A)[0])
    wtsv, _ = calc_neighbor_weightmatrix(nside)

    pg = jaxopt.ProximalGradient(fun = LS_TSV_objective, prox = jaxopt.prox.prox_non_negative_lasso)
    state = pg.init_state(init_params = jnp.array(A[:,k]), hyperparams_prox = laml1 / normxk)
    params, state = pg.update(params = jnp.array(A[:,k]), hyperparams_prox = laml1/normxk, state = state, W = W, p = p, normxk = normxk, wtsv = wtsv, lamtsv = lamtsv)

    return params
