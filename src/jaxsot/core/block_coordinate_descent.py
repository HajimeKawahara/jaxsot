import numpy as np
import jax.numpy as jnp
import jaxopt

def QP_obj_xk(params,W_x,K,bx,shapeA0):
    obj = 0.5*jnp.dot(params, jnp.dot((W_x+K),params)) + jnp.dot(bx,params)
    return obj/shapeA0

def QP_obj_ak(params,W_a,T_a,b,shapeA0):
    obj = 0.5*jnp.dot(params, jnp.dot((W_a+T_a),params)) + jnp.dot(b,params)
    return obj/shapeA0

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

    pg = jaxopt.ProjectedGradient(fun=QP_obj_xk, projection=jaxopt.projection.projection_non_negative)
    state=pg.init_state(init_params=jnp.array(X[k,:]))
    params,state=pg.update(params=jnp.array(X[k,:]),state=state,W_x=W_x,K=K,bx=bx,shapeA0=A.shape[0])

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

    pg = jaxopt.ProjectedGradient(fun=QP_obj_ak, projection=jaxopt.projection.projection_non_negative)
    #res = pg.run(init_params=jnp.array(A[:,k]))
    state=pg.init_state(init_params=jnp.array(A[:,k]))
    params,state=pg.update(params=jnp.array(A[:,k]),state=state,W_a=W_a,T_a=T_a,b=b,shapeA0=A.shape[0])

    return params
