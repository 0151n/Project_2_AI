import cvxopt as cvx
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial

def makeB(K,lambdas,X,t,zero_tol=1e-10, precision=6):
    """Make bias under kernel K.  Lambdas and ts cvxopt column vectors.
       X is a cvxopt matrix with 1 training vector per row."""
    supports = 0
    b = 0
    for s in range(len(lambdas)):
        if lambdas[s] > zero_tol:  # lambdas[s] is a support vector if > 0.
            supports += 1
            b += t[s] - sum(cvx.mul(cvx.mul(lambdas,t),K(X,X[s,:])))
    return round(b / supports, precision)


def Krbf(x,y,s2):
    """RBF kernel on two CVXOPT row vectors, or matrices. s2 is the RBF variance parameter."""
    return cvx.matrix(np.exp(-scipy.spatial.distance.cdist(x,y,metric='sqeuclidean')/(2*s2)))

def kernClassify(xvec,K,lambdas,Xs,ts,b,zero_tol=1e-10):
    """Requires X to be a matrix of training inputs arranged as cvxopt row vectors.
       'xvec', the input to be classified can be a cvxopt row vector, a Python list
       representing a row vector, or a NumPy 1-d array."""
    # Do conversions on xvec if needed to coerce to a cvxopt matrix with 1 row and n cols
    # (i.e., a cvxopt row vector).
    if isinstance(xvec,list):          # Convert Python list to cvxopt row vector.
        xvec = cvx.matrix(xvec).T
    elif isinstance(xvec,np.ndarray):  # Convert NumPy array to cvxopt row vector.
        xv = xvec
        xvec = cvx.matrix(xv) if xv.shape[0] == 1 else cvx.matrix(xv).T
    #-----------------------------------------------------------
    # Actual calculation.  y is activation level.
    y = b + sum(cvx.mul(cvx.mul(lambdas,ts),K(Xs,xvec)))
    return +1 if y > 0 else -1


#load training dataset

dataset = np.loadtxt("train.txt",dtype='float')
points,labels = dataset[:,:2], dataset[:,2]

#plot dataset
##plt.figure(figsize=(5,5))
##plt.scatter(points.T[0], points.T[1], c=labels.T, cmap='bwr')
##plt.grid()
##plt.axis('equal');
##plt.show()

N = len(labels)
C = 1

#CVXOPT inputs to qp.
bv = cvx.matrix(0.0)
qn = cvx.matrix(-np.ones(N))
Gn = cvx.matrix(np.vstack((-np.eye(N),np.eye(N))))
hn = cvx.matrix(np.hstack((np.zeros(N),C*np.ones(N))))
Xn = cvx.matrix(points)
tn = cvx.matrix(labels)

S2 = 0.25   # Variance for RBF kernel.

Pn_rbf = cvx.mul(tn*tn.T, Krbf(Xn,Xn,S2))

r_n_rbf = cvx.solvers.qp(Pn_rbf,qn,Gn,hn,tn.T,bv)


lambdas_rbf = cvx.matrix([round(li,6) for li in r_n_rbf['x']])

Krbf_025 = lambda x,y: Krbf(x,y,S2)

b_rbf = makeB(Krbf_025,lambdas_rbf,Xn,tn)

misclass = 0
for xvec,lab in zip(points,labels):
    op = kernClassify(xvec, Krbf_025, lambdas_rbf, Xn, tn, b_rbf)
    if op != lab: 
        misclass += 1
        print(f"[{xvec[0]:7.4f},{xvec[1]:7.4f}] --> {op:+2d} ({lab:+2.0f})")

print(f"\n{misclass}/{len(labels)} misclassifications")

print("no. suppport vecs " + str(np.sum(np.array(lambdas_rbf)>1e-10)))
