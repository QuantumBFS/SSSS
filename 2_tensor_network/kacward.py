import numpy as np 

'''
Kac-Ward exact Ising 
See Theorem 1 of https://arxiv.org/abs/1011.3494
'''

phi = np.array([[0., np.pi/2, -np.pi/2, np.nan ],
                [-np.pi/2, 0.0, np.nan, np.pi/2],
                [np.pi/2, np.nan, 0.0, -np.pi/2],
                [np.nan, -np.pi/2, np.pi/2, 0]
                ])

def logcosh(x):
    xp = np.abs(x)
    if (xp< 12):
        return np.log( np.cosh(x) )
    else:
        return xp - np.log(2.)

def neighborsite(i, n, L):
    """
    The coordinate system is geometrically left->right, down -> up
          y|
           |
           |
           |________ x
          (0,0)
    So as a definition, l means x-1, r means x+1, u means y+1, and d means y-1
    """
    x = i%L
    y = i//L # y denotes 
    site = None
    # ludr :  
    if (n==0):
        if (x-1>=0):
            site = (x-1) + y*L
    elif (n==1):
        if (y+1<L):
            site = x + (y+1)*L
    elif (n==2):
        if (y-1>=0):
            site = x + (y-1)*L
    elif (n==3):
        if (x+1<L):
            site = (x+1) + y*L
    return site

#K: ludr 
def kacward_solution(L, K):
    V = L**2    # number of vertex 
    E = 2*(V-L) # number of edge

    D = np.zeros((2*E, 2*E), np.complex128)
    ij = 0
    ijdict = {}
    for i in range(V):
        for j in range(4):
            if neighborsite(i, j, L) is not None:
                D[ij, ij] = np.tanh(K[i, j])
                ijdict[(i,j)] = ij # mapping for (site, neighbor) to index
                ij += 1

    A = np.zeros((2*E, 2*E), np.complex128)
    for i in range(V):
        for j in range(4):
            for l in range(4):
                k = neighborsite(i, j, L)
                if  (not np.isnan(phi[j, l])) and (k is not None) and (neighborsite(k, l, L) is not None):
                    ij = ijdict[(i, j)]
                    kl = ijdict[(k, l)] 
                    A[ij, kl] = np.exp(1J*phi[j, l]/2.) 

    res = V*np.log(2) 
    for i in range(V):
        for j in [1,3]: # only u, r to avoid double counting
            if neighborsite(i, j, L) is not None:
                res += logcosh(K[i, j]) 
    _, logdet = np.linalg.slogdet(np.eye(2*E, 2*E, dtype=np.float64) - A@D )
    res += 0.5*logdet

    return res     

def lnZ_2d_ferro_Ising(L,beta):
    K = np.ones((L**2, 4)) * beta
    lnZ = kacward_solution(L, K) 
    return lnZ

if __name__=='__main__':
    L = 16 
    
    for beta in np.linspace(0.1, 5.0, 50):
        K = np.ones((L**2, 4)) * beta
        lnZ = kacward_solution(L, K) 
        print ("%g"%beta, "%.8f"%(-1.0/beta*lnZ/L**2))
