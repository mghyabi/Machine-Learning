def noisySin(N,v):
    N-=1
    xn=np.arange(0,1+1/N,1/N)
    tn=np.sin(2*np.pi*xn)+np.sqrt(v)*np.random.randn(len(xn))
    E=np.vstack((xn,tn))
    return E
