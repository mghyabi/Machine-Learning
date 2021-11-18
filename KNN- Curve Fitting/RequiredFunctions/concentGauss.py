def concentGauss(N,vc,vo,r):
    c=np.sqrt(vc)*np.random.randn(2,N//2)
    l1=np.ones((1,N//2))
    E1=np.vstack((c,l1))

    ro=np.sqrt(vo)*np.random.randn(1,N//2)+r
    t=np.random.uniform(low=0,high=2*np.pi,size=N//2)
    l2=-1*np.ones((1,N//2))
    E2=np.vstack((ro*np.cos(t),ro*np.sin(t),l2))
    
    E=np.hstack((E1,E2))
    return E
