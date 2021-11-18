def Histogram(X,N):
    b=np.linspace(np.min(X),np.max(X),num=N)
    h=np.zeros(N)
    for m in np.arange(len(X)):
        e=np.abs(b-X[m])
        e=np.where(e==np.min(e),1,0)
        h+=e
    return h,b
