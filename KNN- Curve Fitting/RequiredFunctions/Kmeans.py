def Kmeans(E,K,eta,tol):
    D=E.shape[0]
    N=E.shape[1]
    #initializing mu by shuffling data points and d
    i=np.arange(N)
    np.random.shuffle(i)
    E=E[:,i]

    mu=E[:,0:K]
    mu_old=mu.copy()
    d=np.zeros((K,N))

    l=1
    while l<10000:
        #distances (d) d has K rows (one for each cluster) and N columns (one for each datapoint)
        for i in np.arange(K):
            d[i,:]=np.sum((np.tile(np.reshape(mu[:,i],(D,1),order='C'),(1,N))-E)**2,axis=0)

        #defining tags (r) and index for the mu to change at each data point(r2)
        r=np.equal(d,np.tile(np.amin(d,axis=0),(K,1)))*1
        r2=r*(np.tile(np.arange(D).reshape((D,1)),(1,N))+1)

        #Learning algorithm
        for j in np.arange(N):
            k=np.sum(r2[:,j])
            mu[:,k-1]=mu[:,k-1]+eta*(E[:,j]-mu[:,k-1])
        if np.allclose(mu_old,mu,atol=tol):
            break
        #initializing mu by shuffling data points and d
        i=np.arange(N)
        np.random.shuffle(i)
        E=E[:,i]
        l+=1
        mu_old=mu.copy()
    return mu,l
