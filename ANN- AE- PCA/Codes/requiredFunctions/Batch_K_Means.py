def Batch_K_Means(E,K):
    D=np.shape(E)
    N=D[1]
    D=D[0]
    j=1
    
    #initializing mu
    mu_index=np.random.randint(N,size=K)
    mu_old=E[:,mu_index]
    d=np.zeros((K,N))

    while j<10000:#max iterations
        #distances (d) d has K rows (one for each cluster) and N columns (one for each datapoint)
        for i in np.arange(K):
            d[i,:]=np.sum((np.tile(np.reshape(mu_old[:,i],(D,1),order='C'),(1,N))-E)**2,axis=0)

        #using boolean operators to determine r from d
        r=np.equal(d,np.tile(np.amin(d,axis=0),(K,1)))*1
        
        #checking at least one data point falls into each cluster
        check=np.prod(np.sum(r,axis=1))
        if check==0:
            while check==0:
                mu_index=np.random.randint(N,size=K)
                mu_old=E[:,mu_index]
                for i in np.arange(K):
                    d[i,:]=np.sum((np.tile(np.reshape(mu_old[:,i],(D,1),order='C'),(1,N))-E)**2,axis=0)
                r=np.equal(d,np.tile(np.amin(d,axis=0),(K,1)))*1
                check=np.prod(np.sum(r,axis=1))

        #updating mu 
        mu_new=np.zeros((K,D))
        for i in np.arange(K):
            num=np.sum(np.tile(r[i,:],(D,1))*E,axis=1)
            den=np.sum(np.tile(r[i,:],(D,1)),axis=1)
            mu_new[i,:]=np.squeeze(num/den)
        mu_new=mu_new.T
    
        if np.allclose(mu_old,mu_new):
            break
        mu_old=mu_new.copy()
        j+=1
    
    J=np.zeros((K,N))    
    for i in np.arange(K):
        J[i,:]=np.sum((np.tile(np.reshape(mu_new[:,i],(D,1),order='C'),(1,N))-E)**2,axis=0)

    J=np.sum(J*r)
    for i in np.arange(K):
        d[i,:]=np.sum((np.tile(np.reshape(mu_new[:,i],(D,1),order='C'),(1,N))-E)**2,axis=0)
    r=np.equal(d,np.tile(np.amin(d,axis=0),(K,1)))*1
    return mu_new, J, j, r
