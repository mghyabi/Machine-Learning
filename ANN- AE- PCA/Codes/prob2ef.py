#P2_e
for name in dir():
    del globals()[name]
    
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import time

def trainMLP(X,D,H,eta,alpha,epochMax,MSETarget):
    a = 1.7159
    b = 2/3.
    [p, N] = np.shape(np.array(X))                 #dimension of input vector and number of training data pts
#     m = len(D)                                  #number of output neurons
    m = 1                                  #number of output neurons
    bias = -1                                      #bias value
    Wh=[]
    WhAnt=[]
    X = np.concatenate([bias*np.ones([1,N]),X ],axis=0)                  #add zero'th order terms
    for j in range(len(H)):
        if j ==0:
            Wh.append(np.random.rand(H[j],p+1))                          #initialize first hidden layer weights
            WhAnt.append(np.zeros([H[j],p+1]))                      #initialize variable for weight correction using momentum 
        else:
            Wh.append( np.random.rand(H[j],H[j-1]+1)  ) #initialize hidden layer weights
            WhAnt.append(np.zeros([H[j],H[j-1]+1]) )                #initialize variable for weight correction using momentum 
    Wo = np.random.rand(m,H[-1]+1)                                 #initialize output layer weights
    WoAnt = np.zeros([m,H[-1]+1])                            #initialize variable for weight correction using momentum
    MSETemp = np.zeros([epochMax,1])                   #allocate memory for MSE error for each epoch
    for i in range(epochMax):
        O=[]
        for j in range(len(H)):               #%loop over each hidden layer
            if j==0:
                V = Wh[j]@X               #%weighted sum of inputs [1] Eqn(4.29/30)
            else:
                V = Wh[j]@O[j-1]          #%weighted sum of hidden inputs [1] Eqn(4.29/31)
            PHI = a * np.tanh(b*V)         #%activation function [1] Eqn(4.37)
            O.append(np.concatenate([bias*np.ones([1,N]),PHI],axis=0))   #%add zero'th order terms
        V = Wo@O[-1]                 #%weighted sum of inputs [1] Eqn(4.29)
        Y = a * np.tanh(b*V)       #%activation function [1] Eqn(4.37)
        E = D - Y                  #%calclate error
        mse = np.mean(E**2)    #%calculate mean square error
        MSETemp[i,0] = mse           #%save mse
        #%DISPLAY PROGRESS, BREAK IF ERROR CONSTRAINT MET
#         print('epoch = ' +str(i)+ ' mse = ' +str(mse))
        if (mse < MSETarget):
            MSE = MSETemp
            return(Wh,Wo,MSE)
        PHI_PRMo = b/a *(a-Y)*(a+Y)   #%derivative of activation function [1] Eqn(4.38)
        dGo = E * PHI_PRMo                 #%local gradient [1] Eqn(4.35/39)
        DWo = dGo@O[-1].T                    #%non-scaled weight correction [1] Eqn(4.27)

        Wo = Wo + eta*DWo + alpha*WoAnt  #%weight correction including momentum term [1] Eqn(4.41)
        WoAnt = eta*DWo + alpha*WoAnt                         #%save weight correction for momentum calculation
        for j in np.arange(len(H))[::-1]:
            PHI_PRMh = b/a *(a-O[j])*(a+O[j])         #%derivative of activation function [1] Eqn(4.38)
            if j==(len(H)-1):
                dGh = PHI_PRMh * (Wo.T @ dGo)                   #%local gradient[1] Eqn(4.36/40)
            else:
                dGh = PHI_PRMh * (Wh[j+1].T @ np.matlib.repmat( dGo,Wh[j+1].shape[0],1 ) )         # %local gradient[1] Eqn(4.36/40)
            dGh = dGh[1:,:]                             #%dicard first row of local gradient (bias doesn't update)
            if j==0:
                DWh = dGh@X.T                            #%non-scaled weight correction [1] Eqn(4.27/30)
            else:
                DWh = dGh@O[j-1].T                       #%non-scaled weight correction [1] Eqn(4.27/31)
            Wh[j] =Wh[j]+ eta*DWh + alpha*WhAnt[j]  # %weight correction including momentum term [1] Eqn(4.41)
            WhAnt[j] =eta*DWh + alpha*WhAnt[j]     #%save weight correction for momentum calculation
    MSE = MSETemp
    return(Wh,Wo,MSE)

def MLP(X,Wh,Wo):
    a = 1.7159
    b = 2/3.
    N = len(X[0,:])               #%number of training data pts
    bias = -1                  # %initial bias value
    O=[]
    X = np.concatenate((bias*np.ones([1,N]) , X),axis=0)    #%add zero'th order terms
    H=[]
    for j in range(len(Wh)):
        H.append(len(Wh[j]))
    for j in range(len(H)):               #%loop over each hidden layer
        if j==0:
            V = Wh[j]@X               #%weighted sum of inputs [1] Eqn(4.29/30)
        else:
            V = Wh[j]@O[j-1]          #%weighted sum of hidden inputs [1] Eqn(4.29/31)
        
        PHI = a * np.tanh(b*V)     #%acivation function [1] Eqn(4.37)
        O.append( np.concatenate((bias*np.ones([1,N]),PHI),axis=0))   #%add zero'th order terms
    V = Wo@O[-1]            #%weighted sum of inputs [1] Eqn(4.29)
    Y = a * np.tanh(b*V)    #%activation function [1] Eqn(4.37)
    return Y

def gaussX(N,v):
    E1=np.sqrt(v)*np.random.randn(2,N//2)
    E2=np.sqrt(v)*np.random.randn(2,N//2)
    i=E1[0,:]*E1[1,:]
    E1=E1[:,i<0]
    l1=np.ones((1,E1.shape[1]))
    E1=np.vstack((E1,l1))
    i=E2[0,:]*E2[1,:]
    E2=E2[:,i>0]
    l2=-1*np.ones((1,E2.shape[1]))
    E2=np.vstack((E2,l2))
    E=np.hstack((E1,E2))
    return E

start_time=time.time()

#Data parameter definition
N=300
v=1
#NN parameters
H=[4,7,4]
eta=0.001
alpha=0.1
epochMax=2000
MSETarget=1e-12

numIter=20

MSE=np.zeros((epochMax,numIter))

#iterating the training process
for i in range(numIter):
    E=gaussX(N,v)
    # #shuffling
    # i=np.arange(N)
    # np.random.shuffle(i)
    # E=E[:,i]
    X=E[0:2,:]
    D=E[None,-1,:]
    NN=trainMLP(X,D,H,eta,alpha,epochMax,MSETarget)
    MSE[:,i,None]=NN[2]
#__________________________________________________
#P2_f

Wh=NN[0]
Wo=NN[1]

Y=MLP(X,Wh,Wo)
Y[Y>=0]=1
Y[Y<0]=-1

#indexing data according to lables
i1=np.squeeze(np.equal(Y+D,2*np.ones((Y.shape[0],1))))
i2=np.squeeze(np.equal(Y+D,-2*np.ones((Y.shape[0],1))))
i3=np.squeeze(np.equal(Y+D,np.zeros((Y.shape[0],1))))
#printing the grid
nn=500
nodes=np.linspace(-4,4,nn)
x1, x2 = np.meshgrid(nodes, nodes)
NodeTag=np.zeros((nn,nn))
crd=np.stack((x1,x2),axis=2)
crd=np.reshape(crd,(nn**2,2),order='C')
crd=crd.T
y1=MLP(crd,Wh,Wo)
y1=np.reshape(y1,(nn,nn))

print("--- %s seconds ---" % (time.time() - start_time))

fig1=plt.figure(figsize=[10,5])
plt.plot(np.arange(epochMax),np.mean(MSE,axis=1),c="b")
plt.xlabel('epoch',fontsize=20)
plt.ylabel('MSE',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid('True',linestyle='--', linewidth=1)
plt.show(fig1)
# fig1.savefig('p2_e.svg',format='svg')
#plotting
fig2=plt.figure(figsize=[10,10],dpi=300)
plt.contourf(x1,x2,y1,150,cmap='pink', linewidths=3,linestyles='solid')
plt.scatter(X[0,i1], X[1,i1], alpha=0.8,marker='+', c='b', edgecolors='none', s=75)
plt.scatter(X[0,i2], X[1,i2], alpha=0.8,marker='x', c='g', edgecolors='none', s=55)
plt.scatter(X[0,i3], X[1,i3], alpha=0.8,marker='*', c='r', edgecolors='none', s=95)
plt.axis([-4,4,-4,4])
plt.xlabel('$x_1$',fontsize=20)
plt.ylabel('$x_2$',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show(fig2)
# fig2.savefig('p2_f.svg',format='svg')
