#p4_b

for name in dir():
    del globals()[name]
    
import sympy as sym
import numpy as np
import matplotlib.pyplot as plt

w1=sym.Symbol('w1') # define symbols
w2=sym.Symbol('w2')


bowl = 'bowl'
case2= ''
case = bowl #case select
if(case==bowl):
    j=(w1**2+w1*w2+3*w2**2)# define equation
elif(case==case2):
    #define other surfaces here
    pass
else:
    print('case not recognized')
    

#compute gradient
j_grad1=sym.diff(j,w1)
j_grad2=sym.diff(j,w2)

wStar=np.array([0,0]) # ending point
lim=2000 # number of iterations
LR=np.linspace(0.002,0.32,70)# learning rates
Its=[]
Ew_thresh=1e-2
sigma=3
mu=0
NumIter=10

for i in np.arange(np.size(LR)):
    eta=LR[i]
    counts=0
    for k in np.arange(NumIter):
        w=sigma*np.random.randn(2)+mu # starting point
        count= 0
        while(True):
            #compute gradient matrix and hessian matrix
            g= np.array([float(j_grad1.subs({w1:w[0],w2:w[1]})),float(j_grad2.subs({w1:w[0],w2:w[1]}))])

            wnew =  w-eta*g 
            #loop check
            if( count>lim ):
    #             print('Count Break')
                break
            elif(np.isnan(g).any()):
    #             print('nan break')
                break
            elif(np.linalg.norm(w-wStar)<Ew_thresh):
    #             print('Threshold break')
                break
            else:
                count=count +1
                wprev=w.copy()

                w=wnew.copy()
        counts+=count
    Its.append(counts/NumIter)

print('The optimum learning rate is:')
print(LR[Its==np.min(Its)])
print('number of iterations:')
print(np.min(Its))

fig1=plt.figure(figsize=[10,5])
plt.plot(LR,Its,c="b")
plt.xlabel('Learning rate',fontsize=20)
plt.ylabel('Number of Iterations',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid('True',linestyle='--', linewidth=1)
plt.show(fig1)
# fig1.savefig('p4_b.svg',format='svg')
