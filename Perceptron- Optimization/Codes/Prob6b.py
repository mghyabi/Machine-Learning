#p6_b

for name in dir():
    del globals()[name]
    
import sympy as sym
import numpy as np
import matplotlib.pyplot as plt

w1=sym.Symbol('w1') # define symbols
w2=sym.Symbol('w2')

bowl = 'bowl'
Rosenbrock= 'Rosenbrock'
Himmelblau= 'Himmelblau'
case = Himmelblau #case select
if(case==bowl):
    j=(w1**2+w1*w2+3*w2**2) # define equation
elif(case==Rosenbrock):
    j=(1-w1)**2+100*(w2-w1**2)**2 # define equation
    pass
elif(case==Himmelblau):
    j=(w1**2+w2-11)**2+(w1+w2**2-7)**2 # define equation
    pass
else:
    print('case not recognized')
    

#compute gradient
j_grad1=sym.diff(j,w1)
j_grad2=sym.diff(j,w2)

wStar1=[3,2] # ending point
wStar2=[-2.8,3.13] # ending point
wStar3=[-3.78,-3.28] # ending point
wStar4=[3.58,-1.85] # ending point
lim=100 # number of iterations
eta=0.01 # learning rates
sigma=1
mu=0
NumIter=30
J=np.zeros((NumIter,lim+1))


for k in np.arange(NumIter):
    count=0
#     w=np.array([5,-5]) # starting point
    w=sigma*np.random.randn(2)+mu # starting point
    jw=[]
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
        else:
            count=count +1
            wprev=w.copy()

            w=wnew.copy()
            jw.append(j.subs({w1:w[0],w2:w[1]}))
    J[k,:]=jw
J=np.mean(J,axis=0)

fig1=plt.figure(figsize=[10,5])
plt.plot(np.arange(lim+1),J,c="b")
plt.xlabel('Iteration',fontsize=20)
plt.ylabel('Aveage J',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid('True',linestyle='--', linewidth=1)
plt.show(fig1)
# fig1.savefig('p6_b.svg',format='svg')
