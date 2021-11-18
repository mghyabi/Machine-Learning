#p6_c

for name in dir():
    del globals()[name]
    
import sympy as sym
import numpy as np
import matplotlib.pyplot as plt

HX ,HY  = 50,50 #number of x,y  points for countour
xmin,xmax = -15,15
ymin,ymax = -12,12
x1 = np.linspace(xmin,xmax,HX)
x2 = np.linspace(ymin,ymax,HY)
X1,X2 = np.meshgrid(x1,x2) # genertate mesh grid
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
lim=2000 # number of iterations
LR=np.linspace(0.045,0.9,50)# learning rates
# LR=np.array([0.02,0.03,0.1])
Its=[]
Ew_thresh=1e-2
sigma=1
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
#                 print('Count Break')
                break
            elif(np.isnan(g).any()):
#                 print('nan break')
                break
            elif(np.linalg.norm(w-wStar1)<Ew_thresh) or (np.linalg.norm(w-wStar2)<Ew_thresh) or (np.linalg.norm(w-wStar3)<Ew_thresh) or (np.linalg.norm(w-wStar4)<Ew_thresh):
#                 print('Threshold break')
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
# fig1.savefig('p6_c.svg',format='svg')
