#p9


for name in dir():
    del globals()[name]
    
import sympy as sym
import numpy as np
import matplotlib.pyplot as plt

HX ,HY  = 50,50 #number of x,y  points for countour
xmin,xmax = -6,6
ymin,ymax = -6,6
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
#compute hessian
hess11=sym.diff(j_grad1,w1)
hess12=sym.diff(j_grad1,w2)
hess21=sym.diff(j_grad2,w1)
hess22=sym.diff(j_grad2,w2)

#generate contour map
ConMap=np.zeros((HX,HY))
for i in range(HX):
    for k in range(HY):
        ConMap[i,k]=j.subs({w1:x1[i],w2:x2[k]})


sigma=1
mu=0
w=sigma*np.random.randn(2)+mu # starting point
wStar1=[3,2] # ending point
wStar2=[-2.8,3.13] # ending point
wStar3=[-3.78,-3.28] # ending point
wStar4=[3.58,-1.85] # ending point
ew=[]
jw=[]
eta = 0.9 # learning rate
lim=4000 # number of iterations
count= 0 
line=[]
Ew_thresh=1e-2
Lambda=30

while(True):
    #compute gradient matrix and hessian matrix
    g= np.array([float(j_grad1.subs({w1:w[0],w2:w[1]})),float(j_grad2.subs({w1:w[0],w2:w[1]}))])
    H= np.array([[float(hess11.subs({w1:w[0],w2:w[1]})),float(hess12.subs({w1:w[0],w2:w[1]}))],                 [float(hess21.subs({w1:w[0],w2:w[1]})),float(hess22.subs({w1:w[0],w2:w[1]}))]])
    
    wnew =  w-eta*(np.linalg.inv(H+Lambda*np.eye(H.shape[0], dtype=int))@g) 
    #loop check
    if( count>lim ):
        print('Count Break')
        break
    elif(np.isnan(g).any()):
        print('nan break')
        break
    elif(np.linalg.norm(w-wStar1)<Ew_thresh) or (np.linalg.norm(w-wStar2)<Ew_thresh) or (np.linalg.norm(w-wStar3)<Ew_thresh) or (np.linalg.norm(w-wStar4)<Ew_thresh):
        print('Threshold break')
        break
    else:
        count=count +1
        wprev=w.copy()
        
        line.append(w)
        w=wnew.copy()
        
#         ew.append(np.linalg.norm(w-wStar))
        jw.append(j.subs({w1:w[0],w2:w[1]}))
line=np.array(line)   

fig=plt.figure(figsize=(15,15))
plt.contourf(X1,X2,ConMap,50,cmap='viridis', linewidths=3,linestyles='solid')
plt.plot(line[:,1],line[:,0],color='y')
plt.scatter(line[:,1],line[:,0],color='y')
plt.plot(wStar1[1],wStar1[0],'go')
plt.plot(wStar2[1],wStar2[0],'go')
plt.plot(wStar3[1],wStar3[0],'go')
plt.plot(wStar4[1],wStar4[0],'go')
plt.xlabel('$x_1$',fontsize=25)
plt.ylabel('$x_2$',fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show(fig)
# fig.savefig('p9.svg',format='svg')
