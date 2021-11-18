# p5_a

for name in dir():
    del globals()[name]
    
import sympy as sym
import numpy as np
import matplotlib.pyplot as plt

HX ,HY  = 50,50 #number of x,y  points for countour
xmin,xmax = -1.2,1.2
ymin,ymax = -1.2,1.2
x1 = np.linspace(xmin,xmax,HX)
x2 = np.linspace(ymin,ymax,HY)
X1,X2 = np.meshgrid(x1,x2) # genertate mesh grid
w1=sym.Symbol('w1') # define symbols
w2=sym.Symbol('w2')


bowl = 'bowl'
Rosenbrock= 'Rosenbrock'
case = Rosenbrock #case select
if(case==bowl):
    j=(w1**2+w1*w2+3*w2**2)# define equation
elif(case==Rosenbrock):
    j=(1-w1)**2+100*(w2-w1**2)**2# define equation
    pass
else:
    print('case not recognized')
    

#compute gradient
j_grad1=sym.diff(j,w1)
j_grad2=sym.diff(j,w2)

#generate contour map
ConMap=np.zeros((HX,HY))
for i in range(HX):
    for k in range(HY):
        ConMap[i,k]=j.subs({w1:x1[i],w2:x2[k]})


w=np.array([-0.5,1]) # starting point
wStar=[1,1] # ending point
ew=[]
jw=[]
eta = 0.002 # learning rate
lim=2000 # number of iterations
count= 0 
line=[]
Ew_thresh=1e-2 #define threshold

while(True):
    #compute gradient matrix and hessian matrix
    g= np.array([float(j_grad1.subs({w1:w[0],w2:w[1]})),float(j_grad2.subs({w1:w[0],w2:w[1]}))])
    
    wnew =  w-eta*g 
    #loop check
    if( count>lim ):
        print('Count Break')
        break
    elif(np.isnan(g).any()):
        print('nan break')
        break
    elif(np.linalg.norm(w-wStar)<Ew_thresh):
        print('threshold break')
        break
    else:
        count=count +1
        wprev=w.copy()
        
        line.append(w)
        w=wnew.copy()
        
        ew.append(np.linalg.norm(w-wStar))
        jw.append(j.subs({w1:w[0],w2:w[1]}))
line=np.array(line)   

fig=plt.figure(figsize=(15,10))
plt.contourf(X1,X2,ConMap,50,cmap='viridis', linewidths=3,linestyles='solid')
plt.plot(line[:,1],line[:,0],color='y')
plt.scatter(line[:,1],line[:,0],color='y')
plt.plot(wStar[1],wStar[0],'go')
plt.xlabel('$x_1$',fontsize=25)
plt.ylabel('$x_2$',fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show(fig)
# fig.savefig('p5_a2.svg',format='svg')
