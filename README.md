# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm
1. Import the required libraries. Define a function for costFunction,cost and gradient.
2. Load the dataset.
3. Define X and Y array.
4. Plot the decision boundary and Predict the Regression value.
## Program:
```/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by  : SANJAY S
RegisterNumber: 212221243002
*/
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]
X[:5]
y[:5]
plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()
def sigmoid(z):
    return 1/(1+np.exp(-z))
plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()
def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)
def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J
def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)
def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()
plotDecisionBoundary(res.x,X,y)
prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)
def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)
```
## Output:

![ml5op1](https://user-images.githubusercontent.com/115128955/201039373-0cacd51a-bb75-4e4c-9d31-a65c144f5b2d.png)

![ml5op2](https://user-images.githubusercontent.com/115128955/201039414-8b2c1dfc-46a7-431f-adbd-4dfcd48ae068.png)

![ml5op3](https://user-images.githubusercontent.com/115128955/201039476-03232ab3-c583-42b8-ad09-fdd498d5f461.png)

![ml5op4](https://user-images.githubusercontent.com/115128955/201039518-5ee4e151-595d-48d3-9ad6-36506dc6fb93.png)

![ml5op5](https://user-images.githubusercontent.com/115128955/201039558-553b678e-b20f-40f5-85c9-47b3c7dde1f1.png)

![ml5op6](https://user-images.githubusercontent.com/115128955/201039583-8b3f2ff3-ddf9-4505-ac43-cfb73401cf5b.png)

![ml5op7](https://user-images.githubusercontent.com/115128955/201039691-27713d29-8fc8-469e-9363-84aa90b95e97.png)

![ml5op8](https://user-images.githubusercontent.com/115128955/201039887-9a52a8c2-3143-4bc3-ab33-9d685e790677.png)

![ml5op9](https://user-images.githubusercontent.com/115128955/201039964-2c2e6181-b49b-4c02-a7a8-3e15781ea34d.png)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

