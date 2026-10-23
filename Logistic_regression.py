#Logistic regression
import math as m
import pandas as pd
import numpy as np
from pathlib import Path
file_path = Path(__file__).resolve().parent/ "train.csv"
df= pd.read_csv(file_path)
df["Sex"]= df["Sex"].replace({"male": 0,"female": 1})
features = ['Pclass', 'Sex', 'Age', 'Fare']
x = df[features].fillna(df[features].mean()).to_numpy(dtype=float)
y= df['Survived'].to_numpy(dtype=float)
x_train= x[:600]
y_train= y[:600]
x_valid= x[600:]
y_valid= y[600:]
def sigmoid(z):
   return 1/(1+np.exp(-z))
def Z(x,w,b):
  return x@w + b
def loss_func(x,y,w,b):
   z= Z(x,w,b)
   p= sigmoid(z)
   I = np.ones(len(y))
   return -(y@np.log(p.T)+ (I-y)@np.log((I-p).T))/len(y)
def gradient_descent(x,y,w,b,alpha, iterations):
    i=0
    dw=0
    db=0
    while( i<iterations):
        predictions= sigmoid(Z(x,w,b))
        errors= predictions-y
        dw= (x.T@errors)*(1/len(y))
        db= (np.sum(errors))/(len(y))
        w= w-alpha*dw
        b= b-alpha*db
        i+=1

    return w,b
   
   

w= np.array([0,0,0,0])
b= 0
alpha = 0.0001
iterations= 100000
print("Inital average loss function value:",loss_func(x_train, y_train, w, b))
w,b = gradient_descent(x_train, y_train, w, b, alpha, iterations)
print("Final weight(Pclass, Sex(0 for male and 1 for female), Age, Fare):", w)
print("Final bias:", b)
print("Final cost:", loss_func(x_train, y_train, w, b))

# Validation set
y_pred = sigmoid(Z(x_valid,w,b))
loss = loss_func(x_valid, y_valid, w, b)
print("Average Loss calculated on validation set:",loss)

