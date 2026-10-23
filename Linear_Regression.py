import pandas as pd
import numpy as np
from pathlib import Path
file_path = Path(__file__).resolve().parent/ "train.csv"
df= pd.read_csv(file_path)
features = ["GrLivArea","OverallQual", "GarageArea", "TotalBsmtSF"]

x = df[features].fillna(df[features].mean())
y = df["SalePrice"]
x = x.to_numpy(dtype=float)
y = y.to_numpy(dtype=float)
x_train= x[:1000]
y_train= y[:1000]
x_valid= x[1000:]
y_valid= y[1000:]
# y = wx+b  let initially w=1.0, b= 0.0
def predict(x, w, b):
    return x@w + b
# Cost Function
def cost_func(x, y, w, b):
    sum=0
    predictions = predict(x, w, b)
    for i in range(0,len(y)):
       sum+= ((y[i]-predictions[i])**2)/(2*len(y))
    return sum
def gradient_descent(x,y,w,b,alpha, iterations):
    i=0
    dw=0
    db=0
    while( i<iterations):
        predictions= predict(x,w,b)
        errors= predictions-y
        dw= (x.T@errors)/(len(y))
        db= (np.sum(errors))/(len(y))
        w= w-alpha*dw
        b= b-alpha*db
        i+=1

    return w,b

        
w= np.array([1,1,1,1])
b=0
alpha = 1e-7
iterations= 100000
print("Inital cost function value:",cost_func(x_train, y_train, w, b))
w,b = gradient_descent(x_train, y_train, w, b, alpha, iterations)
print("Final weights (GrLivArea, OverallQual, GarageArea, TotalBsmtSF ):", w)
print("Final bias:", b)
print("Final cost:", cost_func(x_train, y_train, w, b))

# Validation set
y_pred = predict(x_valid,w,b)
rmse = np.sqrt(np.mean((y_pred-y_valid)**2))
print("Validation RMSE:",rmse)

        
 
