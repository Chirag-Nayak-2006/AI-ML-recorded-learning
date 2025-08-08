import numpy as np
    
class LinearRegressionMV():
    def __init__(self): #weights
        self.w = None
        self.b = 0
        
    def f(self, X): #model
        return np.dot(self.w,X) + self.b
    
    def cost_fuction(self, x, Y): #cost 
        return np.mean((self.f(x) - Y)**2) / 2
    
    def gradient_descent(self, X, Y, it, alpha): #fitting the model to the data
        m,n = X.shape
        if self.w is None:
            self.w = np.zeros(n)
            
        for _ in range(it):
            error = self.f(X) - Y
            djdw = np.dot(X.T, error) / m
            djdb = np.mean(error)
            self.w -= alpha*djdw
            self.b -= alpha*djdb