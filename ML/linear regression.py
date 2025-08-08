import numpy as np
    
class LinearRegressionSV():
    def __init__(self): #weights
        self.w = 0
        self.b = 0
        
    def f(self, X): #model
        return self.w*X + self.b
    
    def cost_fuction(self, X, Y): #cost 
        return np.mean((self.f(X) - Y)**2) / 2
    
    def gradient_descent(self, X, Y, it, alpha): #fitting the model to the data
        m = len(X)
        for _ in range(it):
            error = np.mean(self.f(X) - Y)
            djdw = np.dot(error, X) / m
            djdb = np.mean(error)
            self.w, self.b  = self.w - alpha*djdw, self.b - alpha*djdb