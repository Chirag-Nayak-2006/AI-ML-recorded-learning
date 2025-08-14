import numpy as np

class LogisticRegressionMV():
    def __init__(self, mask = None):
        self.w = None
        self.b = 0
        self.mask = mask
    def sigmoid(self,z): # model
        return 1/(1+np.exp(-z))
    def gradient_descent(self, X, y, itr, alpha, lambda_ = 0.0):
        m, n = X.shape
        
        if self.w is None:
            self.w = np.zeros(n)
        if self.mask is None:
            self.mask = np.ones(n)
                    
        for _ in range(itr):
            z = np.dot(X, self.w) + self.b
            f = self.sigmoid(z)
            error = f - y
            djdw = (np.dot(X.T, error) / m) + (lambda_/ m)*(self.w*self.mask)
            djdb = np.mean(error)
            self.w -= alpha*djdw
            self.b -= alpha*djdb
        
    