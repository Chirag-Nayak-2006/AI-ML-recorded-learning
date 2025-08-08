import numpy as np

x = np.array([]) #input features
y = np.array([]) #corresponding output

#weights
w,b = 0,0
def f(num): #model
    global w,b
    return w*num + b

def j(inArr,outArr): #cost function
    return 1/(2*len(inArr)) * sum((f(inArr[i])-outArr[i])**2 for i in range(len(inArr)))

def gradient_descent(inArr, outArr, it, alpha):
    global w,b
    for j in range(it):
        djdw = 1/(len(inArr)) * sum((f(inArr[i])-outArr[i])*inArr[i] for i in range(len(inArr)))
        djdb = 1/(len(inArr)) * sum((f(inArr[i])-outArr[i]) for i in range(len(inArr)))
        w,b = w - alpha*djdw, b - alpha*djdb
    
    