import numpy as np
X = np.random.rand(1000,1)
Y = 4 + 3 * X + 0.5 * np.random.randn(1000, 1) # noise added


def grad(y, yhat, x):
    return (1 / 1000) * np.dot(x.transpose(), (yhat - y))



def batch_GD(epoches, x, y, eta):
    w = np.random.randn(2, 1)
    x = np.insert(x, 1, 1, axis=1)
    for epoch in range(epoches):
        yhat = np.dot(x, w)
        w = w - grad(y, yhat, x) * eta
        if np.linalg.norm(grad(y, np.dot(x, w), x)) / len(w) < 0.001:
            break
    
    print(f'Ket qua: {w.transpose()}')


batch_GD(50, X, Y, 1)