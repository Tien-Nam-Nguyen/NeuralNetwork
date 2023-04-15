import numpy as np
import random

class NeuNet:
  def __init__(self, sizes):
    self.sizes = sizes
    self.num_layer = len(sizes)
    self.wbinit()

  def wbinit(self):
    self.biases = [np.random(next_dim, 1) for next_dim in self.sizes[1:]]
    self.weights = [np.random(x, y) for x,y in zip(self.sizes[1:], self.sizes[:-1])]


  def train(self, x_train, y_train, batch_size, epoch=30, lr=0.001):
    batches = [(x_train[i:i+batch_size], y_train[i:i+batch_size]) for i in range(0, len(x_train), batch_size)]
    for ep in range(epoch):
      for batch in batches:
        self.update_param(batch[0], batch[1], 0.001)



  def sigmoid(self, z):
    return 1.0 / (1 + np.exp(-z))


  def sigmoid_prime(self, z):
    return self.sigmoid(z) * (1 - self.sigmoid(z))


  def update_param(self, batch_x, batch_y, lr):
    zero_w = [np.zeros_like(w) for w in self.weights]
    zero_b = [np.zeros_like(b) for b in self.biases]
    for x, y in zip(batch_x, batch_y):
      delta_w, delta_b = self.back_propagation(x, y)
      zero_w = [zw + dw for zw, dw in zip(zero_w, delta_w)]
      zero_b = [zb + db for zb, db in zip(zero_b, delta_b)]
    
    self.weights = [w - lr * zw / len(batch_x) for w, zw in zip(self.weights, zero_w)]
    self.biases = [b - lr * zb / len(batch_x) for b, zb in zip(self.biases, zero_b)]

    



  def back_propagation(self, x, y):
    list_a = [x]
    nw = []
    nb = []
    zs = []
    for w, b in zip(self.weights, self.biases):
      res = np.dot(w, list_a[-1]) + b
      zs.append(res)
      res = self.sigmoid(res)
      list_a.append(res)
    

    delta = self.delta_output(y, list_a[-1])
    nw.append(np.dot(delta, list_a[-2].transpose()))
    for i in range(2, self.num_layer):
      sp = self.sigmoid_prime(z[-i])
      delta = np.dot(self.weights[-i+1].transpose(), delta) * sp
      nb.insert(0, delta)
      res = np.dot(delta, list_a[-i-1].transpose())
      nw.insert(0, res)
    
    return nw, nb


  
  def delta_output(self, y, a):    
    return (y - a) 

  def feed_forward(self, x):
    a = x
    for w, b in zip(self.weights, self.biases):
      a = self.sigmoid(np.dot(w, a) + b)
    return a

  def evaluate(self, test_data):
        res = [(np.argmax(self.feed_forward(x)), y) for x, y in test_data]
        return sum(int(x == y) for x, y in res)
   

