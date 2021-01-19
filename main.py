#with numpy implementations:
import numpy as np

#f = w * x + b
#f = 2 * x - 3, meaning that we want w to be 5 and b to be -3.
'''Esentially I am giving the program an input X, and a desired output Y. Based of X and Y, it will perform algorithms to learn the difference between X and Y to learn the weight w'''
X = np.array([1.0, 2.0, 3.0, 4.0], dtype = np.float32)
Y = np.array([-1.0, 0.9, 3.1, 4.95], dtype = np.float32)
#So this is our expected output. Based of these inputs it should know that w = 0.5
w = 0.0
b = 0.0

#model prediction
def forward(x):
  return (w * x) + b

#loss function = MSE
def loss(y, y_predicted):
  return ((y_predicted-y)**2).mean() #in linear regression, the loss is this. 

#gradient 
#MSE = 1/N * (w*x - y)**2
#dJ/dw = 1/N * 2x(w*x - y)
#dJ/db = 1/N * 2(w*y - y)
def gradientofW(x, y, y_predicted):
  return np.dot(2*x, y_predicted - y).mean() #what is the significance of the 2?

def gradientofB(x, y, y_predicted):
  return np.dot(2, y_predicted - y).mean()


userInput = int(input("Enter the value you want to find: "))
#print(userInput)
#print(type(userInput))

print(f'Prediction before training: f({userInput}) = {forward(userInput):.3f}')
print()

#Training 
learning_rate = 0.01
n_iters = 1000

for epoch in range(n_iters):
  #prediction = forward pass
  y_pred = forward(X)

  #loss
  l = loss(Y, y_pred)

  #gradients
  dw = gradientofW(X, Y, y_pred)
  db = gradientofB(X, Y, y_pred)

  #update weights. Could also use w = w - learning_rate * dw
  w -= learning_rate * dw
  b -= learning_rate * db

  if epoch % 10 == 0:
    print(f'epoch {epoch+1}: w = {w:.3f}, b = {b:.3f}, loss = {l:.8f}')

print(f'Prediction after training f({userInput}) = {forward(userInput):.3f}')

''''
#with torch implementations: Need package in order to work.
import torch

#f = w * x. Once again this program finds parameter w

#input:
A = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype = torch.float32)
#output
B = torch.tensor([2.0, 4.0, 6.0, 8.0], dytpe = torch.float32)

w = torch.tensor(0.0, dtype = torch.float32, requires_grad = True)

def forward(A):
  return w * A

def loss(B, B_predicted):
  return ((B_predicted-B)**2).mean() #means just find one value

#Training 
learning_rate = 0.01
n_iters = 10

print(f'Prediction after training f(5) = {forward(5):.3f}')

for epoch in range(n_iters):
  #prediction = forward pass
  B_pred = forward(A)

  #loss
  l = loss(B, B_pred)

  #gradients dl/dw
  l.backward()

  with torch.no_grad():
    w -= learning_rate * w

  if epoch % 2 == 0:
    print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training f(5) = {forward(5):.3f}')'''
