# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#import os
import torch
from torch.autograd import Variable #torch Variable module
import torch.nn.functional as F
#import torchvision
import numpy as np
import matplotlib.pyplot as plt

#generate some data for test
x = torch.linspace(-5,5,200)
x = Variable(x)

x_np = x.data.numpy() #convert to numpy array, for showing

#some Activation Function
y_relu      = F.relu(x).data.numpy()
y_sigmoid   = F.sigmoid(x).data.numpy()
y_tanh      = F.tanh(x).data.numpy()
y_softplus  = F.softplus(x).data.numpy()

 
plt.figure(1,figsize=(8,6))

plt.subplot(2,2,1)
plt.plot(x_np,y_relu,c='red',label='relu')
plt.ylim(-1,5)
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x_np,y_sigmoid,c='red',label='sigmoid')
plt.ylim(-0.2,1.2)
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x_np,y_tanh,c='red',label='tanh')
plt.ylim(-1.2,1.2)
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x_np,y_softplus,c='red',label='softplus')
plt.ylim(-0.2,6.0)
plt.legend(loc='best')

plt.show()