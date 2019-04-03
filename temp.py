# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#import os
import torch
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1) # x data(tensor),shape=(100,1)
y = x.pow(2) + 0.2 * torch.rand(x.size())

#draw picture
plt.scatter(x.data.numpy(),y.data.numpy())

plt.show()