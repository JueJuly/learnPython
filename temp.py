# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import torch
from torch.autograd import Variable #torch Variable module
#import torchvision
import numpy as np

# 先生鸡蛋
tensor = torch.FloatTensor([[1,2],[3,4]])
# 把鸡蛋放到篮子里, requires_grad是参不参与误差反向传播, 要不要计算梯度
variable = Variable(tensor, requires_grad=True)

t_out = torch.mean(tensor*tensor)
v_out = torch.mean(variable*variable)

print(t_out)

print(v_out)
