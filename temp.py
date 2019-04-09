# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#import os
import torch
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

#print(torch.__version__)

torch.manual_seed(1)    # reproducible

BATCH_SIZE = 8 #批训练的数据个数

x = torch.linspace(1,10,10) #x data (torch data)
y = torch.linspace(10,1,10) #y data (torch data)

#print('x : ', x)
#print('y : ', y)


#先转换成torch能识别的Dataset
torch_dataset = Data.TensorDataset(x,y)

#print(torch_dataset)

#把dataset放入DataLoader
loader = Data.DataLoader(
        dataset = torch_dataset, #torch TensorDataset format
        batch_size = BATCH_SIZE, #mini batch size
        shuffle = True,          #要不要打乱数据(打乱比较好)
        num_workers = 2,         #多线程来读数据
        )

def show_batch():
    for epoch in range(3):   # train entire dataset 3 times
        for step, (batch_x, batch_y) in enumerate(loader):  # for each training step
            # train your data...
            print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
                  batch_x.numpy(), '| batch y: ', batch_y.numpy())


if __name__ == '__main__':
    show_batch()