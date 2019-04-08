# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#import os
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

#其中x0,x1 : 是测试数据,y0,y1为对应的标签数据
# make fake data
n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1)      # class0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # class0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2*n_data, 1)     # class1 x data (tensor), shape=(100, 2)
y1 = torch.ones(100)                # class1 y data (tensor), shape=(100, 1)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)    # shape (200,) LongTensor = 64-bit integer

x, y = Variable(x), Variable(y)

#plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
#plt.show()



#build Neural Network 

class Net(torch.nn.Module): #继承torch的module
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__() #继承__init__工能
        #定义每层用什么样的形式
        self.hidden = torch.nn.Linear(n_feature,n_hidden) #隐藏层线性输出
        self.predict = torch.nn.Linear(n_hidden,n_output) #输出层线性输出
        
    def forward(self,x): #这同时也是Module中的forward功能
        #正向传播输入值，网络分析出输出值
        x = F.relu(self.hidden(x)) #Activation Function(隐藏层的线性值)
        x = self.predict(x) #输出值
        return x        
    
net = Net(n_feature=2,n_hidden=10,n_output=2)

print(net) #print net struct

#train Neural Network
optimizer = torch.optim.SGD(net.parameters(),lr=0.01) #input net parameters,such as learn rate
#loss_func = torch.nn.MSELoss() #预测值和真实值的误差计算公式(均方差)
loss_func = torch.nn.CrossEntropyLoss() 
      
plt.ion()

for t in range(100):

    out = net(x) #给net训练数据x,输出预测值
    loss = loss_func(out,y) #计算两者之间的误差
    
    optimizer.zero_grad() #清空上一步的残余更新值
    loss.backward() #误差反向传播，计算参数更新值
    optimizer.step() #将参数更新值加到net的parameters上

    # 接着上面来
    if t % 2 == 0:
         # plot and show learning process
        plt.cla()
        prediction = torch.max(out, 1)[1]
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.5)
        

plt.ioff()
plt.show()
