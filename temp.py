# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1) # x data(tensor),shape=(100,1)
y = x.pow(2) + 0.2 * torch.rand(x.size())

#draw picture
#plt.scatter(x.data.numpy(),y.data.numpy())
#
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
    
net = Net(n_feature=1,n_hidden=10,n_output=1)

print(net) #print net struct

#train Neural Network
optimizer = torch.optim.SGD(net.parameters(),lr=0.2) #input net parameters,such as learn rate
loss_func = torch.nn.MSELoss() #预测值和真实值的误差计算公式(均方差)
      
plt.ion()

for t in range(200):

    prediction = net(x) #给net训练数据x,输出预测值
    loss = loss_func(prediction,y) #计算两者之间的误差
    
    optimizer.zero_grad() #清空上一步的残余更新值
    loss.backward() #误差反向传播，计算参数更新值
    optimizer.step() #将参数更新值加到net的parameters上

    # 接着上面来
    if t % 2 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)
        

plt.ioff()
plt.show()
