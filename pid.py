import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1=nn.Linear(3,50)
        self.fc2=nn.Linear(50,1)
    def forward(self,x):
        return(F.tanh(self.fc2(F.relu(self.fc1(x)))))

for i in range(10000):
    