import torch
import torch.nn as nn
import torch.nn. functional as F



#Sequential model
class NN(nn.Module):
    def __init__(self, dropout) :
        super(NN, self).__init__()
        self.fc1 = nn. Linear (512, 256)
        self.d1= nn. Dropout (p=dropout)
        self. fc2 = nn. Linear (256, 128)
        self.d2= nn. Dropout (p=dropout)
        self. fc3 = nn. Linear (128, 2)

        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')


    def forward (self, x):
        x = F.gelu(self.fc1(x) ) 
        x = self.d2(F.gelu(self. fc2(x) ) )
        x= self. fc3(x)
        return x