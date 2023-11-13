import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init
from torch.nn.parameter import Parameter

class KoopmanNetwork(nn.Module):

    def __init__(self, indim, obsdim):
        super(KoopmanNetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(indim, 30),
            nn.ReLU(),
            nn.Linear(30,3)
            # nn.ELU(inplace=True),
            # nn.Linear(15, 15),
            # nn.ELU(inplace=True),
            # nn.Linear(3, obsdim)
        )

        self.decoder = nn.Sequential(
            # nn.Linear(obsdim, 3),
            # nn.ELU(inplace=True),
            # nn.Linear(3, 3),
            # nn.ELU(inplace=True),
            nn.Linear(obsdim, 30),
            nn.ReLU(),
            nn.Linear(30,indim)
        )   

        self.kMatrix = nn.Parameter(torch.rand(obsdim, obsdim))
        self.inputMatrix = nn.Parameter(torch.rand(obsdim, obsdim))        
        self.encoder.apply(self.init_nn_weights)
        self.decoder.apply(self.init_nn_weights)
        init.uniform_(self.kMatrix, -1, 0)
        init.normal_(self.inputMatrix, mean = 0, std = 0.2)
        print('Total number of parameters: {}'.format(self._num_parameters()))

    def init_nn_weights(self, m):
        if(isinstance(m, nn.Linear)):
            init.normal_(m.weight.data, mean=0, std=(2/m.in_features))
            init.zeros_(m.bias.data)

    def forward(self, x):
        g = self.encoder(x)
        x0 = self.decoder(g)

        return g, x0

    def recover(self, g):
        x0 = self.decoder(g)
        return x0
    
    def _num_parameters(self):
        count = 0
        for name, param in self.named_parameters():
            # print(name, param.numel())
            count += param.numel()
        return count

    def koopmanOperation(self, g, s):
        gnext = g
        for i in range(s):
            gnext = torch.mm(gnext, self.kMatrix)
        # gnext = torch.mm(g, self.kMatrix) trying new settings

        return gnext
    
    def getKoopmanMatrix(self):
        return self.kMatrix