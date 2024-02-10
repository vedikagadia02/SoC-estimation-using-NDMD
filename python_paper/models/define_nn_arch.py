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
            nn.ReLU(inplace=True),
            nn.Linear(30,30),
            nn.ReLU(inplace=True),
            nn.Linear(30,obsdim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(obsdim, 30),
            nn.ReLU(inplace=True),
            nn.Linear(30,30),
            nn.ReLU(inplace=True),
            nn.Linear(30,indim)
        )   

        # self.kMatrix = nn.Parameter(torch.rand(obsdim, obsdim))
        # trying out Lusch paper, where koopman matrix is considered to have
        # Jordan form, ie, it is a skew symmetric matrix
        self.obsdim = obsdim
        self.kMatrixDiag = nn.Parameter(torch.rand(obsdim))
        self.kMatrixUT = nn.Parameter(0.01*torch.randn(int(obsdim*(obsdim-1)/2)))

        # self.inputMatrix = nn.Parameter(torch.rand(obsdim, obsdim))        
        self.encoder.apply(self.init_nn_weights)
        self.decoder.apply(self.init_nn_weights)
        # init.uniform_(self.kMatrix, -1, 0)
        # init.normal_(self.inputMatrix, mean = 0, std = 0.2)
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

        kMatrix = Variable(torch.Tensor(self.obsdim, self.obsdim)).to(self.kMatrixUT.device)
        utIdx = torch.triu_indices(self.obsdim, self.obsdim, offset=1)
        diagIdx = torch.stack([torch.arange(0,self.obsdim,dtype=torch.long).unsqueeze(0), \
            torch.arange(0,self.obsdim,dtype=torch.long).unsqueeze(0)], dim=0)
        kMatrix[utIdx[0], utIdx[1]] = self.kMatrixUT
        kMatrix[utIdx[1], utIdx[0]] = -self.kMatrixUT
        kMatrix[diagIdx[0], diagIdx[1]] = torch.nn.functional.relu(self.kMatrixDiag)

        gnext = torch.bmm(g.unsqueeze(1), kMatrix.expand(g.size(0), kMatrix.size(0), kMatrix.size(0)))
        for i in range(s):
            if i == 1 :
                continue
            else :
                gnext = torch.bmm(gnext, kMatrix.expand(g.size(0), kMatrix.size(0), kMatrix.size(0)))
            # print(i, "debug koopman")
            
        return gnext.squeeze(1)
    
    def getKoopmanMatrix(self, requires_grad = False):
        # return self.kMatrix

        kMatrix = Variable(torch.Tensor(self.obsdim, self.obsdim), requires_grad=requires_grad).to(self.kMatrixUT.device)

        utIdx = torch.triu_indices(self.obsdim, self.obsdim, offset=1)
        diagIdx = torch.stack([torch.arange(0,self.obsdim,dtype=torch.long).unsqueeze(0), \
            torch.arange(0,self.obsdim,dtype=torch.long).unsqueeze(0)], dim=0)
        kMatrix[utIdx[0], utIdx[1]] = self.kMatrixUT
        kMatrix[utIdx[1], utIdx[0]] = -self.kMatrixUT
        kMatrix[diagIdx[0], diagIdx[1]] = torch.nn.functional.relu(self.kMatrixDiag)

        return kMatrix
    