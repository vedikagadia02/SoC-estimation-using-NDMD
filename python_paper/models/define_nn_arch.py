import torch
import torch.nn as nn
from torch.autograd import Variable

class KoopmanNetwork(nn.Module):

    def __init__(self, indim, obsdim):
        super(KoopmanNetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(indim, 3),
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
            nn.Linear(3, indim)
        )   

        self.kMatrix = nn.Parameter(torch.eye(obsdim, obsdim))
        print('Total number of parameters: {}'.format(self._num_parameters()))

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

    def koopmanOperation(self, g):
        gnext = torch.mm(g, self.kMatrix)

        return gnext
    
    def getKoopmanMatrix(self):
        return self.kMatrix