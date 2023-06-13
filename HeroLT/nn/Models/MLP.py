from torch import nn

class MLP(nn.Module):
    def __init__(self, nhid, nclass):
        super(MLP, self).__init__()
        self.mlp = nn.Linear(nhid, nclass)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight,std=0.05)

    def forward(self, x):
        x = self.mlp(x)

        return x