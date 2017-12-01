from torch import nn


class DNNModel(nn.ModuleList):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DNNModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)
        self.nonlinear = nn.ReLU()

    def forward(self, x):
        out = self.nonlinear(self.linear1(x))
        out = self.nonlinear(self.linear2(out))
        out = self.linear3(out)
        return out
