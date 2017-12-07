from torch import nn


class DNNModel(nn.ModuleList):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DNNModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.linear4 = nn.Linear(hidden_dim // 2, hidden_dim // 8)
        self.linear5 = nn.Linear(hidden_dim // 8, output_dim)
        self.nonlinear = nn.ReLU()

    def forward(self, x):
        out = self.nonlinear(self.linear1(x))
        out = self.nonlinear(self.linear2(out))
        out = self.nonlinear(self.linear3(out))
        out = self.nonlinear(self.linear4(out))
        out = self.linear5(out)
        return out
