import torch

class MetaStackingHead(torch.nn.Module):
    def __init__(self, input_dim=30, hidden_dims=[128, 64], num_classes=10, dropout=0.3):
        super(MetaStackingHead, self).__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dims[0]),
            #nn.BatchNorm1d(hidden_dims[0]),
            torch.nn.ReLU(inplace=True),
            #nn.Dropout(dropout),

            torch.nn.Linear(hidden_dims[0], hidden_dims[1]),
            #nn.BatchNorm1d(hidden_dims[1]),
            torch.nn.ReLU(inplace=True),
            #nn.Dropout(dropout),

            torch.nn.Linear(hidden_dims[1], num_classes)
        )

    def forward(self, x):
        return self.net(x)