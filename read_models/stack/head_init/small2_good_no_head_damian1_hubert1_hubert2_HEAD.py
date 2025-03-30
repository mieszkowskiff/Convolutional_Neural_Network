import torch

class MetaStackingHead(torch.nn.Module):
    def __init__(self):
        super(MetaStackingHead, self).__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(40, 512),
            #nn.BatchNorm1d(hidden_dims[0]),
            torch.nn.ReLU(inplace=True),
            #nn.Dropout(dropout),

            torch.nn.Linear(512, 512),
            #nn.BatchNorm1d(hidden_dims[1]),
            torch.nn.ReLU(inplace=True),
            #nn.Dropout(dropout),

            
            torch.nn.Linear(512, 256),
            #nn.BatchNorm1d(hidden_dims[1]),
            torch.nn.ReLU(inplace=True),
            #nn.Dropout(dropout),

            torch.nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)