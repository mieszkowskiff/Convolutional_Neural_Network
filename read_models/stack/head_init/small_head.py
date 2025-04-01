import torch.nn as nn


class MetaStackingHead(nn.Module):
    def __init__(self):
        super(MetaStackingHead, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(40, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.2),

            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)