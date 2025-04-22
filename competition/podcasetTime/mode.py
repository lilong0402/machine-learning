import torch.nn as nn
class model(nn.Module):
    def __init__(self, input_size, output_size):
        super(model, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_size),

        )
    def forward(self, x):
        return self.net(x)
