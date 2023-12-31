from torch import nn
import torch

class DDQN(nn.Module):
    """
    mini CNN structure
    input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim
        self.online = self.__build_cnn(3, output_dim)

        self.target = self.__build_cnn(3, output_dim)
        self.target.load_state_dict(self.online.state_dict())

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)

    def __build_cnn(self, c, output_dim):

        return nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=16, kernel_size=4, stride=2, dtype=torch.float64),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, dtype=torch.float64),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, dtype=torch.float64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, dtype=torch.float64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7168, 512, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(512, output_dim, dtype=torch.float64),
        )