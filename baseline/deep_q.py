from torch import nn
import torch

class DDQN(nn.Module):
    """
    Implements a Double Deep Q-Network (DDQN) using Convolutional Neural Networks (CNNs)
    
    input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
    The network architecture consists of three convolutional layers followed by two fully connected layers.

    It maintains two networks: an online network for selecting actions and a target network for stability
    
    Args:
    - input_dim: Dimensions of the input image (channels, height, width)
    - output_dim: Number of actions (size of the output layer)
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Unpack input dimensions
        c, h, w = input_dim

        # Validate input dimensions to match expected size
        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")
        
        # Initialize the online network
        self.online = self.__build_cnn(c, output_dim)

        # Initialize the target network and copy weights from the online network
        self.target = self.__build_cnn(c, output_dim)
        self.target.load_state_dict(self.online.state_dict())

        # Freeze parameters in the target network to prevent them from being updated
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        """
        Defines the forward pass of the DDQN.
        
        Args:
        - input: The input state
        - model: Specifies the network to use ('online' or 'target')
        
        Returns:
        - The output from the specified network
        """
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)

    def __build_cnn(self, c, output_dim):
        """
        Builds the CNN architecture
        
        Args:
        - c: Number of input channels
        - output_dim: The size of the output layer (number of actions)
        
        Returns:
        - A sequential model consisting of convolutional and fully connected layers
        """
        # Define the CNN architecture
        return nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4, dtype=torch.float32),
            nn.BatchNorm2d(32), # Mitigate the problem of internal covariate shift
            nn.LeakyReLU(), # Avoid the issue of "dead neurons"
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, dtype=torch.float32),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, dtype=torch.float32),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Flatten(), # Flatten the output for the fully connected layers
            nn.Linear(3136, 512, dtype=torch.float32), # 3136 = number of features from the last conv layer
            nn.LeakyReLU(),
            nn.Linear(512, output_dim, dtype=torch.float32) # Output layer
        )