from torch import nn

KERNEL_SIZE_UNITS = 3
STRIDE_UNITS = 1
PADDING_UNITS = 1


# Convolutional Neural Network
class FashionMNISTModel(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=KERNEL_SIZE_UNITS,
                      stride=STRIDE_UNITS,
                      padding=PADDING_UNITS
                      ),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=KERNEL_SIZE_UNITS,
                      stride=STRIDE_UNITS,
                      padding=1
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=KERNEL_SIZE_UNITS,
                      stride=STRIDE_UNITS,
                      padding=PADDING_UNITS
                      ),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=KERNEL_SIZE_UNITS,
                      stride=STRIDE_UNITS,
                      padding=PADDING_UNITS
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7, out_features=output_shape)
        )
        
    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return x