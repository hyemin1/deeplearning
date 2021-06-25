import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride = stride)
        self.batch_norm_1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride = 1)
        self.batch_norm_2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride= 1)
        self.batch_norm_3 = nn.BatchNorm2d(out_channels)

    def forward(self, input_tensor):
        output_tensor = self.conv1(input_tensor)
        output_tensor = self.batch_norm_1(output_tensor)
        output_tensor = self.relu1(output_tensor)

        output_tensor = self.conv2(output_tensor)
        output_tensor = self.batch_norm_2(output_tensor)
        output_tensor = self.relu2(output_tensor)

        skip_connection = self.conv3(input_tensor)
        skip_connection = self.batch_norm_3(skip_connection)

        output_tensor = output_tensor + skip_connection
        return output_tensor

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.layers = nn.Sequential(nn.Conv2d(3,64,7,2),
                              nn.BatchNorm2d(),
                              nn.ReLU(),
                              nn.MaxPool2d(3,2),
                              ResBlock(64,64,1),
                              ResBlock(64,128,2),
                              ResBlock(128, 256, 2),
                              ResBlock(256, 512, 2),
                              nn.AvgPool2d(),
                              nn.Flatten(),
                              nn.Linear(512,2),
                              nn.Sigmoid())

    def forward(self, input_tensor):
        output_tensor = self.layers(input_tensor)
        return  output_tensor

