import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResBlock, self).__init__()
        self.stride = stride
        self.input_channels = in_channels
        self.output_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels= out_channels, kernel_size=3, stride = stride, bias=False, padding=1)
        self.batch_norm_1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels = in_channels, out_channels= out_channels, kernel_size=3, stride = 1, bias= False, padding=1)
        self.batch_norm_2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size=1, stride= stride, bias= False, padding=1)
        self.batch_norm_3 = nn.BatchNorm2d(out_channels)

    def forward(self, input_tensor):
        output_tensor = self.conv1(input_tensor)
        output_tensor = self.batch_norm_1(output_tensor)
        output_tensor = self.relu1(output_tensor)

        output_tensor = self.conv2(output_tensor)
        output_tensor = self.batch_norm_2(output_tensor)

        skip_connection = input_tensor
        if (self.stride != 1) or (self.input_channels != self.output_channels):
            skip_connection = self.conv3(input_tensor)
            skip_connection = self.batch_norm_3(skip_connection)

        output_tensor = output_tensor + skip_connection
        output_tensor = self.relu2(output_tensor)

        # output_tensor = self.conv3(output_tensor)
        return output_tensor

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, bias=False, padding=3)
        self.batch_norm = nn.BatchNorm2d(64)
        self.reLU = nn.ReLU()
        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res_block_1 = ResBlock(64,64,1)
        self.res_block_2 = ResBlock(64,128,2)
        self.res_block_3 = ResBlock(128, 256, 2)
        self.res_block_4 = ResBlock(256, 512, 2)
        self.avg_pool = nn.AvgPool2d(1,1)
        self.flatten = nn.Flatten()
        self.fully_connected = nn.Linear(512,2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        output_tensor = self.conv(input_tensor)
        output_tensor = self.batch_norm(output_tensor)
        output_tensor = self.reLU(output_tensor)
        output_tensor = self.maxPool(output_tensor)

        output_tensor = self.res_block_1(output_tensor)
        print("Passed ResBlock 1")
        output_tensor = self.res_block_2(output_tensor)
        print("Passed ResBlock 2")
        output_tensor = self.res_block_3(output_tensor)
        output_tensor = self.res_block_4(output_tensor)
        output_tensor = self.avg_pool(output_tensor)
        output_tensor = self.flatten(output_tensor)
        output_tensor = output_tensor.reshape(output_tensor.shape[0], -1)
        output_tensor = self.fully_connected(output_tensor)
        output_tensor = self.sigmoid(output_tensor)
        return  output_tensor


