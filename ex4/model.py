import torch
# Residual block
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, stride=stride,padding=1,bias=False,kernel_size=3)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels,kernel_size=3,padding=1,bias=False,stride=1)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# ResNet
class ResNet(torch.nn.Module):
    def __init__(self,num_classes=10):
        layers=2
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv = torch.nn.Conv2d(3, 64,kernel_size=7,stride=2,padding=1,bias=False)
        self.bn = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace=True)
        self.mpool=torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(ResidualBlock, 64, layers)
        self.layer2 = self.make_layer(ResidualBlock, 128, layers, 2)
        self.layer3 = self.make_layer(ResidualBlock, 256, layers, 2)
        self.layer4 = self.make_layer(ResidualBlock, 512, layers, 2)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1,1))
        self.fc = torch.nn.Linear(512, 2)
        self.sig=torch.nn.Sigmoid()

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(self.in_channels, out_channels, stride=stride,kernel_size=3,padding=1,bias=False),
                torch.nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out=self.mpool(out)
        out = self.layer1(out)
        print(out.shape)
        out = self.layer2(out)
        print(out.shape)
        out = self.layer3(out)
        print(out.shape)
        out=self.layer4(out)
        print(out.shape)
        # out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.avg_pool(out)
        print(out.shape)
        # out=torch.flatten(out)
        # print(out.shape)
        # # out = out.view(out.size(0), -1)
        # print(out.shape)
        # out=torch.flatten(out)
        # print(out.shape)
        out= out.view(out.size(0), -1)
        print(out.shape)
        out = self.fc(out)
        print(out)
        out=self.sig(out)
        print(out)


        return out
