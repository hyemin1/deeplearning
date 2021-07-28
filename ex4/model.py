import torch
import torch.nn

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, stride=stride,padding=1,bias=False,kernel_size=3)
        torch.nn.init.xavier_uniform(self.conv1.weight)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels,kernel_size=3,padding=1,bias=False,stride=1)
        torch.nn.init.xavier_uniform(self.conv2.weight)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.input_adjust=None
        # for skip connection
        if (stride != 1) or (in_channels != out_channels):
            self.input_adjust = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
                torch.nn.BatchNorm2d(out_channels))

    def forward(self, x):
        connection = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.input_adjust:
            connection = self.input_adjust(x)

        out += connection
        out = self.relu(out)
        return out


# ResNet
class ResNet(torch.nn.Module):
    def __init__(self):
        #define the structure of resnet
        layers=2
        super().__init__()
        self.in_channels = 64
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=64,kernel_size=7,stride=2,padding=3,bias=False)
        torch.nn.init.xavier_uniform(self.conv.weight)
        self.bn = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace=True)
        self.max_pool=torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1=ResidualBlock(64,64)
        self.layer2 = ResidualBlock(64,128,2)
        self.layer3 = ResidualBlock(128,256,2)
        self.layer4 = ResidualBlock(256,512,2)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1,1))
        self.fc = torch.nn.Linear(512, 2)
        self.sig=torch.nn.Sigmoid()


    def forward(self, x):

        if(len(x.shape)==3):
            x=x[None,...]
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out=self.max_pool(out)
        out = self.layer1(out)

        out = self.layer2(out)
        out = self.layer3(out)
        out=self.layer4(out)
        out = self.avg_pool(out)
        out= out.view(out.size(0), -1)
        out = self.fc(out)
        out=self.sig(out)

        return out
