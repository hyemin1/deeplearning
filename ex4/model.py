import torch
import torch.nn
# class ResBlock(torch.nn.Module):
#     def __init__(self,in_channels,out_channels,stride):
#         super().__init__()
#         self.cv_1=torch.nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=stride)
#         self.cv_2=torch.nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=1)
#         self.batch_norm_1=torch.nn.BatchNorm2d(out_channels)
#         self.batch_norm_2 = torch.nn.BatchNorm2d(out_channels)
#         self.relu_1=torch.nn.ReLU(inplace=True)
#         self.relu_2 = torch.nn.ReLU(inplace=True)
#
#         self.cv_3 =torch.nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride)
#         self.batch_norm_3 = torch.nn.BatchNorm2d(out_channels)
#     def forward(self,input_tensor):
#         output=self.cv_1(input_tensor)
#         output=self.batch_norm_1(output)
#         output=self.relu_1(output)
#
#         output=self.cv_2(output)
#         output=self.batch_norm_2(output)
#         # output +=input_tensor
#         extra = self.cv_3(input_tensor)
#         extra = self.batch_norm_3(extra)
#         output+=extra
#         output=self.relu_2(output)
#
#         # # extra = self.cv_3(input_tensor)
#         # # extra = self.batch_norm_3(extra)
#         # output+=extra
#         return output
# class ResNet(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.lay_1=torch.nn.Sequential(torch.nn.Conv2d(3,64,7,2),torch.nn.BatchNorm2d(),torch.torch.nn.ReLU(),torch.nn.MaxPool2d(3,2))
#         self.lay_2=torch.nn.Sequential(ResBlock(64,64,1),ResBlock(64,128,2),ResBlock(128,256,2),ResBlock(256,512,2))
#         # self.avg=torch.nn.AvgPool2d()
#         # self.flat= torch.nn.flatten()
#         self.lay_3=torch.nn.Sequential(torch.nn.AvgPool2d(),torch.nn.flatten(),torch.nn.Linear(512,2),torch.nn.Sigmoid())
# #     def forward(self,input_tensor):
# #         output=self.lay_1(input_tensor)
# #         output=self.lay_2(output)
# #         output=self.lay_3(output)
# #         return output
# class ResBlock(torch.nn.Module):
#     def __init__(self,in_channels=1,out_channels=1,stride=1):
#         super().__init__()
#
#         self.cv_1=torch.nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=stride,padding=1,bias=False)
#         self.cv_2=torch.nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=1,padding=1,bias=False)
#         self.batch_norm_1=torch.nn.BatchNorm2d(out_channels)
#         self.batch_norm_2 = torch.nn.BatchNorm2d(out_channels)
#         self.relu_1=torch.nn.ReLU(inplace=True)
#         self.relu_2 = torch.nn.ReLU(inplace=True)
#
#         self.cv_3 =torch.nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=False)
#         self.batch_norm_3 = torch.nn.BatchNorm2d(out_channels)
#         self.in_channels=in_channels
#         self.out_channels=out_channels
#     def forward(self,input_tensor):
#         output=self.cv_1(input_tensor)
#         output=self.batch_norm_1(output)
#         output=self.relu_1(output)
#         output=self.cv_2(output)
#         output=self.batch_norm_2(output)
#         """
#         add skip connectrion
#         """
#         extra=input_tensor
#         if(self.stride!=1 or self.in_channels !=self.out_channels):
#             extra = self.cv_3(input_tensor)
#             extra = self.batch_norm_3(extra)
#         output+=extra
#
#         output=self.relu_2(output)
#
#
#
#         return output
#
#
# class ResNet(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         self.lay_1 = torch.nn.Sequential(torch.nn.Conv2d(3, 64, kernel_size=7, stride=2,padding=3,bias=False), torch.nn.BatchNorm2d(64), torch.torch.nn.ReLU(),
#         torch.nn.MaxPool2d(kernel_size=3, stride=2,padding=1))
#         # self.lay_2=torch.nn.Sequential(ResBlock(64,64,1),ResBlock(64,128,2),ResBlock(128,256,2),ResBlock(256,512,2))
#         # self.lay_3=torch.nn.Sequential(torch.nn.AvgPool2d((1,1)),torch.flatten(end_dim=512),torch.nn.Linear(512,2),torch.nn.Sigmoid())
#         # self.block_1 =self._make_layer(ResBlock(64,64,1),64,1,1)
#         self.block_1=self._make_layer(64,64,2,1)
#         # self.block_2=self._make_layer(64,128,2,2)
#         self.pool=torch.nn.AvgPool2d((1,1))
#         self.fc=torch.nn.Linear(512,2)
#         self.sig=torch.nn.Sigmoid()
#     def forward(self,input_tensor):
#         self.output=input_tensor
#         print(input_tensor.shape)
#         # self.output=self.lay_1(input_tensor)
#         # print(self.output.shape)
#         self.output=self.block_1(self.output)
#         print(self.output.shape)
#         # self.output=self.block_2(self.output)
#         print(self.output.shape)
#
#         # output=self.lay_2(output)
#         return self.output
#
#     def _make_layer(self, in_channels,out_channels, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#
#             layers.append(ResBlock(in_channels, out_channels, stride))
#             # self.in_channels = out_channels * 1
#
#         return torch.nn.Sequential(*layers)
# Residual block
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, stride=stride,padding=1,bias=False,kernel_size=3)
        torch.nn.init.xavier_uniform(self.conv1.weight)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels,kernel_size=3,padding=1,bias=False,stride=1)
        torch.nn.init.xavier_uniform(self.conv2.weight)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        connection = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            connection = self.downsample(x)

        out += connection
        out = self.relu(out)
        return out


# ResNet
class ResNet(torch.nn.Module):
    def __init__(self):
        layers=2
        super().__init__()
        self.in_channels = 64
        '''
        padding size
        '''
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=64,kernel_size=7,stride=2,padding=3,bias=False)
        torch.nn.init.xavier_uniform(self.conv.weight)
        self.bn = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace=True)
        self.max_pool=torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(64,64, layers)
        self.layer2 = self.make_layer(64,128, layers, 2)
        self.layer3 = self.make_layer(128,256, layers, 2)
        self.layer4 = self.make_layer(256,512, layers, 2)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1,1))
        self.fc = torch.nn.Linear(512, 2)
        self.sig=torch.nn.Sigmoid()

    def make_layer(self, in_channels,out_channels, num_blocks, stride=1):
        downsample = None
        if (stride != 1) or (in_channels != out_channels):
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, stride=stride,kernel_size=3,padding=1,bias=False),
                torch.nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))
        for i in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        # x=x[None,...]
        if(len(x.shape)==3):
            x=x[None,...]
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out=self.max_pool(out)
        out = self.layer1(out)
        # print(out.shape)
        out = self.layer2(out)
        # print(out.shape)
        out = self.layer3(out)
        # print(out.shape)
        out=self.layer4(out)
        # print(out.shape)
        # out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.avg_pool(out)
        # print(out.shape)
        # out=torch.flatten(out)
        # print(out.shape)
        # # out = out.view(out.size(0), -1)
        # print(out.shape)
        # out=torch.flatten(out)
        # print(out.shape)
        out= out.view(out.size(0), -1)
        # print(out.shape)
        out = self.fc(out)
        # print(out)
        out=self.sig(out)
        # print(out)


        return out