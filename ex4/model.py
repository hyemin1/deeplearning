import torch
class ResBlock(torch.nn.Module):
    def __init__(self,in_channels,out_channels,stride):
        super().__init__()
        self.cv_1=torch.nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=stride)
        self.cv_2=torch.nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=1)
        self.batch_norm_1=torch.nn.BatchNorm2d(out_channels)
        self.batch_norm_2 = torch.nn.BatchNorm2d(out_channels)
        self.relu_1=torch.nn.ReLU(inplace=True)
        self.relu_2 = torch.nn.ReLU(inplace=True)

        self.cv_3 =torch.nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride)
        self.batch_norm_3 = torch.nn.BatchNorm2d(out_channels)
    def forward(self,input_tensor):
        output=self.cv_1(input_tensor)
        output=self.batch_norm_1(output)
        output=self.relu_1(output)

        output=self.cv_2(output)
        output=self.batch_norm_2(output)
        # output +=input_tensor
        output=self.relu_2(output)

        extra = self.cv_3(input_tensor)
        extra = self.batch_norm_3(extra)
        output+=extra
        return output
class ResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lay_1=torch.nn.Sequential(torch.nn.Conv2d(3,64,7,2),torch.nn.BatchNorm2d(),torch.torch.nn.ReLU(),torch.nn.MaxPool2d(3,2))
        self.lay_2=torch.nn.Sequential(ResBlock(64,64,1),ResBlock(64,128,2),ResBlock(128,256,2),ResBlock(256,512,2))
        self.lay_3=torch.nn.Sequential(torch.nn.AvgPool2d(),torch.nn.flatten(),torch.nn.Linear(512,2),torch.nn.Sigmoid())
    def forward(self,input_tensor):
        output=self.lay_1(input_tensor)
        output=self.lay_2(output)
        output=self.lay_3(output)
        return output
