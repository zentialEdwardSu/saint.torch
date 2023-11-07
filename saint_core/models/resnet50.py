from torch import nn
# conv module
class BasicConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size,
                      stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

# bottleneck
class Bottleneck(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, stride=1):
        super(Bottleneck, self).__init__()

        self.judge = in_channel == out_channel

        self.bottleneck = nn.Sequential(
            BasicConv2d(in_channel, mid_channel, 1),
            nn.ReLU(True),
            BasicConv2d(mid_channel, mid_channel, 3, padding=1, stride=stride),
            nn.ReLU(True),
            BasicConv2d(mid_channel, out_channel, 1),
        )
        self.relu = nn.ReLU(True)
        # 下采样部分由一个包含BN层的1x1卷积构成：
        if in_channel != out_channel:
            self.downsample = BasicConv2d(
                in_channel, out_channel, 1, stride=stride)

    def forward(self, x):
        out = self.bottleneck(x)
        # 若通道不一致需使用1x1卷积下采样
        if not self.judge:
            self.identity = self.downsample(x)
            # 残差+恒等映射=输出
            out += self.identity
        # 否则直接相加
        else:
            out += x

        out = self.relu(out)

        return out
    

# Resnet50:
class ResNet_50(nn.Module):
    def __init__(self, class_num):
        super(ResNet_50, self).__init__()
        self.conv = BasicConv2d(3, 64, 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        # 卷积组1
        self.block1 = nn.Sequential(
            Bottleneck(64, 64, 256),
            Bottleneck(256, 64, 256),
            Bottleneck(256, 64, 256),
        )
        # 卷积组2
        self.block2 = nn.Sequential(
            Bottleneck(256, 128, 512, stride=2),
            Bottleneck(512, 128, 512),
            Bottleneck(512, 128, 512),
            Bottleneck(512, 128, 512),
        )
        # 卷积组3
        self.block3 = nn.Sequential(
            Bottleneck(512, 256, 1024, stride=2),
            Bottleneck(1024, 256, 1024),
            Bottleneck(1024, 256, 1024),
            Bottleneck(1024, 256, 1024),
            Bottleneck(1024, 256, 1024),
            Bottleneck(1024, 256, 1024),
        )
        # 卷积组4
        self.block4 = nn.Sequential(
            Bottleneck(1024, 512, 2048, stride=2),
            Bottleneck(2048, 512, 2048),
            Bottleneck(2048, 512, 2048),
        )
        self.avgpool = nn.AvgPool2d(4)
        self.classifier = nn.Linear(2048, class_num)

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)

        return out