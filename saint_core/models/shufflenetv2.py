import torch
import torch.nn as nn

class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x:torch.Tensor):
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups

        # 将输入按通道分成groups组
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        
        # 转置和重排通道
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, height, width)

        return x

class ShuffleNetV2Builder:
    def __init__(self):
        self.in_channels = None
        self.num_classes = None
        self.groups = None
        self.num_layers = None

    def from_dict(self, build_info:dict):
        self.in_channels = build_info.get('in_channels', None)
        self.num_classes = build_info.get('num_classes', None)
        self.groups = build_info.get('groups', None)
        self.num_layers = build_info.get('num_layers', None)
        return self

    def build(self):
        if not all([self.in_channels, self.num_classes, self.groups, self.num_layers]):
            raise ValueError("Incomplete configuration for model")

        model = self._create_model()
        return model

    def _create_model(self):
        in_channels = 24
        layers = [conv_1x1(self.in_channels, in_channels)]

        # 添加ShuffleBlock
        for _ in range(self.num_layers):
            shuffle_block = self._create_shuffle_block(in_channels, self.groups)
            layers.append(shuffle_block)

        # 添加全局池化层和全连接层
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Linear(in_channels, self.num_classes))

        return nn.Sequential(*layers)

    def _create_shuffle_block(self, in_channels, groups):
        out_channels = in_channels if in_channels == 24 else in_channels // 2
        return nn.Sequential(
            channel_shuffle(in_channels, groups),
            depthwise_conv3x3(in_channels, out_channels),
            conv_1x1_group(out_channels, in_channels, groups)
        )

def channel_shuffle(num_channels, groups):
    return nn.Sequential(
        nn.Conv2d(num_channels, num_channels, kernel_size=1, groups=groups, bias=False),
        nn.BatchNorm2d(num_channels),
        nn.ReLU(inplace=True)
    )


def depthwise_conv3x3(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def conv_1x1(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def conv_1x1_group(in_channels, out_channels, groups):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

# # 使用建造者模式构建ShuffleNetV2模型
# builder = ShuffleNetV2Builder()
# builder.set_in_channels(3).set_num_classes(10).set_groups(2).set_num_layers(4)
# shufflenetv2_model = builder.build()
# shufflenetv2_model.apply(initialize_weights)  # 初始化模型权重

# # 保存模型检查点
# torch.save(shufflenetv2_model.state_dict(), 'shufflenetv2_checkpoint.pth')

# # 加载模型检查点
# loaded_model = ShuffleNetV2Builder().set_in_channels(3).set_num_classes(10).set_groups(2).set_num_layers(4).build()
# loaded_model.load_state_dict(torch.load('shufflenetv2_checkpoint.pth'))

# # 示例用法
# # 假设输入为128x128
# input_tensor = torch.randn(1, 3, 128, 128)
# output = loaded_model(input_tensor)
# print(output.shape)

# 示例用法 - 从字典中提取建造信息
# build_dict = {
#     'in_channels': 3,
#     'num_classes': 10,
#     'groups': 2,
#     'num_layers': 4
# }

# builder = ShuffleNetV2Builder().from_dict(build_dict)
# shufflenetv2_model = builder.build()

# # 示例用法
# input_tensor = torch.randn(1, 3, 128, 128)
# output = shufflenetv2_model(input_tensor)
# print(output.shape)