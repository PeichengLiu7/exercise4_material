import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.layer_1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(stride, stride), padding=1)
        self.layer_2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1)
        self.layer_3 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(stride, stride))
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.output_relu = nn.ReLU()

    def forward(self, input_tensor):
        skip = input_tensor.clone()
        skip = self.layer_3(skip)
        skip = self.batchnorm(skip)

        out = self.layer_1(input_tensor)
        out = self.batchnorm(out)
        out = self.output_relu(out)

        out = self.layer_2(out)
        out = self.batchnorm(out)

        out += skip

        output = self.output_relu(out)

        return output


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.first_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2)),nn.BatchNorm2d(64),
            # Relu层
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2)
        )

        self.resblock_1 = ResBlock(in_channels=64, out_channels=64, stride=1)
        self.resblock_2 = ResBlock(in_channels=64, out_channels=128, stride=2)
        self.resblock_3 = ResBlock(in_channels=128, out_channels=256, stride=2)
        self.resblock_4 = ResBlock(in_channels=256, out_channels=512, stride=2)

        self.rest_layers = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, 2), nn.Sigmoid())

    def forward(self, input_tensor):
        out = self.first_layers(input_tensor)
        out = self.resblock_1(out)
        out = self.resblock_2(out)
        out = self.resblock_3(out)
        out = self.resblock_4(out)

        out = self.rest_layers(out)

        return out