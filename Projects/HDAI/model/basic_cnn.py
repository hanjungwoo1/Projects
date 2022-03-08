import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.encoding = []
        self.decoding = []
        in_channels = 4
        for i in range(3):
            out_channels = in_channels * 2
            self.encoding.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2))
            self.encoding.append(nn.BatchNorm2d(out_channels))
            self.encoding.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        self.encoding_layer = nn.Sequential(*self.encoding)

        for i in range(3):
            out_channels = in_channels // 2
            self.decoding.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1, output_padding=1, stride=2))
            self.decoding.append(nn.BatchNorm2d(out_channels))
            self.decoding.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        self.decoding_layer = nn.Sequential(*self.decoding)

        self.conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, data):
        data = self.encoding_layer(data)
        data = self.decoding_layer(data)
        data = self.conv1x1(data)

        return data