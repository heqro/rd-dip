import torch
from torch import nn


class ResNet(torch.nn.Module):
    def __init__(self, n_channels_output: int = 3):
        super().__init__()
        # define skip connections processing modules
        self.skips = nn.ModuleList()
        for channels_in in [3] + [128] * 4:
            self.skips.append(
                nn.Sequential(
                    nn.Conv2d(channels_in, 4, kernel_size=(1, 1)),
                    nn.BatchNorm2d(4),
                    nn.LeakyReLU(0.2),
                )
            )

        # define "double-convolution" downsampling modules
        self.downsamplers = nn.ModuleList()
        for channels_in in [3] + [128] * 4:
            self.downsamplers.append(
                nn.Sequential(
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(channels_in, 128, kernel_size=(3, 3), stride=(2, 2)),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU(0.2),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(128, 128, kernel_size=(3, 3)),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU(0.2),
                )
            )

        # define "post-concatenation" processing modules
        self.upsamplers = nn.ModuleList()
        for i in range(5):
            module = nn.Sequential(
                nn.BatchNorm2d(132),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(132, 128, kernel_size=(3, 3)),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                # nn.ReflectionPad2d((0, 0, 0, 0)),
                nn.Conv2d(128, 128, kernel_size=(1, 1)),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
            )
            if i != 0:  # deep layer
                module.append(nn.Upsample(scale_factor=2.0, mode="bilinear"))
            else:  # last layer
                # module.append(nn.ReflectionPad2d((0, 0, 0, 0)))
                module.append(nn.Conv2d(128, n_channels_output, kernel_size=(1, 1)))
                module.append(nn.Sigmoid())
            self.upsamplers.append(module)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x1 = self.downsamplers[0](z)
        x2 = self.downsamplers[1](x1)
        x3 = self.downsamplers[2](x2)
        x4 = self.downsamplers[3](x3)
        x5 = self.downsamplers[4](x4)

        s1 = self.skips[0](z)
        s2 = self.skips[1](x1)
        s3 = self.skips[2](x2)
        s4 = self.skips[3](x3)
        s5 = self.skips[4](x4)

        y5 = torch.cat([s5, nn.Upsample(scale_factor=2.0, mode="bilinear")(x5)], dim=1)
        y4 = self.upsamplers[4](y5)
        y3 = self.upsamplers[3](torch.cat([s4, y4], dim=1))
        y2 = self.upsamplers[2](torch.cat([s3, y3], dim=1))
        y1 = self.upsamplers[1](torch.cat([s2, y2], dim=1))
        y = self.upsamplers[0](torch.cat([s1, y1], dim=1))

        return y
