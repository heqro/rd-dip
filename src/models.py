import torch
from torch import nn, Tensor


class UNet(torch.nn.Module):

    def __init__(
        self,
        n_channels_output: int = 1,
        channels_list: list[int] = [3, 128, 128, 128, 128, 128],
        skip_sizes: list[int] = [4, 4, 4, 4, 4],
    ):
        def get_skip(ch_in: int, skip_size: int):
            return nn.Sequential(
                nn.Conv2d(ch_in, skip_size, kernel_size=(1, 1)),
                nn.BatchNorm2d(skip_size),
                nn.LeakyReLU(0.2),
            )

        def get_downsampler(ch_in: int, ch_out: int):
            return nn.Sequential(
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(ch_in, ch_out, kernel_size=(3, 3), stride=(2, 2)),
                nn.BatchNorm2d(ch_out),
                nn.LeakyReLU(0.2),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(ch_out, ch_out, kernel_size=(3, 3)),
                nn.BatchNorm2d(ch_out),
                nn.LeakyReLU(0.2),
            )

        def get_upsampler(ch_in: int, ch_out: int):
            return nn.Sequential(
                nn.BatchNorm2d(ch_in),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(ch_in, ch_out, kernel_size=(3, 3)),
                nn.BatchNorm2d(ch_out),
                nn.LeakyReLU(0.2),
                nn.Conv2d(ch_out, ch_out, kernel_size=(1, 1)),
                nn.BatchNorm2d(ch_out),
                nn.LeakyReLU(0.2),
            )

        super().__init__()

        if len(channels_list) != len(skip_sizes) + 1:
            raise Exception(
                f"Mismatch between length of channels list ({len(channels_list)}) and skip sizes list ({len(skip_sizes)})"
            )

        self.channels_list = channels_list
        self.skip_sizes = skip_sizes

        self.skips = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        for idx in range(len(channels_list) - 1):  # skip deepest layer
            ch_in, ch_out = channels_list[idx], channels_list[idx + 1]
            self.skips.append(get_skip(ch_in, skip_sizes[idx]))
            self.downsamplers.append(get_downsampler(ch_in, ch_out))

        self.upsamplers = nn.ModuleList()
        for idx in reversed(range(2, len(channels_list))):  # skip shallowest layer
            ch_in, ch_out = channels_list[idx], channels_list[idx - 1]
            self.upsamplers.append(
                get_upsampler(ch_in + skip_sizes[idx - 1], ch_out).append(
                    nn.Upsample(scale_factor=2, mode="bilinear")
                )
            )

        self.upsamplers.append(
            get_upsampler(channels_list[1] + skip_sizes[0], channels_list[1])
            .append(nn.Conv2d(channels_list[1], n_channels_output, kernel_size=(1, 1)))
            .append(nn.Sigmoid())
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = z
        skip_samples = nn.ParameterList()

        for downsampler, skip in zip(self.downsamplers, self.skips):
            skip_samples.append(skip(x))
            x = downsampler(x)

        x = nn.Upsample(scale_factor=2, mode="bilinear")(x)

        for upsampler, skip in zip(self.upsamplers, reversed(skip_samples)):
            x = upsampler(torch.cat([skip, x], dim=1))

        return x


class UNet_Alt(torch.nn.Module):
    def __init__(self, n_channels_output: int = 1):
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


class AttentionGate(nn.Module):
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super(AttentionGate, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gating_signal, x):
        g1 = self.W_g(gating_signal)
        x1 = self.W_x(x)
        attention_coef = nn.Upsample(scale_factor=2, mode="bilinear")(
            self.psi(self.relu(g1 + x1))
        )
        out = x * attention_coef
        return out


class AttentiveUNet(UNet_Alt):

    def __init__(self):
        super().__init__(1)
        self.attention = nn.ModuleList()
        for channels_in in [3] + [128] * 4:
            self.attention.append(AttentionGate(F_g=channels_in, F_l=128, F_int=128))

    def forward(self, z: Tensor) -> Tensor:

        x1 = self.downsamplers[0](z)
        x2 = self.downsamplers[1](x1)
        x3 = self.downsamplers[2](x2)
        x4 = self.downsamplers[3](x3)
        x5 = self.downsamplers[4](x4)

        a5 = self.attention[4](x=x4, gating_signal=x5)
        a4 = self.attention[3](x=x3, gating_signal=x4)
        a3 = self.attention[2](x=x2, gating_signal=x3)
        a2 = self.attention[1](x=x1, gating_signal=x2)
        a1 = self.attention[0](x=z, gating_signal=x1)

        s1 = self.skips[0](a1)
        s2 = self.skips[1](a2)
        s3 = self.skips[2](a3)
        s4 = self.skips[3](a4)
        s5 = self.skips[4](a5)

        y5 = torch.cat([s5, nn.Upsample(scale_factor=2.0, mode="bilinear")(x5)], dim=1)
        y4 = self.upsamplers[4](y5)
        y3 = self.upsamplers[3](torch.cat([s4, y4], dim=1))
        y2 = self.upsamplers[2](torch.cat([s3, y3], dim=1))
        y1 = self.upsamplers[1](torch.cat([s2, y2], dim=1))
        y = self.upsamplers[0](torch.cat([s1, y1], dim=1))

        return y
