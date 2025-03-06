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
