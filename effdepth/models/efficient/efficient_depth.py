from typing import List, Dict

from torch import Tensor, cat
from torch.nn import Module, Sigmoid, ModuleDict
from torch.nn.functional import interpolate

from effdepth.models.layers import ConvBlock, Conv3x3


class EfficientDepthDecoder(Module):
    def __init__(
        self, encoder_output_channels: int, scales: List[int],
    ):
        super().__init__()
        self.scales = scales
        self.decoder_channels = [16, 32, 64, 128, 256]

        self.convs: ModuleDict[str, Module] = ModuleDict()
        self.scaled_convs: ModuleDict[str, Module] = ModuleDict()
        for i in range(4, -1, -1):
            # upconv_0
            input_channels = (
                encoder_output_channels if i == 4 else
                self.decoder_channels[i + 1]
            )
            self.convs[f"{i}0"] = ConvBlock(
                input_channels, self.decoder_channels[i],
            )
            # upconv_1
            input_channels = self.decoder_channels[i]
            self.convs[f"{i}1"] = ConvBlock(
                input_channels, self.decoder_channels[i],
            )
        for s in self.scales:
            self.scaled_convs[f"{s}"] = Conv3x3(self.decoder_channels[s], 1)
        self.sigmoid = Sigmoid()

    def forward(self, input_features: Tensor) -> Dict[int, Tensor]:
        """
        Arguments:
            input_features (Tensor):
                Features from the encoder.
                Each Tensor should be of `(B, C_s, H_s, W_s)` shape.
        Returns:
            Dict[int, Tensor]:
                Decoded disparity maps.
                Keys denote scales, with `0` being the original scale.
                Each tensor will be of `(B, C=1, H_s, W_s)` shape.
        """
        self.outputs: Dict[int, Tensor] = {}
        x = input_features
        for i in range(4, -1, -1):
            x = self.convs[f"{i}0"](x)
            x = interpolate(x, scale_factor=2, mode="nearest")
            x = self.convs[f"{i}1"](x)
            if i in self.scales:
                self.outputs[i] = self.sigmoid(self.scaled_convs[f"{i}"](x))
        return self.outputs
