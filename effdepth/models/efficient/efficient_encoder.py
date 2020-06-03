from torch import randn, Tensor
from torch.nn import Module

from efficientnet_pytorch import EfficientNet


class EfficientEncoder(Module):
    def __init__(
        self, model_name: str = "efficientnet-b2", pretrained: bool = True,
    ):
        super().__init__()
        output_channels = {
            "efficientnet-b0": 1280,
            "efficientnet-b1": 1280,
            "efficientnet-b2": 1408,
        }
        self.output_channels = output_channels[model_name]
        self.model_name = model_name
        self.pretrained = pretrained
        self.encoder = (
            EfficientNet.from_pretrained(model_name)
            if pretrained else
            EfficientNet.from_name(model_name)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.encoder.extract_features(inputs)
