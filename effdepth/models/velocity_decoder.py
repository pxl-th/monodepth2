from typing import List, Tuple
from torch import cat, Tensor
from torch.nn import Conv2d, ReLU, Module, Sequential


class VelocityDecoder(Module):
    def __init__(self, encoder_output_channels: int):
        super().__init__()
        self.input_features = 2
        self.prediction_frames = 1

        self.squeezer = Sequential(
            Conv2d(encoder_output_channels, 256, 1), ReLU(True),
        )
        self.velocity = Sequential(
            Conv2d(self.input_features * 256, 256, 3), ReLU(True),
            Conv2d(256, 256, 3), ReLU(True),
            Conv2d(256, self.prediction_frames, 1),
        )

    def forward(self, features: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """
        length of features is 2: [prev frame, current frame] features
        """
        squeezed = cat([self.squeezer(feature) for feature in features], dim=1)
        velocity = self.velocity(squeezed).mean(3).mean(2)
        return velocity.view(-1, self.prediction_frames)
