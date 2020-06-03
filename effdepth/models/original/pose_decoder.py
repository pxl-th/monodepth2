from typing import List, Tuple
from torch import cat, Tensor
from torch.nn import Conv2d, ReLU, Module, Sequential


class PoseDecoder(Module):
    def __init__(self, encoder_output_channels: int):
        super().__init__()
        self.input_features = 2
        self.prediction_frames = 1

        self.squeezer = Sequential(
            Conv2d(encoder_output_channels, 256, 1), ReLU(True),
        )
        self.pose = Sequential(
            Conv2d(self.input_features * 256, 256, 3, 1, 1), ReLU(True),
            Conv2d(256, 256, 3, 1, 1), ReLU(True),
            Conv2d(256, 6, 1),
        )

    def forward(self, features: List[Tensor]) -> Tuple[Tensor, Tensor]:
        squeezed = cat([self.squeezer(feature) for feature in features], dim=1)
        pose = self.pose(squeezed).mean(3).mean(2)
        pose = 0.01 * pose.view(-1, self.prediction_frames, 6)
        axisangle, translation = pose[..., :3], pose[..., 3:]
        return axisangle, translation
