from typing import List, Tuple
from torch import cat, Tensor
from torch.nn import Conv2d, ReLU, Module, Sequential


class PoseDecoder(Module):
    def __init__(
        self, encoder_channels: List[int], input_features: int,
        prediction_frames: int = None, stride: int = 1,
    ):
        super().__init__()
        self.encoder_channels = encoder_channels
        self.input_features = input_features

        if prediction_frames is None:
            prediction_frames = input_features - 1
        self.prediction_frames = prediction_frames

        self.squeezer = Sequential(
            Conv2d(self.encoder_channels[-1], 256, 1), ReLU(True),
        )
        self.pose = Sequential(
            Conv2d(input_features * 256, 256, 3, stride, 1), ReLU(True),
            Conv2d(256, 256, 3, stride, 1), ReLU(True),
            Conv2d(256, 6 * prediction_frames, 1),
        )

    def forward(self, features: List[Tensor]) -> Tuple[Tensor, Tensor]:
        squeezed = cat([self.squeezer(feature) for feature in features], 1)
        out = self.pose(squeezed).mean(3).mean(2)
        out = 0.01 * out.view(-1, self.prediction_frames, 1, 6)
        axisangle = out[..., :3]
        translation = out[..., 3:]
        return axisangle, translation
