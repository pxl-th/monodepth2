from typing import List

from numpy import array
from torch import cat, Tensor
from torch.nn import BatchNorm2d, ReLU, MaxPool2d, Conv2d, init, Module
from torchvision.models import ResNet, resnet, resnet18, resnet50
import torch.utils.model_zoo as model_zoo


class ResNetMultiImageInput(ResNet):
    def __init__(
        self, block, layers: List[int],
        num_classes: int = 1000, num_input_images: int = 1,
    ):
        super().__init__(block, layers)

        self.inplanes = 64
        self.conv1 = Conv2d(
            num_input_images * 3, 64,
            kernel_size=7, stride=2, padding=3, bias=False,
        )
        self.bn1 = BatchNorm2d(64)
        self.relu = ReLU(True)
        self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, Conv2d):
                init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu',
                )
            elif isinstance(m, BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


def _resnet_multi_image_input(
    num_layers: int, pretrained: bool = False, num_input_images: int = 1,
) -> ResNetMultiImageInput:
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: resnet.BasicBlock, 50: resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(
        block_type, blocks, num_input_images=num_input_images,
    )
    if not pretrained:
        return model

    loaded = model_zoo.load_url(resnet.model_urls[f"resnet{num_layers}"])
    loaded["conv1.weight"] = cat(
        [loaded["conv1.weight"]] * num_input_images, 1,
    ) / num_input_images
    model.load_state_dict(loaded)
    return model


class ResnetEncoder(Module):
    def __init__(
        self, num_layers: int, pretrained: bool, num_input_images: int = 1,
    ):
        super(ResnetEncoder, self).__init__()
        self.layers = array([64, 64, 128, 256, 512])
        resnets = {18: resnet18, 50: resnet50}
        if num_input_images > 1:
            self.encoder = _resnet_multi_image_input(
                num_layers, pretrained, num_input_images,
            )
        else:
            self.encoder = resnets[num_layers](pretrained)
        if num_layers > 34:
            self.layers[1:] *= 4

    def forward(self, input_image: Tensor) -> List[Tensor]:
        """
        Arguments:
            `input_image` (Tensor):
                Input images.
                Tensor should be of `(B, C * L, H, W)` shape.
                `B` --- batch size. `C` --- channels in single image.
                `L` --- length of the sequence.
        Returns:
        List[Tensor]:
            Each Tensor will be of `(B, C_s, H_s, W_s)` shape.
            List of extracted features at each level. Number of levels is 5.
        """
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))
        return self.features
