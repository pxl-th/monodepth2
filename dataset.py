from argparse import Namespace
from typing import Tuple, List, Union
from tqdm import tqdm
from os.path import join

from numpy import ndarray, array, tan, pi
from skimage.io import imread, imsave
from skimage.transform import resize
from torch import Tensor, tensor, cat, float32, pinverse, full
from torch.utils.data import Dataset


def preprocess(
    input_template: str, output_template: str, length: int,
    size: Tuple[int, int],
) -> None:
    for i in tqdm(range(length)):
        image: ndarray = imread(input_template.format(i))
        image = image[158:350]
        if size is not None:
            image = resize(image, size)
            image *= 255
            image = image.round().astype("uint8")
        imsave(output_template.format(i), image, check_contrast=False)


def compute_intrinsics(height: int, width: int) -> Tuple[Tensor, Tensor]:
    intrinsics: Tensor = tensor(
        [[0.58, 0, 0.5, 0],
         [0, 1.92, 0.5, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]], dtype=float32,
    )
    intrinsics[0, :] *= width
    intrinsics[1, :] *= height
    # scaling = width / 1164
    # original_focal = 910
    # focal = original_focal * scaling
    # intrinsics: Tensor = tensor(
    #     [[focal, 0, 0.5 * width, 0],
    #      [0, focal, 0.5 * height, 0],
    #      [0, 0, 1, 0],
    #      [0, 0, 0, 1]], dtype=float32,
    # )
    inv_intrinsics: Tensor = pinverse(intrinsics).unsqueeze_(0)
    intrinsics.unsqueeze_(0)
    return intrinsics, inv_intrinsics


class SequenceData(Dataset):
    def __init__(
        self,
        frame_template: str, targets: Tensor, length: int,
        sequence_length: int, target_id: int, sources_ids: List[int],
    ):
        super().__init__()
        self.frame_template = frame_template
        self.targets = targets
        self.sequence_length = sequence_length
        self.length = length

        self.target_id = target_id
        self.sources_ids = sources_ids
        self.ids = array(sorted([self.target_id] + self.sources_ids))

    @staticmethod
    def no_target_dataset(
        frame_template: str, length: int, hparams: Namespace,
    ) -> "SequenceData":
        targets = full((length,), -1, dtype=float32)
        length = length // hparams.sequence_length - 1
        return SequenceData(
            frame_template, targets, length,
            hparams.sequence_length, hparams.target_id, hparams.sources_ids,
        )

    @staticmethod
    def target_dataset(
        frame_template: str, targets: Tensor, hparams: Namespace,
    ) -> "SequenceData":
        length = targets.shape[0] // hparams.sequence_length - 1
        return SequenceData(
            frame_template, targets, length,
            hparams.sequence_length, hparams.target_id, hparams.sources_ids,
        )

    def __len__(self):
        return self.length

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        start = index * self.sequence_length
        ids = self.ids + start
        return self.load_images(ids), self.targets[ids]

    def load_images(self, ids: ndarray) -> Tensor:
        """
        Returns:
            (l, c, w, h) Tensor:
                Tensor of images.
        """
        images: List[Tensor] = []
        for i in ids:
            image = imread(self.frame_template.format(i))
            image = image.astype("float32") / 255
            images.append(tensor(image).permute(2, 0, 1).unsqueeze_(0))
        return cat(images, dim=0)


def datasets_config(hparams: Namespace):
    datasets = [
        {  # speedchallenge train data
            "frame_template": r"C:\Users\tonys\projects\python\comma\speedchallenge\train\frames-192x640\frame-{:05}.jpg",
            "targets": r"C:\Users\tonys\projects\python\comma\speedchallenge\train\train.txt",
        },
        # {  # speedchallenge test data
        #     "frame_template": r"C:\Users\tonys\projects\python\comma\speedchallenge\test\frames-192x640\frame-{:05}.jpg",
        #     "length": 10798,
        # },
    ]
    # comma_2k19_base = r"C:\Users\tonys\projects\python\comma\2k19"
    # comma_2k19_subs = (
    #     (r"2018-07-29--11-17-20", range(3, 8)),
    #     (r"2018-07-29--12-02-42", range(27, 32)),
    #     (r"2018-07-29--16-37-17", range(4, 6)),
    #     (r"2018-07-30--13-03-07", range(14, 22)),
    # )
    # frames_sub = "frames"
    # frame_name = "frame-{}.jpg"
    # target_name = "speed.txt"
    # for sub, ranges in comma_2k19_subs:
    #     for i in ranges:
    #         sub_base = join(comma_2k19_base, sub, str(i))
    #         frame_template = join(sub_base, frames_sub, frame_name)
    #         target_path = join(sub_base, target_name)
    #         datasets.append({
    #             "frame_template": frame_template,
    #             "targets": target_path,
    #         })
    return datasets
