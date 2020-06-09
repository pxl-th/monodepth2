from argparse import Namespace
from typing import Tuple, List, Union
from os.path import join

from numpy import (
    array, fromfile, ndarray, hstack, linspace, ceil, logical_and, zeros,
    repeat, arange, argwhere,
)
from skimage.io import imread
from torch import Tensor, tensor, cat, float32, pinverse, full, from_numpy
from torch.utils.data import Dataset, Subset, ConcatDataset

from imgaug.augmenters import (
    Sequential, Sometimes, GaussianBlur, LinearContrast, Multiply,
)
from imgaug.parameters import Uniform, Deterministic, DiscreteUniform

from effdepth.preprocess import get_paths


def compute_intrinsics(height: int, width: int) -> Tuple[Tensor, Tensor]:
    original_width = 1164
    original_focal = 910
    scaling = width / original_width
    focal = original_focal * scaling
    intrinsics: Tensor = tensor(
        [[focal, 0, 0.5 * width, 0],
         [0, focal, 0.5 * height, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]], dtype=float32,
    )
    inv_intrinsics: Tensor = pinverse(intrinsics).unsqueeze_(0)
    intrinsics.unsqueeze_(0)
    return intrinsics, inv_intrinsics


class SequenceData(Dataset):
    def __init__(
        self,
        frame_template: str, targets: Tensor, length: int,
        sequence_length: int, target_id: int, sources_ids: List[int],
        augment: bool = False,
    ):
        super().__init__()
        self.frame_template = frame_template
        self.targets = targets
        self.sequence_length = sequence_length
        self.length = length

        self.target_id = target_id
        self.sources_ids = sources_ids
        self.ids = array(sorted([self.target_id] + self.sources_ids))

        self.augment = augment

    @staticmethod
    def no_target_dataset(
        frame_template: str, length: int, hparams: Namespace, **kwargs,
    ) -> "SequenceData":
        targets = full((length,), -1, dtype=float32)
        length = length // hparams.sequence_length - 1
        return SequenceData(
            frame_template, targets, length,
            hparams.sequence_length, hparams.target_id, hparams.sources_ids,
            **kwargs,
        )

    @staticmethod
    def target_dataset(
        frame_template: str, targets: Tensor, hparams: Namespace, **kwargs,
    ) -> "SequenceData":
        length = targets.shape[0] // hparams.sequence_length - 1
        return SequenceData(
            frame_template, targets, length,
            hparams.sequence_length, hparams.target_id, hparams.sources_ids,
            **kwargs,
        )

    def __len__(self):
        return self.length

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        start = index * self.sequence_length
        ids = self.ids + start
        return self.load_images(ids), self.targets[ids]

    def load_images(self, ids: List[int]) -> Tensor:
        """
        Returns:
            (l, c, w, h) Tensor:
                Tensor of images.
        """
        images: List[Tensor] = [
            imread(self.frame_template.format(i)) for i in ids
        ]
        # Using deterministic parameters to apply exactly
        # the same augmentations to all images in the sequence.
        if self.augment:
            augmenter = Sequential([
                LinearContrast(Deterministic(Uniform(0.7, 1.3))),
                Multiply(
                    Deterministic(Uniform(0.7, 1.3)),
                    per_channel=Deterministic(DiscreteUniform(0, 1)),
                ),
                Sometimes(
                    Deterministic(DiscreteUniform(0, 1)),
                    GaussianBlur(sigma=Deterministic(Uniform(0, 0.7))),
                ),
            ], random_order=True)
            images = augmenter(images=images)
        for i in range(len(images)):
            image = images[i].astype("float32") / 255
            images[i] = tensor(image).permute(2, 0, 1).unsqueeze_(0)
        return cat(images, dim=0)

    def valid_ids(self) -> Tensor:
        """
        Valid ids are those that have no `-1`s in targets.
        """
        ids_shifts = repeat(arange(self.length, dtype="int64"), 3)
        ids_shifts *= self.sequence_length

        ids_span = hstack([self.ids] * self.length)
        ids_span += ids_shifts

        valid_mask = self.targets[ids_span] != -1
        valid_mask = valid_mask.reshape(-1, 3)
        valid_mask = logical_and(
            logical_and(valid_mask[:, 0], valid_mask[:, 1]),
            valid_mask[:, 2],
        ).bool()
        valid_ids = argwhere(valid_mask)[0]
        if isinstance(valid_ids, Tensor):
            valid_ids = valid_ids.numpy()
        return valid_ids


def datasets_config():
    datasets = [
        {  # speedchallenge train data
            "frame_template": r"C:\Users\tonys\projects\python\comma\speedchallenge\train\frames-160x320\frame-{}.jpg",
            "targets": r"C:\Users\tonys\projects\python\comma\speedchallenge\train\train.txt",
            "augment": True,
        },
        # {  # speedchallenge test data
        #     "frame_template": r"C:\Users\tonys\projects\python\comma\speedchallenge\test\frames-160x320\frame-{}.jpg",
        #     "length": 10798,
        #     # "augment": True,
        # },
    ]
    comma_2k19_base = r"C:\Users\tonys\projects\python\comma\2k19"
    frames_sub = "frames-160x320"
    frame_name = "frame-{}.jpg"
    target_name = "speed.txt"
    for sub in get_paths(comma_2k19_base):
        datasets.append({
            "frame_template": join(sub, frames_sub, frame_name),
            "targets": join(sub, target_name),
            "augment": True,
        })
    print("train", len(datasets))
    return datasets


def validation_dataset_config():
    datasets = []
    comma_2k19_base = r"C:\Users\tonys\projects\python\comma\2k19-validation"
    frames_sub = "frames-160x320"
    frame_name = "frame-{}.jpg"
    target_name = "speed.txt"
    for sub in get_paths(comma_2k19_base):
        datasets.append({
            "frame_template": join(sub, frames_sub, frame_name),
            "targets": join(sub, target_name),
        })
    print("val", len(datasets))
    return datasets


def load_targets(path: str, numpy: bool = False) -> Union[ndarray, Tensor]:
    if numpy:
        return fromfile(path, dtype="float32", sep="\n")
    return from_numpy(fromfile(path, dtype="float32", sep="\n"))


def create_dataset(config, hparams: Namespace) -> SequenceData:
    augment = "augment" in config and config["augment"]
    if "targets" in config:
        return SequenceData.target_dataset(
            config["frame_template"], load_targets(config["targets"]),
            hparams, augment=augment,
        )
    return SequenceData.no_target_dataset(
        config["frame_template"], config["length"],
        hparams, augment=augment,
    )


def calculate_splits(
    targets: Union[ndarray, Tensor], splits: ndarray,
) -> Tuple[ndarray, List[ndarray]]:
    if isinstance(targets, Tensor):
        targets = targets.numpy()

    split_sizes, split_masks = [], []
    for i in range(len(splits) - 1):
        mask = logical_and(targets > splits[i], targets < splits[i + 1])
        split_sizes.append(mask.sum())
        split_masks.append(mask)

    return array(split_sizes, dtype="int64"), split_masks


def create_splits(configs, parts: int) -> ndarray:
    targets = [
        load_targets(config["targets"], True)
        for config in configs
        if "targets" in config
    ]
    joined_targets = hstack(targets)
    tmin, tmax = joined_targets.min(), joined_targets.max()

    splits = linspace(tmin, tmax, parts)
    split_sizes, _ = calculate_splits(joined_targets, splits)
    return splits, split_sizes


def filter_out_splits(
    targets: ndarray, loaded_split_sizes: ndarray,
    splits: ndarray, target_split_size: int,
):
    split_sizes, split_masks = calculate_splits(targets, splits)
    for i in range(len(split_sizes)):
        if loaded_split_sizes[i] < target_split_size:
            loaded_split_sizes[i] += split_sizes[i]
            continue
        targets[split_masks[i]] = -1


def even_training_targets(
    hparams: Namespace, parts: int = 9,
) -> List[SequenceData]:
    splits, split_sizes = create_splits(datasets_config(), parts)
    target_split_size = min(split_sizes)
    print(f"Target splits: {splits}")
    print(f"Current datasets splits: {split_sizes}")
    print(f"Minimum split size: {target_split_size}")

    loaded_split_sizes = zeros(len(split_sizes), dtype="int64")
    train_datasets = []
    for config in datasets_config():
        dataset = create_dataset(config, hparams)
        if "length" in config:
            train_datasets.append(dataset)
            continue
        filter_out_splits(
            dataset.targets, loaded_split_sizes, splits, target_split_size,
        )
        train_datasets.append(Subset(dataset, dataset.valid_ids()))

    evened_targets = hstack([d.dataset.targets for d in train_datasets])
    evened_split_sizes, _ = calculate_splits(evened_targets, splits)
    print(f"Evened datasets splits: {evened_split_sizes}")
    return train_datasets


def get_training_dataset(hparams: Namespace) -> Dataset:
    if "even_parts" in hparams:
        datasets = even_training_targets(hparams, hparams.even_parts)
    else:
        datasets = [
            create_dataset(config, hparams) for config in datasets_config()
        ]
    return ConcatDataset(datasets)


def main():
    hparams = Namespace(
        pretrained=True, encoder_layers=18,
        scales=[0, 1, 2, 3],
        disparity_smoothness=1e-3,  # control sharpness of disparity
        lr=1e-4, step_size=10, batch_size=4,
        height=160, width=320,  # have to be divisible by 2**5
        min_depth=0.1, max_depth=100.0,
        target_id=7, sources_ids=[0, 14], sequence_length=15,
        device="cuda", dt=1 / 20,
        even_parts=9,
    )
    get_training_dataset(hparams)


if __name__ == "__main__":
    main()
