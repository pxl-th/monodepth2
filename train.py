from argparse import Namespace
from os.path import join
from typing import List, Dict, Tuple

from skimage.io import imsave
from numpy import fromfile, transpose, round_, sort, argwhere, array
from numpy.random import seed
from torch import (
    Tensor, abs as tabs, cat, randn, min as tmin, from_numpy, manual_seed,
    zeros, norm, matmul, float32, tensor,
)
from torch.nn.functional import interpolate, grid_sample, mse_loss
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam, lr_scheduler
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from models.resnet_encoder import ResnetEncoder
from models.depth_decoder import DepthDecoder
from models.pose_decoder import PoseDecoder
from models.layers import (
    SSIM, BackprojectDepth, Project3D, disp_to_depth,
    transformation_from_parameters, get_smooth_loss,
)
from dataset import SequenceData, compute_intrinsics, datasets_config


seed(0)
manual_seed(0)


class EffDepthTraining(LightningModule):
    def __init__(self, hparams: Namespace):
        super().__init__()
        self.hparams = hparams

        self.encoder = ResnetEncoder(
            hparams.encoder_layers, hparams.pretrained,
        )
        self.depth_decoder = DepthDecoder(self.encoder.layers, hparams.scales)
        self.pose_decoder = PoseDecoder(
            self.encoder.layers, hparams.pose_sequence_length,
        )
        # Set target id and sources ids as they will appear in the sequence.
        ids = sort(array(
            self.hparams.sources_ids + [self.hparams.target_id], dtype="int32",
        ))
        self.batch_target_id = argwhere(ids == self.hparams.target_id)[0, 0]
        self.batch_sources_id = [
            argwhere(ids == sid)[0, 0] for sid in self.hparams.sources_ids
        ]
        self.time_delta = tensor([[
            abs(sid - self.hparams.target_id)
            for sid in self.hparams.sources_ids
        ]], dtype=float32, device=hparams.device)

        self.intrinsics, self.inv_intrinsics = compute_intrinsics(
            hparams.height, hparams.width,
        )
        self.intrinsics = self.intrinsics.to(hparams.device)
        self.inv_intrinsics = self.inv_intrinsics.to(hparams.device)

        self.ssim = SSIM()
        self.projections: Project3D = Project3D(
            hparams.batch_size, hparams.height, hparams.width,
        )
        self.backprojections: BackprojectDepth = BackprojectDepth(
            hparams.batch_size, hparams.height, hparams.width,
        )
        print(
            f"Target id: {self.batch_target_id}, "
            f"Sources ids: {self.batch_sources_id}"
        )
        print(f"Intrinsics: {self.intrinsics}")


    def forward(
        self, inputs: Tensor
    ) -> Tuple[Dict[int, Tensor], Dict[int, Tuple[Tensor, Tensor, Tensor]]]:
        """
        inputs: [b, l, c, w, h]:
            c - channels
            l - seq length
            b - batch size
            0 - target, -1 - left, 1 - right
        """
        # = forward:
        # | encode all images
        # | depth decode only target images
        # | predict poses between each (prev, curr) & (curr, next) pairs in sequence
        # = prepare projections:
        # | generate new images by backprojecting depth to point cloud
        # | then projecting point cloud back to the near image
        # | using estimated transformation
        # | again for each (prev, curr) & (curr, next) pairs in sequence
        # = calculate losses:
        # | reprojection (SSIM + L1) **multi-scale training section**
        # | automasking (SSIM + L1)
        # -> get min loss
        b, l, c, w, h = inputs.shape
        inputs = inputs.view(b * l, c, w, h)
        features: List[Tensor] = []
        for feature in self.encoder(inputs):
            _, ce, he, we = feature.shape
            features.append(feature.view(b, l, ce, he, we))

        disparities: Dict[int, Tensor] = self.depth_decoder([
            feature[:, self.batch_target_id] for feature in features
        ])
        poses: Dict[int, Tuple[Tensor, Tensor, Tensor]] = self._estimate_poses(
            features,
        )
        return disparities, poses

    def _estimate_poses(
        self, features: List[Tensor],
    ) -> Dict[int, Tuple[Tensor, Tensor]]:
        """
        Estimate poses between each (prev -> target) & (target -> next) frames
        given their respective ids.
        *Note* that `prev` and `next` do not need to be adjacent w.r.t. time,
        but should be not "too far", so that estimation can be done.

        Arguments:
            features (List[Tensor]):
                Features from the encoder.
            target_id (int):
                Id of the target features in `features`.
            sources_ids (List[int]):
                Ids of source frames in the `features`.

        Returns:
            Dict[int, Tuple[Tensor, Tensor]]:
                Mapping of `source id` to its
                euler angles and translation vector.
        """
        poses: Dict[int, Tuple[Tensor, Tensor, Tensor]] = {}
        for bsid in self.batch_sources_id:
            pose_inputs = (
                [features[-1][:, bsid], features[-1][:, self.batch_target_id]]
                if bsid < self.batch_target_id else
                [features[-1][:, self.batch_target_id], features[-1][:, bsid]]
            )
            axisangle, translation = self.pose_decoder(pose_inputs)
            # axisangle, translation, velocity = self.pose_decoder(pose_inputs)
            poses[bsid] = (
                axisangle.squeeze_(1), translation.squeeze_(1),
                # velocity.squeeze_(1),
            )
        return poses

    def _reprojection_loss(self, target: Tensor, predicted: Tensor) -> Tensor:
        """
        Calculate reprojection loss between target and predicted images.
        Which is a combination of SSIM and L1 losses.
        """
        l1_loss = tabs(target - predicted).mean(dim=1, keepdim=True)
        ssim_loss = self.ssim(predicted, target).mean(dim=1, keepdim=True)
        return 0.85 * ssim_loss + 0.15 * l1_loss

    def _compute_identity_reprojection_loss(self, inputs: Tensor) -> Tensor:
        """
        Compute identity reprojection losses between source & target images.
        """
        identity_reprojection_loss: Tensor = None
        for bsid in self.batch_sources_id:
            il = self._reprojection_loss(
                inputs[:, self.batch_target_id], inputs[:, bsid],
            )
            il += randn(il.shape, device=il.device) * 1e-5
            if identity_reprojection_loss is None:
                identity_reprojection_loss = il
                continue
            identity_reprojection_loss = tmin(
                cat((il, identity_reprojection_loss), dim=1),
                dim=1, keepdim=True,
            )[0]
        return identity_reprojection_loss

    def _compute_smooth_loss(
        self, inputs: Tensor, scale_disparity: Tensor, scale: int,
    ) -> Tensor:
        """
        Compute smooth loss between scale disparity
        and normalized scaled target input image.
        """
        scale_factor = 2 ** scale
        if scale > 0:
            scaled_target = interpolate(inputs[:, self.batch_target_id], (
                self.hparams.height // scale_factor,
                self.hparams.width // scale_factor,
            ), mode="bilinear", align_corners=False)
        else:
            scaled_target = inputs[:, self.batch_target_id]

        normalized_disparity = scale_disparity / (
            scale_disparity.mean(2, True).mean(3, True) + 1e-7
        )
        return (
            get_smooth_loss(normalized_disparity, scaled_target)
            * self.hparams.disparity_smoothness
            / scale_factor
        )

    def _warp_image(
        self, image: Tensor, disparity: Tensor, transformation: Tensor,
    ) -> Tensor:
        """
        Given transformation between `disparity` and `image`,
        project disparity onto `image` and sample new image from there.
        """
        scaled_disparity = interpolate(
            disparity, (self.hparams.height, self.hparams.width),
            mode="bilinear", align_corners=False,
        )
        depth = disp_to_depth(
            scaled_disparity, self.hparams.min_depth, self.hparams.max_depth,
        )[1]

        point_cloud = self.backprojections(depth, self.inv_intrinsics)
        pixel_coordinates: Tensor = self.projections(
            point_cloud, self.intrinsics, transformation,
        )
        return grid_sample(
            image, pixel_coordinates,
            padding_mode="border", align_corners=False,
        )

    def _compute_scale_reprojection_loss(
        self, inputs: Tensor, poses: Dict[int, Tuple[Tensor, Tensor]],
        scale_disparity: Tensor,
    ) -> Tensor:
        """
        Compute reprojection loss between
        scaled disparity (upscaled to original input size)
        and target image.
        """
        scale_reprojection_loss: Tensor = None
        for bsid in self.batch_sources_id:
            axisangle, translation = poses[bsid]
            transformation = transformation_from_parameters(
                axisangle, translation, invert=bsid < self.batch_target_id,
            )
            warped = self._warp_image(
                inputs[:, bsid], scale_disparity, transformation,
            )
            reprojection_loss = self._reprojection_loss(
                inputs[:, self.batch_target_id], warped,
            )
            if scale_reprojection_loss is None:
                scale_reprojection_loss = reprojection_loss
                continue
            scale_reprojection_loss = tmin(cat(
                (scale_reprojection_loss, reprojection_loss), dim=1),
                dim=1, keepdim=True,
            )[0]
        return scale_reprojection_loss

    def _compute_photometric_losses(
        self, inputs: Tensor, disparities: Dict[int, Tensor],
        poses: Dict[int, Tuple[Tensor, Tensor]],
    ) -> Tensor:
        identity_reprojection_loss = (
            self._compute_identity_reprojection_loss(inputs)
        )
        total_loss: Tensor = 0
        for scale in self.hparams.scales:
            scale_disparity = disparities[scale]
            scale_reprojection_loss = self._compute_scale_reprojection_loss(
                inputs, poses, scale_disparity,
            )
            total_loss += tmin(cat(
                (identity_reprojection_loss, scale_reprojection_loss), dim=1,
            ), dim=1)[0].mean()
            total_loss += self._compute_smooth_loss(
                inputs, scale_disparity, scale,
            )
        total_loss /= len(self.hparams.scales)
        return total_loss

    def _velocity_loss(
        self,
        poses: Dict[int, Tuple[Tensor, Tensor, Tensor]], velocities: Tensor,
    ) -> Tensor:
        target_velocities = velocities[:, [
            self.batch_target_id, self.batch_sources_id[1]
        ]]
        estimated_velocities = cat([
            poses[bsid][2] for bsid in self.batch_sources_id
        ]).unsqueeze_(0)
        return mse_loss(estimated_velocities, target_velocities)

    def _pose_constraint(
        self,
        poses: Dict[int, Tuple[Tensor, Tensor]], velocities: Tensor,
    ) -> Tensor:
        constraint: Tensor = 0
        starting_point = zeros(1, 4, 1, dtype=float32).to(velocities.device)
        starting_point[:, 3] = 1.0

        for bsid, sid in zip(self.batch_sources_id, self.hparams.sources_ids):
            invert = bsid < self.batch_target_id

            target_velocity = velocities[
                :, [self.batch_target_id if invert else bsid]
            ]
            time_delta = abs(self.hparams.target_id - sid)
            distance = target_velocity * self.hparams.dt * time_delta * 0.01

            axisangle, translation = poses[bsid]
            transformation = transformation_from_parameters(
                axisangle, translation, invert=invert,
            )
            transformed_point = matmul(transformation, starting_point)

            constraint += mse_loss(
                norm(transformed_point[:, :3], dim=1), distance,
            )
        constraint /= len(self.batch_sources_id)
        return constraint

    def _pose_constraint_z(
        self,
        poses: Dict[int, Tuple[Tensor, Tensor]], velocities: Tensor,
    ) -> Tensor:
        tvid = [
            self.batch_target_id if bsid < self.batch_target_id else bsid
            for bsid in self.batch_sources_id
        ]
        distances = (
            velocities[:, tvid] * self.hparams.dt * self.time_delta * 0.01
        )
        translations = cat(
            [poses[bsid][1] for bsid in self.batch_sources_id], dim=1,
        )
        return (
            mse_loss(tabs(translations[..., 2]), distances)
            + mse_loss(norm(translations, dim=2), distances ** 2)
        )

    def configure_optimizers(self):
        train_parameters = (
            list(self.encoder.parameters())
            + list(self.depth_decoder.parameters())
            + list(self.pose_decoder.parameters())
        )
        optimizer = Adam(train_parameters, self.hparams.lr)
        scheduler = lr_scheduler.StepLR(optimizer, self.hparams.step_size, 0.1)
        return [optimizer], [scheduler]

    def prepare_data(self):
        datasets = []
        for config in datasets_config(self.hparams):
            if "targets" in config:
                targets = from_numpy(fromfile(
                    config["targets"], dtype="float32", sep="\n",
                ))
                datasets.append(SequenceData.target_dataset(
                    config["frame_template"], targets, self.hparams,
                ))
            else:
                datasets.append(SequenceData.no_target_dataset(
                    config["frame_template"], config["length"], self.hparams,
                ))
        self.train_dataset = ConcatDataset(datasets)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, self.hparams.batch_size,
            shuffle=True, drop_last=True,
        )

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        inputs, velocities = batch
        disparities, poses = self.forward(inputs)
        loss = self._compute_photometric_losses(inputs, disparities, poses)

        log = {"loss": loss}
        # if (velocities != -1).all():
        #     velocity_loss = self._velocity_loss(poses, velocities)
        #     log["velocity_loss"] = velocity_loss
        #     loss += velocity_loss * 0.0001
        # if (velocities != -1).all():
        #     pose_constraint = self._pose_constraint_z(poses, velocities)
        #     log["pose_constraint"] = pose_constraint
        #     loss += pose_constraint

        if batch_idx % 100 == 0:
            base = r"C:\Users\tonys\projects\python\comma\effdepth-models\disp"
            disp_path = join(base, f"disp-{self.global_step}.jpg")

            disp_image = disparities[0].detach()
            disp_image = disp_image.cpu().numpy()
            disp_image = round_(disp_image[0, 0] * 255).astype("uint8")
            imsave(disp_path, disp_image, check_contrast=False)

            for bsid in self.batch_sources_id:
                rotation, translation = poses[bsid]
                transformation = transformation_from_parameters(
                    rotation, translation, invert=bsid < self.batch_target_id,
                )
                warped_image = self._warp_image(
                    inputs[:, bsid], disparities[0], transformation,
                )
                warped_image = warped_image.detach().cpu().numpy()[0]
                warped_image = transpose(warped_image, (1, 2, 0))
                warped_image = round_(warped_image * 255).astype("uint8")

                warp_path = join(base, f"warp-{self.global_step}-{bsid}.jpg")
                imsave(warp_path, warped_image, check_contrast=False)

            print(
                "R", rotation.detach().cpu().numpy()[0, 0],
                " | t", translation.detach().cpu().numpy()[0, 0],
            )
            print("Target velocity", velocities[0, 0].detach().cpu().numpy())

        return {"loss": loss, "log": log}


def main():
    hparams = Namespace(
        pretrained=True,
        encoder_layers=18,
        scales=[0, 1, 2, 3],
        input_images=1,  # how many images in channel dim to feed to encoder
        pose_sequence_length=2,  # how many frames to feed to pose NN at once
        disparity_smoothness=1e-3,  # control sharpness of disparity
        lr=1e-4, step_size=10, batch_size=4,
        height=160, width=320,  # have to be divisible by 2**5
        min_depth=0.1, max_depth=100.0,
        target_id=4, sources_ids=[0, 8], sequence_length=9,
        # kitti at 10 hz, maybe increase length
        device="cuda", dt=1 / 20,
    )
    loggin_dir = r"C:\Users\tonys\projects\python\comma\effdepth-models"
    checkpoint_path = join(loggin_dir, r"manual-velocity\depth-{epoch:02d}")
    model = EffDepthTraining(hparams)
    # load_checkpoint_path = join(
    #     loggin_dir, r"manual-velocity-better\depth-epoch=01.ckpt",
    # )
    # model = EffDepthTraining.load_from_checkpoint(load_checkpoint_path)
    checkpoint_callback = ModelCheckpoint(checkpoint_path, save_top_k=-1)
    trainer = Trainer(
        logger=TensorBoardLogger(loggin_dir),
        checkpoint_callback=checkpoint_callback, early_stop_callback=False,
        max_epochs=20,
        accumulate_grad_batches=3,
        gpus=1 if hparams.device == "cuda" else 0,
        benchmark=True,
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
