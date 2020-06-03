from argparse import Namespace
from os.path import join
from typing import List, Dict, Tuple

from skimage.io import imsave
from numpy import fromfile, transpose, round_, sort, argwhere, array
from numpy.random import seed
from torch import (
    Tensor, abs as tabs, cat, randn, min as tmin, from_numpy, manual_seed,
    norm, float32, tensor,
)
from torch.nn.functional import interpolate, grid_sample, mse_loss
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam, lr_scheduler
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from effdepth.models.original import ResnetEncoder, DepthDecoder, PoseDecoder
from effdepth.models.velocity_decoder import VelocityDecoder
from effdepth.models.layers import (
    SSIM, BackprojectDepth, Project3D, disp_to_depth,
    transformation_from_parameters, get_smooth_loss,
)
from effdepth.dataset import SequenceData, compute_intrinsics, datasets_config


seed(0)
manual_seed(0)


class DepthTraining(LightningModule):
    def __init__(self, hparams: Namespace):
        super().__init__()
        self.hparams = hparams

        self.encoder = ResnetEncoder(
            hparams.encoder_layers, hparams.pretrained,
        )
        self.depth_decoder = DepthDecoder(self.encoder.layers, hparams.scales)
        self.pose_decoder = PoseDecoder(self.encoder.layers[-1])
        self.velocity_decoder = VelocityDecoder(self.encoder.layers[-1])

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
    ) -> Tuple[
        Dict[int, Tensor],
        Dict[int, Tuple[Tensor, Tensor, Tensor]],
        Dict[int, Tensor]
    ]:
        features = self.extract_features(inputs)
        disparities: Dict[int, Tensor] = self.depth_decoder([
            feature[:, self.batch_target_id] for feature in features
        ])
        poses: Dict[int, Tuple[Tensor, Tensor, Tensor]] = self.estimate_poses(
            features[-1],
        )
        velocities: Dict[int, Tensor] = self.estimate_velocities(features[-1])
        return disparities, poses, velocities

    def extract_features(self, inputs: Tensor) -> List[Tensor]:
        b, l, c, w, h = inputs.shape
        inputs = inputs.view(b * l, c, w, h)
        features: List[Tensor] = []
        for feature in self.encoder(inputs):
            _, ce, he, we = feature.shape
            features.append(feature.view(b, l, ce, he, we))
        return features

    def estimate_poses(
        self, features: Tensor,
    ) -> Dict[int, Tuple[Tensor, Tensor]]:
        """
        Estimate poses between each (prev -> target) & (target -> next) frames
        given their respective ids.
        *Note* that `prev` and `next` do not need to be adjacent w.r.t. time,
        but should be not "too far", so that estimation can be done.

        Arguments:
            features (Tensor):
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
        poses: Dict[int, Tuple[Tensor, Tensor]] = {}
        for bsid in self.batch_sources_id:
            pose_inputs = (
                [features[:, bsid], features[:, self.batch_target_id]]
                if bsid < self.batch_target_id else
                [features[:, self.batch_target_id], features[:, bsid]]
            )
            poses[bsid] = self.pose_decoder(pose_inputs)
        return poses

    def estimate_velocities(self, features: Tensor) -> Dict[int, Tensor]:
        velocities: Dict[int, Tensor] = {}
        for bsid in self.batch_sources_id:
            features_inputs = (
                [features[:, bsid], features[:, self.batch_target_id]]
                if bsid < self.batch_target_id else
                [features[:, self.batch_target_id], features[:, bsid]]
            )
            velocities[bsid] = self.velocity_decoder(features_inputs)
        return velocities

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
        self, estimated_velocities: Dict[int, Tensor], velocities: Tensor,
    ) -> Tensor:
        # target_velocities = velocities[:, [self.batch_target_id, self.batch_sources_id[1]]]
        # estimated_velocities = cat([
        #     estimated_velocities[bsid] for bsid in self.batch_sources_id
        # ], dim=1)
        target_velocities = cat([
            velocities[:, [self.batch_target_id]],
            velocities[:, [self.batch_sources_id[1]]],
        ], dim=0)
        estimated_velocities = cat([
            estimated_velocities[bsid] for bsid in self.batch_sources_id
        ], dim=0)
        return mse_loss(estimated_velocities, target_velocities)

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
            + list(self.velocity_decoder.parameters())
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
        inputs, target_velocities = batch
        disparities, poses, estimated_velocities = self.forward(inputs)
        loss: Tensor = 0
        loss += self._compute_photometric_losses(inputs, disparities, poses)
        log = {"loss": loss}

        if (target_velocities != -1).all():
            velocity_loss = self._velocity_loss(
                estimated_velocities, target_velocities,
            )
            loss += velocity_loss
            log["velocity_loss"] = velocity_loss

            # pose_constraint = self._pose_constraint_z(poses, target_velocities)
            # loss += pose_constraint
            # log["pose_constraint"] = pose_constraint

        if batch_idx % 100 == 0:
            base = r"C:\Users\tonys\projects\python\comma\effdepth-models\disp"
            disp_path = join(base, f"disp-{self.global_step}.jpg")

            disp_image = disparities[0].detach().cpu().numpy()
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

            # print("R", rotation.detach().cpu().numpy()[0, 0])
            # print("t", translation.detach().cpu().numpy()[0, 0])
            # print(
            #     "Target velocity",
            #     target_velocities[0, 0].detach().cpu().numpy(),
            # )
            # print(
            #     "Estimated velocity",
            #     estimated_velocities[0][0, 0].detach().cpu().numpy(),
            # )

        return {"loss": loss, "log": log}


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
    )
    loggin_dir = r"C:\Users\tonys\projects\python\comma\effdepth-models"
    checkpoint_path = join(
        loggin_dir, r"velo-160x320\depth-{epoch:02d}",
    )
    # load_checkpoint_path = join(
    #     loggin_dir, r"manual-velocity-better\depth-epoch=01.ckpt",
    # )
    # model = EffDepthTraining.load_from_checkpoint(load_checkpoint_path)
    model = DepthTraining(hparams)
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
