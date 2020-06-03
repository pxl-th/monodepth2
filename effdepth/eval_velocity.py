from argparse import Namespace
from os.path import join

from numpy import array, savetxt, fromfile
from torch import device, from_numpy, tensor
from torch.nn.functional import mse_loss
from tqdm import tqdm
from matplotlib import pyplot

from dataset import SequenceData
from train import EffDepthTraining


def velocity():
    dev = device("cuda")

    loggin_dir = r"C:\Users\tonys\projects\python\comma\effdepth-models"
    checkpoint_path = join(
        loggin_dir,
        r"velo-and-constraint-z-160x320\depth-epoch=19.ckpt",
    )
    model = EffDepthTraining.load_from_checkpoint(checkpoint_path)
    model = model.to(dev)
    print(model.hparams.sequence_length)

    # frame_template = (
    #     r"C:\Users\tonys\projects\python\comma\speedchallenge"
    #     r"\test\frames-160x320\frame-{}.jpg"
    # )
    # test_dataset = SequenceData.no_target_dataset(
    #     frame_template, 10798, model.hparams,
    # )
    frame_template = (
        r"C:\Users\tonys\projects\python\comma\speedchallenge"
        r"\train\frames-160x320\frame-{}.jpg"
    )
    targets_path = (
        r"C:\Users\tonys\projects\python\comma\speedchallenge"
        r"\train\train.txt"
    )
    targets = from_numpy(fromfile(targets_path, dtype="float32", sep="\n"))
    test_dataset = SequenceData.target_dataset(
        frame_template, targets, model.hparams,
    )

    velocities = []
    target_velocities = []
    for i in tqdm(range(len(test_dataset))):
        inputs, tvelo = test_dataset[i]
        inputs = inputs.unsqueeze_(0).to(dev)
        velo = model.estimate_velocities(model.extract_features(inputs))

        for bsid in model.batch_sources_id:
            velocities.append(velo[bsid].detach().cpu()[0, 0])
            target_velocities.append(tvelo[
                model.batch_target_id if bsid < model.batch_target_id else bsid
            ].tolist())

    velocities = tensor(velocities)
    target_velocities = tensor(target_velocities)

    loss = mse_loss(velocities, target_velocities)
    pyplot.plot(velocities, label="Estimates")
    pyplot.plot(target_velocities, label="Targets")
    pyplot.title(f"Loss {loss}")
    pyplot.show()


def poses():
    dev = device("cuda")

    loggin_dir = r"C:\Users\tonys\projects\python\comma\effdepth-models"
    checkpoint_path = join(loggin_dir, r"constraint-z-160x320\depth-epoch=06.ckpt")
    model = EffDepthTraining.load_from_checkpoint(checkpoint_path)
    model = model.to(dev)
    print(model.hparams.sequence_length)

    frame_template = (
        r"C:\Users\tonys\projects\python\comma\speedchallenge"
        r"\test\frames-160x320\frame-{}.jpg"
    )
    test_dataset = SequenceData.no_target_dataset(
        frame_template, 10798, model.hparams,
    )

    velocities = []
    for i in tqdm(range(len(test_dataset))):
        inputs, _ = test_dataset[i]
        inputs = inputs.unsqueeze_(0).to(dev)
        _, poses = model(inputs)
        for bsid in model.batch_sources_id:
            translation = poses[bsid][1][:, 0, 2]
            velocities.extend(translation.detach().cpu().numpy())

    # savetxt(
    #     join(loggin_dir, "test-00.txt"),
    #     array(velocities), delimiter="\n", fmt="%.7f",
    # )
    pyplot.plot(velocities)
    pyplot.show()


if __name__ == "__main__":
    velocity()
    # poses()
