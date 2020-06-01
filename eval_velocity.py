from argparse import Namespace
from os.path import join

from numpy import array, savetxt
from torch import device
from tqdm import tqdm
from matplotlib import pyplot

from dataset import SequenceData
from train import EffDepthTraining


def velocity():
    dev = device("cuda")

    loggin_dir = r"C:\Users\tonys\projects\python\comma\effdepth-models"
    checkpoint_path = join(loggin_dir, r"manual-velocity\depth-epoch=00.ckpt")
    model = EffDepthTraining.load_from_checkpoint(checkpoint_path)
    model = model.to(dev)
    print(model.hparams.sequence_length)

    frame_template = (
        r"C:\Users\tonys\projects\python\comma\speedchallenge"
        r"\test\frames-192x640\frame-{:05}.jpg"
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
            velocities.extend(poses[bsid][2].detach().cpu().numpy())

    savetxt(
        join(loggin_dir, "test-00.txt"),
        array(velocities), delimiter="\n", fmt="%.7f",
    )
    pyplot.plot(velocities)
    pyplot.show()


def poses():
    dev = device("cuda")

    loggin_dir = r"C:\Users\tonys\projects\python\comma\effdepth-models"
    checkpoint_path = join(loggin_dir, r"manual-velocity\depth-epoch=16.ckpt")
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
    # velocity()
    poses()
