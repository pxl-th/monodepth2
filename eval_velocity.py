from argparse import Namespace
from os.path import join

from numpy import array, savetxt
from torch import device
from tqdm import tqdm
from matplotlib import pyplot

from dataset import SequenceData
from train import EffDepthTraining


def main():
    dev = device("cuda")

    hparams = Namespace(
        pretrained=True,
        encoder_layers=18,
        scales=[0, 1, 2, 3],
        input_images=1,  # how many images in channel dim to feed to encoder
        pose_sequence_length=2,  # how many frames to feed to pose NN at once
        disparity_smoothness=1e-3,
        lr=3e-4, step_size=10, batch_size=1,
        height=192, width=640,
        min_depth=0.1, max_depth=100,
        target_id=4, sources_ids=[0, 8], sequence_length=9,
        device="cuda",
    )

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
        frame_template, 10798, hparams,
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


if __name__ == "__main__":
    main()
