from os.path import join

from numpy import round_
from torch import device
from tqdm import tqdm
from skvideo.io import FFmpegWriter

from dataset import SequenceData
from train import EffDepthTraining


def main():
    dev = device("cuda")
    loggin_dir = r"C:\Users\tonys\projects\python\comma\effdepth-models"
    checkpoint_path = join(loggin_dir, r"manual-velocity\depth-epoch=16.ckpt")
    model = EffDepthTraining.load_from_checkpoint(checkpoint_path)
    model = model.to(dev)

    # frame_template = (
    #     r"C:\Users\tonys\projects\python\comma\2k19\2018-07-29--12-02-42"
    #     r"\30\frames-160x320\frame-{}.jpg"
    # )
    # output_path = (
    #     r"C:\Users\tonys\projects\python\comma\2k19\2018-07-29--12-02-42"
    #     r"\30\disparity.mp4"
    # )
    frame_template = (
        r"C:\Users\tonys\projects\python\comma\speedchallenge"
        r"\test\frames-160x320\frame-{}.jpg"
    )
    output_path = (
        r"C:\Users\tonys\projects\python\comma\speedchallenge"
        r"\test\output.mp4"
    )
    dataset = SequenceData.no_target_dataset(
        frame_template, 961, model.hparams,
    )

    writer = FFmpegWriter(output_path)
    for i in tqdm(range(961)):
        image = dataset.load_images([i]).to(dev)
        disparity = model.depth_decoder(model.encoder(image))[0]
        disparity = disparity.detach().cpu().numpy()[0, 0]
        disparity = round_(disparity * 255).astype("uint8")
        writer.writeFrame(disparity)
    writer.close()


if __name__ == "__main__":
    main()
