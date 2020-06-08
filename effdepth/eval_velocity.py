from os.path import join

from cv2 import (
    VideoCapture, VideoWriter, VideoWriter_fourcc, putText,
    FONT_HERSHEY_SIMPLEX,
)

from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

from numpy import array, savetxt, ndarray, dot, diag, loadtxt
from matplotlib import pyplot
from torch import device
from tqdm import tqdm

from effdepth.dataset import SequenceData, load_targets
from effdepth.training.train import DepthTraining


def state_to_measurement(state: ndarray) -> ndarray:
    return state[[1]]


def state_transition(state: ndarray, dt: float) -> ndarray:
    transition = array([
        [1, dt, dt * dt],
        [0, 1, dt],
        [0, 0, 1],
    ], dtype="float32")
    return dot(transition, state)


def smooth(data: ndarray, dt: float):
    points = MerweScaledSigmaPoints(3, alpha=1e-3, beta=2.0, kappa=0)
    noisy_kalman = UnscentedKalmanFilter(
        dim_x=3, dim_z=1, dt=dt,
        hx=state_to_measurement, fx=state_transition, points=points,
    )

    noisy_kalman.x = array([0, data[1], data[1] - data[0]], dtype="float32")
    noisy_kalman.R *= 20 ** 2  # sensor variance
    noisy_kalman.P = diag([5 ** 2, 5 ** 2, 1 ** 2])  # variance of the system
    noisy_kalman.Q = Q_discrete_white_noise(3, dt=dt, var=0.05)

    means, covariances = noisy_kalman.batch_filter(data)
    means[:, 1][means[:, 1] < 0] = 0  # clip velocity
    return means[:, 1]


def draw_speed(video_path: str, speed_path: str, output_video: str) -> None:
    reader = VideoCapture(video_path)
    writer = VideoWriter(
        output_video, VideoWriter_fourcc(*"mp4v"), 20, (640, 480),
    )
    data = loadtxt(speed_path, delimiter="\n", dtype="float32")

    frame_id = 0
    while reader.isOpened():
        ret, frame = reader.read()
        if not ret:
            break

        putText(
            frame, f"{data[frame_id]:0.3f}",
            (250, 420), FONT_HERSHEY_SIMPLEX, 0.7,
            (255, 255, 255), 2,
        )
        writer.write(frame)

        frame_id += 1
        if frame_id == data.shape[0]:
            break

    reader.release()
    writer.release()


def predict_velocity():
    dev = device("cuda")

    loggin_dir = r"C:\Users\tonys\projects\python\comma\effdepth-models"
    checkpoint_path = join(loggin_dir, r"velo-chunk-12-160x320\depth-epoch=05.ckpt")
    model = DepthTraining.load_from_checkpoint(checkpoint_path).to(dev)
    print(model.hparams.sequence_length)

    frame_template = (
        r"C:\Users\tonys\projects\python\comma\speedchallenge"
        r"\test\frames-160x320\frame-{}.jpg"
    )
    targets = load_targets(r"C:\Users\tonys\Downloads\test.txt")
    dataset = SequenceData.target_dataset(
        frame_template, targets, model.hparams,
    )
    total_frames = targets.shape[0]
    ids = array([
        model.hparams.sources_ids[0],
        model.hparams.target_id,
        model.hparams.sources_ids[1],
    ], dtype="int64")

    velocities = []
    delta = model.hparams.sources_ids[1] - model.hparams.sources_ids[0]
    for i in tqdm(range(total_frames - delta)):
        inputs = dataset.load_images(ids).unsqueeze_(0).to(dev)
        ids += 1

        velo = model.estimate_velocities(model.extract_features(inputs)[-1])
        if i < model.hparams.target_id:
            for bsid in model.batch_sources_id:
                velocities.append(velo[bsid].detach().cpu()[0, 0])
        else:
            velocities.append(
                velo[model.batch_sources_id[1]].detach().cpu()[0, 0]
            )

    velocities = array(velocities)
    smoothed_velocities = smooth(
        velocities, dt=model.hparams.dt * model.hparams.target_id,
    )

    print(f"Smoothed: {smoothed_velocities.shape[0]}, Submitted: {targets.shape[0]}")

    save_path = (
        r"C:\Users\tonys\projects\python\comma\effdepth-models\velo-chunk-12-160x320"
        r"\velocity-00_v1.txt"
    )
    savetxt(save_path, smoothed_velocities, "%.08f", "\n")

    print("Drawing speed...")
    draw_speed(
        r"C:\Users\tonys\projects\python\comma\speedchallenge\test\test.mp4",
        save_path,
        r"C:\Users\tonys\projects\python\comma\speedchallenge\test\test-speed.mp4"
    )
    print("Done.")

    pyplot.plot(velocities, label="Estimates")
    pyplot.plot(smoothed_velocities, label="Smoothed")
    pyplot.plot(targets, label="Submitted")
    pyplot.legend()
    pyplot.show()


if __name__ == "__main__":
    predict_velocity()
