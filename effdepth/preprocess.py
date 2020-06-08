from os import makedirs, listdir
from os.path import join, exists
from subprocess import run
from typing import List, Tuple
from tqdm import tqdm

from skimage.io import imsave, imread
from skimage.transform import resize
from skvideo.io import ffprobe, vreader
from numpy import round_, load, linspace, savetxt, ndarray


def video_to_frames(video_path: str, frame_template: str):
    for i, frame in enumerate(vreader(video_path)):
        frame = resize(frame, (240, 320))
        frame = frame[10:170]
        frame = round_(frame * 255).astype("uint8")
        imsave(frame_template.format(i), frame, check_contrast=False)


def get_paths(base_dir: str, exist_tag: str = None) -> List[str]:
    paths = []
    for pdir in listdir(base_dir):
        apdir = join(base_dir, pdir)
        for sdir in listdir(apdir):
            fdir = join(apdir, sdir)
            if exist_tag is not None and exists(join(fdir, exist_tag)):
                continue
            paths.append(fdir)
    return paths


def hevc_to_mpeg(base_dir: str):
    input_file = "video.hevc"
    output_file = "video.mp4"
    command = "ffmpeg -i {} -q:v 0 -filter:v fps=fps=20 {}"
    for rec_dir in tqdm(listdir(base_dir)):
        for part_dir in tqdm(listdir(join(base_dir, rec_dir))):
            input_path = join(base_dir, rec_dir, part_dir, input_file)
            output_path = join(base_dir, rec_dir, part_dir, output_file)
            conversion_command = (
                command.format(input_path, output_path).split(" ")
            )
            run(conversion_command)


def comma(base_dir):
    speed_path = r"processed_log\CAN\speed"
    video_file = r"video.mp4"
    speeds_file = "value"
    output_speed_file = "speed.txt"

    folders_paths = get_paths(base_dir, output_speed_file)
    print(f"Total folders to process {len(folders_paths)}")

    bar = tqdm(folders_paths)
    for fpath in bar:
        video_path = join(fpath, video_file)
        frame_path = join(fpath, "frames-160x320")
        frame_template = join(frame_path, "frame-{}.jpg")
        makedirs(frame_path, exist_ok=True)
        # Convert video to frames.
        bar.write(f"Processing video {video_path}")
        video_to_frames(video_path=video_path, frame_template=frame_template)
        # Convert speed.
        video_meta = ffprobe(video_path)["video"]
        output_speed_path = join(fpath, output_speed_file)
        speeds = load(join(fpath, speed_path, speeds_file))
        video_frames = int(video_meta["@nb_frames"])

        speed_ids = round_(
            linspace(0, speeds.shape[0] - 1, num=video_frames),
        ).astype("int32")
        savetxt(output_speed_path, speeds[speed_ids], "%.8f", "\n")


if __name__ == "__main__":
    # base_dir = r"C:\Users\tonys\projects\python\comma\2k19"
    base_dir = r"C:\Users\tonys\Downloads\comma2k19\Chunk_2"
    # hevc_to_mpeg(base_dir)
    comma(base_dir)
