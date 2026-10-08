"""Run frozen R3M on 16 frames from one success and one failure of each task.

This does not train anything, and it does not change the episode files or videos.
"""

import os

import h5py
import imageio.v3 as iio
import numpy as np
import torch
from r3m import load_r3m

DATA = r"C:\Users\sunee\oopsiedata-v0.1"
OUT = r"C:\Users\sunee\oopsiedata-work\r3m_two_tasks.pt"


def success_of(h5_path):
    with h5py.File(h5_path, "r") as h:
        ann = h["episode_annotations"]
        name = next(iter(ann.keys()))
        return float(ann[name].attrs["success"]), str(h.attrs["language_instruction"])


def video_for(h5_path):
    """Main camera only: table view for the oven, base view for the cups."""
    folder = os.path.dirname(h5_path)
    with h5py.File(h5_path, "r") as h:
        paths = h["observations/video_paths"]
        if "image" in paths:
            name = paths["image"][()].decode()
        else:
            name = paths["base_image"][()].decode()
    return os.path.join(folder, name)


def find_pair(task_dir):
    success_h5 = failure_h5 = None
    for dirpath, _, files in os.walk(task_dir):
        for name in files:
            if not name.endswith(".h5"):
                continue
            path = os.path.join(dirpath, name)
            value, _ = success_of(path)
            if value == 1.0 and success_h5 is None:
                success_h5 = path
            elif value == 0.0 and failure_h5 is None:
                failure_h5 = path
            if success_h5 and failure_h5:
                return success_h5, failure_h5
    raise RuntimeError("Could not find both a success and a failure in " + task_dir)


def sample_16(video_path):
    # Count frames first. The file's fps times duration is not always exact.
    n = 0
    for _ in iio.imiter(video_path):
        n += 1
    if n < 2:
        raise RuntimeError("Video has too few frames: " + video_path)
    wanted = [int(x) for x in np.linspace(0, n - 1, 16)]
    wanted_set = set(wanted)
    grabbed = {}
    for index, frame in enumerate(iio.imiter(video_path)):
        if index in wanted_set and index not in grabbed:
            grabbed[index] = frame
        if len(grabbed) == len(wanted_set):
            break
    frames = [grabbed[i] for i in wanted]
    return np.stack(frames), wanted


def main():
    tasks = {
        "open_oven": os.path.join(DATA, r"labs\Araya\flower_open_oven"),
        "stack_cups": os.path.join(DATA, r"labs\Harvard_EML\stack-and-pack-cups"),
    }

    print("Loading frozen R3M resnet50. This does not train the model.")
    model = load_r3m("resnet50")
    model.eval()

    saved = {}
    for task, folder in tasks.items():
        success_h5, failure_h5 = find_pair(folder)
        for label, h5_path in (("success", success_h5), ("failure", failure_h5)):
            video = video_for(h5_path)
            frames, indexes = sample_16(video)
            # R3M expects pixel values from 0 to 255, shaped batch x 3 x height x width.
            batch = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
            with torch.no_grad():
                embedding = model(batch).detach().cpu()
            norms = embedding.norm(dim=1)
            first = embedding[0:1]
            drift = 1 - torch.nn.functional.cosine_similarity(embedding, first)
            key = task + "_" + label
            saved[key] = {
                "video": os.path.relpath(video, DATA),
                "frame_indexes": indexes,
                "embedding": embedding,
            }
            print()
            print(key)
            print("  video:", os.path.relpath(video, DATA))
            print("  frames used:", indexes)
            print("  embedding shape:", tuple(embedding.shape))
            moved = float((embedding[-1] - embedding[0]).norm())
            print("  vector length, first frame:", round(float(norms[0]), 2))
            print("  vector length, last frame:", round(float(norms[-1]), 2))
            print("  distance from first frame to last:", round(moved, 3))
            print("  drift from first frame to last (0 means same direction):", round(float(drift[-1]), 3))

    torch.save(saved, OUT)
    print()
    print("Saved embeddings to", OUT)


if __name__ == "__main__":
    main()
