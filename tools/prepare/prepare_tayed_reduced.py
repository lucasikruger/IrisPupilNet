import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

# https://www.kaggle.com/datasets/shivanshpachnanda/teyed-reduced
# https://www.kaggle.com/code/shivanshpachnanda/dataset-access-help

DATA_PATH = Path("/media/agot-lkruger/X9 Pro/facu/facu/tesis/tayed_reduced")
NEW_SAVE_PATH = Path("/media/agot-lkruger/X9 Pro/facu/facu/tesis/tayed_reduced_frames")

for npz_path in tqdm(DATA_PATH.glob("*.npz"), total=280):
    video_name = npz_path.stem
    data = dict(np.load(npz_path))
    data_combine = np.zeros_like(data['video'], dtype=np.uint8)
    data["pupil_seg_2D.mp4_path"] = data["pupil_seg_2D.mp4_path"]*255
    data["iris_seg_2D.mp4_path"] = data["iris_seg_2D.mp4_path"]*255
    data["lid_seg_2D.mp4_path"] = data["lid_seg_2D.mp4_path"]*255
    data_combine[..., 0] = data["pupil_seg_2D.mp4_path"]
    data_combine[..., 1] = data["iris_seg_2D.mp4_path"]
    data_combine[..., 2] = data["lid_seg_2D.mp4_path"]
    data['combine'] = data_combine
    data['video'] = (data['video'][...,0]).astype(np.uint8)
    frame_index = data['frames']
    for key in data.keys():
        if key == 'frames':
            continue
        for index, frame in enumerate(data[key]):
            frame_number = frame_index[index]
            new_path = NEW_SAVE_PATH / key / f"{video_name}_frame_{frame_number:05d}.png"
            new_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "L"
            if frame.shape[-1] == 3:
                mode = "RGB"
            img = Image.fromarray(frame, mode=mode)
            img.save(new_path)