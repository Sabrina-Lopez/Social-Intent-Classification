import av
import numpy as np
from torch.utils.data import Dataset
from utils import read_video_pyav

class VideoDataset(Dataset):
    def __init__(self, hf_dataset, image_processor, sample_frame_indices_fn, transform=None):
        # hf_dataset is a slice of the main dataset
        self.dataset = hf_dataset
        self.image_processor = image_processor
        self.sample_frame_indices = sample_frame_indices_fn
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        video_path = item["video_path"]
        label = item["label"]
        start = item["start"]
        end = item["end"]


        container = av.open(video_path)
        stream = container.streams.video[0]
        fps, start_frame, end_frame = None, None, None

        if stream.average_rate is not None:
            fps = float(stream.average_rate)
        else:
            fps = 30.0

        if end is None:
            start_frame = 0
            end_frame = stream.frames
        else:
            start_frame = int(start * fps)
            end_frame = int(end * fps)


        seg_len = end_frame - start_frame

        indices = self.sample_frame_indices(
                clip_len=16, frame_rate_bool=False, frame_sample_rate=4, seg_len=seg_len
        )

        frames = read_video_pyav(container, indices)  # your custom read func

        # Convert frames -> pixel_values
        inputs = self.image_processor(list(frames), return_tensors="pt")
        pixel_values = inputs["pixel_values"][0]  # shape [T, C, H, W]

        return {
            "pixel_values": pixel_values,  # torch.Tensor
            "label": label
        }
