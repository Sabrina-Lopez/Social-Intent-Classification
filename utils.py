import numpy as np
import torch
import torch.nn as nn
import av

def read_video_pyav(container, indices, clip_len=16):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    # Convert frames to numpy arrays
    frames = [x.to_ndarray(format="rgb24") for x in frames]

    # If fewer frames were decoded than desired, duplicate the last frame until reaching clip_len
    if len(frames) < clip_len:
        if len(frames) == 0:
            raise ValueError("No frames decoded from video!")
        last_frame = frames[-1]
        while len(frames) < clip_len:
            frames.append(last_frame)

    return np.stack(frames)


def sample_frame_indices(clip_len, frame_rate_bool, frame_sample_rate, seg_len):
    """
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    """
    indicies = None

    if frame_rate_bool:
        converted_len = int(clip_len * frame_sample_rate)

        if seg_len >= converted_len:
            end_idx = np.random.randint(converted_len, seg_len)
            start_idx = end_idx - converted_len
            indicies = np.linspace(start_idx, end_idx, num=clip_len)
            indicies = np.clip(indicies, start_idx, end_idx - 1).astype(np.int64)
        else:
            indicies = np.linspace(0, seg_len, num=clip_len)
            indicies = np.clip(indicies, 0, seg_len - 1).astype(np.int64)
    else:
        if seg_len >= clip_len:
            indicies = np.linspace(0, seg_len, num=clip_len)
            indicies = np.clip(indicies, 0, seg_len - 1).astype(np.int64)
        else:
            indicies = np.concatenate(
                [np.arange(seg_len), np.full((clip_len - seg_len), seg_len - 1)]
            ).astype(np.int64)

    return indicies


def preprocess_function(example):
    example["label"] = int(example["label"])
    return example


def custom_collate_fn(samples):
    batch = {key: [sample[key] for sample in samples] for key in samples[0]}
    return batch


def evaluate(model, test_loader, image_processor):
    model.eval()

    all_preds = []
    all_labels = []
    batch_losses = []
    batch_accuracies = []

    criterion = nn.CrossEntropyLoss()
    device = next(model.parameters()).device

    # Turn off gradient computation
    with torch.no_grad():
        for batch in test_loader:
            batch_video_paths = batch["video_path"]
            
            # Convert list -> tensor
            batch_labels = torch.tensor(batch["label"], dtype=torch.long)
            
            # This will hold the processed inputs for the entire batch
            batch_pixel_values = []

            for i in range(len(batch_video_paths)):
                video_path = batch_video_paths[i]

                # Open the video
                container = av.open(video_path)

                # Sample frames from the video
                seg_len = container.streams.video[0].frames

                indices = sample_frame_indices(
                    clip_len=16, frame_rate_bool=False, frame_sample_rate=4, seg_len=seg_len
                )

                video_frames = read_video_pyav(container=container, indices=indices)

                # Preprocess frames with the image processor
                inputs = image_processor([list(video_frames)], return_tensors="pt")

                # Store pre-processed pixel_values to feed into the model in a single big batch
                batch_pixel_values.append(inputs["pixel_values"][0])

            # Stack pixel values for each video along the batch dimension:
            batch_pixel_values = torch.stack(batch_pixel_values, dim=0).to(device)  # shape [B, T, C, H, W]

            # Forward pass
            output = model(pixel_values=batch_pixel_values)
            logits = output.logits  # shape [B, 3] if you have 3 classes
            predictions = torch.argmax(logits, dim=-1)

            batch_loss = criterion(logits, batch_labels)
            batch_losses.append(batch_loss.item())
            batch_accuracy = (predictions == batch_labels).float().mean().item()
            batch_accuracies.append(batch_accuracy)

            # Store preds & labels for evaluation
            all_preds.extend(predictions.cpu().tolist())
            all_labels.extend(batch_labels.cpu().tolist())

    return all_preds, all_labels, batch_losses, batch_accuracies
