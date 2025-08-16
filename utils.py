import numpy as np
import torch
import torch.nn as nn
import av

def read_video_pyav(container, indices, clip_len=16):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]

    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)

    frames = [x.to_ndarray(format="rgb24") for x in frames]

    # If fewer frames were decoded than desired, duplicate the last frame until reaching clip_len
    if len(frames) < clip_len:
        if len(frames) == 0:
            raise ValueError("No frames decoded from video.")

        last_frame = frames[-1]

        while len(frames) < clip_len:
            frames.append(last_frame)

    return np.stack(frames) # (num_frames, height, width, 3)


def sample_frame_indices(clip_len, seg_len, max_len=960):
    # Uniform positions across the longest video
    base_indices = np.linspace(0, max_len - 1, num=clip_len)
    
    # Round to nearest frame and cast to ints
    sampled = np.round(base_indices).astype(np.int64)
    
    # Any index beyond the actual video length seg_len will result in padding with last frame
    sampled = np.where(sampled < seg_len, sampled, seg_len - 1)
    
    return sampled

"""
def sample_frame_indices(clip_len, frame_sample_rate, seg_len, jitter_factor=0.5):
    # Uniformly sample indices over the full video duration with temporal jittering
    
    # If the video is too short, sample what is available and pad with the last frame
    if seg_len < clip_len:
        indices = np.concatenate([np.arange(seg_len), np.full((clip_len - seg_len), seg_len - 1)])
        return indices.astype(np.int64)
    
    # Uniformly space clip_len indices from 0 to seg_len-1
    base_indices = np.linspace(0, seg_len - 1, num=clip_len)
    
    # Define a jitter range based on the frame_sample_rate and jitter_factor
    max_jitter = frame_sample_rate * jitter_factor
    
    # Sample a random offset for each index from a uniform distribution in [-max_jitter, max_jitter]
    jitter = np.random.uniform(-max_jitter, max_jitter, size=clip_len)
    
    # Apply the jitter to the uniform indices
    jittered_indices = base_indices + jitter
    
    # Round to the nearest integer and ensure all indices fall within valid range [0, seg_len-1]
    jittered_indices = np.round(jittered_indices).astype(np.int64)
    jittered_indices = np.clip(jittered_indices, 0, seg_len - 1)

    return jittered_indices
"""

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

    with torch.no_grad():
        for batch in test_loader:
            batch_video_paths = batch["video_path"]
            
            # Convert list -> tensor
            batch_labels = torch.tensor(batch["label"], dtype=torch.long).to(device)
            
            # This will hold the processed inputs for the entire batch
            batch_pixel_values = []

            for i in range(len(batch_video_paths)):
                video_path = batch_video_paths[i]

                # Open the video
                container = av.open(video_path)

                # Sample frames from the video
                seg_len = container.streams.video[0].frames

                indices = sample_frame_indices(
                    clip_len=16, seg_len=seg_len
                )

                video_frames = read_video_pyav(container=container, indices=indices)

                # Preprocess frames with the image processor
                inputs = image_processor([list(video_frames)], return_tensors="pt")

                if hasattr(image_processor, "video_processor_type"):
                    # Store pre-processed pixel_values to feed into the model in a single big batch
                    batch_pixel_values.append(inputs["pixel_values_videos"][0])
                else: 
                    # Store pre-processed pixel_values to feed into the model in a single big batch
                    batch_pixel_values.append(inputs["pixel_values"][0])


            # Stack pixel values for each video along the batch dimension:
            batch_pixel_values = torch.stack(batch_pixel_values, dim=0).to(device)  # shape [B, T, C, H, W]

            # Forward pass
            if hasattr(image_processor, "video_processor_type"):
                output = model(pixel_values_videos=batch_pixel_values)
            else:
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
