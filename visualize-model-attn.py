import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import argparse
import math
from tqdm import tqdm
from transformers import AutoVideoProcessor, VJEPA2ForVideoClassification, VJEPA2Config
from datasets import load_dataset
from torch.utils.data import DataLoader
from utils import read_video_pyav, preprocess_function, custom_collate_fn, sample_frame_indices
from scipy.ndimage import zoom
import av
import torch.nn.functional as F


def overlay_attention_on_video(frames, attn_maps, output_path, fps):
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    n_frames = min(len(frames), len(attn_maps))
    for i in range(n_frames):
        frame_np = frames[i].astype(np.float32) / 255.0

        cam = attn_maps[i]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
        cam_overlay = (cam * 255).astype(np.uint8)
        cam_overlay = cv2.applyColorMap(cam_overlay, cv2.COLORMAP_JET)
        cam_overlay = cv2.cvtColor(cam_overlay, cv2.COLOR_BGR2RGB)

        # Resize attention overlay to match original frame size
        cam_overlay = cv2.resize(cam_overlay, (width, height), interpolation=cv2.INTER_CUBIC)
        
        overlayed = (0.5 * frame_np + 0.5 * cam_overlay / 255.0)
        overlayed = (overlayed * 255).astype(np.uint8)
        writer.write(cv2.cvtColor(overlayed, cv2.COLOR_RGB2BGR))

    writer.release()


def vjepa2_process_chunks_and_visualize(model, frames, video_path, image_processor, output_dir, head_selection, fps):
    device = next(model.parameters()).device
    inputs = image_processor([frames], return_tensors="pt").to(device)

    print(frames[0].shape)  # (700, 700, 3)

    # ---- geometry from config ----
    cfg = model.config
    S = cfg.image_size          # e.g., 256
    P = cfg.patch_size          # e.g., 16
    tube = cfg.tubelet_size     # e.g., 2
    T = cfg.num_frames          # e.g., 16
    H_p = S // P                # e.g., 16
    W_p = S // P                # e.g., 16
    T_tube = T // tube          # e.g., 8

    with torch.no_grad():
        outputs = model(
            pixel_values_videos=inputs["pixel_values_videos"],
            output_attentions=True,
            output_hidden_states=True,   # not strictly needed below, but handy if you switch saliency type
            return_dict=True
        )

        # Last attention: [B, H, N, N]
        attn = outputs.attentions[-1]    # keep on device for now
        B, H, N, _ = attn.shape

        # Select/aggregate heads
        if head_selection == "mean":
            A = attn.mean(dim=1)                 # [B, N, N]
        elif head_selection == "max":
            A = attn.max(dim=1).values           # [B, N, N]
        else:
            h = int(head_selection)
            if not (0 <= h < H):
                raise ValueError(f"head_selection={head_selection} out of range [0,{H-1}]")
            A = attn[:, h]                        # [B, N, N]

        # Incoming attention per token (avg over all queries)
        token_scores = A.mean(dim=1)              # [B, N]

    # Sanity check and reshape to tubelet grids
    expected_N = T_tube * H_p * W_p
    if token_scores.shape[1] != expected_N:
        raise RuntimeError(
            f"Seq len mismatch: N={token_scores.shape[1]} vs {T_tube}*{H_p}*{W_p}={expected_N}. "
            "Check num_frames/tubelet_size/patch_size/image_size."
        )

    scores = token_scores[0]                                  # [N]
    tubelet_grids = scores.view(T_tube, H_p, W_p)             # [T_tube, H_p, W_p]

    # Normalize per tubelet for stable colors
    tubelet_grids = (tubelet_grids - tubelet_grids.amin(dim=(1,2), keepdim=True)) / (
        tubelet_grids.amax(dim=(1,2), keepdim=True) - tubelet_grids.amin(dim=(1,2), keepdim=True) + 1e-6
    )

    tubelet_grids = tubelet_grids.detach().cpu().numpy()      # -> numpy for OpenCV

    # ---- Upsample spatially and temporally to per-frame maps ----
    upsampled_attn_maps = []
    for i in range(T_tube):
        upsampled = cv2.resize(tubelet_grids[i], (S, S), interpolation=cv2.INTER_CUBIC)
        # Optional: light smoothing to avoid blockiness
        # upsampled = cv2.GaussianBlur(upsampled, (0,0), sigmaX=1.0, sigmaY=1.0, borderType=cv2.BORDER_REFLECT101)
        upsampled = np.clip(upsampled, a_min=0, a_max=None)
        # Duplicate for each frame covered by this tubelet
        for _ in range(tube):
            upsampled_attn_maps.append(upsampled.copy())

    attn_maps = np.stack(upsampled_attn_maps, axis=0)         # [T, S, S]

    # ---- Overlay and save ----
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(output_dir, f"{video_name}_attn_{head_selection}.mp4")
    os.makedirs(output_dir, exist_ok=True)
    overlay_attention_on_video(frames, attn_maps, out_path, fps)

    np.save(os.path.join(output_dir, f"{video_name}_attn_{head_selection}.npy"), attn_maps)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="test_data_100_unmod_latest.csv")
    # parser.add_argument("--model_name", type=str, default="./saved_model_dir/facebook/vjepa2-vitl-fpc32-256-diving48-default-finetuned-train_data_100_unmod_latest-method-default-bs-4-k-1-hd-00-ad-00-lr-1e-05-wd-005-ls-00-sd-00_1753925744")
    parser.add_argument("--model_name", type=str, default="./saved_model_dir/facebook/vjepa2-vitl-fpc16-256-ssv2-default-finetuned-train_data_100_unmod_latest-method-default-bs-4-k-1-hd-00-ad-00-lr-5e-05-wd-001-ls-00-sd-00_1755214499")
    parser.add_argument("--output_dir", type=str, default="./attn_outputs")
    parser.add_argument("--head_selection", type=str, default="mean", help="mean, max or head index")
    parser.add_argument("--test_output_name", type=str, default="test_results.txt", help="Name of text file name where testing information is stored")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation"
    )
    args = parser.parse_args()

    dataset = load_dataset('csv', data_files={"test": args.data_file}, sep=',')
    dataset["test"] = dataset["test"].map(preprocess_function)
    test_dataset = dataset["test"].with_format("torch", columns=["video_path", "start", "end", "label"])

    batch_size = args.batch_size
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn
    )

    label2id = {"help": 0, "hinder": 1, "physical": 2}
    id2label = {0: "help", 1: "hinder", 2: "physical"}

    base_model_name = args.model_name
    config = None

    # Load the base config from the pretrained checkpoint
    config = VJEPA2Config.from_pretrained(base_model_name, attn_implementation="eager")

    # Ensure config has the correct configurations
    config.num_labels = len(label2id)
    config.label2id = label2id
    config.id2label = id2label
    config.num_frames = 16

    # Get the pretrained image processor and model
    image_processor = AutoVideoProcessor.from_pretrained(base_model_name)
    model = VJEPA2ForVideoClassification.from_pretrained(base_model_name, config=config, ignore_mismatched_sizes=True)

    # print(model.config)

    model.eval()
    model_name = "vjepa2"
    if ("finetuned-train" in base_model_name):
        model_name = os.path.basename(args.model_name.rstrip("/")) # Gets the substring of the model's saved directory path
    # print(model_name)
    output_dir_folder = f"attn_{args.head_selection}_{model_name}"
    os.makedirs(os.path.join(args.output_dir, output_dir_folder), exist_ok=True)

    
    for example in tqdm(test_dataset):
        path = example["video_path"]
        container = av.open(path)
        raw_frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        fps = float(container.streams.video[0].average_rate)
        seg_len = container.streams.video[0].frames

        # Compute sampled indices using the same parameters as training
        sample_indices = sample_frame_indices(clip_len=config.num_frames,
                                              seg_len=seg_len)
        
        sampled_frames = [raw_frames[i] for i in sample_indices]

        vjepa2_process_chunks_and_visualize(
            model=model,
            frames=sampled_frames,
            video_path=path,
            image_processor=image_processor,
            output_dir=os.path.join(args.output_dir, output_dir_folder),
            head_selection=args.head_selection,
            fps=fps
        )

        torch.cuda.empty_cache()
 

if __name__ == "__main__":
    main()