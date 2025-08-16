import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import argparse
import math
from tqdm import tqdm
from transformers import VivitImageProcessor, VivitForVideoClassification, VivitConfig
from transformers import AutoImageProcessor, TimesformerForVideoClassification, TimesformerConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
from utils import read_video_pyav, preprocess_function, custom_collate_fn, evaluate, sample_frame_indices
from scipy.ndimage import zoom
from sklearn.metrics import accuracy_score
import av


def get_cls_attention(attn_weights, head_selection="mean"):
    if head_selection == "mean":
        attn = attn_weights.mean(dim=1)
    elif head_selection == "max":
        attn = attn_weights.max(dim=1).values
    else:
        attn = attn_weights[:, int(head_selection)]

    cls_attn = attn[:, 0, 1:]
    return cls_attn


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


def vivit_process_chunks_and_visualize(model, frames, video_path, image_processor, output_dir, head_selection, fps):
    # Process the frames
    inputs = image_processor([frames], return_tensors="pt").to(model.device)

    # print(model)

    with torch.no_grad():
        outputs = model(pixel_values=inputs["pixel_values"], output_attentions=True)
        attn = outputs.attentions[-1].detach().cpu() 
        cls_attn = get_cls_attention(attn, head_selection=head_selection) 
    
    # print(outputs, attn.shape, cls_attn.shape) # outputs, torch.Size([1, 12, 1569, 1569]) torch.Size([1, 1568])
    # attn.shape = [B, heads, seq_len, seq_len]
        # seq_len = CLS + spatio-temporal tokens
    # cls_attn.shape = [B, seq_len - 1]
        # 16 frames --> 8 tubelets; each tubelet into 14x14 grid (224/16 = 14) --> 196 spatial tokens
        # total tokens = 8 * 196 = 1568, + 1 CLS token --> 1569

    model_frames = inputs["pixel_values"].shape[1]  
    tokens_total = cls_attn.shape[-1]
    tokens_per_frame = tokens_total // model_frames
    trimmed_tokens = tokens_per_frame * model_frames
    cls_attn = cls_attn[..., :trimmed_tokens].reshape(model_frames, tokens_per_frame)
    # print(inputs["pixel_values"].shape, tokens_total, tokens_per_frame, trimmed_tokens, cls_attn.shape) # torch.Size([1, 16, 3, 224, 224]) 1568 98 1568 torch.Size([16, 98])
    # inputs["pixel_values"].shape = [B, frames, C, H, W]
    # cls_attn.shape = [frames, tokens]
        # 1568/16 = 98 tokens/frame

    attn_maps = cls_attn.numpy()
    # print(attn_maps.shape) # (16, 98)

    # Each 2-frame tubelet has 196 spatial tokens → 98 = 196 / 2, so first regroup:
    tubelet_tokens = attn_maps.reshape(8, 196)  # 8 tubelets

    # Now reshape spatially
    tubelet_grids = tubelet_tokens.reshape(8, 14, 14)

    # Upsample spatially and temporally
    upsampled_attn_maps = []
    for i in range(8):
        upsampled = cv2.resize(tubelet_grids[i], (224, 224), interpolation=cv2.INTER_CUBIC)
        upsampled = np.clip(upsampled, a_min=0, a_max=None)  # Zero out all negative values
        # Fill two frames with the same upsampled map (due to tubelet_size temporal = 2)
        upsampled_attn_maps.extend([upsampled.copy(), upsampled.copy()])

    attn_maps = np.stack(upsampled_attn_maps)  # shape: (16, 224, 224)
    # print(attn_maps.shape)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(output_dir, f"{video_name}_attn_{head_selection}.mp4")
    overlay_attention_on_video(frames, attn_maps, out_path, fps)
    np.save(os.path.join(output_dir, f"{video_name}_attn_{head_selection}.npy"), attn_maps)


def timesformer_process_chunks_and_visualize(model, frames, video_path, image_processor, output_dir, head_selection, fps):
    # Process the frames
    inputs = image_processor([frames], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(pixel_values=inputs["pixel_values"], output_attentions=True)
        attn = outputs.attentions[-1].detach().cpu()
        
        # For TimeSformer, collapse over heads as needed
        # Here we use the same helper function to select the cls token attention
        cls_attn = get_cls_attention(attn, head_selection=head_selection)

    # print(outputs, attn.shape, cls_attn.shape) # outputs, torch.Size([16, 12, 197, 197]) torch.Size([16, 196])
    # attn.shape = [frames, heads, seq_len, seq_len]
        # seq_len = CLS + 196 (14 x 14 patches)
    # cls_attn.shape = [frames, tokens]
        # one row per frame, one weight per spatial patch
    
    attn_maps = cls_attn.numpy()
    # print(attn_maps.shape) # (16, 196)

    # Attempt reshaping to 2D
    grid_h, grid_w = 14, 14 
    assert grid_h * grid_w == attn_maps.shape[1], "Token shape mismatch with 196 tokens"

    upsampled_attn_maps = []
    for frame_attn in attn_maps:
        attn_grid = frame_attn.reshape(grid_h, grid_w)

        # Upsample to 224x224
        upsampled = cv2.resize(attn_grid, (224, 224), interpolation=cv2.INTER_CUBIC)
        upsampled = np.clip(upsampled, a_min=0, a_max=None)  # Zero out all negative values
        upsampled_attn_maps.append(upsampled)

    attn_maps = np.stack(upsampled_attn_maps)  # shape: (16, 224, 224)
    print(attn_maps.shape)
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(output_dir, f"{video_name}_attn_{head_selection}.mp4")
    overlay_attention_on_video(frames, attn_maps, out_path, fps)
    np.save(os.path.join(output_dir, f"{video_name}_attn_{head_selection}.npy"), attn_maps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="test_data_100_unmod_latest.csv")
    parser.add_argument("--model_name", type=str, default="./saved_model_dir/google/vivit-b-16x2-kinetics400-default-finetuned-train_data_100_unmod_latest-method-default-bs-4-k-1-hd-005-ad-015-lr-00001-wd-001-ls-00-sd-00_1753943189")
    # parser.add_argument("--model_name", type=str, default="./saved_model_dir/facebook/timesformer-base-finetuned-ssv2-default-finetuned-train_data_100_unmod_latest-method-default-bs-4-k-1-hd-00-ad-01-lr-1e-05-wd-005-ls-00-sd-00_1753948910")
    # parser.add_argument("--model_name", type=str, default="./saved_model_dir/facebook/timesformer-base-finetuned-k600-default-finetuned-train_data_100_unmod_latest-method-default-bs-4-k-1-hd-00-ad-005-lr-5e-05-wd-00001-ls-00-sd-00_1753957488")
    # parser.add_argument("--model_name", type=str, default="./saved_model_dir/facebook/timesformer-base-finetuned-k400-default-finetuned-train_data_100_unmod_latest-method-default-bs-4-k-1-hd-00-ad-01-lr-5e-05-wd-01-ls-00-sd-00_1753922975")
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
    if ("vivit" in args.model_name):
        config = VivitConfig.from_pretrained(base_model_name, attn_implementation="eager")
    elif ("timesformer" in args.model_name):
        config = TimesformerConfig.from_pretrained(base_model_name, attn_implementation="eager")

    # Ensure config has the correct configurations
    config.num_labels = len(label2id)
    config.label2id = label2id
    config.id2label = id2label
    config.num_frames = 16

    # Get the pretrained image processor and model
    if ("vivit" in args.model_name):
        image_processor = VivitImageProcessor.from_pretrained(base_model_name)
        model = VivitForVideoClassification.from_pretrained(base_model_name, config=config, ignore_mismatched_sizes=True)
    elif ("timesformer" in args.model_name):
        image_processor = AutoImageProcessor.from_pretrained(base_model_name)
        model = TimesformerForVideoClassification.from_pretrained(base_model_name, config=config, ignore_mismatched_sizes=True)

    # print(model.config)
    # print(image_processor, image_processor.crop_size["height"])

    model.eval()
    model_name = "vivit" if ("vivit" in base_model_name) else "timesformer"
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

        if ("vivit" in base_model_name):
            vivit_process_chunks_and_visualize(
                model=model,
                frames=sampled_frames,
                video_path=path,
                image_processor=image_processor,
                output_dir=os.path.join(args.output_dir, output_dir_folder),
                head_selection=args.head_selection,
                fps=fps
            )
        elif ("timesformer" in base_model_name):
            timesformer_process_chunks_and_visualize(
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