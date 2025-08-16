import os, cv2, numpy as np, random, hashlib
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

# Configuration 
augmentation_values = {
    'random_crop': [0.0, 0.5, 1.0],
    'random_flip': [0.0, 0.5, 1.0],
    'scale_jitter': [0.0, 0.5, 1.0],
    'color_jitter': [0.0, 0.5, 1.0],
    'rand_augment_layers': [0, 2, 4],
    'rand_augment_magnitude': [5, 10, 15, 20]
}

data_path_name = 'data_100_unmod_latest'
data_path = os.path.join('./', data_path_name)
output_path_base = 'data_100_unmod_latest_augmented'
os.makedirs(output_path_base, exist_ok=True)

class_folders = ['Help', 'Hinder', 'Physical']
splits = ['train', 'test']
VIDEO_EXTS = {'.mp4'}

# Helpers
def is_video(fname):
    return os.path.splitext(fname)[1] in VIDEO_EXTS

def decode_video_cv2(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], 30.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames, float(fps)

def encode_video_cv2(frames_bgr, output_path, fps):
    if not frames_bgr:
        return
    h, w = frames_bgr[0].shape[:2]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames_bgr:
        out.write(f)
    out.release()

def build_transform(aug_dict):
    """Compose PIL->PIL transforms. We’ll make them deterministic per-clip via seeding."""
    t = []

    # Scale jitter: up/down then crop back (keeps aspect; avoids stretching)
    if aug_dict['scale_jitter'] > 0:
        size = int(224 * (1 + aug_dict['scale_jitter']))
        t.append(T.Resize(size, interpolation=InterpolationMode.BILINEAR))

    # Random crop (as RandomResizedCrop with tight ratio to avoid distortion)
    if aug_dict['random_crop'] > 0:
        lo = max(0.05, 1.0 - aug_dict['random_crop'])  # keep some sensible min
        t.append(T.RandomResizedCrop(224, scale=(lo, 1.0), ratio=(0.95, 1.05)))
    else:
        # If we didn’t crop, standardize resolution
        t.append(T.Resize((224, 224), interpolation=InterpolationMode.BILINEAR))

    # Horizontal flip
    if aug_dict['random_flip'] > 0:
        t.append(T.RandomHorizontalFlip(p=aug_dict['random_flip']))

    # Color jitter
    if aug_dict['color_jitter'] > 0:
        c = aug_dict['color_jitter']
        t.append(T.ColorJitter(brightness=c, contrast=c, saturation=c, hue=min(0.1, c)))

    # RandAugment (kept last to act on 224x224)
    if aug_dict['rand_augment_layers'] > 0:
        t.append(T.RandAugment(num_ops=aug_dict['rand_augment_layers'],
                               magnitude=aug_dict['rand_augment_magnitude']))

    return T.Compose(t)

def apply_transform_clip(frames_pil, transform, seed):
    """Apply the same random params to every frame (deterministic per clip)."""
    out = []
    # Seed all RNGs before each frame so the same ops/params get used
    for img in frames_pil:
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed % (2**32 - 1))
            out_img = transform(img)
        # Guarantee size after all ops
        if out_img.size != (224, 224):
            out_img = F.resize(out_img, (224, 224), interpolation=InterpolationMode.BILINEAR)
        out.append(cv2.cvtColor(np.array(out_img), cv2.COLOR_RGB2BGR))
    return out

def is_effectively_noop(aug_type, val):
    if aug_type in ('random_crop', 'random_flip', 'scale_jitter', 'color_jitter') and val == 0.0:
        return True
    if aug_type == 'rand_augment_layers' and val == 0:
        return True
    # Magnitude alone without layers does nothing
    if aug_type == 'rand_augment_magnitude':
        return True
    return False

def stable_seed(*parts):
    s = '|'.join(map(str, parts))
    return int(hashlib.sha1(s.encode()).hexdigest(), 16) % (2**31)

# Main
for split in splits:
    for class_name in class_folders:
        input_folder = os.path.join(data_path, split, class_name)
        if not os.path.isdir(input_folder):
            continue

        for video in sorted(os.listdir(input_folder)):
            if not is_video(video): 
                continue
            video_path = os.path.join(input_folder, video)

            # Decode once; reuse frames for all variants
            frames_pil, fps = decode_video_cv2(video_path)
            if not frames_pil:
                print(f"[WARN] Could not read: {video_path}")
                continue

            # Individual augmentations
            for aug_type, values in augmentation_values.items():
                for val in values:
                    if is_effectively_noop(aug_type, val):
                        # Copy original instead of re-encoding (faster + no quality loss)
                        out_folder = os.path.join(output_path_base, f'{split}_{class_name}_{aug_type}_{val}')
                        os.makedirs(out_folder, exist_ok=True)
                        out_path = os.path.join(out_folder, f'{os.path.splitext(video)[0]}.mp4')
                        if not os.path.exists(out_path):
                            # Re-encode once to 224 if you want uniform size; else do a file copy
                            t_identity = T.Compose([T.Resize((224, 224), interpolation=InterpolationMode.BILINEAR)])
                            seed = stable_seed(video, aug_type, val)
                            frames_bgr = apply_transform_clip(frames_pil, t_identity, seed)
                            encode_video_cv2(frames_bgr, out_path, fps)
                        continue

                    # Build aug dict with all zeros except the one we’re sweeping
                    aug_dict = {k: 0 for k in augmentation_values}
                    aug_dict[aug_type] = val

                    transform = build_transform(aug_dict)
                    out_folder = os.path.join(output_path_base, f'{split}_{class_name}_{aug_type}_{val}')
                    os.makedirs(out_folder, exist_ok=True)
                    out_path = os.path.join(out_folder, f'{os.path.splitext(video)[0]}.mp4')

                    seed = stable_seed(video, aug_type, val)
                    frames_bgr = apply_transform_clip(frames_pil, transform, seed)
                    encode_video_cv2(frames_bgr, out_path, fps)

            # Combined augmentations (optional)
            """
            from itertools import product
            for combo in product(*augmentation_values.values()):
                aug_dict = dict(zip(augmentation_values.keys(), combo))
                transform = build_transform(aug_dict)
                combo_name = "_".join([f"{k}-{v}" for k, v in aug_dict.items()])
                out_folder = os.path.join(output_path_base, f'{split}_{class_name}_combo_{combo_name}')
                os.makedirs(out_folder, exist_ok=True)
                out_path = os.path.join(out_folder, f'{os.path.splitext(video)[0]}.mp4')
                seed = stable_seed(video, combo_name)
                frames_bgr = apply_transform_clip(frames_pil, transform, seed)
                encode_video_cv2(frames_bgr, out_path, fps)
            """