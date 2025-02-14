import os
import cv2
import numpy as np
import random
from PIL import Image
import torch
import torchvision.transforms.functional as TF


seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Video length and augmentation mode
vid_length = 33
mix_bool = False

# Setup data paths
data_path_name = 'data_' + str(vid_length) + '_unmod'
data_path = os.path.join('./', data_path_name)
train_folder = os.path.join(data_path, 'train')
test_folder = os.path.join(data_path, 'test')
data_folders = [train_folder, test_folder]
class_folders = ['Help', 'Hinder', 'Physical']

if mix_bool:
    output_data_path_name = 'data_' + str(vid_length) + '_mod' + '_mix'
    output_data_path = os.path.join('./', output_data_path_name)
    output_train_folder = os.path.join(output_data_path, 'train')
    output_test_folder = os.path.join(output_data_path, 'test')
    output_data_folders = [output_train_folder, output_test_folder]
else:
    output_data_path_name = 'data_' + str(vid_length) + '_mod' + '_single'
    output_data_path = os.path.join('./', output_data_path_name)
    output_train_folder = os.path.join(output_data_path, 'train')
    output_test_folder = os.path.join(output_data_path, 'test')
    output_data_folders = [output_train_folder, output_test_folder]

# Create output directories if needed
if not os.path.exists(output_data_path):
    os.mkdir(output_data_path)
for folder in output_data_folders:
    if not os.path.exists(folder):
        os.mkdir(folder)
    for class_name in class_folders:
        class_out_path = os.path.join(folder, class_name)
        if not os.path.exists(class_out_path):
            os.mkdir(class_out_path)

# Define augmentation parameter lists
angles = [90, 180, 270]  
# For lighting, these are added to 1 (e.g., 1 + 0.1 = 1.1 means a slight brightness boost)
lighting = [0.1, 0.3, 0.5, 0.7, 0.9]  
zoom = [0.5, 0.7, 0.9]

# Helper function to apply all three torchvision transforms
def apply_torchvision_transforms(frame, angle, brightness_factor, zoom_factor):
    # Convert BGR to RGB then to PIL image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Rotate image
    img = TF.rotate(img, angle)

    # Adjust brightness
    img = TF.adjust_brightness(img, brightness_factor)

    # Resize image for zoom
    # PIL's size returns (width, height)
    orig_width, orig_height = img.size
    new_width = int(orig_width * zoom_factor)
    new_height = int(orig_height * zoom_factor)
    img = TF.resize(img, (new_height, new_width))

    # Convert back to OpenCV BGR image
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# Process each video in both train and test folders
for idx, data_folder in enumerate(data_folders):
    output_data_folder = output_data_folders[idx]

    for class_folder in class_folders:
        class_path = os.path.join(data_folder, class_folder)
        output_class_path = os.path.join(output_data_folder, class_folder)

        for video in os.listdir(class_path):
            video_path = os.path.join(class_path, video)
            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret: 
                    break
                frames.append(frame)
            cap.release()

            if mix_bool:
                angle_idx = random.choice(range(len(angles)))
                light_idx = random.choice(range(len(lighting)))
                zoom_idx = random.choice(range(len(zoom)))
                
                chosen_angle = angles[angle_idx]
                chosen_light = lighting[light_idx]
                chosen_zoom = zoom[zoom_idx]
                brightness_factor = 1 + chosen_light

                # Apply all augmentations to each frame
                frames_aug = [
                    apply_torchvision_transforms(frame, chosen_angle, brightness_factor, chosen_zoom)
                    for frame in frames
                ]

                new_video_path = os.path.join(
                    output_class_path,
                    f'{video[:-4]}_mix_{angle_idx}_{light_idx}_{zoom_idx}.mp4'
                )
            else:
                aug_idx = random.choice([0, 1, 2])
                frames_aug = []
                if aug_idx == 0:
                    # Rotation only
                    angle_idx = random.choice(range(len(angles)))
                    chosen_angle = angles[angle_idx]

                    for frame in frames:
                        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        img = TF.rotate(img, chosen_angle)
                        frame_aug = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                        frames_aug.append(frame_aug)

                    new_video_path = os.path.join(output_class_path, f'{video[:-4]}_single_0_{angle_idx}.mp4')
                elif aug_idx == 1:
                    # Lighting only
                    light_idx = random.choice(range(len(lighting)))
                    chosen_light = lighting[light_idx]
                    brightness_factor = 1 + chosen_light

                    for frame in frames:
                        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        img = TF.adjust_brightness(img, brightness_factor)
                        frame_aug = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                        frames_aug.append(frame_aug)

                    new_video_path = os.path.join(output_class_path, f'{video[:-4]}_single_1_{light_idx}.mp4')
                elif aug_idx == 2:
                    # Zoom only
                    zoom_idx = random.choice(range(len(zoom)))
                    chosen_zoom = zoom[zoom_idx]

                    for frame in frames:
                        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        orig_width, orig_height = img.size
                        new_width = int(orig_width * chosen_zoom)
                        new_height = int(orig_height * chosen_zoom)
                        img = TF.resize(img, (new_height, new_width))
                        frame_aug = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                        frames_aug.append(frame_aug)

                    new_video_path = os.path.join(output_class_path, f'{video[:-4]}_single_2_{zoom_idx}.mp4')

            # Write out the augmented video (using the size of the first frame)
            height, width = frames_aug[0].shape[:2]
            out = cv2.VideoWriter(new_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

            for frame in frames_aug:
                out.write(frame)

            out.release()