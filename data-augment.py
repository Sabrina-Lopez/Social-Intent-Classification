import os
import cv2
import numpy as np
import random


random.seed(0)
np.random.seed(0)


# Video length to get all train/test videos of that length
vid_length = 100
# Boolean if want all combinations of augmentations to be applied to the videos or just once at a time
mix_bool = True


data_path_name = 'data_' + str(vid_length) + '_unmod'
data_path = os.path.join('./', data_path_name)
train_folder = os.path.join(data_path, 'train')
test_folder = os.path.join(data_path, 'test')
data_folders = [train_folder, test_folder]
class_folders = ['Help', 'Hinder', 'Physical']

output_data_path_name = None
output_data_path = None
output_train_folder = None
output_test_folder = None
output_data_folders = None

if mix_bool:
    output_data_path_name = 'data_' + str(vid_length) + '_mod' + '_mix'
    output_data_path = os.path.join('./', output_data_path_name)
    output_train_folder = os.path.join(output_data_path, 'train')
    output_test_folder = os.path.join(output_data_path, 'test')
    output_data_folders = [output_train_folder, output_test_folder]

    if not os.path.exists(output_data_path):
        os.mkdir(output_data_path)

    if not os.path.exists(output_train_folder):
        os.mkdir(output_train_folder)

    if not os.path.exists(output_test_folder):
        os.mkdir(output_test_folder)
else:
    output_data_path_name = 'data_' + str(vid_length) + '_mod' + '_single'
    output_data_path = os.path.join('./', output_data_path_name)
    output_train_folder = os.path.join(output_data_path, 'train')
    output_test_folder = os.path.join(output_data_path, 'test')
    output_data_folders = [output_train_folder, output_test_folder]

    if not os.path.exists(output_data_path):
        os.mkdir(output_data_path)

    if not os.path.exists(output_train_folder):
        os.mkdir(output_train_folder)

    if not os.path.exists(output_test_folder):
        os.mkdir(output_test_folder)

for class_name in class_folders:
    if not os.path.exists(os.path.join(output_train_folder, class_name)):
        os.mkdir(os.path.join(output_train_folder, class_name))
    
    if not os.path.exists(os.path.join(output_test_folder, class_name)):
        os.mkdir(os.path.join(output_test_folder, class_name))


angle_map = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE
}
lighting = [0.1, 0.3, 0.5, 0.7, 0.9]
zoom = [0.5, 0.7, 0.9]
augmentations = [angle_map, lighting, zoom]

num_help, num_hinder, num_phy = 12, 7, 9

# For each video in the train and test directories, create new videos for each of the augmentation options and save them to data directory
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
                if not ret: break
                frames.append(frame)
            cap.release()

            if mix_bool:
                # Randomly choose one rotation, one lighting value, and one zoom value
                chosen_angle = random.choice(list(angle_map.keys()))
                chosen_light = random.choice([0, 1, 2, 3, 4])
                chosen_zoom = random.choice([0, 1, 2])

                # First, apply the rotation to all frames
                frames_aug = [cv2.rotate(frame, angle_map[chosen_angle]) for frame in frames]

                # Then, apply the lighting augmentation using cv2.addWeighted
                frames_aug = [
                    cv2.addWeighted(frame, 1 + lighting[chosen_light], np.zeros(frame.shape, frame.dtype), 0, 0)
                    for frame in frames_aug
                ]

                # Finally, apply the zoom augmentation
                frames_aug = [
                    cv2.resize(frame, (int(frame.shape[1] * zoom[chosen_zoom]), int(frame.shape[0] * zoom[chosen_zoom])))
                    for frame in frames_aug
                ]

                new_video_path = os.path.join(output_class_path, f'{video[:-4]}_mix_{chosen_angle}_{chosen_light}_{chosen_zoom}.mp4')
            else:
                aug_idx = random.choice([0, 1, 2])
                if aug_idx == 0:
                    # Rotation only
                    chosen_angle = random.choice(list(angle_map.keys()))
                    frames_aug = [cv2.rotate(frame, angle_map[chosen_angle]) for frame in frames]
                    new_video_path = os.path.join(output_class_path, f'{video[:-4]}_single_{0}_{aug_idx}.mp4')
                elif aug_idx == 1:
                    # Lighting only
                    chosen_light = random.choice([0, 1, 2, 3, 4])
                    frames_aug = [
                        cv2.addWeighted(frame, 1 + lighting[chosen_light], np.zeros(frame.shape, frame.dtype), 0, 0)
                        for frame in frames
                    ]
                    new_video_path = os.path.join(output_class_path, f'{video[:-4]}_single_{1}_{chosen_light}.mp4')
                elif aug_idx == 2:
                    # Zoom only
                    chosen_zoom = random.choice([0, 1, 2])
                    frames_aug = [
                        cv2.resize(frame, (int(frame.shape[1] * zoom[chosen_zoom]), int(frame.shape[0] * zoom[chosen_zoom])))
                        for frame in frames
                    ]
                    new_video_path = os.path.join(output_class_path, f'{video[:-4]}_single_{2}_{chosen_zoom}.mp4')

            out = cv2.VideoWriter(new_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frames_aug[0].shape[1], frames_aug[0].shape[0]))
            for frame in frames_aug:
                out.write(frame)
            out.release()