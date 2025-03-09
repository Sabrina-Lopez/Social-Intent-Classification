import os
import shutil
from collections import defaultdict


train_txt_files = ['fold1.txt', 'fold2.txt', 'fold3.txt', 'fold4.txt']
test_txt_file = 'balanced_test.txt'

# Text files
train_txt_paths = [os.path.join('./text-files', 'balanced', file) for file in train_txt_files]
test_txt_path = os.path.join('./text-files', 'balanced', test_txt_file)

train_vids_names = defaultdict(list)
train_vids_labels = defaultdict(list)

test_vids_names = []
test_vids_labels = []


# Open and read the train.txt file, saving the video name as an array
for i, path in enumerate(train_txt_paths):
    with open(path, 'r') as train_file:
        for line in train_file:
            temp = line.strip()
            name, label = temp.split(' ')
            train_vids_names[i].append(name)
            train_vids_labels[i].append(label)

# Open and read the test.txt file, saving the video name as an array
with open(test_txt_path, 'r') as test_file:
    for line in test_file:
        temp = line.strip()
        name, label = temp.split(' ')
        test_vids_names.append(name)
        test_vids_labels.append(label)


# Path to all the videos and path to save the organized dataset for later use
vid_path = os.path.join('videos')
all_vid_folders = os.listdir(vid_path)

data_path_name = 'balanced_data_unmod'
data_path = os.path.join('./', data_path_name)

if not os.path.exists(data_path):
    os.mkdir(data_path)

train_folder = os.path.join(data_path, 'train')
test_folder = os.path.join(data_path, 'test')

if not os.path.exists(train_folder):
    os.mkdir(train_folder)

if not os.path.exists(test_folder):
    os.mkdir(test_folder)


output_fold_folder_paths = []


for i, fold_file in enumerate(train_txt_files):
    fold_file_name = os.path.splitext(fold_file)[0]
    fold_folder = os.path.join(train_folder, fold_file_name)

    if not os.path.exists(fold_folder):
        os.mkdir(fold_folder)
    
    output_fold_folder_paths.append(fold_folder)


train_vid_paths, test_vid_paths = defaultdict(list), []


# For each video listed in the train text file, find its path from the videos directory
for i in range(len(train_txt_files)):
    for j, vid_name in enumerate(train_vids_names[i]):
        concat_vid_name = None
        if int(train_vids_labels[i][j]) == 0:
            concat_vid_name = vid_name[:12]
        elif int(train_vids_labels[i][j]) == 1:
            concat_vid_name = vid_name[:14]
        elif int(train_vids_labels[i][j]) == 2:
            concat_vid_name = vid_name[:16]
        else: print('The label is not valid or is missing.')

        # print('Video name:', vid_name)
        # print('Folder name:', concat_vid_name)

        if concat_vid_name in all_vid_folders:
            folder_path = os.path.join(vid_path, concat_vid_name)
            for vid in os.listdir(folder_path):
                vid = os.path.splitext(vid)[0]
                if vid == vid_name:
                    # print('Found the video:', vid)
                    train_vid_paths[i].append(os.path.join(folder_path, vid + '.mp4'))
                    break
        else: print('The video name is not in the data folders.')

# For each video listed in the test file, find its path from the videos directory
for i, vid_name in enumerate(test_vids_names):
    concat_vid_name = None
    if int(test_vids_labels[i]) == 0:
        concat_vid_name = vid_name[:12]
    elif int(test_vids_labels[i]) == 1:
        concat_vid_name = vid_name[:14]
    elif int(test_vids_labels[i]) == 2:
        concat_vid_name = vid_name[:16]
    else: print('The label is not valid or is missing.')

    # print('Video name:', vid_name)
    # print('Folder name:', concat_vid_name)

    if concat_vid_name in all_vid_folders:
        folder_path = os.path.join(vid_path, concat_vid_name)
        for vid in os.listdir(folder_path):
            vid = os.path.splitext(vid)[0]
            if vid == vid_name:
                # print('Found the video:', vid)
                test_vid_paths.append(os.path.join(folder_path, vid + '.mp4'))
                break
    else: print('The video name is not in the data folders.')


fold_dict = {0: 'fold1', 1: 'fold2', 2: 'fold3', 3: 'fold4'}


# Create the data directory for all the train and test videos specified from the text files
for i, _ in enumerate(train_vid_paths):
    for j, vid_path in enumerate(train_vid_paths[i]):
        label = None
        if int(train_vids_labels[i][j]) == 0: label = 'Help'
        elif int(train_vids_labels[i][j]) == 1: label = 'Hinder'
        elif int(train_vids_labels[i][j]) == 2: label = 'Physical'

        vid_name = os.path.basename(vid_path)

        # print('Video name:', vid_name)

        if not os.path.exists(os.path.join(output_fold_folder_paths[i], label)):
            os.mkdir(os.path.join(output_fold_folder_paths[i], label))

        shutil.copy(vid_path, os.path.join(output_fold_folder_paths[i], label, vid_name))


for i, vid_path in enumerate(test_vid_paths):
    label = None
    if int(test_vids_labels[i]) == 0: label = 'Help'
    elif int(test_vids_labels[i]) == 1: label = 'Hinder'
    elif int(test_vids_labels[i]) == 2: label = 'Physical'

    vid_name = os.path.basename(vid_path)

    # print('Video name:', vid_name)

    if not os.path.exists(os.path.join(test_folder, label)):
        os.mkdir(os.path.join(test_folder, label))

    shutil.copy(vid_path, os.path.join(test_folder, label, vid_name))