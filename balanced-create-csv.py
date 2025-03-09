import os
import cv2
import datetime
import csv


modBool = False
mixBool = False

data_path_name = None
train_f1_data_csv_name = None
train_f2_data_csv_name = None
train_f3_data_csv_name = None
train_f4_data_csv_name = None
test_data_csv_name = None 


fold_dict = {0: 'fold1', 1: 'fold2', 2: 'fold3', 3: 'fold4'}


if modBool:
  if mixBool:
    data_path_name = 'balanced_data_mod_mix'
    train_f1_data_csv_name = 'balanced_train_f1_data_mod_mix.csv'
    train_f2_data_csv_name = 'balanced_train_f2_data_mod_mix.csv'
    train_f3_data_csv_name = 'balanced_train_f3_data_mod_mix.csv'
    train_f4_data_csv_name = 'balanced_train_f4_data_mod_mix.csv'
    test_data_csv_name =  'balanced_test_data_mod_mix.csv'
  else:
    data_path_name = 'balanced_data_mod_single'
    train_f1_data_csv_name = 'balanced_train_f1_data_mod_single.csv'
    train_f2_data_csv_name = 'balanced_train_f2_data_mod_single.csv'
    train_f3_data_csv_name = 'balanced_train_f3_data_mod_single.csv'
    train_f4_data_csv_name = 'balanced_train_f4_data_mod_single.csv'
    test_data_csv_name =  'balanced_test_data_mod_single.csv'
else:
  data_path_name = 'balanced_data_unmod'
  train_f1_data_csv_name = 'balanced_train_f1_data_unmod.csv'
  train_f2_data_csv_name = 'balanced_train_f2_data_unmod.csv'
  train_f3_data_csv_name = 'balanced_train_f3_data_unmod.csv'
  train_f4_data_csv_name = 'balanced_train_f4_data_unmod.csv'
  test_data_csv_name =  'balanced_test_data_unmod.csv'


dataset = os.path.join('./', data_path_name)


# Define output CSV files for train, test, and validation
csv_outputs = {
    'train_f1': train_f1_data_csv_name,
    'train_f2': train_f2_data_csv_name,
    'train_f3': train_f3_data_csv_name,
    'train_f4': train_f4_data_csv_name,
    'test': test_data_csv_name
}


data_dirs = {
    'train_f1': os.path.join(dataset, 'train', 'fold1'),
    'train_f2': os.path.join(dataset, 'train', 'fold2'),
    'train_f3': os.path.join(dataset, 'train', 'fold3'),
    'train_f4': os.path.join(dataset, 'train', 'fold4'),
    'test': os.path.join(dataset, 'test')
}


# Create CSV files and write headers
for key, csv_output in csv_outputs.items():
  data_dir = data_dirs[key]
  with open(csv_output, 'w', newline='') as file:
    writer = csv.writer(file)
    field = ['video_path', 'start', 'end', 'label']
    writer.writerow(field)

    for cur_class in os.listdir(data_dir):
      cur_video_class = os.path.join(data_dir, cur_class)

      if not os.path.isdir(cur_video_class):
        continue
      
      for video in os.listdir(cur_video_class):
        cur_video = os.path.join(cur_video_class, video)
        
        # Check if the file exists
        if not os.path.exists(cur_video):
            print('File not found: {}'.format(cur_video))
            continue
        
        cap_video = cv2.VideoCapture(cur_video)
        
        frames = cap_video.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap_video.get(cv2.CAP_PROP_FPS)

        if fps == 0:
          print(f"FPS is 0 for video: {cur_video}")
          continue
        
        seconds = round(frames / fps) # Convert to total seconds
        video_time = datetime.timedelta(seconds=seconds)
        seconds_as_float = float(video_time.total_seconds())
        
        start_time = 0.0

        label = None
        if cur_class == 'Help': label = 0
        elif cur_class == 'Hinder': label = 1
        elif cur_class == 'Physical': label = 2
        
        writer.writerow([cur_video, start_time, seconds_as_float, label])