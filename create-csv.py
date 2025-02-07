import os
import cv2
import datetime
import csv


vid_length = 100
modBool = True
mixBool = True

data_path_name = None
train_data_csv_name = None
test_data_csv_name = None 

if modBool:
  if mixBool:
    data_path_name = 'data_' + str(vid_length) + '_mod' + '_mix'
    train_data_csv_name = 'train_data_' + str(vid_length) + '_mod' + '_mix' + '.csv'
    test_data_csv_name =  'test_data_' + str(vid_length) + '_mod' + '_mix' + '.csv'
  else:
    data_path_name = 'data_' + str(vid_length) + '_mod' + '_single'
    train_data_csv_name = 'train_data_' + str(vid_length) + '_mod' + '_single' + '.csv'
    test_data_csv_name =  'test_data_' + str(vid_length) + '_mod' + '_single' + '.csv'
else:
  data_path_name = 'data_' + str(vid_length) + '_unmod'
  train_data_csv_name = 'train_data_' + str(vid_length) + '_unmod' + '.csv'
  test_data_csv_name =  'test_data_' + str(vid_length) + '_unmod' + '.csv'

dataset = os.path.join('./', data_path_name)

# Define output CSV files for train, test, and validation
csv_outputs = {
    'train': train_data_csv_name,
    'test': test_data_csv_name
}


# Create CSV files and write headers
for data_type, csv_output in csv_outputs.items():
  with open(csv_output, 'w', newline='') as file:
    writer = csv.writer(file)
    field = ['video_path', 'start', 'end', 'label']
    writer.writerow(field)

    cur_data = os.path.join(dataset, data_type)
    
    for cur_class in os.listdir(cur_data):  # This is the label for all the videos in the third for-loop
      cur_video_class = os.path.join(dataset, data_type, cur_class)
      
      for video in os.listdir(cur_video_class):
        cur_video = os.path.join(dataset, data_type, cur_class, video)
        
        # Check if the file exists
        if not os.path.exists(cur_video):
            print('File not found: {}'.format(cur_video))
            continue
        
        cap_video = cv2.VideoCapture(cur_video)
        
        frames = cap_video.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap_video.get(cv2.CAP_PROP_FPS)
        
        seconds = round(frames / fps) # Convert to total seconds
        video_time = datetime.timedelta(seconds=seconds)
        seconds_as_float = float(video_time.total_seconds())
        
        start_time = 0.0

        label = None
        if cur_class == 'Help': label = 0
        elif cur_class == 'Hinder': label = 1
        elif cur_class == 'Physical': label = 2
        
        path = '{}/{}/{}/{}'.format(dataset, data_type, cur_class, video)
        writer.writerow([f'{path}', f'{start_time}', f'{seconds_as_float}', f'{label}'])