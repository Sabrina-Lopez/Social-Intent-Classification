import cv2
import os
import sys

# Specify the video file name or path relative to this script's directory
# video_file = "data_100_mod_single/test/Help/3_ca_SW_Help__r6__100_single_0_0.mp4"  # Replace with your actual video file name
# video_file = "data_100_mod_single/test/Hinder/1_bc_WS_Hinder__r1__100_single_2_1.mp4"
# video_file = "data_100_mod_single/test/Physical/1_ab_WS_Physical__r2__100_single_1_2.mp4"
# video_file = "data_100_mod_mix/test/Help/3_ab_WS_Help__r5__100_mix_270_4_0.mp4"
# video_file = "data_100_mod_mix/test/Hinder/3_ab_WS_Hinder__r5__100_mix_180_0_1.mp4"
# video_file = "data_100_mod_mix/test/Physical/1_ab_WS_Physical__r1__100_mix_90_1_0.mp4"
video_file = "videos/1_bc_SW_Physical/1_bc_SW_Physical__r3__100.mp4" # 340 frames total, 30.0 frame rate

# Get the absolute path of the video file
video_path = os.path.join('./', video_file)

# Check if the video file exists
if not os.path.exists(video_path):
    print(f"Error: The video file '{video_path}' does not exist!")
    sys.exit(1)

# Open the video file using OpenCV
cap = cv2.VideoCapture(video_path)

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(length)

if not cap.isOpened():
    print("Error: Could not open video.")
    sys.exit(1)

print("Press 'q' to quit the video player.")

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if no more frames are available

    # Display the frame in a window named "Video Player"
    cv2.imshow("Video Player", frame)

    # Wait 25ms and check if the 'q' key is pressed to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
