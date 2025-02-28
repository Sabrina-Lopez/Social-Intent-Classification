import av
import torch
import numpy as np
import os
from transformers import AutoImageProcessor, TimesformerForVideoClassification, TimesformerConfig
from utils import sample_frame_indices, read_video_pyav


np.random.seed(0)

# Retrieve a video clip example
file_path = os.path.join('./data', 'test', 'Help', '1_ab_WS_Help_V_100.mp4')
container = av.open(file_path)

# Sample 8 frames
indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
video = read_video_pyav(container, indices)


# Define custom labels
custom_labels = ["help", "hinder", "physical"]
num_custom_labels = len(custom_labels)

# Create label2id and id2label dicts
label2id = {"help": 0, "hinder": 1, "physical": 2}
id2label = {0: "help", 1: "hinder", 2: "physical"}

# Define the desired base model
base_model_name = "facebook/timesformer-base-finetuned-k400"

# Load the base config from the pretrained checkpoint
config = TimesformerConfig.from_pretrained(base_model_name)

# Update number of labels and label mappings
config.num_labels = num_custom_labels
config.id2label = id2label
config.label2id = label2id

# Get the pretrained image processor and model
image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
model = TimesformerForVideoClassification.from_pretrained(base_model_name, config=config, ignore_mismatched_sizes=True)


inputs = image_processor(list(video), return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)  # Forward pass
    logits = outputs.logits
    predicted_id = logits.argmax(-1).item()

predicted_label = model.config.id2label[predicted_id]
print("Predicted label:", predicted_label)