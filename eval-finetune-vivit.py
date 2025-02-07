import numpy as np
from transformers import VivitImageProcessor, VivitForVideoClassification, VivitConfig
from utils import preprocess_function, custom_collate_fn, evaluate
from torch.utils.data import DataLoader
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
import random
import torch
import os
import matplotlib.pyplot as plt


seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Retrieve and enable the loading of the test data
data_files = {"test": "test_data_100_mod_mix.csv"}
data_csv = load_dataset('csv', data_files=data_files, sep=',')

data_csv["test"] = data_csv["test"].map(preprocess_function)
test_dataset = data_csv["test"].with_format(
    "torch", columns=["video_path", "start", "end", "label"]
)

batch_size = 4
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn
)


dataset_type = os.path.splitext(data_files["test"])[0]


# Define custom labels
custom_labels = ["help", "hinder", "physical"]
num_custom_labels = len(custom_labels)

# Create label2id and id2label dicts
label2id = {"help": 0, "hinder": 1, "physical": 2}
id2label = {0: "help", 1: "hinder", 2: "physical"}

# Load the base config from the pretrained checkpoint
base_model_name = "vivit-finetuned-train" + dataset_type[4:]

# Load the base config from the pretrained checkpoint
config = VivitConfig.from_pretrained(base_model_name)

# Update number of labels and label mappings
config.num_labels = num_custom_labels
config.id2label = id2label
config.label2id = label2id
config.num_frames = 16

# Get the pretrained image processor and model
image_processor = VivitImageProcessor.from_pretrained(base_model_name)
model = VivitForVideoClassification.from_pretrained(base_model_name, config=config, ignore_mismatched_sizes=True)


# Evaluate the testing dataset
all_preds, all_labels, losses, accuracies = evaluate(model, test_loader, image_processor)

accuracy = accuracy_score(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=["help", "hinder", "physical"])

print("Accuracy:", accuracy)
print("Classification Report:")
print(report)


# Plot and save loss/accuracy graph for testing
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(losses) + 1), losses, 'r-o', label='Test Loss')
plt.plot(range(1, len(accuracies) + 1), accuracies, 'b-o', label='Test Accuracy')
plt.xlabel('Batch Number')
plt.ylabel('Metric Value')
plt.title(f'Evaluation - Testing Loss and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(f'Finetune_Vivit_{dataset_type}_metrics.png')
plt.close()  # Close the figure to free memory