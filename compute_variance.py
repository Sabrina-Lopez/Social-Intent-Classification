import wandb
import numpy as np
from collections import defaultdict
from dotenv import load_dotenv
import os

load_dotenv()
wandb.login(key=os.environ.get("WANDB_API_KEY"))

api = wandb.Api()

train_project = "sabrinameganlopez015-live-robotics/Fine-tuned Intent Classification 6-Model Training"
test_project = "sabrinameganlopez015-live-robotics/Fine-tuned Intent Classification 6-Model Testing"

# ========== Helper Functions ==========

def extract_test_base_name(model_name):
    return model_name.rsplit("_", 1)[0]

def compute_variance_summary(groups):
    summary = []
    for key, values in groups.items():
        np_vals = np.array(values)
        if len(np_vals) > 1:
            summary.append((key, np.mean(np_vals), np.var(np_vals, ddof=1)))  # sample variance
    return sorted(summary, key=lambda x: x[1], reverse=True)

def print_grouped_variance(title, summary):
    print(f"=== {title.upper()} ===")
    for key, mean, var in summary:
        print(f"{key}: Mean = {mean:.5f}, Variance = {var:.6f}")
    print()

# ========== TRAINING / VALIDATION VARIANCE ==========

runs = api.runs(train_project)
grouped_train = defaultdict(list)
grouped_val = defaultdict(list)

for run in runs:
    config = run.config
    if all(k in config for k in ["model_name", "batch_size", "attn_dropout", "hidden_dropout", "lr", "w_decay"]):
        model_name = config["model_name"]

        model_type = None
        if "vivit" in model_name.lower():
            model_type = "vivit"
        elif "timesformer" in model_name.lower():
            model_type = "timesformer"
        elif "vjepa2" in model_name.lower():
            model_type = "vjepa2"
        else:
            continue

        group_key = (
            model_type,
            model_name,
            config["batch_size"],
            config["attn_dropout"],
            config["hidden_dropout"],
            config["lr"],
            config["w_decay"]
        )

        history = run.history(keys=["train_accuracy", "val_accuracy"])
        if not history.empty:
            if "train_accuracy" in history.columns:
                train_val = history["train_accuracy"].dropna().values
                if len(train_val) > 0:
                    grouped_train[group_key].append(train_val[-1])
            if "val_accuracy" in history.columns:
                val_val = history["val_accuracy"].dropna().values
                if len(val_val) > 0:
                    grouped_val[group_key].append(val_val[-1])

# Sort and print training/validation variance
train_vivit = compute_variance_summary({k: v for k, v in grouped_train.items() if k[0] == "vivit"})
train_times = compute_variance_summary({k: v for k, v in grouped_train.items() if k[0] == "timesformer"})
train_vjepa2 = compute_variance_summary({k: v for k, v in grouped_train.items() if k[0] == "vjepa2"})
val_vivit = compute_variance_summary({k: v for k, v in grouped_val.items() if k[0] == "vivit"})
val_times = compute_variance_summary({k: v for k, v in grouped_val.items() if k[0] == "timesformer"})
val_vjepa2 = compute_variance_summary({k: v for k, v in grouped_val.items() if k[0] == "vjepa2"})

print_grouped_variance("ViViT Train Accuracy", train_vivit)
print_grouped_variance("ViViT Validation Accuracy", val_vivit)
print_grouped_variance("TimeSformer Train Accuracy", train_times)
print_grouped_variance("TimeSformer Validation Accuracy", val_times)
print_grouped_variance("VJEPA2 Train Accuracy", train_vjepa2)
print_grouped_variance("VJEPA2 Validation Accuracy", val_vjepa2)

# ========== TESTING VARIANCE ==========

test_runs = api.runs(test_project)
grouped_test = defaultdict(list)

for run in test_runs:
    config = run.config
    if "model_name" not in config:
        continue

    model_name = config["model_name"]
    model_type = None
    if "vivit" in model_name.lower():
        model_type = "vivit"
    elif "timesformer" in model_name.lower():
        model_type = "timesformer"
    elif "vjepa2" in model_name.lower():
        model_type = "vjepa2"
    else:
        continue

    base_name = extract_test_base_name(model_name)
    history = run.history(keys=["accuracy"])
    if not history.empty and "accuracy" in history.columns:
        final_acc = history["accuracy"].dropna().values
        if len(final_acc) > 0:
            grouped_test[(model_type, base_name)].append(final_acc[-1])

test_vivit = compute_variance_summary({k: v for k, v in grouped_test.items() if k[0] == "vivit"})
test_times = compute_variance_summary({k: v for k, v in grouped_test.items() if k[0] == "timesformer"})
test_vjepa2 = compute_variance_summary({k: v for k, v in grouped_test.items() if k[0] == "vjepa2"})

print_grouped_variance("ViViT Test Accuracy", test_vivit)
print_grouped_variance("TimeSformer Test Accuracy", test_times)
print_grouped_variance("VJEPA2 Test Accuracy", test_vjepa2)