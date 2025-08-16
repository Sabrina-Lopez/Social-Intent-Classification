import subprocess
import itertools
from collections import defaultdict
import wandb
from dotenv import load_dotenv
import os


load_dotenv()
wandb.login(key = os.environ.get("WANDB_API_KEY"), relogin=True)


def main():
    api = wandb.Api()
    project_path = "sabrinameganlopez015-live-robotics/Fine-tuned Intent Classification 6-Model Training"

    project_exists = False
    # Figure out once whether the project actually exists
    try:
        _ = api.project(project_path)
        project_exists = True
    except ValueError:
        project_exists = False

    # Define the parameter lists
    batch_sizes = [
        4,
        # 8,
        # 16
        ]
    # batch_sizes.reverse()
    epoch_lens = [
        60]
    ks = [1]
    data_files = [
        "train_data_100_unmod_latest.csv"
    ]
    model_names = [
        # "facebook/timesformer-base-finetuned-k600",
        # "google/vivit-b-16x2-kinetics400",
        # "google/vivit-b-16x2-kinetics400-fe", # Not available via Hugging Face
        # "facebook/timesformer-base-finetuned-k400",
        # "facebook/timesformer-base-finetuned-ssv2",
        "facebook/vjepa2-vitl-fpc16-256-ssv2",
        # "facebook/vjepa2-vitl-fpc32-256-diving48"
    ]
    finetunes = [
        "default" # Fine-tuning that involves manually freezing model layers 
    ]
    finetune_scripts = [
        # "finetune.py",
        "seedless-finetune.py",
        "seedless-finetune-no-vjepa2.py"
    ]

    hidden_dropout = [0.0, 
                    # 0.05,
                    # 0.1, 
                    # 0.15,
                    ]
    # hidden_dropout.reverse()
    attn_dropout = [0.0, 
                    # 0.05,
                    # 0.1, 
                    # 0.15,
                    ]
    lrs = [# 0.0001,
           0.00005,
           # 0.00001
           ]
    w_decays = [# 0.1,
                0.01,
                # 0.05,
                # 0.001,
                # 0.0001
                ]
    stoch_depths = [0.0,
                    # 0.1,
                    # 0.2,
                    # 0.3
                ]
    label_smooth_vals = [0.0,
                         # 0.015,
                         # 0.03,
                         # 0.05,
                         # 0.1,
                         # 0.15,
                         # 0.2,
                         # 0.25,
                         # 0.3
                        ]

    for i in range(0, 5):
        for model_name, finetune, script, dataset, bs, epochs, k, h_dropout, a_dropout, lr, decay, stoch_depth, label_smooth_val in itertools.product(model_names, finetunes, finetune_scripts, data_files, batch_sizes, epoch_lens, ks, hidden_dropout, attn_dropout, lrs, w_decays, stoch_depths, label_smooth_vals):
            
            if "timesformer" in model_name and "no-vjepa2" not in script: continue
            if "vjepa2" in model_name and "no-vjepa2" in script: continue
            if "vivit" in model_name and "no-vjepa2" not in script: continue

            data_file = {"train": dataset}
            dataset_type = os.path.splitext(data_file["train"])[0]
            
            print(f"\nRunning fine-tune with finetune={finetune}, batch_size={bs}, data_file={dataset}, model_name={model_name}, lr={lr}, decay={decay}, hd={h_dropout}, ad={a_dropout}")

            cmd = [
                "python", script,
                "--batch_size", str(bs),
                "--epochs", str(epochs),
                "--k", str(k),
                "--data_file", dataset,
                "--model_name", model_name,
                "--finetune", finetune,
                "--hidden_dropout", str(h_dropout),
                "--attn_dropout", str(a_dropout),
                "--lr", str(lr),
                "--decay", str(decay),
                "--stoch_depth", str(stoch_depth),
                "--label_smoothing", str(label_smooth_val),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(result.stdout)
            
            if result.stderr:
                print("Error:", result.stderr)  

if __name__ == "__main__":
    main()